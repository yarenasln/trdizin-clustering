import os
import json
import time
import requests
import pandas as pd


# ============================================================
# AYARLAR
# ============================================================

API_URL = (
    "https://search.trdizin.gov.tr/"
    "api/defaultSearch/publication/"
)

TARGET_PER_SUBJECT = 150

PAGE_SIZE = 50

REQUEST_SLEEP = 0.25

OUTPUT_DIR = "data"

ARTICLE_FILE = os.path.join(
    OUTPUT_DIR,
    "balanced_articles.csv"
)

SUBJECT_FILE = os.path.join(
    OUTPUT_DIR,
    "article_subjects.csv"
)

BALANCE_FILE = os.path.join(
    OUTPUT_DIR,
    "subject_balance_report.csv"
)


# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================

def clean_text(value):

    if value is None:
        return ""

    return str(value).strip()


def is_leaf_subject(subject_path):

    parts = [
        part.strip()
        for part in str(subject_path).split(">")
        if part.strip()
    ]

    # Fen > Tıp > Onkoloji gibi
    return len(parts) >= 3


def get_subject_buckets():

    params = {
        "q": "",
        "order": "publicationYear-DESC",
        "page": 1,
        "limit": 1
    }

    response = requests.get(
        API_URL,
        params=params,
        timeout=60
    )

    response.raise_for_status()

    data = response.json()

    subject_facet = (
        data
        .get("aggregations", {})
        .get("facet-subject", {})
    )

    if "buckets" in subject_facet:

        buckets = subject_facet["buckets"]

    elif "values" in subject_facet:

        buckets = (
            subject_facet
            .get("values", {})
            .get("buckets", [])
        )

    else:

        buckets = []

    return buckets


# ============================================================
# MAKALE METNİ SEÇ
# ============================================================
#
# Öncelik:
#
# 1. Türkçe
# 2. İngilizce
# 3. Varsa diğer ilk dil
#
# Ortak embedding aşamasında bu tekilleştirilmiş
# metni kullanacağız.
# ============================================================

def choose_article_text(source):

    abstracts = (
        source.get("abstracts")
        or []
    )

    usable = []

    for item in abstracts:

        language = clean_text(
            item.get("language")
        ).upper()

        title = clean_text(
            item.get("title")
        )

        abstract = clean_text(
            item.get("abstract")
        )

        keywords = (
            item.get("keywords")
            or []
        )

        if not title:
            continue

        usable.append(
            {
                "language":
                    language,

                "title":
                    title,

                "abstract":
                    abstract,

                "keywords":
                    keywords
            }
        )

    if not usable:

        return {
            "language": "",
            "title": "",
            "abstract": "",
            "keywords": []
        }

    tur = [
        row
        for row in usable
        if row["language"] == "TUR"
    ]

    eng = [
        row
        for row in usable
        if row["language"] == "ENG"
    ]

    if tur:

        return tur[0]

    if eng:

        return eng[0]

    return usable[0]


# ============================================================
# MAKALEYİ ORTAK FORMATA ÇEVİR
# ============================================================

def build_article_record(source):

    text = choose_article_text(
        source
    )

    keywords = (
        text["keywords"]
        if isinstance(
            text["keywords"],
            list
        )
        else []
    )

    keywords_text = ", ".join(
        str(keyword)
        for keyword in keywords
    )

    embedding_text = (
        text["title"].strip()
        + ". "
        + text["abstract"].strip()
        + ". Keywords: "
        + keywords_text.strip()
    ).strip()

    return {
        "external_id":
            clean_text(
                source.get("id")
            ),

        "doi":
            clean_text(
                source.get("doi")
            ),

        "publication_year":
            source.get(
                "publicationYear"
            ),

        "publication_type":
            clean_text(
                source.get(
                    "publicationType"
                )
            ),

        "language":
            text["language"],

        "title":
            text["title"],

        "abstract":
            text["abstract"],

        "keywords":
            json.dumps(
                keywords,
                ensure_ascii=False
            ),

        "embedding_text":
            embedding_text
    }


# ============================================================
# MAKALEDEKİ TÜM LEAF ETİKETLERİ
# ============================================================

def extract_leaf_subjects(source):

    subjects = (
        source.get("subjects")
        or []
    )

    rows = []

    for subject in subjects:

        if not isinstance(
            subject,
            dict
        ):
            continue

        fullname = clean_text(
            subject.get("fullName")
        )

        if not fullname:
            continue

        if not is_leaf_subject(
            fullname
        ):
            continue

        rows.append(
            {
                "subject_id":
                    subject.get("id"),

                "subject_name":
                    clean_text(
                        subject.get("name")
                    ),

                "subject_fullname":
                    fullname,

                "root_name":
                    clean_text(
                        subject.get("rootName")
                    )
            }
        )

    return rows


# ============================================================
# SUBJECT LİSTESİNİ AL
# ============================================================

print("=" * 110)
print("TR DİZİN DENGELİ ORTAK VERİ SETİ")
print("=" * 110)

print(
    "\nSubject listesi alınıyor..."
)

subject_buckets = (
    get_subject_buckets()
)


leaf_subjects = []


for bucket in subject_buckets:

    subject_path = clean_text(
        bucket.get("key")
    )

    if not subject_path:
        continue

    if not is_leaf_subject(
        subject_path
    ):
        continue

    leaf_subjects.append(
        {
            "subject_fullname":
                subject_path,

            "api_total":
                bucket.get(
                    "doc_count",
                    bucket.get(
                        "count",
                        None
                    )
                )
        }
    )


# Aynı subject tekrar gelirse temizle
subject_map = {}

for row in leaf_subjects:

    subject_map[
        row["subject_fullname"]
    ] = row


leaf_subjects = list(
    subject_map.values()
)


print(
    "Bulunan leaf subject:",
    len(leaf_subjects)
)

print(
    "Konu başına hedef:",
    TARGET_PER_SUBJECT
)


# ============================================================
# ORTAK DEPOLAR
# ============================================================

articles_by_id = {}

subject_memberships = set()

balance_rows = []


# ============================================================
# HER LEAF SUBJECT'TEN DENGELİ VERİ ÇEK
# ============================================================

for subject_index, subject_info in enumerate(
    leaf_subjects,
    start=1
):

    subject_path = (
        subject_info[
            "subject_fullname"
        ]
    )

    print("\n" + "=" * 110)

    print(
        f"[{subject_index}/{len(leaf_subjects)}]"
    )

    print(
        subject_path
    )

    print("-" * 110)


    selected_for_subject = set()

    page = 1

    no_new_counter = 0


    while (
        len(
            selected_for_subject
        )
        <
        TARGET_PER_SUBJECT
    ):

        params = {
            "q": "",
            "order":
                "publicationYear-DESC",

            "page":
                page,

            "limit":
                PAGE_SIZE,

            "facet-subject":
                subject_path
        }


        try:

            response = requests.get(
                API_URL,
                params=params,
                timeout=60
            )

            response.raise_for_status()

            result = (
                response.json()
            )

        except Exception as error:

            print(
                "API hatası:",
                error
            )

            break


        hits = (
            result
            .get("hits", {})
            .get("hits", [])
        )


        if not hits:

            print(
                "Başka yayın bulunamadı."
            )

            break


        before_count = len(
            selected_for_subject
        )


        for hit in hits:

            source = (
                hit.get("_source")
                or {}
            )

            external_id = clean_text(
                source.get("id")
            )


            if not external_id:

                continue


            # ------------------------------------------------
            # Bu subject için zaten seçildiyse tekrar alma
            # ------------------------------------------------

            if (
                external_id
                in
                selected_for_subject
            ):

                continue


            # ------------------------------------------------
            # Makale metni kullanılabilir mi?
            # ------------------------------------------------

            article_record = (
                build_article_record(
                    source
                )
            )


            if (
                not article_record[
                    "title"
                ]
            ):

                continue


            if (
                len(
                    article_record[
                        "embedding_text"
                    ]
                )
                <
                50
            ):

                continue


            # ------------------------------------------------
            # SUBJECT HEDEFİNE EKLE
            # ------------------------------------------------

            selected_for_subject.add(
                external_id
            )


            # ------------------------------------------------
            # GLOBAL MAKALE HAVUZU
            #
            # Aynı makale başka subject'te tekrar gelse bile
            # balanced_articles.csv içinde tek satır olacak.
            # ------------------------------------------------

            if (
                external_id
                not in
                articles_by_id
            ):

                articles_by_id[
                    external_id
                ] = article_record


            # ------------------------------------------------
            # MAKALEDEKİ TÜM LEAF SUBJECTLERİ KORU
            # ------------------------------------------------

            article_subjects = (
                extract_leaf_subjects(
                    source
                )
            )


            for article_subject in article_subjects:

                membership = (
                    external_id,
                    article_subject[
                        "subject_id"
                    ],
                    article_subject[
                        "subject_name"
                    ],
                    article_subject[
                        "subject_fullname"
                    ],
                    article_subject[
                        "root_name"
                    ]
                )

                subject_memberships.add(
                    membership
                )


            # Hedef tamamlandıysa dur
            if (
                len(
                    selected_for_subject
                )
                >=
                TARGET_PER_SUBJECT
            ):

                break


        after_count = len(
            selected_for_subject
        )


        added_this_page = (
            after_count
            -
            before_count
        )


        print(
            f"Sayfa {page} | "
            f"Yeni: {added_this_page} | "
            f"Toplam: {after_count}/"
            f"{TARGET_PER_SUBJECT}"
        )


        # ----------------------------------------------------
        # Aynı sayfalarda dönüp kalma koruması
        # ----------------------------------------------------

        if added_this_page == 0:

            no_new_counter += 1

        else:

            no_new_counter = 0


        if no_new_counter >= 2:

            print(
                "Yeni benzersiz yayın gelmedi."
            )

            break


        page += 1

        time.sleep(
            REQUEST_SLEEP
        )


    # ========================================================
    # SUBJECT RAPORU
    # ========================================================

    selected_count = len(
        selected_for_subject
    )

    balance_rows.append(
        {
            "subject_fullname":
                subject_path,

            "target":
                TARGET_PER_SUBJECT,

            "selected_count":
                selected_count,

            "target_reached":
                (
                    selected_count
                    >=
                    TARGET_PER_SUBJECT
                ),

            "api_total":
                subject_info[
                    "api_total"
                ]
        }
    )


    print(
        "Konu tamamlandı:",
        selected_count
    )


# ============================================================
# DATAFRAME'LER
# ============================================================

articles_df = pd.DataFrame(
    list(
        articles_by_id.values()
    )
)


subject_df = pd.DataFrame(
    list(
        subject_memberships
    ),
    columns=[
        "external_id",
        "subject_id",
        "subject_name",
        "subject_fullname",
        "root_name"
    ]
)


balance_df = pd.DataFrame(
    balance_rows
)


# ============================================================
# SIRALA
# ============================================================

if not articles_df.empty:

    articles_df = (
        articles_df
        .sort_values(
            "external_id"
        )
        .reset_index(
            drop=True
        )
    )


if not subject_df.empty:

    subject_df = (
        subject_df
        .sort_values(
            [
                "external_id",
                "subject_fullname"
            ]
        )
        .reset_index(
            drop=True
        )
    )


if not balance_df.empty:

    balance_df = (
        balance_df
        .sort_values(
            [
                "selected_count",
                "subject_fullname"
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# KAYDET
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


articles_df.to_csv(
    ARTICLE_FILE,
    index=False,
    encoding="utf-8-sig"
)


subject_df.to_csv(
    SUBJECT_FILE,
    index=False,
    encoding="utf-8-sig"
)


balance_df.to_csv(
    BALANCE_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# SONUÇ
# ============================================================

print("\n" + "=" * 110)
print("DENGELİ VERİ ÇEKME TAMAMLANDI")
print("=" * 110)


print(
    "Leaf subject:",
    len(leaf_subjects)
)


print(
    "Benzersiz makale:",
    len(articles_df)
)


print(
    "Makale-leaf konu ilişkisi:",
    len(subject_df)
)


print(
    "Hedefe ulaşan konu:",
    int(
        balance_df[
            "target_reached"
        ].sum()
    )
)


print(
    "Hedefe ulaşamayan konu:",
    int(
        (
            ~balance_df[
                "target_reached"
            ]
        ).sum()
    )
)


print("\nKONU BAŞINA SAYILAR")

print("-" * 80)

print(
    balance_df[
        "selected_count"
    ]
    .describe()
)


print("\nEN AZ VERİ BULUNAN 20 KONU")

print("-" * 110)

print(
    balance_df[
        [
            "subject_fullname",
            "target",
            "selected_count",
            "target_reached"
        ]
    ]
    .head(20)
    .to_string(
        index=False
    )
)


print("\nDosyalar:")

print(
    ARTICLE_FILE
)

print(
    SUBJECT_FILE
)

print(
    BALANCE_FILE
)