import os
import numpy as np
import pandas as pd


# ============================================================
# AYARLAR
# ============================================================

INDEX_FILE = "embeddings/article_embedding_index.csv"
SUBJECT_FILE = "data/article_subjects.csv"

OUTPUT_DIR = "results/kmeans/holdout"

REFERENCE_RATIO = 0.20
RANDOM_STATE = 42


# ============================================================
# VERİYİ OKU
# ============================================================

print("=" * 100)
print("ORTAK %20 REFERANS / %80 TEST HOLD-OUT")
print("=" * 100)

articles = pd.read_csv(
    INDEX_FILE,
    encoding="utf-8-sig"
)

subjects = pd.read_csv(
    SUBJECT_FILE,
    encoding="utf-8-sig"
)


articles["external_id"] = (
    articles["external_id"]
    .astype(str)
)

subjects["external_id"] = (
    subjects["external_id"]
    .astype(str)
)


rng = np.random.default_rng(
    RANDOM_STATE
)


# ============================================================
# HEDEF REFERANS SAYISI
# ============================================================

target_reference_count = int(
    round(
        len(articles)
        *
        REFERENCE_RATIO
    )
)

print(
    "Toplam makale:",
    len(articles)
)

print(
    "Hedef referans:",
    target_reference_count
)


# ============================================================
# HER MAKALE HANGİ KONULARA SAHİP?
# ============================================================

article_subject_map = (
    subjects
    .groupby("external_id")["subject_fullname"]
    .apply(
        lambda x:
            set(
                x.dropna().astype(str)
            )
    )
    .to_dict()
)


# ============================================================
# HER KONU İÇİN ADAY MAKALELER
# ============================================================

subject_candidates = (
    subjects
    .groupby("subject_fullname")["external_id"]
    .apply(
        lambda x:
            list(
                dict.fromkeys(
                    x.astype(str)
                )
            )
    )
    .to_dict()
)


all_subjects = set(
    subject_candidates.keys()
)


# ============================================================
# 1. ÖNCE TÜM KONULARI KAPSAYACAK REFERANS SEÇ
# ============================================================
#
# En az makalesi olan konulardan başlıyoruz.
#
# Bir makale birden fazla konuyu kapsıyorsa avantaj:
# tek makale ile birkaç konu temsil edilebilir.
# ============================================================

reference_ids = set()

covered_subjects = set()


subjects_by_rarity = sorted(
    all_subjects,
    key=lambda subject:
        len(
            subject_candidates[
                subject
            ]
        )
)


for subject_name in subjects_by_rarity:

    if subject_name in covered_subjects:
        continue


    candidates = subject_candidates[
        subject_name
    ]


    if not candidates:
        continue


    # ---------------------------------------------
    # Henüz seçilmemiş adaylar
    # ---------------------------------------------

    available_candidates = [
        article_id
        for article_id in candidates
        if article_id not in reference_ids
    ]


    if not available_candidates:

        available_candidates = (
            candidates
        )


    # ---------------------------------------------
    # En fazla kapsanmamış konuyu aynı anda
    # temsil eden makaleyi seç
    # ---------------------------------------------

    best_candidates = []

    best_score = -1


    for article_id in available_candidates:

        article_topics = (
            article_subject_map
            .get(
                article_id,
                set()
            )
        )


        new_topics = (
            article_topics
            -
            covered_subjects
        )


        score = len(
            new_topics
        )


        if score > best_score:

            best_score = score

            best_candidates = [
                article_id
            ]


        elif score == best_score:

            best_candidates.append(
                article_id
            )


    selected_id = rng.choice(
        best_candidates
    )


    reference_ids.add(
        selected_id
    )


    covered_subjects.update(
        article_subject_map.get(
            selected_id,
            set()
        )
    )


# ============================================================
# 2. REFERANSI TAM %20'YE TAMAMLA
# ============================================================

remaining_ids = (
    articles[
        ~articles[
            "external_id"
        ].isin(
            reference_ids
        )
    ][
        "external_id"
    ]
    .tolist()
)


needed = (
    target_reference_count
    -
    len(reference_ids)
)


if needed < 0:

    raise ValueError(
        "Bütün konuları temsil etmek için "
        "hedef %20'den fazla makale gerekti."
    )


if needed > 0:

    extra_ids = rng.choice(
        remaining_ids,
        size=needed,
        replace=False
    )

    reference_ids.update(
        extra_ids.tolist()
    )


# ============================================================
# 3. REFERENCE / TEST AYIR
# ============================================================

reference = (
    articles[
        articles[
            "external_id"
        ].isin(
            reference_ids
        )
    ]
    .copy()
)


test = (
    articles[
        ~articles[
            "external_id"
        ].isin(
            reference_ids
        )
    ]
    .copy()
)


reference["split"] = (
    "REFERENCE"
)

test["split"] = (
    "TEST"
)


# ============================================================
# 4. KONU KAPSAMASI
# ============================================================

reference_subjects = (
    subjects[
        subjects[
            "external_id"
        ].isin(
            reference[
                "external_id"
            ]
        )
    ]
)


test_subjects = (
    subjects[
        subjects[
            "external_id"
        ].isin(
            test[
                "external_id"
            ]
        )
    ]
)


total_subject_count = (
    subjects[
        "subject_fullname"
    ]
    .nunique()
)


reference_subject_count = (
    reference_subjects[
        "subject_fullname"
    ]
    .nunique()
)


test_subject_count = (
    test_subjects[
        "subject_fullname"
    ]
    .nunique()
)


# ============================================================
# SONUÇ
# ============================================================

print("\n" + "=" * 100)
print("HOLD-OUT SONUCU")
print("=" * 100)


print(
    "Referans makale:",
    len(reference)
)


print(
    "Test makale:",
    len(test)
)


print(
    "Referans oranı:",
    round(
        len(reference)
        /
        len(articles),
        4
    )
)


print(
    "Test oranı:",
    round(
        len(test)
        /
        len(articles),
        4
    )
)


print("\n" + "=" * 100)
print("KONU KAPSAMASI")
print("=" * 100)


print(
    "Toplam leaf konu:",
    total_subject_count
)


print(
    "Referansta temsil edilen:",
    reference_subject_count
)


print(
    "Referansta eksik:",
    total_subject_count
    -
    reference_subject_count
)


print(
    "Testte temsil edilen:",
    test_subject_count
)


print(
    "Testte bulunmayan konu:",
    total_subject_count
    -
    test_subject_count
)


# ============================================================
# KAYDET
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


reference.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "reference.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


test.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "test.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


print("\nDosyalar oluşturuldu:")

print(
    "results/kmeans/holdout/reference.csv"
)

print(
    "results/kmeans/holdout/test.csv"
)