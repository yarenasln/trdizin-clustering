import os
import numpy as np
import pandas as pd
from config.paths import EMBEDDING_FILE, INDEX_FILE

# ============================================================
# AYARLAR
# ============================================================


SUBJECT_FILE = "data/article_subjects.csv"


OUTPUT_DIR = "results/kmeans"

SEEDS_PER_SUBJECT = 5


# ============================================================
# VERİLERİ OKU
# ============================================================

articles = pd.read_csv(
    INDEX_FILE,
    encoding="utf-8-sig"
)

subjects = pd.read_csv(
    SUBJECT_FILE,
    encoding="utf-8-sig"
)

embeddings = np.load(
    EMBEDDING_FILE
).astype(np.float32)


print("=" * 100)
print("SEEDED K-MEANS V2 - TEMSİL EDİCİ SEED SEÇİMİ")
print("=" * 100)

print("Toplam makale:", len(articles))
print("Embedding shape:", embeddings.shape)
print("Makale-konu ilişkisi:", len(subjects))


# ============================================================
# ID TİPLERİNİ EŞİTLE
# ============================================================

articles["external_id"] = (
    articles["external_id"]
    .astype(str)
)

subjects["external_id"] = (
    subjects["external_id"]
    .astype(str)
)


# ============================================================
# EMBEDDING ROW MAP
# ============================================================

id_to_row = (
    articles
    .set_index("external_id")["embedding_row"]
    .to_dict()
)


# ============================================================
# HER MAKALE KAÇ LEAF ETİKETİNE SAHİP?
# ============================================================

label_counts = (
    subjects
    .groupby("external_id")["subject_fullname"]
    .nunique()
    .to_dict()
)


# ============================================================
# TÜM LEAF KONULAR
# ============================================================

leaf_subjects = sorted(
    subjects["subject_fullname"]
    .dropna()
    .astype(str)
    .unique()
)

print("Leaf konu sayısı:", len(leaf_subjects))


# ============================================================
# YARDIMCI FONKSİYON
# TEMSİL EDİCİ MAKALELERİ SEÇ
# ============================================================

def select_representative_articles(candidate_ids, count):

    valid_ids = [
        article_id
        for article_id in candidate_ids
        if article_id in id_to_row
    ]

    if not valid_ids:
        return []

    rows = [
        int(id_to_row[article_id])
        for article_id in valid_ids
    ]

    vectors = embeddings[rows]

    # Aday grubun genel merkezini bul
    center = vectors.mean(axis=0)

    center_norm = np.linalg.norm(center)

    if center_norm > 0:
        center = center / center_norm

    # Cosine similarity için embeddingleri normalize et
    vector_norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True
    )

    vector_norms[
        vector_norms == 0
    ] = 1

    normalized_vectors = (
        vectors / vector_norms
    )

    # Merkeze benzerlik
    similarities = (
        normalized_vectors @ center
    )

    # En yüksek benzerlik = en temsil edici
    order = np.argsort(
        -similarities
    )

    selected_count = min(
        count,
        len(valid_ids)
    )

    selected_ids = [
        valid_ids[i]
        for i in order[
            :selected_count
        ]
    ]

    return selected_ids


# ============================================================
# HER KONU İÇİN SEED SEÇ
# ============================================================

seed_rows = []
centroids = []
centroid_subjects = []


for subject_name in leaf_subjects:

    subject_article_ids = (
        subjects[
            subjects["subject_fullname"]
            ==
            subject_name
        ]["external_id"]
        .drop_duplicates()
        .tolist()
    )


    # ========================================================
    # 1. ÖNCELİK:
    # TEK LEAF ETİKETLİ MAKALELER
    # ========================================================

    clean_candidates = [
        article_id
        for article_id in subject_article_ids
        if label_counts.get(
            article_id,
            0
        ) == 1
    ]


    selected_ids = []


    # ========================================================
    # YETERLİ SINGLE-LABEL VARSA
    # ========================================================

    if len(clean_candidates) >= SEEDS_PER_SUBJECT:

        selected_ids = (
            select_representative_articles(
                clean_candidates,
                SEEDS_PER_SUBJECT
            )
        )

        selection_type = (
            "REPRESENTATIVE_SINGLE_LABEL"
        )


    # ========================================================
    # YETERLİ SINGLE-LABEL YOKSA
    # ========================================================

    else:

        # Olan temiz makaleleri al
        selected_ids.extend(
            select_representative_articles(
                clean_candidates,
                len(clean_candidates)
            )
        )


        needed = (
            SEEDS_PER_SUBJECT
            -
            len(selected_ids)
        )


        remaining_candidates = [
            article_id
            for article_id in subject_article_ids
            if article_id not in selected_ids
        ]


        extra = (
            select_representative_articles(
                remaining_candidates,
                needed
            )
        )


        selected_ids.extend(
            extra
        )


        if len(clean_candidates) > 0:

            selection_type = (
                "REPRESENTATIVE_MIXED"
            )

        else:

            selection_type = (
                "REPRESENTATIVE_MULTI_LABEL_ONLY"
            )


    # ========================================================
    # EMBEDDING ROWLARI
    # ========================================================

    selected_embedding_rows = []


    for article_id in selected_ids:

        if article_id not in id_to_row:
            continue


        embedding_row = int(
            id_to_row[
                article_id
            ]
        )


        selected_embedding_rows.append(
            embedding_row
        )


        seed_rows.append(
            {
                "subject_fullname":
                    subject_name,

                "external_id":
                    article_id,

                "embedding_row":
                    embedding_row,

                "article_label_count":
                    int(
                        label_counts.get(
                            article_id,
                            0
                        )
                    ),

                "selection_type":
                    selection_type
            }
        )


    # ========================================================
    # CENTROID OLUŞTUR
    # ========================================================

    if not selected_embedding_rows:

        print(
            "UYARI - seed bulunamadı:",
            subject_name
        )

        continue


    subject_vectors = (
        embeddings[
            selected_embedding_rows
        ]
    )


    centroid = (
        subject_vectors.mean(
            axis=0
        )
    )


    # Normalize
    norm = np.linalg.norm(
        centroid
    )


    if norm > 0:

        centroid = (
            centroid / norm
        )


    centroids.append(
        centroid
    )


    centroid_subjects.append(
        subject_name
    )


# ============================================================
# SONUÇLARI HAZIRLA
# ============================================================

seed_df = pd.DataFrame(
    seed_rows
)


centroid_matrix = np.vstack(
    centroids
).astype(
    np.float32
)


centroid_metadata = pd.DataFrame(
    {
        "cluster_id":
            range(
                len(
                    centroid_subjects
                )
            ),

        "subject_fullname":
            centroid_subjects
    }
)


# ============================================================
# SONUÇLARI GÖSTER
# ============================================================

print("\n" + "=" * 100)
print("V2 SEED SONUCU")
print("=" * 100)


print(
    "Centroid oluşturulan konu:",
    len(centroid_subjects)
)


print(
    "Centroid matrix shape:",
    centroid_matrix.shape
)


print(
    "Toplam seed ilişkisi:",
    len(seed_df)
)


print(
    "Benzersiz seed makale:",
    seed_df[
        "external_id"
    ].nunique()
)


print("\nSEÇİM TİPLERİ")
print("-" * 70)


print(
    seed_df[
        "selection_type"
    ]
    .value_counts()
    .to_string()
)


# ============================================================
# SEED SAYISI
# ============================================================

seed_counts = (
    seed_df
    .groupby(
        "subject_fullname"
    )["external_id"]
    .nunique()
)


print("\nSEED SAYISI")
print("-" * 70)

print(
    seed_counts.describe()
)


print("\n5'TEN AZ SEED OLAN KONULAR")
print("-" * 100)


low_seed = (
    seed_counts[
        seed_counts
        <
        SEEDS_PER_SUBJECT
    ]
)


if low_seed.empty:

    print("Yok")

else:

    print(
        low_seed.to_string()
    )


# ============================================================
# DOSYALARI KAYDET
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


seed_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "seed_articles.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


centroid_metadata.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "seed_centroid_subjects.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


np.save(
    os.path.join(
        OUTPUT_DIR,
        "seed_centroids.npy"
    ),
    centroid_matrix
)


print("\n" + "=" * 100)
print("TAMAMLANDI")
print("=" * 100)

print(
    "results/kmeans/seed_articles.csv"
)

print(
    "results/kmeans/seed_centroid_subjects.csv"
)

print(
    "results/kmeans/seed_centroids.npy"
)