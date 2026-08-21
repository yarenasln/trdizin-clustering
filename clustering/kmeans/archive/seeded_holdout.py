import os
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)


# ============================================================
# AYARLAR
# ============================================================

EMBEDDING_FILE = "embeddings/mpnet_multilingual_embeddings.npy"
SUBJECT_FILE = "data/article_subjects.csv"

REFERENCE_FILE = "results/kmeans/holdout/reference.csv"
TEST_FILE = "results/kmeans/holdout/test.csv"

OUTPUT_DIR = "results/kmeans/holdout"

SEEDS_PER_SUBJECT = 10
RANDOM_STATE = 42


# ============================================================
# VERİLERİ OKU
# ============================================================

print("=" * 100)
print("SEEDED K-MEANS - HOLDOUT DENEYİ")
print("=" * 100)

embeddings = np.load(
    EMBEDDING_FILE
).astype(np.float32)

reference = pd.read_csv(
    REFERENCE_FILE,
    encoding="utf-8-sig"
)

test = pd.read_csv(
    TEST_FILE,
    encoding="utf-8-sig"
)

subjects = pd.read_csv(
    SUBJECT_FILE,
    encoding="utf-8-sig"
)


# ID tiplerini eşitle
reference["external_id"] = (
    reference["external_id"]
    .astype(str)
)

test["external_id"] = (
    test["external_id"]
    .astype(str)
)

subjects["external_id"] = (
    subjects["external_id"]
    .astype(str)
)


print("Referans makale:", len(reference))
print("Test makale:", len(test))


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
# EMBEDDING ROW MAP
# ============================================================

reference_row_map = (
    reference
    .set_index("external_id")["embedding_row"]
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
# YARDIMCI:
# TEMSİL EDİCİ MAKALELERİ SEÇ
# ============================================================

def select_representative(candidate_ids, count):

    valid_ids = [
        article_id
        for article_id in candidate_ids
        if article_id in reference_row_map
    ]

    if not valid_ids:
        return []

    rows = [
        int(reference_row_map[x])
        for x in valid_ids
    ]

    vectors = embeddings[rows]

    # Aday grubun merkezi
    center = vectors.mean(axis=0)

    norm = np.linalg.norm(center)

    if norm > 0:
        center = center / norm

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

    similarities = (
        normalized_vectors @ center
    )

    order = np.argsort(
        -similarities
    )

    selected_count = min(
        count,
        len(valid_ids)
    )

    return [
        valid_ids[i]
        for i in order[
            :selected_count
        ]
    ]


# ============================================================
# 1. %20 REFERANSTAN 195 BAŞLANGIÇ CENTROIDI OLUŞTUR
# ============================================================

print("\n" + "=" * 100)
print("BAŞLANGIÇ CENTROIDLERİ OLUŞTURULUYOR")
print("=" * 100)

centroids = []
centroid_subjects = []
seed_rows = []


reference_ids = set(
    reference["external_id"]
)


for subject_name in leaf_subjects:

    # Sadece REFERANS tarafındaki
    # bu konuya ait makaleler
    candidates = (
        subjects[
            (subjects["subject_fullname"] == subject_name)
            &
            (subjects["external_id"].isin(reference_ids))
        ]["external_id"]
        .drop_duplicates()
        .tolist()
    )


    # Önce tek etiketli olanlar
    single_label_candidates = [
        article_id
        for article_id in candidates
        if label_counts.get(article_id, 0) == 1
    ]


    selected_ids = []


    # --------------------------------------------------------
    # Yeterli temiz örnek varsa
    # --------------------------------------------------------

    if len(single_label_candidates) >= SEEDS_PER_SUBJECT:

        selected_ids = select_representative(
            single_label_candidates,
            SEEDS_PER_SUBJECT
        )

        selection_type = "SINGLE_LABEL"


    else:

        # Olan temiz örneklerin hepsi
        selected_ids.extend(
            select_representative(
                single_label_candidates,
                len(single_label_candidates)
            )
        )

        needed = (
            SEEDS_PER_SUBJECT
            -
            len(selected_ids)
        )

        remaining_candidates = [
            article_id
            for article_id in candidates
            if article_id not in selected_ids
        ]

        selected_ids.extend(
            select_representative(
                remaining_candidates,
                needed
            )
        )

        selection_type = "MIXED"


    if not selected_ids:

        raise ValueError(
            f"Bu konu için referans seed bulunamadı: {subject_name}"
        )


    rows = [
        int(
            reference_row_map[
                article_id
            ]
        )
        for article_id in selected_ids
    ]


    vectors = embeddings[
        rows
    ]


    centroid = vectors.mean(
        axis=0
    )


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


    for article_id in selected_ids:

        seed_rows.append(
            {
                "subject_fullname":
                    subject_name,

                "external_id":
                    article_id,

                "embedding_row":
                    int(
                        reference_row_map[
                            article_id
                        ]
                    ),

                "label_count":
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


initial_centroids = np.vstack(
    centroids
).astype(np.float32)


print(
    "Initial centroid shape:",
    initial_centroids.shape
)


seed_df = pd.DataFrame(
    seed_rows
)


print(
    "Benzersiz seed makale:",
    seed_df["external_id"].nunique()
)


# ============================================================
# 2. K-MEANS'İ REFERANS CENTROIDLERDEN BAŞLAT
# ============================================================
#
# Burada TEST etiketleri kullanılmıyor.
#
# K-Means yalnızca test embeddinglerini görüyor.
# ============================================================

print("\n" + "=" * 100)
print("TEST VERİSİ KÜMELENİYOR")
print("=" * 100)


test_rows = (
    test["embedding_row"]
    .astype(int)
    .to_numpy()
)


X_test = embeddings[
    test_rows
]


K = len(
    initial_centroids
)


kmeans = KMeans(
    n_clusters=K,
    init=initial_centroids,
    n_init=1,
    random_state=RANDOM_STATE,
    max_iter=300,
    tol=1e-4
)


test_labels = kmeans.fit_predict(
    X_test
)


print(
    "K-Means tamamlandı."
)


# ============================================================
# 3. TEST SONUÇLARINI HAZIRLA
# ============================================================

cluster_to_subject = {
    cluster_id: subject_name
    for cluster_id, subject_name
    in enumerate(
        centroid_subjects
    )
}


predictions = (
    test.copy()
    .reset_index(drop=True)
)


predictions[
    "cluster_id"
] = test_labels


predictions[
    "predicted_subject"
] = (
    predictions[
        "cluster_id"
    ]
    .map(
        cluster_to_subject
    )
)


# ============================================================
# 4. TEST ETİKETLERİNİ ŞİMDİ AÇ
# ============================================================
#
# Bu noktaya kadar test etiketlerini kullanmadık.
# ============================================================

evaluation = predictions[
    [
        "external_id",
        "cluster_id",
        "predicted_subject"
    ]
].merge(
    subjects[
        [
            "external_id",
            "subject_fullname"
        ]
    ],
    on="external_id",
    how="left"
)


evaluation["subject_match"] = (
    evaluation[
        "predicted_subject"
    ]
    ==
    evaluation[
        "subject_fullname"
    ]
)


article_match = (
    evaluation
    .groupby("external_id")[
        "subject_match"
    ]
    .any()
)


matched = int(
    article_match.sum()
)

total = int(
    article_match.size
)

topic_match_rate = (
    matched / total
)


# ============================================================
# 5. GEOMETRİK METRİKLER
# ============================================================

silhouette = silhouette_score(
    X_test,
    test_labels
)

davies = davies_bouldin_score(
    X_test,
    test_labels
)

calinski = calinski_harabasz_score(
    X_test,
    test_labels
)


# ============================================================
# 6. CLUSTER BOYUTLARI
# ============================================================

cluster_sizes = (
    pd.Series(test_labels)
    .value_counts()
)


singleton = int(
    (cluster_sizes == 1).sum()
)

le5 = int(
    (cluster_sizes <= 5).sum()
)

le10 = int(
    (cluster_sizes <= 10).sum()
)


# ============================================================
# 7. SONUÇLARI YAZDIR
# ============================================================

print("\n" + "=" * 100)
print("SEEDED HOLDOUT SONUCU")
print("=" * 100)

print(
    "Test makale:",
    total
)

print(
    "Konu eşleşen:",
    matched
)

print(
    "Konu uyum yüzdesi:",
    f"{topic_match_rate * 100:.2f}%"
)

print(
    "Silhouette:",
    round(
        silhouette,
        6
    )
)

print(
    "Davies-Bouldin:",
    round(
        davies,
        6
    )
)

print(
    "Calinski-Harabasz:",
    round(
        calinski,
        6
    )
)

print(
    "Singleton:",
    singleton
)

print(
    "<=5 cluster:",
    le5
)

print(
    "<=10 cluster:",
    le10
)


# ============================================================
# 8. CENTROID HAREKETİ
# ============================================================

centroid_shift = np.linalg.norm(
    kmeans.cluster_centers_
    -
    initial_centroids,
    axis=1
)


print(
    "Ortalama centroid hareketi:",
    round(
        centroid_shift.mean(),
        4
    )
)


# ============================================================
# 9. KAYDET
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


predictions.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "seeded_holdout_predictions.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


seed_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "seeded_holdout_seeds.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


np.save(
    os.path.join(
        OUTPUT_DIR,
        "seeded_holdout_initial_centroids.npy"
    ),
    initial_centroids
)


summary = pd.DataFrame(
    [
        {
            "Method":
                "Seeded K-Means",

            "Reference_Articles":
                len(reference),

            "Test_Articles":
                total,

            "K":
                K,

            "Topic_Match_Rate":
                topic_match_rate,

            "Silhouette":
                silhouette,

            "Davies_Bouldin":
                davies,

            "Calinski_Harabasz":
                calinski,

            "Singleton_Clusters":
                singleton,

            "Clusters_LE_5":
                le5,

            "Clusters_LE_10":
                le10,

            "Mean_Centroid_Shift":
                centroid_shift.mean()
        }
    ]
)


summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "seeded_holdout_summary.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


print("\nDosyalar oluşturuldu:")

print(
    "results/kmeans/holdout/seeded_holdout_predictions.csv"
)

print(
    "results/kmeans/holdout/seeded_holdout_summary.csv"
)