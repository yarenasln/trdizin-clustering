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

K = 195
RANDOM_STATE = 42


# ============================================================
# VERİLERİ OKU
# ============================================================

print("=" * 100)
print("NORMAL K-MEANS++ - HOLDOUT DENEYİ")
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


reference["external_id"] = (
    reference["external_id"].astype(str)
)

test["external_id"] = (
    test["external_id"].astype(str)
)

subjects["external_id"] = (
    subjects["external_id"].astype(str)
)


print("Referans makale:", len(reference))
print("Test makale:", len(test))
print("K:", K)


# ============================================================
# 1. TÜM EMBEDDINGLER
# ============================================================
#
# K-Means++ etiket görmeden bütün embedding geometrisini
# kullanarak 195 cluster oluşturuyor.
#
# Referans/test ayrımı burada etiket bilgisinin kullanımı
# açısından önem taşıyor.
# ============================================================

all_data = pd.concat(
    [
        reference,
        test
    ],
    ignore_index=True
)

all_rows = (
    all_data["embedding_row"]
    .astype(int)
    .to_numpy()
)

X_all = embeddings[
    all_rows
]


print(
    "K-Means++ embedding shape:",
    X_all.shape
)


# ============================================================
# 2. NORMAL K-MEANS++
# ============================================================

print("\n" + "=" * 100)
print("K-MEANS++ ÇALIŞIYOR")
print("=" * 100)


kmeans = KMeans(
    n_clusters=K,
    init="k-means++",
    n_init=10,
    random_state=RANDOM_STATE,
    max_iter=300,
    tol=1e-4
)


all_labels = kmeans.fit_predict(
    X_all
)


all_data = all_data.copy()

all_data[
    "cluster_id"
] = all_labels


print("K-Means++ tamamlandı.")


# ============================================================
# 3. REFERANS VE TESTİ GERİ AYIR
# ============================================================

reference_clustered = all_data[
    all_data["split"] == "REFERENCE"
].copy()


test_clustered = all_data[
    all_data["split"] == "TEST"
].copy()


print(
    "Clusterlanmış referans:",
    len(reference_clustered)
)

print(
    "Clusterlanmış test:",
    len(test_clustered)
)


# ============================================================
# 4. SADECE %20 REFERANS ETİKETLERİNİ AÇ
# ============================================================
#
# Test etiketleri burada kullanılmıyor.
#
# Referans makalelerin gerçek konularına bakarak:
#
# Cluster 0 -> en çok hangi konu?
# Cluster 1 -> en çok hangi konu?
# ...
#
# şeklinde clusterlara konu adı veriyoruz.
# ============================================================

reference_with_topics = (
    reference_clustered[
        [
            "external_id",
            "cluster_id"
        ]
    ]
    .merge(
        subjects[
            [
                "external_id",
                "subject_fullname"
            ]
        ],
        on="external_id",
        how="left"
    )
)


topic_counts = (
    reference_with_topics
    .groupby(
        [
            "cluster_id",
            "subject_fullname"
        ]
    )
    .size()
    .reset_index(
        name="count"
    )
)


dominant_topics = (
    topic_counts
    .sort_values(
        [
            "cluster_id",
            "count",
            "subject_fullname"
        ],
        ascending=[
            True,
            False,
            True
        ]
    )
    .drop_duplicates(
        "cluster_id"
    )
    .rename(
        columns={
            "subject_fullname":
                "predicted_subject"
        }
    )
)


cluster_to_subject = (
    dominant_topics
    .set_index(
        "cluster_id"
    )[
        "predicted_subject"
    ]
    .to_dict()
)


named_cluster_count = len(
    cluster_to_subject
)


print("\n" + "=" * 100)
print("REFERANS İLE CLUSTER İSİMLENDİRME")
print("=" * 100)

print(
    "Toplam cluster:",
    K
)

print(
    "Referans ile isim verilebilen:",
    named_cluster_count
)

print(
    "İsim verilemeyen:",
    K - named_cluster_count
)


# ============================================================
# 5. TEST MAKALELERİNE KONU TAHMİNİ
# ============================================================

test_clustered[
    "predicted_subject"
] = (
    test_clustered[
        "cluster_id"
    ]
    .map(
        cluster_to_subject
    )
)


unnamed_test_articles = int(
    test_clustered[
        "predicted_subject"
    ]
    .isna()
    .sum()
)


print(
    "İsimsiz cluster'a düşen test makalesi:",
    unnamed_test_articles
)


# ============================================================
# 6. ŞİMDİ TEST ETİKETLERİNİ AÇ
# ============================================================

evaluation = (
    test_clustered[
        [
            "external_id",
            "cluster_id",
            "predicted_subject"
        ]
    ]
    .merge(
        subjects[
            [
                "external_id",
                "subject_fullname"
            ]
        ],
        on="external_id",
        how="left"
    )
)


evaluation[
    "subject_match"
] = (
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
    .groupby(
        "external_id"
    )[
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
# 7. SADECE TEST SETİNDE GEOMETRİK METRİKLER
# ============================================================

test_rows = (
    test_clustered[
        "embedding_row"
    ]
    .astype(int)
    .to_numpy()
)


X_test = embeddings[
    test_rows
]


test_labels = (
    test_clustered[
        "cluster_id"
    ]
    .astype(int)
    .to_numpy()
)


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
# 8. CLUSTER BOYUTLARI - TEST
# ============================================================

cluster_sizes = (
    pd.Series(
        test_labels
    )
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
# 9. SONUÇ
# ============================================================

print("\n" + "=" * 100)
print("K-MEANS++ HOLDOUT SONUCU")
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
# 10. KAYDET
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


test_clustered.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "baseline_holdout_predictions.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


dominant_topics.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "baseline_cluster_subject_mapping.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


summary = pd.DataFrame(
    [
        {
            "Method":
                "K-Means++",

            "Reference_Articles":
                len(reference),

            "Test_Articles":
                total,

            "K":
                K,

            "Named_Clusters":
                named_cluster_count,

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
                le10
        }
    ]
)


summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "baseline_holdout_summary.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


print("\nDosyalar oluşturuldu:")

print(
    "results/kmeans/holdout/baseline_holdout_predictions.csv"
)

print(
    "results/kmeans/holdout/baseline_holdout_summary.csv"
)