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

SEEDED_ASSIGNMENTS_FILE = (
    "results/kmeans/seeded_cluster_assignments.csv"
)

OUTPUT_DIR = "results/kmeans"

K = 195
RANDOM_STATE = 42


# ============================================================
# VERİYİ YÜKLE
# ============================================================

print("=" * 100)
print("BASELINE - NORMAL K-MEANS++")
print("=" * 100)

embeddings = np.load(
    EMBEDDING_FILE
).astype(np.float32)

articles = pd.read_csv(
    SEEDED_ASSIGNMENTS_FILE,
    encoding="utf-8-sig"
)

print("Makale sayısı:", len(articles))
print("K:", K)


# ============================================================
# AYNI 6004 MAKALEYİ AL
# ============================================================

embedding_rows = (
    articles["embedding_row"]
    .astype(int)
    .to_numpy()
)

X = embeddings[
    embedding_rows
]

print("Embedding shape:", X.shape)


# ============================================================
# NORMAL K-MEANS++
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

labels = kmeans.fit_predict(X)

print("K-Means++ tamamlandı.")


# ============================================================
# METRİKLER
# ============================================================

print("\n" + "=" * 100)
print("BASELINE METRİKLERİ")
print("=" * 100)

silhouette = silhouette_score(
    X,
    labels
)

davies_bouldin = davies_bouldin_score(
    X,
    labels
)

calinski_harabasz = calinski_harabasz_score(
    X,
    labels
)


print(
    "Silhouette:",
    round(silhouette, 6)
)

print(
    "Davies-Bouldin:",
    round(davies_bouldin, 6)
)

print(
    "Calinski-Harabasz:",
    round(calinski_harabasz, 6)
)


# ============================================================
# CLUSTER BOYUTLARI
# ============================================================

cluster_sizes = pd.Series(
    labels
).value_counts()


print("\n" + "=" * 100)
print("CLUSTER İSTATİSTİKLERİ")
print("=" * 100)

print(
    "Kullanılan cluster:",
    len(cluster_sizes)
)

print(
    "En küçük cluster:",
    int(cluster_sizes.min())
)

print(
    "En büyük cluster:",
    int(cluster_sizes.max())
)

print(
    "Ortalama cluster:",
    round(
        cluster_sizes.mean(),
        2
    )
)

print(
    "Singleton cluster:",
    int(
        (cluster_sizes == 1).sum()
    )
)

print(
    "5 veya daha az:",
    int(
        (cluster_sizes <= 5).sum()
    )
)

print(
    "10 veya daha az:",
    int(
        (cluster_sizes <= 10).sum()
    )
)


# ============================================================
# SONUÇLARI KAYDET
# ============================================================

articles = articles.copy()

articles[
    "baseline_cluster_id"
] = labels


articles.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "baseline_cluster_assignments.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


summary = pd.DataFrame(
    [
        {
            "Method":
                "K-Means++",

            "Articles":
                len(articles),

            "Clusters":
                len(cluster_sizes),

            "Silhouette":
                silhouette,

            "Davies_Bouldin":
                davies_bouldin,

            "Calinski_Harabasz":
                calinski_harabasz,

            "Singleton_Clusters":
                int(
                    (cluster_sizes == 1).sum()
                ),

            "Clusters_LE_5":
                int(
                    (cluster_sizes <= 5).sum()
                ),

            "Clusters_LE_10":
                int(
                    (cluster_sizes <= 10).sum()
                )
        }
    ]
)


summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "baseline_evaluation_summary.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


print("\n" + "=" * 100)
print("TAMAMLANDI")
print("=" * 100)

print(
    "results/kmeans/baseline_cluster_assignments.csv"
)

print(
    "results/kmeans/baseline_evaluation_summary.csv"
)