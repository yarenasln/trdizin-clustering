import os
import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from config.paths import EMBEDDING_FILE, INDEX_FILE

# ============================================================
# AYARLAR
# ============================================================


SEED_ARTICLE_FILE = (
    "results/kmeans/seed_articles.csv"
)

SEED_CENTROID_FILE = (
    "results/kmeans/seed_centroids.npy"
)

SEED_SUBJECT_FILE = (
    "results/kmeans/seed_centroid_subjects.csv"
)

OUTPUT_DIR = (
    "results/kmeans"
)

MODEL_DIR = (
    "models/kmeans"
)

RANDOM_STATE = 42


# ============================================================
# 1. VERİLERİ YÜKLE
# ============================================================

print("=" * 100)
print("SEEDED K-MEANS TRAIN")
print("=" * 100)


embeddings = np.load(
    EMBEDDING_FILE
).astype(
    np.float32
)


articles = pd.read_csv(
    INDEX_FILE,
    encoding="utf-8-sig"
)


seed_articles = pd.read_csv(
    SEED_ARTICLE_FILE,
    encoding="utf-8-sig"
)


seed_centroids = np.load(
    SEED_CENTROID_FILE
).astype(
    np.float32
)


seed_subjects = pd.read_csv(
    SEED_SUBJECT_FILE,
    encoding="utf-8-sig"
)


print(
    "Toplam embedding:",
    embeddings.shape
)

print(
    "Toplam makale:",
    len(articles)
)

print(
    "Seed centroid shape:",
    seed_centroids.shape
)

print(
    "Seed konu sayısı:",
    len(seed_subjects)
)


# ============================================================
# 2. KRİTİK KONTROLLER
# ============================================================

if len(articles) != embeddings.shape[0]:

    raise ValueError(
        "Makale sayısı ile embedding satır sayısı eşleşmiyor."
    )


if seed_centroids.shape[0] != len(seed_subjects):

    raise ValueError(
        "Centroid sayısı ile konu sayısı eşleşmiyor."
    )


K = seed_centroids.shape[0]


print(
    "K:",
    K
)


# ============================================================
# 3. SEED MAKALELERİ AYIR
# ============================================================

articles["external_id"] = (
    articles["external_id"]
    .astype(str)
)

seed_articles["external_id"] = (
    seed_articles["external_id"]
    .astype(str)
)


unique_seed_ids = set(
    seed_articles[
        "external_id"
    ]
    .unique()
)


seed_mask = (
    articles[
        "external_id"
    ]
    .isin(
        unique_seed_ids
    )
)


non_seed_mask = (
    ~seed_mask
)


seed_dataset = (
    articles[
        seed_mask
    ]
    .copy()
)


non_seed_dataset = (
    articles[
        non_seed_mask
    ]
    .copy()
)


X_non_seed = (
    embeddings[
        non_seed_mask.to_numpy()
    ]
)


print("\n" + "=" * 100)
print("VERİ AYRIMI")
print("=" * 100)

print(
    "Benzersiz seed makale:",
    len(seed_dataset)
)

print(
    "Etiketsiz kümelenecek makale:",
    len(non_seed_dataset)
)

print(
    "Toplam:",
    len(seed_dataset)
    +
    len(non_seed_dataset)
)


# ============================================================
# 4. SEEDED K-MEANS
# ============================================================

print("\n" + "=" * 100)
print("SEEDED K-MEANS ÇALIŞIYOR")
print("=" * 100)

print(
    "Başlangıç centroidleri:",
    seed_centroids.shape
)


kmeans = KMeans(

    n_clusters=K,

    init=seed_centroids,

    n_init=1,

    random_state=
        RANDOM_STATE,

    max_iter=300,

    tol=1e-4
)


# ============================================================
# 5. KALAN VERİYİ KÜMELE
# ============================================================

labels = (
    kmeans.fit_predict(
        X_non_seed
    )
)


print(
    "K-Means tamamlandı."
)

print(
    "Final centroid shape:",
    kmeans.cluster_centers_.shape
)


# ============================================================
# 6. SONUÇ DATAFRAME
# ============================================================

results = (
    non_seed_dataset
    .copy()
    .reset_index(
        drop=True
    )
)


results[
    "cluster_id"
] = labels


# ============================================================
# 7. CLUSTER → BAŞLANGIÇ KONU EŞLEŞMESİ
# ============================================================
#
# cluster_id sırası, seed_centroid_subjects.csv içindeki
# başlangıç centroid sırasıyla aynıdır.
#
# Fakat dikkat:
# K-Means çalışırken centroidler hareket eder.
#
# Bu yüzden bu kolon:
# "başlangıçta hangi konudan başlatıldı?"
# anlamına gelir.
# ============================================================

cluster_to_subject = (
    seed_subjects
    .set_index(
        "cluster_id"
    )[
        "subject_fullname"
    ]
    .to_dict()
)


results[
    "initial_seed_subject"
] = (
    results[
        "cluster_id"
    ]
    .map(
        cluster_to_subject
    )
)


# ============================================================
# 8. CLUSTER BOYUTLARI
# ============================================================

cluster_sizes = (
    results[
        "cluster_id"
    ]
    .value_counts()
    .sort_index()
)


cluster_stats = pd.DataFrame(
    {
        "cluster_id":
            range(K)
    }
)


cluster_stats[
    "initial_seed_subject"
] = (
    cluster_stats[
        "cluster_id"
    ]
    .map(
        cluster_to_subject
    )
)


cluster_stats[
    "article_count"
] = (
    cluster_stats[
        "cluster_id"
    ]
    .map(
        cluster_sizes
    )
    .fillna(0)
    .astype(int)
)


# ============================================================
# 9. İSTATİSTİKLER
# ============================================================

print("\n" + "=" * 100)
print("CLUSTER İSTATİSTİKLERİ")
print("=" * 100)


print(
    "Cluster sayısı:",
    K
)


print(
    "Kullanılan cluster:",
    results[
        "cluster_id"
    ].nunique()
)


print(
    "Boş cluster:",
    int(
        (
            cluster_stats[
                "article_count"
            ]
            ==
            0
        ).sum()
    )
)


print(
    "En küçük cluster:",
    int(
        cluster_stats[
            "article_count"
        ].min()
    )
)


print(
    "En büyük cluster:",
    int(
        cluster_stats[
            "article_count"
        ].max()
    )
)


print(
    "Ortalama cluster büyüklüğü:",
    round(
        cluster_stats[
            "article_count"
        ].mean(),
        2
    )
)


print(
    "Singleton cluster:",
    int(
        (
            cluster_stats[
                "article_count"
            ]
            ==
            1
        ).sum()
    )
)


print(
    "5 veya daha az makaleli:",
    int(
        (
            cluster_stats[
                "article_count"
            ]
            <=
            5
        ).sum()
    )
)


print(
    "10 veya daha az makaleli:",
    int(
        (
            cluster_stats[
                "article_count"
            ]
            <=
            10
        ).sum()
    )
)


# ============================================================
# 10. DOSYALARI KAYDET
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "seeded_cluster_assignments.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


cluster_stats.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "seeded_cluster_sizes.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


np.save(
    os.path.join(
        MODEL_DIR,
        "seeded_final_centroids.npy"
    ),
    kmeans.cluster_centers_
)


joblib.dump(
    kmeans,
    os.path.join(
        MODEL_DIR,
        "seeded_kmeans_model.joblib"
    )
)


# ============================================================
# 11. BAŞLANGIÇ VE FİNAL CENTROID FARKI
# ============================================================

centroid_shift = (
    np.linalg.norm(
        kmeans.cluster_centers_
        -
        seed_centroids,
        axis=1
    )
)


shift_df = pd.DataFrame(
    {
        "cluster_id":
            range(K),

        "subject_fullname":
            [
                cluster_to_subject[i]
                for i in range(K)
            ],

        "centroid_shift":
            centroid_shift
    }
)


shift_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "seeded_centroid_shift.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


print("\n" + "=" * 100)
print("CENTROID HAREKETİ")
print("=" * 100)


print(
    "Ortalama centroid hareketi:",
    round(
        centroid_shift.mean(),
        4
    )
)


print(
    "Medyan centroid hareketi:",
    round(
        np.median(
            centroid_shift
        ),
        4
    )
)


print(
    "En fazla hareket:",
    round(
        centroid_shift.max(),
        4
    )
)


# ============================================================
# TAMAMLANDI
# ============================================================

print("\n" + "=" * 100)
print("TAMAMLANDI")
print("=" * 100)

print(
    "results/kmeans/seeded_cluster_assignments.csv"
)

print(
    "results/kmeans/seeded_cluster_sizes.csv"
)

print(
    "results/kmeans/seeded_centroid_shift.csv"
)

print(
    "models/kmeans/seeded_final_centroids.npy"
)

print(
    "models/kmeans/seeded_kmeans_model.joblib"
)