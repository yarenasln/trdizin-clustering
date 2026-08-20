import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


# ============================================================
# AYARLAR
# ============================================================

EMBEDDING_FILE = "embeddings/mpnet_multilingual_embeddings.npy"
SUBJECT_FILE = "data/article_subjects.csv"

ASSIGNMENT_FILE = (
    "results/kmeans/seeded_cluster_assignments.csv"
)

OUTPUT_DIR = "results/kmeans"


# ============================================================
# VERİLERİ OKU
# ============================================================

print("=" * 100)
print("SEEDED K-MEANS DEĞERLENDİRME")
print("=" * 100)

embeddings = np.load(
    EMBEDDING_FILE
).astype(np.float32)

results = pd.read_csv(
    ASSIGNMENT_FILE,
    encoding="utf-8-sig"
)

subjects = pd.read_csv(
    SUBJECT_FILE,
    encoding="utf-8-sig"
)


# ID tiplerini eşitle
results["external_id"] = (
    results["external_id"]
    .astype(str)
)

subjects["external_id"] = (
    subjects["external_id"]
    .astype(str)
)


print("Değerlendirilen makale:", len(results))
print("Cluster sayısı:", results["cluster_id"].nunique())


# ============================================================
# 1. EMBEDDINGLERİ DOĞRU SATIRLARDAN AL
# ============================================================

embedding_rows = (
    results["embedding_row"]
    .astype(int)
    .to_numpy()
)

X = embeddings[
    embedding_rows
]

labels = (
    results["cluster_id"]
    .astype(int)
    .to_numpy()
)


print("Değerlendirme embedding shape:", X.shape)


# ============================================================
# 2. CLUSTERING İÇ METRİKLERİ
# ============================================================

print("\n" + "=" * 100)
print("CLUSTERING METRİKLERİ")
print("=" * 100)

silhouette = silhouette_score(
    X,
    labels,
    metric="euclidean"
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
# 3. GERÇEK TR DİZİN ETİKETLERİNİ EKLE
# ============================================================

evaluation = results[
    [
        "external_id",
        "cluster_id",
        "initial_seed_subject"
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


# ============================================================
# 4. SEED KONUSU GERÇEK ETİKETLERDEN BİRİ Mİ?
# ============================================================
#
# Multi-label nedeniyle:
#
# Makalenin 3 gerçek konusu varsa ve
# cluster'ın başlangıç konusu bunlardan
# herhangi biriyle eşleşiyorsa doğru kabul ediyoruz.
# ============================================================

evaluation["subject_match"] = (
    evaluation["initial_seed_subject"]
    ==
    evaluation["subject_fullname"]
)


article_match = (
    evaluation
    .groupby("external_id")["subject_match"]
    .any()
)


matched_articles = int(
    article_match.sum()
)

total_articles = int(
    article_match.size
)

topic_match_rate = (
    matched_articles
    /
    total_articles
)


print("\n" + "=" * 100)
print("TR DİZİN KONU UYUMU")
print("=" * 100)

print(
    "Değerlendirilen benzersiz makale:",
    total_articles
)

print(
    "Seed konusu gerçek etiketlerden biri olan:",
    matched_articles
)

print(
    "Konu uyum oranı:",
    round(
        topic_match_rate,
        4
    )
)

print(
    "Konu uyum yüzdesi:",
    f"{topic_match_rate * 100:.2f}%"
)


# ============================================================
# 5. HER CLUSTER İÇİN UYUM
# ============================================================

article_cluster = (
    results[
        [
            "external_id",
            "cluster_id",
            "initial_seed_subject"
        ]
    ]
    .copy()
)


article_match_df = (
    article_match
    .rename("subject_match")
    .reset_index()
)


article_cluster = (
    article_cluster
    .merge(
        article_match_df,
        on="external_id",
        how="left"
    )
)


cluster_evaluation = (
    article_cluster
    .groupby(
        [
            "cluster_id",
            "initial_seed_subject"
        ]
    )
    .agg(
        article_count=(
            "external_id",
            "nunique"
        ),
        matched_articles=(
            "subject_match",
            "sum"
        )
    )
    .reset_index()
)


cluster_evaluation[
    "match_rate"
] = (
    cluster_evaluation[
        "matched_articles"
    ]
    /
    cluster_evaluation[
        "article_count"
    ]
)


# ============================================================
# 6. EN İYİ / EN KÖTÜ CLUSTERLAR
# ============================================================

print("\n" + "=" * 100)
print("KONU UYUMU EN YÜKSEK 10 CLUSTER")
print("=" * 100)

print(
    cluster_evaluation
    .sort_values(
        [
            "match_rate",
            "article_count"
        ],
        ascending=[
            False,
            False
        ]
    )
    .head(10)
    .to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("KONU UYUMU EN DÜŞÜK 10 CLUSTER")
print("=" * 100)

print(
    cluster_evaluation
    .sort_values(
        [
            "match_rate",
            "article_count"
        ],
        ascending=[
            True,
            False
        ]
    )
    .head(10)
    .to_string(
        index=False
    )
)


# ============================================================
# 7. ÖZET SONUÇ
# ============================================================

summary = pd.DataFrame(
    [
        {
            "Method":
                "Seeded K-Means",

            "Articles":
                total_articles,

            "Clusters":
                results[
                    "cluster_id"
                ].nunique(),

            "Silhouette":
                silhouette,

            "Davies_Bouldin":
                davies_bouldin,

            "Calinski_Harabasz":
                calinski_harabasz,

            "Topic_Match_Rate":
                topic_match_rate,

            "Matched_Articles":
                matched_articles
        }
    ]
)


# ============================================================
# 8. KAYDET
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "seeded_evaluation_summary.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


cluster_evaluation.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "seeded_cluster_topic_evaluation.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


print("\n" + "=" * 100)
print("TAMAMLANDI")
print("=" * 100)

print(
    "results/kmeans/seeded_evaluation_summary.csv"
)

print(
    "results/kmeans/seeded_cluster_topic_evaluation.csv"
)