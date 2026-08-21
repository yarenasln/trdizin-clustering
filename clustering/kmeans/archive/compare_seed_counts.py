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

OUTPUT_FILE = "results/kmeans/holdout/seed_count_comparison.csv"

SEED_COUNTS = [3, 5, 7, 10]
RANDOM_STATE = 42


# ============================================================
# VERİLERİ OKU
# ============================================================

print("=" * 100)
print("SEED SAYISI KARŞILAŞTIRMA DENEYİ")
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

reference["external_id"] = reference["external_id"].astype(str)
test["external_id"] = test["external_id"].astype(str)
subjects["external_id"] = subjects["external_id"].astype(str)


print("Referans:", len(reference))
print("Test:", len(test))


# ============================================================
# HAZIRLIK
# ============================================================

reference_ids = set(
    reference["external_id"]
)

reference_row_map = (
    reference
    .set_index("external_id")["embedding_row"]
    .to_dict()
)

label_counts = (
    subjects
    .groupby("external_id")["subject_fullname"]
    .nunique()
    .to_dict()
)

leaf_subjects = sorted(
    subjects["subject_fullname"]
    .dropna()
    .astype(str)
    .unique()
)

test_rows = (
    test["embedding_row"]
    .astype(int)
    .to_numpy()
)

X_test = embeddings[test_rows]


# ============================================================
# TEMSİL EDİCİ MAKALE SEÇİMİ
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

    center = vectors.mean(axis=0)

    norm = np.linalg.norm(center)

    if norm > 0:
        center = center / norm

    vector_norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True
    )

    vector_norms[vector_norms == 0] = 1

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
        for i in order[:selected_count]
    ]


# ============================================================
# TEK BİR SEED SAYISINI DENE
# ============================================================

def run_experiment(seeds_per_subject):

    print("\n" + "=" * 100)
    print(
        f"SEED SAYISI: {seeds_per_subject}"
    )
    print("=" * 100)

    centroids = []
    centroid_subjects = []

    unique_seed_ids = set()

    subjects_with_full_seed_count = 0
    subjects_with_less_seeds = 0


    # ========================================================
    # BAŞLANGIÇ CENTROIDLERİ
    # ========================================================

    for subject_name in leaf_subjects:

        candidates = (
            subjects[
                (subjects["subject_fullname"] == subject_name)
                &
                (subjects["external_id"].isin(reference_ids))
            ]["external_id"]
            .drop_duplicates()
            .tolist()
        )

        single_label_candidates = [
            article_id
            for article_id in candidates
            if label_counts.get(article_id, 0) == 1
        ]

        selected_ids = []


        # Önce single-label
        if len(single_label_candidates) >= seeds_per_subject:

            selected_ids = select_representative(
                single_label_candidates,
                seeds_per_subject
            )

        else:

            selected_ids.extend(
                select_representative(
                    single_label_candidates,
                    len(single_label_candidates)
                )
            )

            needed = (
                seeds_per_subject
                -
                len(selected_ids)
            )

            remaining = [
                article_id
                for article_id in candidates
                if article_id not in selected_ids
            ]

            selected_ids.extend(
                select_representative(
                    remaining,
                    needed
                )
            )


        if not selected_ids:

            raise ValueError(
                f"Seed bulunamadı: {subject_name}"
            )


        if len(selected_ids) == seeds_per_subject:
            subjects_with_full_seed_count += 1
        else:
            subjects_with_less_seeds += 1


        unique_seed_ids.update(
            selected_ids
        )


        rows = [
            int(reference_row_map[x])
            for x in selected_ids
        ]

        vectors = embeddings[rows]

        centroid = vectors.mean(
            axis=0
        )

        norm = np.linalg.norm(
            centroid
        )

        if norm > 0:
            centroid = centroid / norm

        centroids.append(
            centroid
        )

        centroid_subjects.append(
            subject_name
        )


    initial_centroids = np.vstack(
        centroids
    ).astype(np.float32)


    print(
        "Centroid shape:",
        initial_centroids.shape
    )

    print(
        "Benzersiz seed makale:",
        len(unique_seed_ids)
    )

    print(
        "Tam seed sayısına ulaşan konu:",
        subjects_with_full_seed_count
    )

    print(
        "Yetersiz örneği olan konu:",
        subjects_with_less_seeds
    )


    # ========================================================
    # K-MEANS
    # ========================================================

    kmeans = KMeans(
        n_clusters=len(leaf_subjects),
        init=initial_centroids,
        n_init=1,
        random_state=RANDOM_STATE,
        max_iter=300,
        tol=1e-4
    )

    labels = kmeans.fit_predict(
        X_test
    )


    # ========================================================
    # CLUSTER → KONU
    # ========================================================

    cluster_to_subject = {
        cluster_id: subject_name
        for cluster_id, subject_name
        in enumerate(centroid_subjects)
    }


    predictions = test.copy()

    predictions["cluster_id"] = labels

    predictions["predicted_subject"] = (
        predictions["cluster_id"]
        .map(cluster_to_subject)
    )


    # ========================================================
    # TEST GERÇEK ETİKETLERİ
    # ========================================================

    evaluation = (
        predictions[
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

    evaluation["subject_match"] = (
        evaluation["predicted_subject"]
        ==
        evaluation["subject_fullname"]
    )

    article_match = (
        evaluation
        .groupby("external_id")["subject_match"]
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


    # ========================================================
    # GEOMETRİK METRİKLER
    # ========================================================

    silhouette = silhouette_score(
        X_test,
        labels
    )

    davies = davies_bouldin_score(
        X_test,
        labels
    )

    calinski = calinski_harabasz_score(
        X_test,
        labels
    )


    # ========================================================
    # CLUSTER İSTATİSTİKLERİ
    # ========================================================

    cluster_sizes = (
        pd.Series(labels)
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


    # ========================================================
    # CENTROID HAREKETİ
    # ========================================================

    centroid_shift = np.linalg.norm(
        kmeans.cluster_centers_
        -
        initial_centroids,
        axis=1
    )

    mean_shift = float(
        centroid_shift.mean()
    )


    # ========================================================
    # SONUÇ
    # ========================================================

    print(
        "Konu uyumu:",
        f"{topic_match_rate * 100:.2f}%"
    )

    print(
        "Silhouette:",
        round(silhouette, 6)
    )

    print(
        "Davies-Bouldin:",
        round(davies, 6)
    )

    print(
        "Calinski-Harabasz:",
        round(calinski, 6)
    )

    print(
        "Singleton:",
        singleton
    )

    print(
        "<=5:",
        le5
    )

    print(
        "<=10:",
        le10
    )

    print(
        "Ortalama centroid hareketi:",
        round(mean_shift, 4)
    )


    return {
        "Seed_Count":
            seeds_per_subject,

        "Unique_Seed_Articles":
            len(unique_seed_ids),

        "Subjects_With_Full_Seed_Count":
            subjects_with_full_seed_count,

        "Subjects_With_Less_Seeds":
            subjects_with_less_seeds,

        "Topic_Match_Rate":
            topic_match_rate,

        "Topic_Match_Percent":
            topic_match_rate * 100,

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
            mean_shift
    }


# ============================================================
# TÜM SEED SAYILARINI DENE
# ============================================================

results = []

for seed_count in SEED_COUNTS:

    result = run_experiment(
        seed_count
    )

    results.append(
        result
    )


# ============================================================
# GENEL KARŞILAŞTIRMA
# ============================================================

results_df = pd.DataFrame(
    results
)


print("\n" + "=" * 120)
print("SEED SAYISI GENEL KARŞILAŞTIRMA")
print("=" * 120)

print(
    results_df[
        [
            "Seed_Count",
            "Unique_Seed_Articles",
            "Topic_Match_Percent",
            "Silhouette",
            "Davies_Bouldin",
            "Calinski_Harabasz",
            "Singleton_Clusters",
            "Clusters_LE_5",
            "Clusters_LE_10",
            "Mean_Centroid_Shift"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# EN İYİLER
# ============================================================

best_topic = results_df.loc[
    results_df[
        "Topic_Match_Rate"
    ].idxmax()
]

best_silhouette = results_df.loc[
    results_df[
        "Silhouette"
    ].idxmax()
]

best_davies = results_df.loc[
    results_df[
        "Davies_Bouldin"
    ].idxmin()
]


print("\n" + "=" * 100)
print("KAZANANLAR")
print("=" * 100)

print(
    "En yüksek konu uyumu:",
    int(best_topic["Seed_Count"]),
    "seed ->",
    f'{best_topic["Topic_Match_Percent"]:.2f}%'
)

print(
    "En iyi Silhouette:",
    int(best_silhouette["Seed_Count"]),
    "seed ->",
    round(
        best_silhouette["Silhouette"],
        6
    )
)

print(
    "En iyi Davies-Bouldin:",
    int(best_davies["Seed_Count"]),
    "seed ->",
    round(
        best_davies["Davies_Bouldin"],
        6
    )
)


# ============================================================
# KAYDET
# ============================================================

os.makedirs(
    os.path.dirname(
        OUTPUT_FILE
    ),
    exist_ok=True
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\nDosya oluşturuldu:")
print(OUTPUT_FILE)