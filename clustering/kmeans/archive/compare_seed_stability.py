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
INDEX_FILE = "embeddings/article_embedding_index.csv"
SUBJECT_FILE = "data/article_subjects.csv"

OUTPUT_FILE = "results/kmeans/holdout/seed_stability_comparison.csv"

REFERENCE_RATIO = 0.20

RANDOM_STATES = [
    42,
    123,
    2026,
    7,
    99
]

SEED_COUNTS = [
    7,
    10
]


# ============================================================
# VERİLERİ OKU
# ============================================================

print("=" * 110)
print("SEEDED K-MEANS - SEED STABILITY TEST")
print("=" * 110)

embeddings = np.load(
    EMBEDDING_FILE
).astype(np.float32)

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


print("Toplam makale:", len(articles))
print("Embedding shape:", embeddings.shape)


# ============================================================
# ORTAK HAZIRLIK
# ============================================================

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

article_subject_map = (
    subjects
    .groupby("external_id")["subject_fullname"]
    .apply(
        lambda values:
            set(
                values
                .dropna()
                .astype(str)
            )
    )
    .to_dict()
)

subject_candidates = (
    subjects
    .groupby("subject_fullname")["external_id"]
    .apply(
        lambda values:
            list(
                dict.fromkeys(
                    values.astype(str)
                )
            )
    )
    .to_dict()
)


# ============================================================
# HOLDOUT OLUŞTUR
# ============================================================

def create_holdout(random_state):

    rng = np.random.default_rng(
        random_state
    )

    target_reference_count = int(
        round(
            len(articles)
            *
            REFERENCE_RATIO
        )
    )

    reference_ids = set()

    covered_subjects = set()

    subjects_by_rarity = sorted(
        leaf_subjects,
        key=lambda subject:
            len(
                subject_candidates[
                    subject
                ]
            )
    )


    # --------------------------------------------------------
    # Önce tüm konuları kapsa
    # --------------------------------------------------------

    for subject_name in subjects_by_rarity:

        if subject_name in covered_subjects:
            continue

        candidates = subject_candidates[
            subject_name
        ]

        available_candidates = [
            article_id
            for article_id in candidates
            if article_id not in reference_ids
        ]

        if not available_candidates:
            available_candidates = candidates

        best_candidates = []
        best_score = -1

        for article_id in available_candidates:

            new_topics = (
                article_subject_map.get(
                    article_id,
                    set()
                )
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


    # --------------------------------------------------------
    # %20'ye tamamla
    # --------------------------------------------------------

    remaining_ids = (
        articles[
            ~articles[
                "external_id"
            ].isin(
                reference_ids
            )
        ]["external_id"]
        .tolist()
    )

    needed = (
        target_reference_count
        -
        len(reference_ids)
    )

    if needed < 0:

        raise ValueError(
            "Konu kapsamı %20 hedefini aştı."
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


    return reference, test


# ============================================================
# TEMSİL EDİCİ SEED SEÇ
# ============================================================

def select_representative(
    candidate_ids,
    count,
    row_map
):

    valid_ids = [
        article_id
        for article_id in candidate_ids
        if article_id in row_map
    ]

    if not valid_ids:
        return []

    rows = [
        int(
            row_map[
                article_id
            ]
        )
        for article_id in valid_ids
    ]

    vectors = embeddings[
        rows
    ]

    center = vectors.mean(
        axis=0
    )

    center_norm = np.linalg.norm(
        center
    )

    if center_norm > 0:

        center = (
            center /
            center_norm
        )

    vector_norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True
    )

    vector_norms[
        vector_norms == 0
    ] = 1

    normalized_vectors = (
        vectors /
        vector_norms
    )

    similarities = (
        normalized_vectors
        @
        center
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
# TEK DENEY
# ============================================================

def run_experiment(
    reference,
    test,
    seed_count,
    random_state
):

    reference_ids = set(
        reference[
            "external_id"
        ]
    )

    reference_row_map = (
        reference
        .set_index(
            "external_id"
        )[
            "embedding_row"
        ]
        .to_dict()
    )


    centroids = []

    centroid_subjects = []

    unique_seed_ids = set()

    subjects_with_less_seeds = 0


    # --------------------------------------------------------
    # Her konu için centroid
    # --------------------------------------------------------

    for subject_name in leaf_subjects:

        candidates = (
            subjects[
                (
                    subjects[
                        "subject_fullname"
                    ]
                    ==
                    subject_name
                )
                &
                (
                    subjects[
                        "external_id"
                    ]
                    .isin(
                        reference_ids
                    )
                )
            ][
                "external_id"
            ]
            .drop_duplicates()
            .tolist()
        )


        single_label_candidates = [
            article_id
            for article_id in candidates
            if label_counts.get(
                article_id,
                0
            ) == 1
        ]


        selected_ids = []


        if (
            len(
                single_label_candidates
            )
            >=
            seed_count
        ):

            selected_ids = (
                select_representative(
                    single_label_candidates,
                    seed_count,
                    reference_row_map
                )
            )

        else:

            selected_ids.extend(
                select_representative(
                    single_label_candidates,
                    len(
                        single_label_candidates
                    ),
                    reference_row_map
                )
            )

            needed = (
                seed_count
                -
                len(
                    selected_ids
                )
            )

            remaining = [
                article_id
                for article_id in candidates
                if article_id
                not in selected_ids
            ]

            selected_ids.extend(
                select_representative(
                    remaining,
                    needed,
                    reference_row_map
                )
            )


        if not selected_ids:

            raise ValueError(
                f"Seed bulunamadı: {subject_name}"
            )


        if (
            len(
                selected_ids
            )
            <
            seed_count
        ):

            subjects_with_less_seeds += 1


        unique_seed_ids.update(
            selected_ids
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
                centroid /
                norm
            )


        centroids.append(
            centroid
        )

        centroid_subjects.append(
            subject_name
        )


    initial_centroids = np.vstack(
        centroids
    ).astype(
        np.float32
    )


    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    test_rows = (
        test[
            "embedding_row"
        ]
        .astype(int)
        .to_numpy()
    )

    X_test = embeddings[
        test_rows
    ]


    kmeans = KMeans(
        n_clusters=len(
            leaf_subjects
        ),
        init=initial_centroids,
        n_init=1,
        random_state=random_state,
        max_iter=300,
        tol=1e-4
    )


    labels = kmeans.fit_predict(
        X_test
    )


    cluster_to_subject = {
        cluster_id:
            subject_name

        for cluster_id,
            subject_name

        in enumerate(
            centroid_subjects
        )
    }


    predictions = (
        test.copy()
        .reset_index(
            drop=True
        )
    )


    predictions[
        "cluster_id"
    ] = labels


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


    # --------------------------------------------------------
    # ETİKETLERİ AÇ
    # --------------------------------------------------------

    evaluation = (
        predictions[
            [
                "external_id",
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
        matched /
        total
    )


    # --------------------------------------------------------
    # METRİKLER
    # --------------------------------------------------------

    silhouette = (
        silhouette_score(
            X_test,
            labels
        )
    )

    davies = (
        davies_bouldin_score(
            X_test,
            labels
        )
    )

    calinski = (
        calinski_harabasz_score(
            X_test,
            labels
        )
    )


    cluster_sizes = (
        pd.Series(
            labels
        )
        .value_counts()
    )


    singleton = int(
        (
            cluster_sizes
            ==
            1
        ).sum()
    )


    le10 = int(
        (
            cluster_sizes
            <=
            10
        ).sum()
    )


    centroid_shift = (
        np.linalg.norm(
            kmeans.cluster_centers_
            -
            initial_centroids,
            axis=1
        )
    )


    return {
        "Random_State":
            random_state,

        "Seed_Count":
            seed_count,

        "Reference_Articles":
            len(reference),

        "Test_Articles":
            len(test),

        "Unique_Seed_Articles":
            len(
                unique_seed_ids
            ),

        "Subjects_With_Less_Seeds":
            subjects_with_less_seeds,

        "Topic_Match_Rate":
            topic_match_rate,

        "Topic_Match_Percent":
            topic_match_rate
            *
            100,

        "Silhouette":
            silhouette,

        "Davies_Bouldin":
            davies,

        "Calinski_Harabasz":
            calinski,

        "Singleton_Clusters":
            singleton,

        "Clusters_LE_10":
            le10,

        "Mean_Centroid_Shift":
            float(
                centroid_shift.mean()
            )
    }


# ============================================================
# TÜM DENEYLER
# ============================================================

results = []


for random_state in RANDOM_STATES:

    print("\n" + "=" * 110)
    print(
        "RANDOM STATE:",
        random_state
    )
    print("=" * 110)


    reference, test = (
        create_holdout(
            random_state
        )
    )


    print(
        "Reference:",
        len(reference),
        "| Test:",
        len(test)
    )


    for seed_count in SEED_COUNTS:

        result = run_experiment(
            reference,
            test,
            seed_count,
            random_state
        )

        results.append(
            result
        )


        print(
            f"{seed_count} seed"
            f" | Topic: "
            f"{result['Topic_Match_Percent']:.2f}%"
            f" | Sil: "
            f"{result['Silhouette']:.6f}"
            f" | DB: "
            f"{result['Davies_Bouldin']:.6f}"
        )


# ============================================================
# DETAYLI SONUÇ
# ============================================================

results_df = pd.DataFrame(
    results
)


print("\n" + "=" * 140)
print("TÜM DENEYLER")
print("=" * 140)


print(
    results_df[
        [
            "Random_State",
            "Seed_Count",
            "Unique_Seed_Articles",
            "Topic_Match_Percent",
            "Silhouette",
            "Davies_Bouldin",
            "Calinski_Harabasz",
            "Singleton_Clusters",
            "Clusters_LE_10",
            "Mean_Centroid_Shift"
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# ORTALAMALAR
# ============================================================

summary = (
    results_df
    .groupby(
        "Seed_Count"
    )
    .agg(
        Topic_Match_Mean=(
            "Topic_Match_Percent",
            "mean"
        ),

        Topic_Match_Std=(
            "Topic_Match_Percent",
            "std"
        ),

        Silhouette_Mean=(
            "Silhouette",
            "mean"
        ),

        Silhouette_Std=(
            "Silhouette",
            "std"
        ),

        Davies_Mean=(
            "Davies_Bouldin",
            "mean"
        ),

        Davies_Std=(
            "Davies_Bouldin",
            "std"
        ),

        Calinski_Mean=(
            "Calinski_Harabasz",
            "mean"
        ),

        Mean_Seed_Articles=(
            "Unique_Seed_Articles",
            "mean"
        )
    )
    .reset_index()
)


print("\n" + "=" * 120)
print("STABILITY ÖZETİ")
print("=" * 120)


print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# KAZANAN
# ============================================================

best_topic = (
    summary.loc[
        summary[
            "Topic_Match_Mean"
        ].idxmax()
    ]
)


print("\n" + "=" * 100)
print("SONUÇ")
print("=" * 100)


print(
    "En yüksek ortalama konu uyumu:",
    int(
        best_topic[
            "Seed_Count"
        ]
    ),
    "seed"
)


print(
    "Ortalama konu uyumu:",
    f"{best_topic['Topic_Match_Mean']:.2f}%"
)


print(
    "Standart sapma:",
    f"{best_topic['Topic_Match_Std']:.2f}"
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


summary.to_csv(
    "results/kmeans/holdout/seed_stability_summary.csv",
    index=False,
    encoding="utf-8-sig"
)


print("\nDosyalar oluşturuldu:")

print(
    OUTPUT_FILE
)

print(
    "results/kmeans/holdout/seed_stability_summary.csv"
)