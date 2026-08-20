import os
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans


# ============================================================
# AYARLAR
# ============================================================

EMBEDDING_FILE = "embeddings/mpnet_multilingual_embeddings.npy"
SUBJECT_FILE = "data/article_subjects.csv"

REFERENCE_FILE = "results/kmeans/holdout/reference.csv"
TEST_FILE = "results/kmeans/holdout/test.csv"

OUTPUT_DIR = "results/kmeans/holdout"

SEEDS_PER_SUBJECT = 10
PROTOTYPE_COUNTS = [1, 2, 3]

RANDOM_STATE = 42


# ============================================================
# VERİLERİ OKU
# ============================================================

print("=" * 100)
print("MULTI-CENTROID + COSINE DENEYİ")
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


print("Referans makale:", len(reference))
print("Test makale:", len(test))
print("Embedding shape:", embeddings.shape)


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


true_subjects = (
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


print(
    "Leaf konu sayısı:",
    len(leaf_subjects)
)


# ============================================================
# NORMALIZE
# ============================================================

def normalize_vectors(vectors):

    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True
    )

    norms[norms == 0] = 1

    return vectors / norms


# ============================================================
# TEMSİL EDİCİ 10 SEED SEÇ
# ============================================================

def select_representative(
    candidate_ids,
    count
):

    valid_ids = [
        article_id
        for article_id in candidate_ids
        if article_id in reference_row_map
    ]

    if not valid_ids:
        return []


    rows = [
        int(
            reference_row_map[
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


    normalized_vectors = normalize_vectors(
        vectors
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
# HER KONU İÇİN 10 SEED HAZIRLA
# ============================================================

print("\n" + "=" * 100)
print("SEEDLER HAZIRLANIYOR")
print("=" * 100)


subject_seed_vectors = {}

unique_seed_ids = set()


for subject_name in leaf_subjects:

    candidates = (
        subjects[
            (
                subjects["subject_fullname"]
                ==
                subject_name
            )
            &
            (
                subjects["external_id"]
                .isin(reference_ids)
            )
        ]["external_id"]
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


    # Önce temiz single-label makaleler
    if (
        len(single_label_candidates)
        >=
        SEEDS_PER_SUBJECT
    ):

        selected_ids = (
            select_representative(
                single_label_candidates,
                SEEDS_PER_SUBJECT
            )
        )

    else:

        selected_ids.extend(
            select_representative(
                single_label_candidates,
                len(
                    single_label_candidates
                )
            )
        )


        needed = (
            SEEDS_PER_SUBJECT
            -
            len(selected_ids)
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
                needed
            )
        )


    if not selected_ids:

        raise ValueError(
            f"Seed bulunamadı: {subject_name}"
        )


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


    subject_seed_vectors[
        subject_name
    ] = embeddings[
        rows
    ]


print(
    "Benzersiz seed makale:",
    len(unique_seed_ids)
)


# ============================================================
# TEST EMBEDDINGLERİ
# ============================================================

test_rows = (
    test["embedding_row"]
    .astype(int)
    .to_numpy()
)


X_test = embeddings[
    test_rows
]


X_test_normalized = normalize_vectors(
    X_test
)


# ============================================================
# PROTOTYPE OLUŞTUR
# ============================================================

def create_prototypes(
    vectors,
    prototype_count
):

    # Seed sayısı prototype sayısından azsa
    # mümkün olan kadar prototype oluştur.
    actual_count = min(
        prototype_count,
        len(vectors)
    )


    # --------------------------------------------------------
    # 1 prototype:
    # Eski fixed-centroid yöntemiyle aynı.
    # --------------------------------------------------------

    if actual_count == 1:

        prototype = vectors.mean(
            axis=0
        )

        norm = np.linalg.norm(
            prototype
        )

        if norm > 0:

            prototype = (
                prototype /
                norm
            )

        return np.array(
            [prototype],
            dtype=np.float32
        )


    # --------------------------------------------------------
    # 2 veya 3 prototype:
    # Konunun kendi seedlerini küçük K-Means ile ayır.
    # --------------------------------------------------------

    kmeans = KMeans(
        n_clusters=actual_count,
        init="k-means++",
        n_init=10,
        random_state=RANDOM_STATE
    )


    kmeans.fit(
        vectors
    )


    prototypes = (
        kmeans.cluster_centers_
        .astype(np.float32)
    )


    prototypes = normalize_vectors(
        prototypes
    )


    return prototypes


# ============================================================
# TEK PROTOTYPE SAYISINI DEĞERLENDİR
# ============================================================

def evaluate_prototype_count(
    prototype_count
):

    print("\n" + "=" * 100)

    print(
        f"{prototype_count} PROTOTYPE / KONU"
    )

    print("=" * 100)


    all_prototypes = []

    prototype_subjects = []


    # --------------------------------------------------------
    # Tüm konu prototypelarını oluştur
    # --------------------------------------------------------

    for subject_name in leaf_subjects:

        vectors = (
            subject_seed_vectors[
                subject_name
            ]
        )


        prototypes = create_prototypes(
            vectors,
            prototype_count
        )


        for prototype in prototypes:

            all_prototypes.append(
                prototype
            )

            prototype_subjects.append(
                subject_name
            )


    prototype_matrix = np.vstack(
        all_prototypes
    ).astype(np.float32)


    print(
        "Toplam prototype:",
        len(prototype_matrix)
    )


    # --------------------------------------------------------
    # Cosine similarity
    # --------------------------------------------------------

    similarities = (
        X_test_normalized
        @
        prototype_matrix.T
    )


    # ========================================================
    # ÖNEMLİ:
    #
    # Bir konu birden fazla prototype'a sahip.
    # Aynı konunun prototypelarından EN YÜKSEK similarity
    # o konunun skoru kabul edilir.
    #
    # Böylece 390/585 ayrı sınıf oluşmaz.
    # Hâlâ 195 konu vardır.
    # ========================================================

    subject_score_matrix = np.full(
        (
            len(test),
            len(leaf_subjects)
        ),
        -np.inf,
        dtype=np.float32
    )


    subject_index_map = {
        subject_name: index
        for index, subject_name
        in enumerate(leaf_subjects)
    }


    for prototype_index, subject_name in enumerate(
        prototype_subjects
    ):

        subject_index = (
            subject_index_map[
                subject_name
            ]
        )


        subject_score_matrix[
            :,
            subject_index
        ] = np.maximum(
            subject_score_matrix[
                :,
                subject_index
            ],
            similarities[
                :,
                prototype_index
            ]
        )


    # --------------------------------------------------------
    # TOP-5 konu
    # --------------------------------------------------------

    top5_indices = np.argsort(
        -subject_score_matrix,
        axis=1
    )[:, :5]


    top5_scores = np.take_along_axis(
        subject_score_matrix,
        top5_indices,
        axis=1
    )


    top1_correct = 0
    top3_correct = 0
    top5_correct = 0


    prediction_rows = []


    reset_test = test.reset_index(
        drop=True
    )


    for i, row in reset_test.iterrows():

        external_id = str(
            row["external_id"]
        )


        real_topics = true_subjects.get(
            external_id,
            set()
        )


        predicted_topics = [
            leaf_subjects[index]
            for index
            in top5_indices[i]
        ]


        predicted_scores = [
            float(score)
            for score
            in top5_scores[i]
        ]


        top1_match = any(
            topic in real_topics
            for topic
            in predicted_topics[:1]
        )


        top3_match = any(
            topic in real_topics
            for topic
            in predicted_topics[:3]
        )


        top5_match = any(
            topic in real_topics
            for topic
            in predicted_topics[:5]
        )


        if top1_match:
            top1_correct += 1

        if top3_match:
            top3_correct += 1

        if top5_match:
            top5_correct += 1


        prediction_rows.append(
            {
                "external_id":
                    external_id,

                "prototype_count":
                    prototype_count,

                "top1_subject":
                    predicted_topics[0],

                "top1_score":
                    predicted_scores[0],

                "top2_subject":
                    predicted_topics[1],

                "top2_score":
                    predicted_scores[1],

                "top3_subject":
                    predicted_topics[2],

                "top3_score":
                    predicted_scores[2],

                "top4_subject":
                    predicted_topics[3],

                "top4_score":
                    predicted_scores[3],

                "top5_subject":
                    predicted_topics[4],

                "top5_score":
                    predicted_scores[4],

                "top1_match":
                    top1_match,

                "top3_match":
                    top3_match,

                "top5_match":
                    top5_match
            }
        )


    total = len(
        test
    )


    top1_rate = (
        top1_correct /
        total
    )

    top3_rate = (
        top3_correct /
        total
    )

    top5_rate = (
        top5_correct /
        total
    )


    print(
        "Top-1:",
        f"{top1_correct}/{total}",
        f"= {top1_rate * 100:.2f}%"
    )


    print(
        "Top-3:",
        f"{top3_correct}/{total}",
        f"= {top3_rate * 100:.2f}%"
    )


    print(
        "Top-5:",
        f"{top5_correct}/{total}",
        f"= {top5_rate * 100:.2f}%"
    )


    return (
        {
            "Prototype_Count":
                prototype_count,

            "Total_Prototypes":
                len(
                    prototype_matrix
                ),

            "Test_Articles":
                total,

            "Top1_Correct":
                top1_correct,

            "Top1_Percent":
                top1_rate * 100,

            "Top3_Correct":
                top3_correct,

            "Top3_Percent":
                top3_rate * 100,

            "Top5_Correct":
                top5_correct,

            "Top5_Percent":
                top5_rate * 100
        },

        pd.DataFrame(
            prediction_rows
        )
    )


# ============================================================
# 1 / 2 / 3 PROTOTYPE DENEYLERİ
# ============================================================

results = []

all_predictions = []


for prototype_count in PROTOTYPE_COUNTS:

    result, predictions = (
        evaluate_prototype_count(
            prototype_count
        )
    )


    results.append(
        result
    )


    all_predictions.append(
        predictions
    )


# ============================================================
# KARŞILAŞTIRMA
# ============================================================

results_df = pd.DataFrame(
    results
)


print("\n" + "=" * 110)
print("MULTI-CENTROID GENEL KARŞILAŞTIRMA")
print("=" * 110)


print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# KAZANAN
# ============================================================

best = (
    results_df.loc[
        results_df[
            "Top1_Percent"
        ].idxmax()
    ]
)


print("\n" + "=" * 100)
print("KAZANAN")
print("=" * 100)


print(
    "En iyi prototype sayısı:",
    int(
        best[
            "Prototype_Count"
        ]
    )
)


print(
    "Top-1:",
    f"{best['Top1_Percent']:.2f}%"
)


print(
    "Top-3:",
    f"{best['Top3_Percent']:.2f}%"
)


print(
    "Top-5:",
    f"{best['Top5_Percent']:.2f}%"
)


# ============================================================
# KAYDET
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "multi_centroid_comparison.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


all_predictions_df = pd.concat(
    all_predictions,
    ignore_index=True
)


all_predictions_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "multi_centroid_predictions.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


print("\nDosyalar oluşturuldu:")

print(
    "results/kmeans/holdout/"
    "multi_centroid_comparison.csv"
)

print(
    "results/kmeans/holdout/"
    "multi_centroid_predictions.csv"
)