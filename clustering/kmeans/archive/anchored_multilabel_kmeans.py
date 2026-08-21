import os
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)
from sklearn.preprocessing import MultiLabelBinarizer


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
VALIDATION_RATIO = 0.50

# Seed merkezinin ağırlığı
# 1.00 = tamamen seed merkezi
# 0.00 = tamamen K-Means final merkezi
ANCHOR_WEIGHTS = [
    1.00,
    0.75,
    0.50,
    0.25,
    0.00
]

THRESHOLDS = np.arange(
    0.35,
    0.801,
    0.01
)


# ============================================================
# VERİLERİ OKU
# ============================================================

print("=" * 110)
print("ANCHORED MULTI-LABEL SEEDED K-MEANS")
print("=" * 110)

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

print(
    "Referans makale:",
    len(reference)
)

print(
    "Test havuzu:",
    len(test)
)

print(
    "Embedding shape:",
    embeddings.shape
)


# ============================================================
# HAZIRLIK
# ============================================================

reference_ids = set(
    reference["external_id"]
)

reference_row_map = (
    reference
    .set_index(
        "external_id"
    )["embedding_row"]
    .to_dict()
)

label_counts = (
    subjects
    .groupby(
        "external_id"
    )["subject_fullname"]
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
    .groupby(
        "external_id"
    )["subject_fullname"]
    .apply(
        lambda values:
            sorted(
                set(
                    values
                    .dropna()
                    .astype(str)
                )
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

    norms[
        norms == 0
    ] = 1

    return (
        vectors / norms
    )


# ============================================================
# TEMSİL EDİCİ SEED SEÇİMİ
# ============================================================

def select_representative(
    candidate_ids,
    count
):

    valid_ids = [
        article_id
        for article_id in candidate_ids
        if article_id
        in reference_row_map
    ]

    if not valid_ids:
        return []

    rows = [
        int(
            reference_row_map[
                article_id
            ]
        )
        for article_id
        in valid_ids
    ]

    vectors = embeddings[
        rows
    ]

    center = (
        vectors.mean(
            axis=0
        )
    )

    center_norm = (
        np.linalg.norm(
            center
        )
    )

    if center_norm > 0:

        center = (
            center /
            center_norm
        )

    normalized_vectors = (
        normalize_vectors(
            vectors
        )
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
        for i
        in order[
            :selected_count
        ]
    ]


# ============================================================
# 1. BAŞLANGIÇ SEED CENTROIDLERİ
# ============================================================

print()
print("=" * 110)
print("SEED CENTROIDLERİ OLUŞTURULUYOR")
print("=" * 110)

initial_centroids = []
centroid_subjects = []

unique_seed_ids = set()


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
        ]["external_id"]
        .drop_duplicates()
        .tolist()
    )


    single_label_candidates = [
        article_id
        for article_id
        in candidates
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
            len(
                selected_ids
            )
        )

        remaining = [
            article_id
            for article_id
            in candidates
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
        for article_id
        in selected_ids
    ]


    vectors = embeddings[
        rows
    ]


    centroid = (
        vectors.mean(
            axis=0
        )
    )


    norm = np.linalg.norm(
        centroid
    )


    if norm > 0:

        centroid = (
            centroid /
            norm
        )


    initial_centroids.append(
        centroid
    )

    centroid_subjects.append(
        subject_name
    )


initial_centroids = np.vstack(
    initial_centroids
).astype(np.float32)


print(
    "Initial centroid matrix:",
    initial_centroids.shape
)

print(
    "Benzersiz seed makale:",
    len(
        unique_seed_ids
    )
)


# ============================================================
# 2. SEEDED K-MEANS
# ============================================================

print()
print("=" * 110)
print("SEEDED K-MEANS REFERENCE ÜZERİNDE ÇALIŞIYOR")
print("=" * 110)


reference_rows = (
    reference[
        "embedding_row"
    ]
    .astype(int)
    .to_numpy()
)


X_reference = embeddings[
    reference_rows
]


kmeans = KMeans(
    n_clusters=
        len(
            leaf_subjects
        ),
    init=
        initial_centroids,
    n_init=1,
    random_state=
        RANDOM_STATE,
    max_iter=300,
    tol=1e-4
)


reference_labels = (
    kmeans.fit_predict(
        X_reference
    )
)


kmeans_centroids = (
    kmeans.cluster_centers_
    .astype(np.float32)
)


print(
    "K-Means final centroid matrix:",
    kmeans_centroids.shape
)

print(
    "Kullanılan cluster:",
    len(
        np.unique(
            reference_labels
        )
    )
)


centroid_shift = (
    np.linalg.norm(
        kmeans_centroids
        -
        initial_centroids,
        axis=1
    )
)


print(
    "Ortalama K-Means hareketi:",
    round(
        float(
            centroid_shift.mean()
        ),
        4
    )
)

print(
    "Maximum K-Means hareketi:",
    round(
        float(
            centroid_shift.max()
        ),
        4
    )
)


# ============================================================
# 3. VALIDATION / FINAL TEST
# ============================================================

validation_df, final_test_df = (
    train_test_split(
        test,
        test_size=(
            1
            -
            VALIDATION_RATIO
        ),
        random_state=
            RANDOM_STATE,
        shuffle=True
    )
)


validation_df = (
    validation_df
    .reset_index(
        drop=True
    )
)


final_test_df = (
    final_test_df
    .reset_index(
        drop=True
    )
)


print()
print("=" * 110)
print("VALIDATION / FINAL TEST")
print("=" * 110)


print(
    "Validation:",
    len(
        validation_df
    )
)

print(
    "Final test:",
    len(
        final_test_df
    )
)


# ============================================================
# 4. GERÇEK MULTI-LABEL MATRİSLER
# ============================================================

mlb = MultiLabelBinarizer(
    classes=
        leaf_subjects
)

mlb.fit(
    [
        leaf_subjects
    ]
)


def get_true_binary(df):

    labels = [
        true_subjects.get(
            str(
                external_id
            ),
            []
        )
        for external_id
        in df[
            "external_id"
        ]
    ]

    return (
        mlb.transform(
            labels
        )
    )


y_validation_true = (
    get_true_binary(
        validation_df
    )
)


y_final_true = (
    get_true_binary(
        final_test_df
    )
)


# ============================================================
# 5. TEST EMBEDDINGLERİNİ HAZIRLA
# ============================================================

def get_normalized_vectors(df):

    rows = (
        df[
            "embedding_row"
        ]
        .astype(int)
        .to_numpy()
    )

    vectors = embeddings[
        rows
    ]

    return (
        normalize_vectors(
            vectors
        )
    )


X_validation = (
    get_normalized_vectors(
        validation_df
    )
)


X_final = (
    get_normalized_vectors(
        final_test_df
    )
)


# ============================================================
# 6. MULTI-LABEL TAHMİN
# ============================================================

def predictions_from_threshold(
    scores,
    threshold
):

    predictions = (
        scores
        >=
        threshold
    ).astype(int)


    # Hiçbir cluster geçmezse
    # en yakın cluster'ı yine ver.
    empty_rows = np.where(
        predictions.sum(
            axis=1
        )
        ==
        0
    )[0]


    for row_index in empty_rows:

        best_index = int(
            np.argmax(
                scores[
                    row_index
                ]
            )
        )

        predictions[
            row_index,
            best_index
        ] = 1


    return predictions


# ============================================================
# 7. METRİKLER
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    return {

        "Micro_Precision":
            precision_score(
                y_true,
                y_pred,
                average="micro",
                zero_division=0
            ),

        "Micro_Recall":
            recall_score(
                y_true,
                y_pred,
                average="micro",
                zero_division=0
            ),

        "Micro_F1":
            f1_score(
                y_true,
                y_pred,
                average="micro",
                zero_division=0
            ),

        "Macro_Precision":
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            ),

        "Macro_Recall":
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            ),

        "Macro_F1":
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            ),

        "Sample_F1":
            f1_score(
                y_true,
                y_pred,
                average="samples",
                zero_division=0
            ),

        "Average_Predicted_Labels":
            float(
                y_pred
                .sum(
                    axis=1
                )
                .mean()
            )
    }


# ============================================================
# 8. TEK ANCHOR AĞIRLIĞINI DENE
# ============================================================

def run_anchor_experiment(
    anchor_weight
):

    print()
    print("=" * 110)

    print(
        f"ANCHOR WEIGHT: {anchor_weight:.2f}"
    )

    print("=" * 110)


    # ========================================================
    # Anchored centroid:
    #
    # anchor_weight = seed payı
    #
    # 1.00 -> %100 seed
    # 0.00 -> %100 K-Means
    # ========================================================

    anchored_centroids = (
        anchor_weight
        *
        initial_centroids
        +
        (
            1
            -
            anchor_weight
        )
        *
        kmeans_centroids
    )


    # Cosine için normalize
    anchored_centroids = (
        normalize_vectors(
            anchored_centroids
        )
    )


    # ========================================================
    # Anchored merkez gerçekten başlangıçtan
    # ne kadar uzak?
    # ========================================================

    anchored_shift = (
        np.linalg.norm(
            anchored_centroids
            -
            normalize_vectors(
                initial_centroids
            ),
            axis=1
        )
    )


    print(
        "Ortalama anchored hareket:",
        round(
            float(
                anchored_shift.mean()
            ),
            4
        )
    )


    # ========================================================
    # COSINE MATRİSLER
    # ========================================================

    validation_scores = (
        X_validation
        @
        anchored_centroids.T
    )


    final_scores = (
        X_final
        @
        anchored_centroids.T
    )


    # ========================================================
    # VALIDATION'DA THRESHOLD ARA
    # ========================================================

    threshold_results = []


    for threshold in THRESHOLDS:

        y_validation_pred = (
            predictions_from_threshold(
                validation_scores,
                threshold
            )
        )


        metrics = (
            calculate_metrics(
                y_validation_true,
                y_validation_pred
            )
        )


        threshold_results.append(
            {
                "Threshold":
                    float(
                        threshold
                    ),

                **metrics
            }
        )


    threshold_df = pd.DataFrame(
        threshold_results
    )


    best_row = (
        threshold_df
        .sort_values(
            [
                "Micro_F1",
                "Macro_F1"
            ],
            ascending=[
                False,
                False
            ]
        )
        .iloc[0]
    )


    best_threshold = float(
        best_row[
            "Threshold"
        ]
    )


    print(
        "En iyi validation threshold:",
        round(
            best_threshold,
            4
        )
    )

    print(
        "Validation Micro F1:",
        f"{best_row['Micro_F1'] * 100:.2f}%"
    )


    # ========================================================
    # FINAL TEST
    # ========================================================

    y_final_pred = (
        predictions_from_threshold(
            final_scores,
            best_threshold
        )
    )


    final_metrics = (
        calculate_metrics(
            y_final_true,
            y_final_pred
        )
    )


    real_average_labels = float(
        y_final_true
        .sum(
            axis=1
        )
        .mean()
    )


    exact_match = float(
        np.all(
            y_final_true
            ==
            y_final_pred,
            axis=1
        )
        .mean()
    )


    at_least_one = float(
        (
            (
                y_final_true
                &
                y_final_pred
            )
            .sum(
                axis=1
            )
            >
            0
        )
        .mean()
    )


    print(
        "Final Micro Precision:",
        f"{final_metrics['Micro_Precision'] * 100:.2f}%"
    )

    print(
        "Final Micro Recall:",
        f"{final_metrics['Micro_Recall'] * 100:.2f}%"
    )

    print(
        "Final Micro F1:",
        f"{final_metrics['Micro_F1'] * 100:.2f}%"
    )

    print(
        "Final Macro F1:",
        f"{final_metrics['Macro_F1'] * 100:.2f}%"
    )

    print(
        "Ortalama tahmin edilen etiket:",
        round(
            final_metrics[
                "Average_Predicted_Labels"
            ],
            2
        )
    )

    print(
        "Gerçek ortalama etiket:",
        round(
            real_average_labels,
            2
        )
    )

    print(
        "Exact set match:",
        f"{exact_match * 100:.2f}%"
    )

    print(
        "En az 1 gerçek etiket:",
        f"{at_least_one * 100:.2f}%"
    )


    return {

        "Anchor_Weight":
            anchor_weight,

        "Seed_Percent":
            anchor_weight * 100,

        "KMeans_Percent":
            (
                1
                -
                anchor_weight
            )
            *
            100,

        "Mean_Anchored_Shift":
            float(
                anchored_shift.mean()
            ),

        "Best_Threshold":
            best_threshold,

        "Validation_Micro_F1":
            float(
                best_row[
                    "Micro_F1"
                ]
            ),

        **final_metrics,

        "Average_True_Labels":
            real_average_labels,

        "Exact_Match_Rate":
            exact_match,

        "At_Least_One_Match_Rate":
            at_least_one
    }


# ============================================================
# 9. TÜM ANCHOR AĞIRLIKLARINI DENE
# ============================================================

results = []


for anchor_weight in ANCHOR_WEIGHTS:

    result = (
        run_anchor_experiment(
            anchor_weight
        )
    )

    results.append(
        result
    )


results_df = pd.DataFrame(
    results
)


# ============================================================
# 10. GENEL KARŞILAŞTIRMA
# ============================================================

print()
print("=" * 140)
print("ANCHORED K-MEANS GENEL KARŞILAŞTIRMA")
print("=" * 140)


display_columns = [
    "Anchor_Weight",
    "Seed_Percent",
    "KMeans_Percent",
    "Mean_Anchored_Shift",
    "Best_Threshold",
    "Micro_Precision",
    "Micro_Recall",
    "Micro_F1",
    "Macro_F1",
    "Sample_F1",
    "Average_Predicted_Labels",
    "At_Least_One_Match_Rate"
]


print(
    results_df[
        display_columns
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# 11. KAZANAN
# ============================================================

best = (
    results_df.loc[
        results_df[
            "Micro_F1"
        ]
        .idxmax()
    ]
)


print()
print("=" * 110)
print("KAZANAN")
print("=" * 110)


print(
    "Seed ağırlığı:",
    f"{best['Seed_Percent']:.0f}%"
)

print(
    "K-Means ağırlığı:",
    f"{best['KMeans_Percent']:.0f}%"
)

print(
    "Micro Precision:",
    f"{best['Micro_Precision'] * 100:.2f}%"
)

print(
    "Micro Recall:",
    f"{best['Micro_Recall'] * 100:.2f}%"
)

print(
    "Micro F1:",
    f"{best['Micro_F1'] * 100:.2f}%"
)

print(
    "Macro F1:",
    f"{best['Macro_F1'] * 100:.2f}%"
)

print(
    "Threshold:",
    round(
        best[
            "Best_Threshold"
        ],
        4
    )
)


# ============================================================
# 12. KAYDET
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "anchored_multilabel_comparison.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


print()
print("Dosya oluşturuldu:")

print(
    "results/kmeans/holdout/"
    "anchored_multilabel_comparison.csv"
)