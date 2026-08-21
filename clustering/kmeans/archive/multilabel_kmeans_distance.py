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

# İkinci/üçüncü cluster'ı kabul ederken
# en yakın cluster uzaklığına göre oran kullanacağız.
#
# Örnek:
# nearest_distance = 0.50
# ratio = 1.20
#
# kabul sınırı = 0.60
#
# distance <= 0.60 olan diğer clusterlar da etiket olur.
DISTANCE_RATIOS = np.arange(
    1.00,
    2.01,
    0.05
)

MAX_LABELS = 5


# ============================================================
# VERİLERİ OKU
# ============================================================

print("=" * 110)
print("MULTI-LABEL SEEDED K-MEANS + EUCLIDEAN DISTANCE")
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

reference["external_id"] = reference["external_id"].astype(str)
test["external_id"] = test["external_id"].astype(str)
subjects["external_id"] = subjects["external_id"].astype(str)

print("Referans makale:", len(reference))
print("Test havuzu:", len(test))
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

print("Leaf konu sayısı:", len(leaf_subjects))


# ============================================================
# TEMSİL EDİCİ SEED SEÇ
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

    vectors = embeddings[
        rows
    ]

    center = vectors.mean(
        axis=0
    )

    # Merkeze Öklid uzaklığı
    distances = np.linalg.norm(
        vectors - center,
        axis=1
    )

    order = np.argsort(
        distances
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
# 1. BAŞLANGIÇ CENTROIDLERİNİ OLUŞTUR
# ============================================================

print()
print("=" * 110)
print("BAŞLANGIÇ CENTROIDLERİ OLUŞTURULUYOR")
print("=" * 110)

initial_centroids = []
centroid_subjects = []

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

    if len(single_label_candidates) >= SEEDS_PER_SUBJECT:

        selected_ids = select_representative(
            single_label_candidates,
            SEEDS_PER_SUBJECT
        )

    else:

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
    len(unique_seed_ids)
)


# ============================================================
# 2. SEEDED K-MEANS
# ============================================================

print()
print("=" * 110)
print("SEEDED K-MEANS REFERENCE ÜZERİNDE ÇALIŞIYOR")
print("=" * 110)

reference_rows = (
    reference["embedding_row"]
    .astype(int)
    .to_numpy()
)

X_reference = embeddings[
    reference_rows
]


kmeans = KMeans(
    n_clusters=len(leaf_subjects),
    init=initial_centroids,
    n_init=1,
    random_state=RANDOM_STATE,
    max_iter=300,
    tol=1e-4
)

reference_labels = kmeans.fit_predict(
    X_reference
)

final_centroids = (
    kmeans.cluster_centers_
    .astype(np.float32)
)


print(
    "Final centroid matrix:",
    final_centroids.shape
)

print(
    "Kullanılan cluster:",
    len(
        np.unique(
            reference_labels
        )
    )
)


centroid_shift = np.linalg.norm(
    final_centroids
    -
    initial_centroids,
    axis=1
)


print(
    "Ortalama centroid hareketi:",
    round(
        float(
            centroid_shift.mean()
        ),
        4
    )
)


# ============================================================
# 3. VALIDATION / FINAL TEST
# ============================================================

validation_df, final_test_df = train_test_split(
    test,
    test_size=(
        1 - VALIDATION_RATIO
    ),
    random_state=RANDOM_STATE,
    shuffle=True
)

validation_df = (
    validation_df
    .reset_index(drop=True)
)

final_test_df = (
    final_test_df
    .reset_index(drop=True)
)


print()
print("=" * 110)
print("VALIDATION / FINAL TEST")
print("=" * 110)

print(
    "Validation:",
    len(validation_df)
)

print(
    "Final test:",
    len(final_test_df)
)


# ============================================================
# 4. GERÇEK MULTI-LABEL ETİKET MATRİSLERİ
# ============================================================

mlb = MultiLabelBinarizer(
    classes=leaf_subjects
)

mlb.fit(
    [leaf_subjects]
)


def get_true_binary(df):

    labels = [
        true_subjects.get(
            str(external_id),
            []
        )
        for external_id
        in df["external_id"]
    ]

    return mlb.transform(
        labels
    )


y_validation_true = get_true_binary(
    validation_df
)

y_final_true = get_true_binary(
    final_test_df
)


# ============================================================
# 5. K-MEANS CENTROID UZAKLIKLARI
# ============================================================

def create_distance_matrix(df):

    rows = (
        df["embedding_row"]
        .astype(int)
        .to_numpy()
    )

    vectors = embeddings[
        rows
    ]

    # Shape:
    # article x cluster x embedding_dim
    differences = (
        vectors[:, np.newaxis, :]
        -
        final_centroids[np.newaxis, :, :]
    )

    distances = np.linalg.norm(
        differences,
        axis=2
    )

    return distances


validation_distances = (
    create_distance_matrix(
        validation_df
    )
)

final_distances = (
    create_distance_matrix(
        final_test_df
    )
)


print(
    "Validation distance matrix:",
    validation_distances.shape
)

print(
    "Final distance matrix:",
    final_distances.shape
)


# ============================================================
# 6. DISTANCE RATIO İLE MULTI-LABEL TAHMİN
# ============================================================

def predictions_from_ratio(
    distances,
    ratio
):

    n_articles = distances.shape[0]

    predictions = np.zeros(
        distances.shape,
        dtype=int
    )

    for i in range(
        n_articles
    ):

        row_distances = distances[i]

        sorted_indices = np.argsort(
            row_distances
        )

        nearest_index = (
            sorted_indices[0]
        )

        nearest_distance = (
            row_distances[
                nearest_index
            ]
        )

        # En yakın her zaman verilir
        predictions[
            i,
            nearest_index
        ] = 1

        max_distance = (
            nearest_distance
            *
            ratio
        )

        label_count = 1

        # En fazla MAX_LABELS tane
        for cluster_index in sorted_indices[1:]:

            if label_count >= MAX_LABELS:
                break

            if (
                row_distances[
                    cluster_index
                ]
                <=
                max_distance
            ):

                predictions[
                    i,
                    cluster_index
                ] = 1

                label_count += 1

            else:
                break

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
                .sum(axis=1)
                .mean()
            )
    }


# ============================================================
# 8. VALIDATION'DA EN İYİ DISTANCE RATIO
# ============================================================

print()
print("=" * 110)
print("VALIDATION - DISTANCE RATIO ARAMASI")
print("=" * 110)


ratio_results = []


for ratio in DISTANCE_RATIOS:

    y_pred = predictions_from_ratio(
        validation_distances,
        ratio
    )

    metrics = calculate_metrics(
        y_validation_true,
        y_pred
    )

    ratio_results.append(
        {
            "Distance_Ratio":
                float(ratio),

            **metrics
        }
    )


ratio_df = pd.DataFrame(
    ratio_results
)


best_row = (
    ratio_df
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


best_ratio = float(
    best_row[
        "Distance_Ratio"
    ]
)


print(
    "En iyi distance ratio:",
    round(
        best_ratio,
        4
    )
)

print(
    "Validation Micro Precision:",
    f"{best_row['Micro_Precision'] * 100:.2f}%"
)

print(
    "Validation Micro Recall:",
    f"{best_row['Micro_Recall'] * 100:.2f}%"
)

print(
    "Validation Micro F1:",
    f"{best_row['Micro_F1'] * 100:.2f}%"
)

print(
    "Validation Macro F1:",
    f"{best_row['Macro_F1'] * 100:.2f}%"
)

print(
    "Ortalama tahmin edilen etiket:",
    round(
        float(
            best_row[
                "Average_Predicted_Labels"
            ]
        ),
        2
    )
)


# ============================================================
# 9. FINAL TEST
# ============================================================

print()
print("=" * 110)
print("FINAL MULTI-LABEL K-MEANS DISTANCE TEST")
print("=" * 110)


y_final_pred = (
    predictions_from_ratio(
        final_distances,
        best_ratio
    )
)


final_metrics = calculate_metrics(
    y_final_true,
    y_final_pred
)


real_average_labels = float(
    y_final_true
    .sum(axis=1)
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
        .sum(axis=1)
        >
        0
    )
    .mean()
)


print(
    "Final test makale:",
    len(final_test_df)
)

print()

print(
    "Micro Precision:",
    f"{final_metrics['Micro_Precision'] * 100:.2f}%"
)

print(
    "Micro Recall:",
    f"{final_metrics['Micro_Recall'] * 100:.2f}%"
)

print(
    "Micro F1:",
    f"{final_metrics['Micro_F1'] * 100:.2f}%"
)

print()

print(
    "Macro Precision:",
    f"{final_metrics['Macro_Precision'] * 100:.2f}%"
)

print(
    "Macro Recall:",
    f"{final_metrics['Macro_Recall'] * 100:.2f}%"
)

print(
    "Macro F1:",
    f"{final_metrics['Macro_F1'] * 100:.2f}%"
)

print()

print(
    "Sample F1:",
    f"{final_metrics['Sample_F1'] * 100:.2f}%"
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


# ============================================================
# 10. TOP-K ANALİZİ
# ============================================================

print()
print("=" * 110)
print("TOP-K K-MEANS CENTROID ANALİZİ")
print("=" * 110)


for top_k in [
    1,
    2,
    3,
    5
]:

    top_indices = np.argsort(
        final_distances,
        axis=1
    )[:, :top_k]

    top_predictions = np.zeros(
        final_distances.shape,
        dtype=int
    )

    for i in range(
        len(
            final_test_df
        )
    ):

        top_predictions[
            i,
            top_indices[i]
        ] = 1

    metrics = calculate_metrics(
        y_final_true,
        top_predictions
    )

    print(
        f"Top-{top_k}"
        f" | Micro F1: "
        f"{metrics['Micro_F1'] * 100:.2f}%"
        f" | Precision: "
        f"{metrics['Micro_Precision'] * 100:.2f}%"
        f" | Recall: "
        f"{metrics['Micro_Recall'] * 100:.2f}%"
    )


# ============================================================
# 11. KAYDET
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


ratio_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "kmeans_distance_ratio_search.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


summary = pd.DataFrame(
    [
        {
            "Method":
                "Multi-label Seeded K-Means Distance",

            "Seeds_Per_Subject":
                SEEDS_PER_SUBJECT,

            "Best_Distance_Ratio":
                best_ratio,

            "Reference_Articles":
                len(reference),

            "Final_Test_Articles":
                len(final_test_df),

            "Mean_Centroid_Shift":
                float(
                    centroid_shift.mean()
                ),

            **final_metrics,

            "Average_True_Labels":
                real_average_labels,

            "Exact_Match_Rate":
                exact_match,

            "At_Least_One_Match_Rate":
                at_least_one
        }
    ]
)


summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "kmeans_distance_multilabel_summary.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


print()
print("=" * 110)
print("TAMAMLANDI")
print("=" * 110)

print(
    "results/kmeans/holdout/"
    "kmeans_distance_ratio_search.csv"
)

print(
    "results/kmeans/holdout/"
    "kmeans_distance_multilabel_summary.csv"
)