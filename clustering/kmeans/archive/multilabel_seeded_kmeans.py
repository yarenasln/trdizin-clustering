import os
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
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

# Validation setinde denenecek cosine threshold değerleri
THRESHOLDS = np.arange(0.35, 0.801, 0.01)


# ============================================================
# VERİLERİ OKU
# ============================================================

print("=" * 110)
print("MULTI-LABEL SEEDED K-MEANS + COSINE")
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
# KONU BİLGİLERİNİ HAZIRLA
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
        lambda values: sorted(
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
# NORMALIZATION
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
        int(reference_row_map[article_id])
        for article_id in valid_ids
    ]

    vectors = embeddings[rows]

    center = vectors.mean(axis=0)

    center_norm = np.linalg.norm(center)

    if center_norm > 0:
        center = center / center_norm

    normalized_vectors = normalize_vectors(
        vectors
    )

    similarities = (
        normalized_vectors
        @ center
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

    # Öncelikle yalnızca tek etiketi olan temiz örnekleri seç
    single_label_candidates = [
        article_id
        for article_id in candidates
        if label_counts.get(article_id, 0) == 1
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
        int(reference_row_map[article_id])
        for article_id in selected_ids
    ]

    vectors = embeddings[rows]

    centroid = vectors.mean(axis=0)

    norm = np.linalg.norm(
        centroid
    )

    if norm > 0:
        centroid = centroid / norm

    initial_centroids.append(
        centroid
    )

    # Cluster sırası ile konu sırası birlikte tutuluyor.
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
#
# ÖNEMLİ:
# K-Means yalnızca %20 reference verisini görüyor.
# Test havuzu K-Means eğitimine girmiyor.
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

reference_cluster_labels = (
    kmeans.fit_predict(
        X_reference
    )
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
    "Reference'ta kullanılan cluster:",
    len(
        np.unique(
            reference_cluster_labels
        )
    )
)


# ============================================================
# 3. CENTROID HAREKETİ
# ============================================================

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

print(
    "Medyan centroid hareketi:",
    round(
        float(
            np.median(
                centroid_shift
            )
        ),
        4
    )
)

print(
    "En fazla centroid hareketi:",
    round(
        float(
            centroid_shift.max()
        ),
        4
    )
)


# ============================================================
# 4. FINAL K-MEANS CENTROIDLERİNİ NORMALIZE ET
# ============================================================

final_centroids_normalized = (
    normalize_vectors(
        final_centroids
    )
)


# ============================================================
# 5. TEST HAVUZUNU VALIDATION / FINAL TEST AYIR
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
# 6. FINAL K-MEANS CENTROIDLERİNE COSINE SIMILARITY
# ============================================================

def create_similarity_matrix(df):

    rows = (
        df["embedding_row"]
        .astype(int)
        .to_numpy()
    )

    vectors = embeddings[
        rows
    ]

    normalized_vectors = (
        normalize_vectors(
            vectors
        )
    )

    similarity_matrix = (
        normalized_vectors
        @
        final_centroids_normalized.T
    )

    return similarity_matrix


validation_scores = (
    create_similarity_matrix(
        validation_df
    )
)

final_scores = (
    create_similarity_matrix(
        final_test_df
    )
)

print(
    "Validation similarity matrix:",
    validation_scores.shape
)

print(
    "Final test similarity matrix:",
    final_scores.shape
)


# ============================================================
# 7. GERÇEK MULTI-LABEL MATRİSLERİ
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
# 8. THRESHOLD İLE MULTI-LABEL TAHMİN
# ============================================================

def predictions_from_threshold(
    scores,
    threshold
):

    predictions = (
        scores >= threshold
    ).astype(int)

    # Hiçbir centroid threshold'u geçmezse
    # makaleyi tamamen etiketsiz bırakmıyoruz.
    # En yakın K-Means centroidini veriyoruz.
    empty_rows = np.where(
        predictions.sum(axis=1)
        ==
        0
    )[0]

    for row_index in empty_rows:

        best_index = int(
            np.argmax(
                scores[row_index]
            )
        )

        predictions[
            row_index,
            best_index
        ] = 1

    return predictions


# ============================================================
# 9. METRİKLER
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
# 10. VALIDATION'DA THRESHOLD BUL
# ============================================================

print()
print("=" * 110)
print("VALIDATION - THRESHOLD ARAMASI")
print("=" * 110)

threshold_results = []


for threshold in THRESHOLDS:

    y_pred = (
        predictions_from_threshold(
            validation_scores,
            threshold
        )
    )

    metrics = calculate_metrics(
        y_validation_true,
        y_pred
    )

    threshold_results.append(
        {
            "Threshold":
                float(threshold),

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
    best_row["Threshold"]
)

print(
    "En iyi threshold:",
    round(
        best_threshold,
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
# 11. FINAL TEST
# ============================================================

print()
print("=" * 110)
print("FINAL MULTI-LABEL SEEDED K-MEANS TEST")
print("=" * 110)

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
    .sum(axis=1)
    .mean()
)

exact_matches = np.all(
    y_final_true
    ==
    y_final_pred,
    axis=1
)

intersection_counts = (
    (
        y_final_true
        &
        y_final_pred
    )
    .sum(axis=1)
)

at_least_one = (
    intersection_counts > 0
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
    f"{exact_matches.mean() * 100:.2f}%"
)

print(
    "En az 1 gerçek etiketi yakalama:",
    f"{at_least_one.mean() * 100:.2f}%"
)


# ============================================================
# 12. TAHMİNLERİ OKUNABİLİR HALE GETİR
# ============================================================

prediction_rows = []


for i, row in final_test_df.iterrows():

    predicted_indices = np.where(
        y_final_pred[i] == 1
    )[0]

    ordered = sorted(
        [
            (
                centroid_subjects[index],
                float(
                    final_scores[
                        i,
                        index
                    ]
                )
            )
            for index in predicted_indices
        ],
        key=lambda item: item[1],
        reverse=True
    )

    external_id = str(
        row["external_id"]
    )

    prediction_rows.append(
        {
            "external_id":
                external_id,

            "predicted_label_count":
                len(ordered),

            "predicted_subjects":
                " || ".join(
                    [
                        subject
                        for subject, score
                        in ordered
                    ]
                ),

            "predicted_scores":
                " || ".join(
                    [
                        f"{score:.4f}"
                        for subject, score
                        in ordered
                    ]
                ),

            "true_subjects":
                " || ".join(
                    true_subjects.get(
                        external_id,
                        []
                    )
                ),

            "exact_match":
                bool(
                    exact_matches[i]
                ),

            "at_least_one_match":
                bool(
                    at_least_one[i]
                )
        }
    )


prediction_df = pd.DataFrame(
    prediction_rows
)


# ============================================================
# 13. SONUÇLARI KAYDET
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "multilabel_seeded_final_centroids.npy"
    ),
    final_centroids
)

threshold_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "multilabel_seeded_threshold_search.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

prediction_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "multilabel_seeded_predictions.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

summary = pd.DataFrame(
    [
        {
            "Method":
                "Multi-label Seeded K-Means + Cosine",

            "Seeds_Per_Subject":
                SEEDS_PER_SUBJECT,

            "Reference_Articles":
                len(reference),

            "Final_Test_Articles":
                len(final_test_df),

            "Best_Threshold":
                best_threshold,

            "Mean_Centroid_Shift":
                float(
                    centroid_shift.mean()
                ),

            **final_metrics,

            "Average_True_Labels":
                real_average_labels,

            "Exact_Match_Rate":
                float(
                    exact_matches.mean()
                ),

            "At_Least_One_Match_Rate":
                float(
                    at_least_one.mean()
                )
        }
    ]
)

summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "multilabel_seeded_summary.csv"
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
    "multilabel_seeded_summary.csv"
)

print(
    "results/kmeans/holdout/"
    "multilabel_seeded_predictions.csv"
)