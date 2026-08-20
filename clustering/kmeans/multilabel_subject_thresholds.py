import os
import numpy as np
import pandas as pd

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

# Her konu için ayrı ayrı denenecek thresholdlar
THRESHOLDS = np.arange(
    0.35,
    0.801,
    0.01
)

# Validation'da çok az pozitif örneği olan konularda
# threshold aşırı ezberlenmesin diye fallback kullanacağız.
MIN_POSITIVE_SAMPLES = 5

# Önceki deneyde bulunan global threshold
GLOBAL_FALLBACK_THRESHOLD = 0.64


# ============================================================
# VERİLERİ OKU
# ============================================================

print("=" * 110)
print("MULTI-LABEL + SUBJECT-SPECIFIC THRESHOLD")
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
    reference["external_id"].astype(str)
)

test["external_id"] = (
    test["external_id"].astype(str)
)

subjects["external_id"] = (
    subjects["external_id"].astype(str)
)

print("Referans makale:", len(reference))
print("Test havuzu:", len(test))
print("Embedding shape:", embeddings.shape)


# ============================================================
# KONU VE ETİKET BİLGİLERİ
# ============================================================

reference_ids = set(
    reference["external_id"]
)

reference_row_map = (
    reference
    .set_index("external_id")["embedding_row"]
    .to_dict()
)

leaf_subjects = sorted(
    subjects["subject_fullname"]
    .dropna()
    .astype(str)
    .unique()
)

label_counts = (
    subjects
    .groupby("external_id")["subject_fullname"]
    .nunique()
    .to_dict()
)

true_subjects = (
    subjects
    .groupby("external_id")["subject_fullname"]
    .apply(
        lambda x:
            sorted(
                set(
                    x.dropna().astype(str)
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
# TEMSİL EDİCİ SEED SEÇİMİ
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
# 195 KONU CENTROIDI
# ============================================================

print("\n" + "=" * 110)
print("KONU CENTROIDLERİ OLUŞTURULUYOR")
print("=" * 110)

centroids = []
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

    # Önce mümkün olduğunca single-label
    if (
        len(single_label_candidates)
        >=
        SEEDS_PER_SUBJECT
    ):

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

        remaining_candidates = [
            article_id
            for article_id in candidates
            if article_id not in selected_ids
        ]

        selected_ids.extend(
            select_representative(
                remaining_candidates,
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


centroid_matrix = np.vstack(
    centroids
).astype(np.float32)


print(
    "Centroid matrix:",
    centroid_matrix.shape
)

print(
    "Benzersiz seed makale:",
    len(unique_seed_ids)
)


# ============================================================
# VALIDATION / FINAL TEST
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


print("\n" + "=" * 110)
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
# COSINE SCORE MATRIX
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

    normalized_vectors = normalize_vectors(
        vectors
    )

    return (
        normalized_vectors
        @
        centroid_matrix.T
    )


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
    "Validation score matrix:",
    validation_scores.shape
)

print(
    "Final score matrix:",
    final_scores.shape
)


# ============================================================
# GERÇEK MULTI-LABEL MATRİSLER
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
# HER KONU İÇİN AYRI THRESHOLD BUL
# ============================================================

print("\n" + "=" * 110)
print("KONUYA ÖZEL THRESHOLD ARAMASI")
print("=" * 110)


subject_thresholds = []

threshold_rows = []


for subject_index, subject_name in enumerate(
    leaf_subjects
):

    y_true_subject = (
        y_validation_true[
            :,
            subject_index
        ]
    )

    subject_scores = (
        validation_scores[
            :,
            subject_index
        ]
    )

    positive_count = int(
        y_true_subject.sum()
    )


    # --------------------------------------------------------
    # Çok az örnek varsa global threshold kullan.
    # --------------------------------------------------------

    if positive_count < MIN_POSITIVE_SAMPLES:

        best_threshold = (
            GLOBAL_FALLBACK_THRESHOLD
        )

        y_pred_subject = (
            subject_scores
            >=
            best_threshold
        ).astype(int)

        best_f1 = f1_score(
            y_true_subject,
            y_pred_subject,
            zero_division=0
        )

        threshold_source = "GLOBAL_FALLBACK"


    # --------------------------------------------------------
    # Yeterli örnek varsa bu konuya özel threshold ara.
    # --------------------------------------------------------

    else:

        best_threshold = (
            GLOBAL_FALLBACK_THRESHOLD
        )

        best_f1 = -1


        for threshold in THRESHOLDS:

            y_pred_subject = (
                subject_scores
                >=
                threshold
            ).astype(int)

            current_f1 = f1_score(
                y_true_subject,
                y_pred_subject,
                zero_division=0
            )


            if current_f1 > best_f1:

                best_f1 = current_f1

                best_threshold = float(
                    threshold
                )


        threshold_source = (
            "SUBJECT_SPECIFIC"
        )


    subject_thresholds.append(
        best_threshold
    )


    threshold_rows.append(
        {
            "subject_fullname":
                subject_name,

            "positive_validation_samples":
                positive_count,

            "threshold":
                best_threshold,

            "validation_subject_f1":
                best_f1,

            "threshold_source":
                threshold_source
        }
    )


subject_thresholds = np.array(
    subject_thresholds,
    dtype=np.float32
)


threshold_df = pd.DataFrame(
    threshold_rows
)


print(
    "Konuya özel threshold:",
    int(
        (
            threshold_df[
                "threshold_source"
            ]
            ==
            "SUBJECT_SPECIFIC"
        ).sum()
    )
)

print(
    "Global fallback kullanan:",
    int(
        (
            threshold_df[
                "threshold_source"
            ]
            ==
            "GLOBAL_FALLBACK"
        ).sum()
    )
)

print(
    "Ortalama threshold:",
    round(
        float(
            subject_thresholds.mean()
        ),
        4
    )
)

print(
    "Minimum threshold:",
    round(
        float(
            subject_thresholds.min()
        ),
        4
    )
)

print(
    "Maximum threshold:",
    round(
        float(
            subject_thresholds.max()
        ),
        4
    )
)


# ============================================================
# SUBJECT-SPECIFIC TAHMİN
# ============================================================

def predict_with_subject_thresholds(
    scores
):

    predictions = (
        scores
        >=
        subject_thresholds[
            np.newaxis,
            :
        ]
    ).astype(int)


    # Hiçbir etiketi geçemeyen makale varsa
    # en yakın bir konuyu yine ver.
    empty_rows = np.where(
        predictions.sum(
            axis=1
        ) == 0
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
# METRİKLER
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
                y_pred.sum(
                    axis=1
                ).mean()
            )
    }


# ============================================================
# VALIDATION SONUCU
# ============================================================

y_validation_pred = (
    predict_with_subject_thresholds(
        validation_scores
    )
)


validation_metrics = (
    calculate_metrics(
        y_validation_true,
        y_validation_pred
    )
)


print("\n" + "=" * 110)
print("VALIDATION SONUCU")
print("=" * 110)

print(
    "Micro Precision:",
    f"{validation_metrics['Micro_Precision'] * 100:.2f}%"
)

print(
    "Micro Recall:",
    f"{validation_metrics['Micro_Recall'] * 100:.2f}%"
)

print(
    "Micro F1:",
    f"{validation_metrics['Micro_F1'] * 100:.2f}%"
)

print(
    "Macro F1:",
    f"{validation_metrics['Macro_F1'] * 100:.2f}%"
)

print(
    "Ortalama tahmin edilen etiket:",
    round(
        validation_metrics[
            "Average_Predicted_Labels"
        ],
        2
    )
)


# ============================================================
# FINAL TEST
# ============================================================

y_final_pred = (
    predict_with_subject_thresholds(
        final_scores
    )
)


final_metrics = calculate_metrics(
    y_final_true,
    y_final_pred
)


real_average_labels = float(
    y_final_true.sum(
        axis=1
    ).mean()
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


print("\n" + "=" * 110)
print("FINAL SUBJECT-SPECIFIC MULTI-LABEL TEST")
print("=" * 110)

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
# GLOBAL BASELINE İLE AYNI FINAL SETTE KARŞILAŞTIR
# ============================================================

global_pred = (
    final_scores
    >=
    GLOBAL_FALLBACK_THRESHOLD
).astype(int)


empty_rows = np.where(
    global_pred.sum(
        axis=1
    ) == 0
)[0]


for row_index in empty_rows:

    best_index = int(
        np.argmax(
            final_scores[
                row_index
            ]
        )
    )

    global_pred[
        row_index,
        best_index
    ] = 1


global_metrics = calculate_metrics(
    y_final_true,
    global_pred
)


print("\n" + "=" * 110)
print("GLOBAL 0.64 vs SUBJECT-SPECIFIC")
print("=" * 110)

print(
    "Global Micro F1:",
    f"{global_metrics['Micro_F1'] * 100:.2f}%"
)

print(
    "Subject-specific Micro F1:",
    f"{final_metrics['Micro_F1'] * 100:.2f}%"
)

print(
    "Fark:",
    f"{(final_metrics['Micro_F1'] - global_metrics['Micro_F1']) * 100:+.2f} puan"
)

print()

print(
    "Global Macro F1:",
    f"{global_metrics['Macro_F1'] * 100:.2f}%"
)

print(
    "Subject-specific Macro F1:",
    f"{final_metrics['Macro_F1'] * 100:.2f}%"
)


# ============================================================
# KAYDET
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


threshold_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "subject_specific_thresholds.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


summary_df = pd.DataFrame(
    [
        {
            "Method":
                "Global Threshold",

            "Threshold":
                GLOBAL_FALLBACK_THRESHOLD,

            **global_metrics
        },

        {
            "Method":
                "Subject-Specific Threshold",

            "Threshold":
                "195 separate thresholds",

            **final_metrics
        }
    ]
)


summary_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "subject_threshold_comparison.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


print("\n" + "=" * 110)
print("TAMAMLANDI")
print("=" * 110)

print(
    "results/kmeans/holdout/"
    "subject_specific_thresholds.csv"
)

print(
    "results/kmeans/holdout/"
    "subject_threshold_comparison.csv"
)