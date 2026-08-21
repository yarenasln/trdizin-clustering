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

# %80 test kısmının yarısı threshold seçimi için,
# yarısı final test için kullanılacak.
VALIDATION_RATIO = 0.50

# Denenecek cosine similarity eşikleri
THRESHOLDS = np.arange(
    0.35,
    0.801,
    0.01
)


# ============================================================
# VERİLERİ OKU
# ============================================================

print("=" * 110)
print("MULTI-LABEL FIXED CENTROID + COSINE")
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
    "Toplam test havuzu:",
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


print(
    "Leaf konu sayısı:",
    len(leaf_subjects)
)


# ============================================================
# GERÇEK MULTI-LABEL ETİKETLER
# ============================================================

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
# TEMSİL EDİCİ SEED SEÇ
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
# 195 KONU CENTROIDI OLUŞTUR
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
                .isin(
                    reference_ids
                )
            )
        ]["external_id"]
        .drop_duplicates()
        .tolist()
    )


    # Önce tek leaf etiketi olan temiz örnekler
    single_label_candidates = [
        article_id
        for article_id in candidates
        if label_counts.get(
            article_id,
            0
        ) == 1
    ]


    selected_ids = []


    # --------------------------------------------------------
    # Yeterli single-label varsa
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Yetmiyorsa multi-label ile tamamla
    # --------------------------------------------------------

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


        remaining_candidates = [
            article_id
            for article_id in candidates
            if article_id
            not in selected_ids
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
).astype(
    np.float32
)


print(
    "Centroid matrix:",
    centroid_matrix.shape
)

print(
    "Benzersiz seed makale:",
    len(
        unique_seed_ids
    )
)


# ============================================================
# TEST HAVUZUNU VALIDATION / FINAL TEST AYIR
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


print("\n" + "=" * 110)
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
# COSINE SCORE MATRIX
# ============================================================

def create_similarity_matrix(df):

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


    normalized = normalize_vectors(
        vectors
    )


    return (
        normalized
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
    "Final test score matrix:",
    final_scores.shape
)


# ============================================================
# MULTILABEL BINARIZER
# ============================================================

mlb = MultiLabelBinarizer(
    classes=leaf_subjects
)


# Classes sabitlensin
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
# THRESHOLD'DAN MULTI-LABEL TAHMİN ÜRET
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


    # --------------------------------------------------------
    # Bir makalede hiçbir skor threshold'u geçmezse
    # etiketsiz bırakmıyoruz.
    #
    # En yüksek skorlu 1 konuyu veriyoruz.
    # --------------------------------------------------------

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
# METRİK HESAPLA
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    micro_precision = (
        precision_score(
            y_true,
            y_pred,
            average="micro",
            zero_division=0
        )
    )


    micro_recall = (
        recall_score(
            y_true,
            y_pred,
            average="micro",
            zero_division=0
        )
    )


    micro_f1 = (
        f1_score(
            y_true,
            y_pred,
            average="micro",
            zero_division=0
        )
    )


    macro_precision = (
        precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )
    )


    macro_recall = (
        recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )
    )


    macro_f1 = (
        f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )
    )


    sample_f1 = (
        f1_score(
            y_true,
            y_pred,
            average="samples",
            zero_division=0
        )
    )


    average_labels = float(
        y_pred.sum(
            axis=1
        ).mean()
    )


    return {

        "Micro_Precision":
            micro_precision,

        "Micro_Recall":
            micro_recall,

        "Micro_F1":
            micro_f1,

        "Macro_Precision":
            macro_precision,

        "Macro_Recall":
            macro_recall,

        "Macro_F1":
            macro_f1,

        "Sample_F1":
            sample_f1,

        "Average_Predicted_Labels":
            average_labels
    }


# ============================================================
# VALIDATION'DA EN İYİ THRESHOLD'U BUL
# ============================================================

print("\n" + "=" * 110)
print("VALIDATION - THRESHOLD ARAMASI")
print("=" * 110)


threshold_results = []


for threshold in THRESHOLDS:

    y_pred = predictions_from_threshold(
        validation_scores,
        threshold
    )


    metrics = calculate_metrics(
        y_validation_true,
        y_pred
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


# Ana seçim metriğimiz Micro-F1
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
    "Ortalama verilen etiket:",
    round(
        best_row[
            "Average_Predicted_Labels"
        ],
        2
    )
)


# ============================================================
# FINAL TEST
# ============================================================

print("\n" + "=" * 110)
print("FINAL MULTI-LABEL TEST")
print("=" * 110)


y_final_pred = (
    predictions_from_threshold(
        final_scores,
        best_threshold
    )
)


final_metrics = calculate_metrics(
    y_final_true,
    y_final_pred
)


print(
    "Final test makale:",
    len(
        final_test_df
    )
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


# ============================================================
# GERÇEK ORTALAMA ETİKET SAYISI
# ============================================================

real_average_labels = float(
    y_final_true.sum(
        axis=1
    ).mean()
)


print(
    "Gerçek ortalama etiket:",
    round(
        real_average_labels,
        2
    )
)


# ============================================================
# EXACT MATCH
# ============================================================
#
# Tahmin edilen etiket seti ile gerçek etiket seti
# tamamen birebir aynı mı?
# ============================================================

exact_matches = np.all(
    y_final_true
    ==
    y_final_pred,
    axis=1
)


exact_match_rate = float(
    exact_matches.mean()
)


print(
    "Exact set match:",
    f"{exact_match_rate * 100:.2f}%"
)


# ============================================================
# EN AZ BİR DOĞRU ETİKET
# ============================================================

intersection_counts = (
    (
        y_final_true
        &
        y_final_pred
    )
    .sum(
        axis=1
    )
)


at_least_one = (
    intersection_counts
    >
    0
)


at_least_one_rate = float(
    at_least_one.mean()
)


print(
    "En az 1 gerçek etiketi yakalama:",
    f"{at_least_one_rate * 100:.2f}%"
)


# ============================================================
# FINAL TAHMİNLERİ YAZIYA ÇEVİR
# ============================================================

prediction_rows = []


for i, row in final_test_df.iterrows():

    predicted_indices = np.where(
        y_final_pred[i]
        ==
        1
    )[0]


    predicted_labels = [
        leaf_subjects[index]
        for index in predicted_indices
    ]


    predicted_scores = [
        float(
            final_scores[
                i,
                index
            ]
        )
        for index in predicted_indices
    ]


    # Skora göre sırala
    ordered = sorted(
        zip(
            predicted_labels,
            predicted_scores
        ),
        key=lambda x:
            x[1],
        reverse=True
    )


    prediction_rows.append(
        {
            "external_id":
                str(
                    row[
                        "external_id"
                    ]
                ),

            "predicted_label_count":
                len(
                    ordered
                ),

            "predicted_subjects":
                " || ".join(
                    [
                        item[0]
                        for item
                        in ordered
                    ]
                ),

            "predicted_scores":
                " || ".join(
                    [
                        f"{item[1]:.4f}"
                        for item
                        in ordered
                    ]
                ),

            "true_subjects":
                " || ".join(
                    true_subjects.get(
                        str(
                            row[
                                "external_id"
                            ]
                        ),
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
# KAYDET
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


threshold_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "multilabel_threshold_search.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


prediction_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "multilabel_final_predictions.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


summary = pd.DataFrame(
    [
        {
            "Method":
                "Multi-label Fixed Centroid + Cosine",

            "Seeds_Per_Subject":
                SEEDS_PER_SUBJECT,

            "Best_Threshold":
                best_threshold,

            "Final_Test_Articles":
                len(
                    final_test_df
                ),

            "Micro_Precision":
                final_metrics[
                    "Micro_Precision"
                ],

            "Micro_Recall":
                final_metrics[
                    "Micro_Recall"
                ],

            "Micro_F1":
                final_metrics[
                    "Micro_F1"
                ],

            "Macro_Precision":
                final_metrics[
                    "Macro_Precision"
                ],

            "Macro_Recall":
                final_metrics[
                    "Macro_Recall"
                ],

            "Macro_F1":
                final_metrics[
                    "Macro_F1"
                ],

            "Sample_F1":
                final_metrics[
                    "Sample_F1"
                ],

            "Average_Predicted_Labels":
                final_metrics[
                    "Average_Predicted_Labels"
                ],

            "Average_True_Labels":
                real_average_labels,

            "Exact_Match_Rate":
                exact_match_rate,

            "At_Least_One_Match_Rate":
                at_least_one_rate
        }
    ]
)


summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "multilabel_final_summary.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


print("\n" + "=" * 110)
print("TAMAMLANDI")
print("=" * 110)

print(
    "results/kmeans/holdout/"
    "multilabel_threshold_search.csv"
)

print(
    "results/kmeans/holdout/"
    "multilabel_final_predictions.csv"
)

print(
    "results/kmeans/holdout/"
    "multilabel_final_summary.csv"
)