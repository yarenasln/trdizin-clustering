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

# Adaptive Top-K iÃ§in denenecek oranlar.
# MantÄ±k:
#
# d1 = en yakÄ±n centroid uzaklÄ±ÄŸÄ±
# d2 = ikinci centroid uzaklÄ±ÄŸÄ±
# d3 = Ã¼Ã§Ã¼ncÃ¼ centroid uzaklÄ±ÄŸÄ±
#
# d2 / d1 kÃ¼Ã§Ã¼kse ikinci etiketi de ekle
# d3 / d1 kÃ¼Ã§Ã¼kse Ã¼Ã§Ã¼ncÃ¼ etiketi de ekle
SECOND_RATIOS = np.arange(
    1.01,
    1.31,
    0.01
)

THIRD_RATIOS = np.arange(
    1.01,
    1.41,
    0.01
)


# ============================================================
# VERÄ°LERÄ° OKU
# ============================================================

print("=" * 110)
print("ADAPTIVE MULTI-LABEL SEEDED K-MEANS")
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
    "Leaf konu sayÄ±sÄ±:",
    len(leaf_subjects)
)


# ============================================================
# BAÅLIK AYIRMA
# ============================================================
#
# Ã–rnek:
# Fen > MÃ¼hendislik > Bilgisayar Bilimleri, Yapay Zeka
#
# main_level_1 = Fen
# main_level_2 = Fen > MÃ¼hendislik
# leaf = Bilgisayar Bilimleri, Yapay Zeka
#
# Bu sadece Ã§Ä±ktÄ± dÃ¼zenleme iÃ§in.
# HiyerarÅŸik karar sistemi DEÄÄ°L.
# ============================================================

def parse_subject(subject_name):

    parts = [
        part.strip()
        for part in str(
            subject_name
        ).split(">")
        if part.strip()
    ]

    level_1 = (
        parts[0]
        if len(parts) >= 1
        else ""
    )

    level_2 = (
        " > ".join(
            parts[:2]
        )
        if len(parts) >= 2
        else level_1
    )

    leaf = (
        parts[-1]
        if parts
        else ""
    )

    return (
        level_1,
        level_2,
        leaf
    )


# ============================================================
# TEMSÄ°L EDÄ°CÄ° SEED SEÃ‡
# ============================================================

def select_representative(
    candidate_ids,
    count
):

    valid_ids = [
        article_id
        for article_id
        in candidate_ids
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

    distances = (
        np.linalg.norm(
            vectors
            -
            center,
            axis=1
        )
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
        for i
        in order[
            :selected_count
        ]
    ]


# ============================================================
# 1. BAÅLANGIÃ‡ CENTROIDLERÄ°
# ============================================================

print()
print("=" * 110)
print("BAÅLANGIÃ‡ CENTROIDLERÄ° OLUÅTURULUYOR")
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
            f"Seed bulunamadÄ±: {subject_name}"
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

    initial_centroids.append(
        centroid
    )

    centroid_subjects.append(
        subject_name
    )


initial_centroids = (
    np.vstack(
        initial_centroids
    )
    .astype(
        np.float32
    )
)

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
print("SEEDED K-MEANS REFERENCE ÃœZERÄ°NDE Ã‡ALIÅIYOR")
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
    n_clusters=len(
        leaf_subjects
    ),
    init=initial_centroids,
    n_init=1,
    random_state=RANDOM_STATE,
    max_iter=300,
    tol=1e-4
)

reference_labels = (
    kmeans.fit_predict(
        X_reference
    )
)

final_centroids = (
    kmeans
    .cluster_centers_
    .astype(
        np.float32
    )
)

print(
    "Final centroid matrix:",
    final_centroids.shape
)

print(
    "KullanÄ±lan cluster:",
    len(
        np.unique(
            reference_labels
        )
    )
)

centroid_shift = (
    np.linalg.norm(
        final_centroids
        -
        initial_centroids,
        axis=1
    )
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
# 4. GERÃ‡EK MULTI-LABEL MATRÄ°SLER
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
# 5. DISTANCE MATRIX
# ============================================================

def create_distance_matrix(df):

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

    differences = (
        vectors[
            :,
            np.newaxis,
            :
        ]
        -
        final_centroids[
            np.newaxis,
            :,
            :
        ]
    )

    distances = (
        np.linalg.norm(
            differences,
            axis=2
        )
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
# 6. ADAPTIVE TOP-K
# ============================================================
#
# 1. konu HER ZAMAN verilir.
#
# Ä°kinci merkezin uzaklÄ±ÄŸÄ±:
# d2 <= d1 * second_ratio ise
# ikinci etiket eklenir.
#
# ÃœÃ§Ã¼ncÃ¼ merkezin uzaklÄ±ÄŸÄ±:
# d3 <= d1 * third_ratio ise
# Ã¼Ã§Ã¼ncÃ¼ etiket eklenir.
#
# BÃ¶ylece bazÄ± makaleler:
# 1 etiket
# bazÄ±larÄ±:
# 2 etiket
# bazÄ±larÄ±:
# 3 etiket
#
# alabilir.
# ============================================================

def adaptive_predictions(
    distances,
    second_ratio,
    third_ratio
):

    predictions = np.zeros(
        distances.shape,
        dtype=int
    )

    predicted_counts = []

    for i in range(
        distances.shape[0]
    ):

        sorted_indices = (
            np.argsort(
                distances[i]
            )
        )

        first_index = (
            sorted_indices[0]
        )

        second_index = (
            sorted_indices[1]
        )

        third_index = (
            sorted_indices[2]
        )

        d1 = float(
            distances[
                i,
                first_index
            ]
        )

        d2 = float(
            distances[
                i,
                second_index
            ]
        )

        d3 = float(
            distances[
                i,
                third_index
            ]
        )


        # Her zaman ilk konu
        predictions[
            i,
            first_index
        ] = 1

        count = 1


        # Ä°kinci konu yeterince yakÄ±n mÄ±?
        if (
            d2
            <=
            d1
            *
            second_ratio
        ):

            predictions[
                i,
                second_index
            ] = 1

            count = 2


            # ÃœÃ§Ã¼ncÃ¼ etiketi ancak
            # ikinciyi de seÃ§miÅŸsek deÄŸerlendir.
            if (
                d3
                <=
                d1
                *
                third_ratio
            ):

                predictions[
                    i,
                    third_index
                ] = 1

                count = 3


        predicted_counts.append(
            count
        )


    return (
        predictions,
        np.array(
            predicted_counts
        )
    )


# ============================================================
# 7. METRÄ°KLER
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
# 8. VALIDATION'DA EN Ä°YÄ° ADAPTIVE KURAL
# ============================================================

print()
print("=" * 110)
print("VALIDATION - ADAPTIVE TOP-K ARAMASI")
print("=" * 110)

search_results = []


for second_ratio in SECOND_RATIOS:

    for third_ratio in THIRD_RATIOS:

        # 3. etiket eÅŸiÄŸi
        # 2. etiketten daha sÄ±kÄ± olmasÄ±n.
        if third_ratio < second_ratio:
            continue

        (
            y_pred,
            predicted_counts
        ) = adaptive_predictions(
            validation_distances,
            second_ratio,
            third_ratio
        )

        metrics = calculate_metrics(
            y_validation_true,
            y_pred
        )

        count_1 = int(
            (
                predicted_counts
                ==
                1
            ).sum()
        )

        count_2 = int(
            (
                predicted_counts
                ==
                2
            ).sum()
        )

        count_3 = int(
            (
                predicted_counts
                ==
                3
            ).sum()
        )

        search_results.append(
            {
                "Second_Ratio":
                    float(
                        second_ratio
                    ),

                "Third_Ratio":
                    float(
                        third_ratio
                    ),

                "Count_1_Label":
                    count_1,

                "Count_2_Labels":
                    count_2,

                "Count_3_Labels":
                    count_3,

                **metrics
            }
        )


search_df = pd.DataFrame(
    search_results
)


best_row = (
    search_df
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


best_second_ratio = float(
    best_row[
        "Second_Ratio"
    ]
)

best_third_ratio = float(
    best_row[
        "Third_Ratio"
    ]
)


print(
    "En iyi second ratio:",
    round(
        best_second_ratio,
        4
    )
)

print(
    "En iyi third ratio:",
    round(
        best_third_ratio,
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

print(
    "1 etiket verilen:",
    int(
        best_row[
            "Count_1_Label"
        ]
    )
)

print(
    "2 etiket verilen:",
    int(
        best_row[
            "Count_2_Labels"
        ]
    )
)

print(
    "3 etiket verilen:",
    int(
        best_row[
            "Count_3_Labels"
        ]
    )
)


# ============================================================
# 9. FINAL TEST
# ============================================================

print()
print("=" * 110)
print("FINAL ADAPTIVE MULTI-LABEL K-MEANS TEST")
print("=" * 110)


(
    y_y_final_pred,
    y_final_predicted_counts
) = adaptive_predictions(
    final_distances,
    best_second_ratio,
    best_third_ratio
)


final_metrics = (
    calculate_metrics(
        y_final_true,
        y_y_final_pred
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
        y_y_final_pred,
        axis=1
    )
    .mean()
)


at_least_one = float(
    (
        (
            y_final_true
            &
            y_y_final_pred
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
    "Macro F1:",
    f"{final_metrics['Macro_F1'] * 100:.2f}%"
)

print(
    "Sample F1:",
    f"{final_metrics['Sample_F1'] * 100:.2f}%"
)

print()

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
    "GerÃ§ek ortalama etiket:",
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
    "En az 1 gerÃ§ek etiket:",
    f"{at_least_one * 100:.2f}%"
)

print()

print(
    "1 etiket verilen:",
    int(
        (
            y_final_predicted_counts
            ==
            1
        ).sum()
    )
)

print(
    "2 etiket verilen:",
    int(
        (
            y_final_predicted_counts
            ==
            2
        ).sum()
    )
)

print(
    "3 etiket verilen:",
    int(
        (
            y_final_predicted_counts
            ==
            3
        ).sum()
    )
)


# ============================================================
# 10. ANA BAÅLIK + ALT KONU Ã‡IKTISI
# ============================================================

prediction_rows = []


for i, row in final_test_df.iterrows():

    selected_indices = np.where(
        y_y_final_pred[i]
        ==
        1
    )[0]


    selected_indices = sorted(
        selected_indices,
        key=lambda index:
            final_distances[
                i,
                index
            ]
    )


    full_topics = [
        centroid_subjects[index]
        for index
        in selected_indices
    ]


    level_1_topics = []

    level_2_topics = []

    leaf_topics = []


    for full_topic in full_topics:

        (
            level_1,
            level_2,
            leaf
        ) = parse_subject(
            full_topic
        )

        if (
            level_1
            not in level_1_topics
        ):
            level_1_topics.append(
                level_1
            )

        if (
            level_2
            not in level_2_topics
        ):
            level_2_topics.append(
                level_2
            )

        leaf_topics.append(
            leaf
        )


    external_id = str(
        row[
            "external_id"
        ]
    )


    prediction_rows.append(
        {
            "external_id":
                external_id,

            "predicted_label_count":
                len(
                    full_topics
                ),

            "main_level_1_predictions":
                " || ".join(
                    level_1_topics
                ),

            "main_level_2_predictions":
                " || ".join(
                    level_2_topics
                ),

            "leaf_predictions":
                " || ".join(
                    leaf_topics
                ),

            "full_topic_predictions":
                " || ".join(
                    full_topics
                ),

            "true_subjects":
                " || ".join(
                    true_subjects.get(
                        external_id,
                        []
                    )
                )
        }
    )


prediction_df = pd.DataFrame(
    prediction_rows
)


# ============================================================
# 10B. A/B TEST: SABÄ°T KONU KÄ°MLÄ°ÄÄ° vs RELABEL
# AynÄ± K-Means, aynÄ± centroidler, aynÄ± final test ve aynÄ±
# Adaptive 1/2/3 cluster seÃ§imi kullanÄ±lÄ±r.
# Tek fark: clusterÄ±n konu adÄ± sabit mi kalÄ±yor, yoksa
# reference kÃ¼mesindeki baskÄ±n gerÃ§ek konuya mÄ± dÃ¶nÃ¼ÅŸÃ¼yor?
# ============================================================

from collections import Counter

subject_to_index = {
    subject: i
    for i, subject in enumerate(leaf_subjects)
}

reference_cluster_ids = kmeans.labels_
relabel_subjects = []
relabel_rows = []

for cluster_id in range(len(centroid_subjects)):
    member_positions = np.where(
        reference_cluster_ids == cluster_id
    )[0]

    counter = Counter()

    for position in member_positions:
        external_id = str(
            reference.iloc[position]["external_id"]
        )
        for subject in true_subjects.get(external_id, []):
            if subject in subject_to_index:
                counter[subject] += 1

    old_subject = centroid_subjects[cluster_id]

    if counter:
        new_subject, dominant_count = counter.most_common(1)[0]
    else:
        new_subject, dominant_count = old_subject, 0

    relabel_subjects.append(new_subject)
    relabel_rows.append({
        "Cluster_ID": cluster_id,
        "Cluster_Size": int(len(member_positions)),
        "Old_Subject": old_subject,
        "New_Subject": new_subject,
        "Changed": old_subject != new_subject,
        "Dominant_Count": int(dominant_count)
    })

relabel_df = pd.DataFrame(relabel_rows)

def relabel_cluster_predictions(cluster_predictions):
    result = np.zeros(
        (cluster_predictions.shape[0], len(leaf_subjects)),
        dtype=int
    )
    for row_index in range(cluster_predictions.shape[0]):
        selected = np.where(
            cluster_predictions[row_index] == 1
        )[0]
        for cluster_id in selected:
            subject = relabel_subjects[cluster_id]
            result[
                row_index,
                subject_to_index[subject]
            ] = 1
    return result

relabel_y_final_pred = relabel_cluster_predictions(
    y_final_pred
)

fixed_ab_metrics = calculate_metrics(
    y_final_true,
    y_final_pred
)
relabel_ab_metrics = calculate_metrics(
    y_final_true,
    relabel_y_final_pred
)

def ab_extra(y_true, y_pred):
    exact = float(
        np.all(y_true == y_pred, axis=1).mean()
    )
    at_least_one = float(
        (((y_true * y_pred).sum(axis=1)) > 0).mean()
    )
    return exact, at_least_one

fixed_exact, fixed_one = ab_extra(
    y_final_true, y_final_pred
)
relabel_exact, relabel_one = ab_extra(
    y_final_true, relabel_y_final_pred
)

comparison_df = pd.DataFrame([
    {
        "Model": "Fixed Topic Identity",
        **fixed_ab_metrics,
        "Exact_Match_Rate": fixed_exact,
        "At_Least_One_Match_Rate": fixed_one,
        "Average_Predicted_Labels":
            float(y_final_pred.sum(axis=1).mean()),
        "Unique_Topics": len(set(centroid_subjects))
    },
    {
        "Model": "Relabel Dominant Topic",
        **relabel_ab_metrics,
        "Exact_Match_Rate": relabel_exact,
        "At_Least_One_Match_Rate": relabel_one,
        "Average_Predicted_Labels":
            float(relabel_y_final_pred.sum(axis=1).mean()),
        "Unique_Topics": len(set(relabel_subjects))
    }
])

print()
print("=" * 110)
print("A/B TEST - SABÄ°T KONU KÄ°MLÄ°ÄÄ° vs RELABEL")
print("=" * 110)
print(
    "Konu kimliÄŸi deÄŸiÅŸen cluster:",
    int(relabel_df["Changed"].sum())
)
print(
    "Relabel sonrasÄ± benzersiz konu:",
    len(set(relabel_subjects)),
    "/",
    len(leaf_subjects)
)

show = comparison_df.copy()
for col in [
    "Micro_Precision", "Micro_Recall", "Micro_F1",
    "Macro_F1", "Sample_F1", "Exact_Match_Rate",
    "At_Least_One_Match_Rate"
]:
    show[col] = (show[col] * 100).round(2)

print(
    show[
        [
            "Model",
            "Micro_Precision",
            "Micro_Recall",
            "Micro_F1",
            "Macro_F1",
            "Sample_F1",
            "Exact_Match_Rate",
            "At_Least_One_Match_Rate",
            "Average_Predicted_Labels",
            "Unique_Topics"
        ]
    ].to_string(index=False)
)

winner = comparison_df.sort_values(
    ["Micro_F1", "Macro_F1"],
    ascending=[False, False]
).iloc[0]

print()
print("KAZANAN:", winner["Model"])
print(
    "Micro F1:",
    f"{winner['Micro_F1'] * 100:.2f}%"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

comparison_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "fixed_vs_relabel_comparison.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

relabel_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "cluster_relabel_map.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

print()
print(
    "A/B sonuÃ§: results/kmeans/holdout/"
    "fixed_vs_relabel_comparison.csv"
)
print(
    "Relabel haritasÄ±: results/kmeans/holdout/"
    "cluster_relabel_map.csv"
)
print()

# ============================================================
# 11. Ã–RNEK TAHMÄ°NLER
# ============================================================

print()
print("=" * 110)
print("Ä°LK 10 TAHMÄ°N Ã–RNEÄÄ°")
print("=" * 110)


print(
    prediction_df[
        [
            "external_id",
            "main_level_2_predictions",
            "leaf_predictions"
        ]
    ]
    .head(10)
    .to_string(
        index=False
    )
)


# ============================================================
# 12. KAYDET
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


search_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "adaptive_kmeans_rule_search.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


prediction_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "adaptive_kmeans_predictions.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


summary = pd.DataFrame(
    [
        {
            "Method":
                "Adaptive Multi-label Seeded K-Means",

            "Seeds_Per_Subject":
                SEEDS_PER_SUBJECT,

            "Best_Second_Ratio":
                best_second_ratio,

            "Best_Third_Ratio":
                best_third_ratio,

            "Reference_Articles":
                len(reference),

            "Final_Test_Articles":
                len(
                    final_test_df
                ),

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
        "adaptive_kmeans_summary.csv"
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
    "adaptive_kmeans_predictions.csv"
)

print(
    "results/kmeans/holdout/"
    "adaptive_kmeans_summary.csv"
)
