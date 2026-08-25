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

# Adaptive Top-K için denenecek oranlar.
# Mantık:
#
# d1 = en yakın centroid uzaklığı
# d2 = ikinci centroid uzaklığı
# d3 = üçüncü centroid uzaklığı
#
# d2 / d1 küçükse ikinci etiketi de ekle
# d3 / d1 küçükse üçüncü etiketi de ekle
SECOND_RATIOS = np.arange(1.01, 1.31, 0.01)

THIRD_RATIOS = np.arange(1.01, 1.31, 0.01)


# ============================================================
# VERİLERİ OKU
# ============================================================

print("=" * 110)
print("ADAPTIVE V2 - 20K MEMORY-SAFE MULTI-LABEL SEEDED K-MEANS")
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
# BAŞLIK AYIRMA
# ============================================================
#
# Örnek:
# Fen > Mühendislik > Bilgisayar Bilimleri, Yapay Zeka
#
# main_level_1 = Fen
# main_level_2 = Fen > Mühendislik
# leaf = Bilgisayar Bilimleri, Yapay Zeka
#
# Bu sadece çıktı düzenleme için.
# Hiyerarşik karar sistemi DEĞİL.
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
# TEMSİL EDİCİ SEED SEÇ
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
# 1. BAŞLANGIÇ CENTROIDLERİ
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
    "Kullanılan cluster:",
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
# 5. DISTANCE MATRIX
# ============================================================

def create_distance_matrix(df):
    """
    Bellek-dostu Öklid uzaklık matrisi.

    Eski sürüm:
        vectors[:, None, :] - centroids[None, :, :]
    şeklinde (makale x cluster x 768) dev bir 3B tensor
    oluşturuyordu. 20K veri setinde bu birkaç GB RAM
    gerektirip Docker prosesinin "Killed" olmasına yol açıyordu.

    Bu sürüm aynı Öklid uzaklığını matematiksel olarak:
        ||x-c||^2 = ||x||^2 + ||c||^2 - 2*x.c
    formülüyle hesaplar ve sadece
    (makale x cluster) matrisi tutar.
    """

    rows = (
        df[
            "embedding_row"
        ]
        .astype(int)
        .to_numpy()
    )

    vectors = np.asarray(
        embeddings[rows],
        dtype=np.float32
    )

    centroids = np.asarray(
        final_centroids,
        dtype=np.float32
    )

    x2 = np.sum(
        vectors * vectors,
        axis=1,
        keepdims=True
    )

    c2 = np.sum(
        centroids * centroids,
        axis=1
    )[np.newaxis, :]

    squared_distances = (
        x2
        +
        c2
        -
        2.0 * (
            vectors
            @
            centroids.T
        )
    )

    # Float yuvarlama nedeniyle çok küçük negatif değerler oluşabilir.
    np.maximum(
        squared_distances,
        0.0,
        out=squared_distances
    )

    np.sqrt(
        squared_distances,
        out=squared_distances
    )

    return squared_distances


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
# İkinci merkezin uzaklığı:
# d2 <= d1 * second_ratio ise
# ikinci etiket eklenir.
#
# Üçüncü merkezin uzaklığı:
# d3 <= d1 * third_ratio ise
# üçüncü etiket eklenir.
#
# Böylece bazı makaleler:
# 1 etiket
# bazıları:
# 2 etiket
# bazıları:
# 3 etiket
#
# alabilir.
# ============================================================

def adaptive_predictions(
    distances,
    second_ratio,
    third_ratio
):
    """
    Adaptive V2:

    d1 = en yakın centroid
    d2 = ikinci en yakın centroid
    d3 = üçüncü en yakın centroid

    1) İlk konu her zaman seçilir.
    2) d2 / d1 <= second_ratio ise ikinci konu seçilir.
    3) Üçüncü konu artık d1'e değil d2'ye göre değerlendirilir:
       d3 / d2 <= third_ratio ise üçüncü konu seçilir.

    Böylece ilk iki konu birbirine yakın, üçüncü konu belirgin
    şekilde uzaksa sistem doğal olarak 2 etikette durabilir.
    """

    predictions = np.zeros(
        distances.shape,
        dtype=int
    )

    predicted_counts = []

    for i in range(distances.shape[0]):

        sorted_indices = np.argsort(
            distances[i]
        )

        first_index = sorted_indices[0]
        second_index = sorted_indices[1]
        third_index = sorted_indices[2]

        d1 = float(
            distances[i, first_index]
        )

        d2 = float(
            distances[i, second_index]
        )

        d3 = float(
            distances[i, third_index]
        )

        # 1. konu her zaman var.
        predictions[
            i,
            first_index
        ] = 1

        count = 1

        # 2. konu, 1. konuya gerçekten yakın mı?
        ratio_2_to_1 = (
            d2 / max(d1, 1e-12)
        )

        if ratio_2_to_1 <= second_ratio:

            predictions[
                i,
                second_index
            ] = 1

            count = 2

            # 3. konu, bu kez 2. konuya göre değerlendirilir.
            # Amaç: 2 -> 3 geçişini daha seçici yapmak.
            ratio_3_to_2 = (
                d3 / max(d2, 1e-12)
            )

            if ratio_3_to_2 <= third_ratio:

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
# 8. VALIDATION'DA ADAPTIVE V2 KURAL ARAMASI
# ============================================================

print()
print("=" * 110)
print("VALIDATION - ADAPTIVE V2 ETİKET SAYISI ARAMASI")
print("=" * 110)

search_results = []

validation_true_average = float(
    y_validation_true
    .sum(axis=1)
    .mean()
)

for second_ratio in SECOND_RATIOS:

    for third_ratio in THIRD_RATIOS:

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
            (predicted_counts == 1).sum()
        )

        count_2 = int(
            (predicted_counts == 2).sum()
        )

        count_3 = int(
            (predicted_counts == 3).sum()
        )

        average_predicted = float(
            predicted_counts.mean()
        )

        search_results.append(
            {
                "Second_Ratio":
                    float(second_ratio),

                "Third_Ratio":
                    float(third_ratio),

                "Count_1_Label":
                    count_1,

                "Count_2_Labels":
                    count_2,

                "Count_3_Labels":
                    count_3,

                "Average_Count":
                    average_predicted,

                "Average_Count_Difference":
                    abs(
                        average_predicted
                        -
                        validation_true_average
                    ),

                **metrics
            }
        )

search_df = pd.DataFrame(
    search_results
)

# ------------------------------------------------------------
# Seçim mantığı
#
# 1) Önce validation'daki maksimum Micro F1'i buluyoruz.
# 2) En iyi F1'in en fazla 0.005 (0.5 yüzde puan) altında kalan
#    adayları "performans olarak yakın" kabul ediyoruz.
# 3) Bu adaylar arasından gerçek ortalama etiket sayısına
#    en yakın olanı seçiyoruz.
# 4) Hâlâ eşitlik varsa daha yüksek Micro F1 / Macro F1 kazanır.
#
# Böylece sırf F1'i birkaç binde yükseltmek için herkese 3 konu
# verme eğilimini azaltıyoruz.
# ------------------------------------------------------------

max_validation_f1 = float(
    search_df[
        "Micro_F1"
    ].max()
)

F1_TOLERANCE = 0.005

candidate_df = search_df[
    search_df[
        "Micro_F1"
    ]
    >=
    (
        max_validation_f1
        -
        F1_TOLERANCE
    )
].copy()

best_row = (
    candidate_df
    .sort_values(
        [
            "Average_Count_Difference",
            "Micro_F1",
            "Macro_F1",
            "Count_2_Labels"
        ],
        ascending=[
            True,
            False,
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
    "Validation gerçek ortalama etiket:",
    round(
        validation_true_average,
        2
    )
)

print(
    "Validation maksimum Micro F1:",
    f"{max_validation_f1 * 100:.2f}%"
)

print(
    "F1 toleransı:",
    f"{F1_TOLERANCE * 100:.2f} yüzde puan"
)

print(
    "Seçilen second ratio (d2/d1):",
    round(
        best_second_ratio,
        4
    )
)

print(
    "Seçilen third ratio (d3/d2):",
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
print("FINAL ADAPTIVE V2 MULTI-LABEL K-MEANS TEST")
print("=" * 110)


(
    y_final_pred,
    final_predicted_counts
) = adaptive_predictions(
    final_distances,
    best_second_ratio,
    best_third_ratio
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

print()

print(
    "1 etiket verilen:",
    int(
        (
            final_predicted_counts
            ==
            1
        ).sum()
    )
)

print(
    "2 etiket verilen:",
    int(
        (
            final_predicted_counts
            ==
            2
        ).sum()
    )
)

print(
    "3 etiket verilen:",
    int(
        (
            final_predicted_counts
            ==
            3
        ).sum()
    )
)


# ============================================================
# 10. ANA BAŞLIK + ALT KONU ÇIKTISI
# ============================================================

prediction_rows = []


for i, row in final_test_df.iterrows():

    selected_indices = np.where(
        y_final_pred[i]
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
# 10B. A/B TEST - SABİT KONU KİMLİĞİ vs RELABEL
# ============================================================
#
# ÖNEMLİ:
# - K-Means aynıdır.
# - Final centroidler aynıdır.
# - Final test aynıdır.
# - Adaptive 1/2/3 cluster seçimi aynıdır.
# - Sadece seçilen clusterın kullanıcıya hangi konu adıyla
#   gösterildiği değişir.
#
# SABİT:
#   Cluster -> başlangıçta verilen centroid_subjects konusu
#
# RELABEL:
#   Cluster -> K-Means sonunda reference kümesinde o clustera
#   düşen makalelerin gerçek etiketleri arasındaki baskın konu
# ============================================================

from collections import Counter

subject_to_index = {
    subject: index
    for index, subject
    in enumerate(leaf_subjects)
}

relabel_subjects = []
relabel_rows = []

for cluster_id in range(len(centroid_subjects)):

    member_positions = np.where(
        reference_labels == cluster_id
    )[0]

    topic_counter = Counter()

    for position in member_positions:

        external_id = str(
            reference.iloc[position][
                "external_id"
            ]
        )

        article_topics = true_subjects.get(
            external_id,
            []
        )

        for topic in article_topics:
            if topic in subject_to_index:
                topic_counter[topic] += 1

    old_subject = centroid_subjects[
        cluster_id
    ]

    if topic_counter:
        (
            new_subject,
            dominant_count
        ) = topic_counter.most_common(1)[0]
    else:
        new_subject = old_subject
        dominant_count = 0

    relabel_subjects.append(
        new_subject
    )

    relabel_rows.append(
        {
            "Cluster_ID":
                cluster_id,

            "Cluster_Size":
                int(
                    len(
                        member_positions
                    )
                ),

            "Old_Subject":
                old_subject,

            "New_Subject":
                new_subject,

            "Changed":
                old_subject
                !=
                new_subject,

            "Dominant_Count":
                int(
                    dominant_count
                )
        }
    )

relabel_df = pd.DataFrame(
    relabel_rows
)


# ------------------------------------------------------------
# Cluster tahmin matrisini RELABEL konu matrisine dönüştür.
#
# y_final_pred'in sütunları cluster sırasıyla aynıdır.
# Sabit modelde cluster i -> centroid_subjects[i].
# Relabel modelde cluster i -> relabel_subjects[i].
#
# Birden fazla cluster aynı yeni konuya dönüşürse aynı sütunda
# birleşir. Bu nedenle Relabel modelin tahmin ettiği etiket
# sayısı Sabit modelden daha düşük olabilir.
# ------------------------------------------------------------

def convert_to_relabel_predictions(
    cluster_prediction_matrix
):

    relabel_prediction_matrix = (
        np.zeros(
            (
                cluster_prediction_matrix.shape[0],
                len(leaf_subjects)
            ),
            dtype=int
        )
    )

    for row_index in range(
        cluster_prediction_matrix.shape[0]
    ):

        selected_cluster_indices = (
            np.where(
                cluster_prediction_matrix[
                    row_index
                ]
                ==
                1
            )[0]
        )

        for cluster_id in (
            selected_cluster_indices
        ):

            new_subject = (
                relabel_subjects[
                    cluster_id
                ]
            )

            new_subject_index = (
                subject_to_index[
                    new_subject
                ]
            )

            relabel_prediction_matrix[
                row_index,
                new_subject_index
            ] = 1

    return (
        relabel_prediction_matrix
    )


# Sabit model:
# y_final_pred zaten mevcut Adaptive K-Means tahminidir.
fixed_final_pred = (
    y_final_pred.copy()
)

# Relabel model:
# Aynı seçilmiş clusterlar, farklı konu kimlikleri.
relabel_final_pred = (
    convert_to_relabel_predictions(
        y_final_pred
    )
)


fixed_metrics_ab = calculate_metrics(
    y_final_true,
    fixed_final_pred
)

relabel_metrics_ab = calculate_metrics(
    y_final_true,
    relabel_final_pred
)


def calculate_extra_metrics(
    y_true,
    y_pred
):

    exact = float(
        np.all(
            y_true
            ==
            y_pred,
            axis=1
        )
        .mean()
    )

    at_least_one_match = float(
        (
            (
                y_true
                &
                y_pred
            )
            .sum(
                axis=1
            )
            >
            0
        )
        .mean()
    )

    return (
        exact,
        at_least_one_match
    )


(
    fixed_exact_ab,
    fixed_at_least_one_ab
) = calculate_extra_metrics(
    y_final_true,
    fixed_final_pred
)

(
    relabel_exact_ab,
    relabel_at_least_one_ab
) = calculate_extra_metrics(
    y_final_true,
    relabel_final_pred
)


fixed_unique_topics = len(
    set(
        centroid_subjects
    )
)

relabel_unique_topics = len(
    set(
        relabel_subjects
    )
)

changed_clusters = int(
    relabel_df[
        "Changed"
    ].sum()
)


comparison_df = pd.DataFrame(
    [
        {
            "Model":
                "Sabit Konu Kimligi",

            **fixed_metrics_ab,

            "Exact_Match_Rate":
                fixed_exact_ab,

            "At_Least_One_Match_Rate":
                fixed_at_least_one_ab,

            "Unique_Topics":
                fixed_unique_topics
        },

        {
            "Model":
                "Relabel - Baskin Konu",

            **relabel_metrics_ab,

            "Exact_Match_Rate":
                relabel_exact_ab,

            "At_Least_One_Match_Rate":
                relabel_at_least_one_ab,

            "Unique_Topics":
                relabel_unique_topics
        }
    ]
)


print()
print("=" * 110)
print(
    "A/B TEST - SABİT KONU KİMLİĞİ vs RELABEL"
)
print("=" * 110)

print(
    "Toplam cluster:",
    len(
        centroid_subjects
    )
)

print(
    "Konu kimliği değişen cluster:",
    changed_clusters
)

print(
    "Sabit model benzersiz konu:",
    fixed_unique_topics
)

print(
    "Relabel model benzersiz konu:",
    relabel_unique_topics
)

print()


for _, result_row in (
    comparison_df.iterrows()
):

    print("-" * 110)

    print(
        result_row[
            "Model"
        ]
    )

    print("-" * 110)

    print(
        "Micro Precision:",
        f"{result_row['Micro_Precision'] * 100:.2f}%"
    )

    print(
        "Micro Recall:",
        f"{result_row['Micro_Recall'] * 100:.2f}%"
    )

    print(
        "Micro F1:",
        f"{result_row['Micro_F1'] * 100:.2f}%"
    )

    print(
        "Macro F1:",
        f"{result_row['Macro_F1'] * 100:.2f}%"
    )

    print(
        "Sample F1:",
        f"{result_row['Sample_F1'] * 100:.2f}%"
    )

    print(
        "Exact Match:",
        f"{result_row['Exact_Match_Rate'] * 100:.2f}%"
    )

    print(
        "En az 1 gerçek etiket:",
        f"{result_row['At_Least_One_Match_Rate'] * 100:.2f}%"
    )

    print(
        "Ortalama tahmin edilen etiket:",
        round(
            float(
                result_row[
                    "Average_Predicted_Labels"
                ]
            ),
            2
        )
    )

    print(
        "Benzersiz konu kapsamı:",
        int(
            result_row[
                "Unique_Topics"
            ]
        ),
        "/",
        len(
            leaf_subjects
        )
    )


winner_row = (
    comparison_df
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


print()
print("=" * 110)
print("A/B TEST KAZANANI")
print("=" * 110)

print(
    winner_row[
        "Model"
    ]
)

print(
    "Micro F1:",
    f"{winner_row['Micro_F1'] * 100:.2f}%"
)


# ------------------------------------------------------------
# A/B sonuçlarını kaydet.
# Bunları daha sonra arayüz doğrudan okuyabilir.
# ------------------------------------------------------------

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

comparison_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "adaptive_v2_fixed_vs_relabel_comparison.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

relabel_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "adaptive_v2_cluster_relabel_map.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


print()
print(
    "A/B sonuç dosyası:"
)
print(
    "results/kmeans/holdout/"
    "adaptive_v2_fixed_vs_relabel_comparison.csv"
)

print(
    "Relabel haritası:"
)
print(
    "results/kmeans/holdout/"
    "adaptive_v2_cluster_relabel_map.csv"
)

print()



# ============================================================
# 10C. RELABEL MODEL - MAKALE BAZLI FINAL TAHMİNLER
# ============================================================

relabel_prediction_rows = []

for i, row in final_test_df.iterrows():

    # Adaptive K-Means'in seçtiği clusterlar.
    selected_cluster_indices = np.where(
        y_final_pred[i] == 1
    )[0]

    # En yakın centroid önce gelsin.
    selected_cluster_indices = sorted(
        selected_cluster_indices,
        key=lambda index: final_distances[i, index]
    )

    # Aynı relabel konusu birden fazla clusterdan gelirse
    # kullanıcıya yalnızca bir kez göster.
    relabel_full_topics = []

    for cluster_id in selected_cluster_indices:
        topic = relabel_subjects[cluster_id]

        if topic not in relabel_full_topics:
            relabel_full_topics.append(topic)

    level_1_topics = []
    level_2_topics = []
    leaf_topics = []

    for full_topic in relabel_full_topics:

        level_1, level_2, leaf = parse_subject(
            full_topic
        )

        if level_1 not in level_1_topics:
            level_1_topics.append(level_1)

        if level_2 not in level_2_topics:
            level_2_topics.append(level_2)

        if leaf not in leaf_topics:
            leaf_topics.append(leaf)

    external_id = str(
        row["external_id"]
    )

    relabel_prediction_rows.append(
        {
            "external_id":
                external_id,

            "predicted_label_count":
                len(relabel_full_topics),

            "main_level_1_predictions":
                " || ".join(level_1_topics),

            "main_level_2_predictions":
                " || ".join(level_2_topics),

            "leaf_predictions":
                " || ".join(leaf_topics),

            "full_topic_predictions":
                " || ".join(relabel_full_topics),

            "true_subjects":
                " || ".join(
                    true_subjects.get(
                        external_id,
                        []
                    )
                )
        }
    )

relabel_prediction_df = pd.DataFrame(
    relabel_prediction_rows
)

relabel_prediction_path = os.path.join(
    OUTPUT_DIR,
    "adaptive_v2_relabel_predictions.csv"
)

relabel_prediction_df.to_csv(
    relabel_prediction_path,
    index=False,
    encoding="utf-8-sig"
)

print()
print("=" * 110)
print("RELABEL MODEL - İLK 10 FINAL TAHMİN")
print("=" * 110)

print(
    relabel_prediction_df[
        [
            "external_id",
            "main_level_2_predictions",
            "leaf_predictions"
        ]
    ]
    .head(10)
    .to_string(index=False)
)

print()
print(
    "Relabel final tahmin dosyası:"
)
print(
    "results/kmeans/holdout/"
    "adaptive_v2_relabel_predictions.csv"
)
print()

# ============================================================
# 11. ÖRNEK TAHMİNLER
# ============================================================

print()
print("=" * 110)
print("İLK 10 TAHMİN ÖRNEĞİ")
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
        "adaptive_v2_rule_search.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


prediction_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "adaptive_v2_predictions.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


summary = pd.DataFrame(
    [
        {
            "Method":
                "Adaptive V2 Multi-label Seeded K-Means",

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
        "adaptive_v2_summary.csv"
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
    "adaptive_v2_predictions.csv"
)

print(
    "results/kmeans/holdout/"
    "adaptive_v2_summary.csv"
)