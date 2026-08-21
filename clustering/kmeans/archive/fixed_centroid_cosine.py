import os
import numpy as np
import pandas as pd


# ============================================================
# AYARLAR
# ============================================================

EMBEDDING_FILE = "embeddings/mpnet_multilingual_embeddings.npy"
SUBJECT_FILE = "data/article_subjects.csv"

REFERENCE_FILE = "results/kmeans/holdout/reference.csv"
TEST_FILE = "results/kmeans/holdout/test.csv"

OUTPUT_DIR = "results/kmeans/holdout"

SEEDS_PER_SUBJECT = 10


# ============================================================
# VERİLERİ OKU
# ============================================================

print("=" * 100)
print("FIXED CENTROID + COSINE DENEYİ")
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
        int(reference_row_map[article_id])
        for article_id in valid_ids
    ]

    vectors = embeddings[rows]

    center = vectors.mean(axis=0)

    center_norm = np.linalg.norm(center)

    if center_norm > 0:
        center = center / center_norm

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
# 195 SABİT KONU CENTROIDI OLUŞTUR
# ============================================================

print("\n" + "=" * 100)
print("SABİT KONU CENTROIDLERİ OLUŞTURULUYOR")
print("=" * 100)

centroids = []
centroid_subjects = []

unique_seed_ids = set()


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

    # Önce tek etiketli temiz örnekleri tercih ediyoruz
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

        # Var olan single-label örnekleri al
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

        # Yetmezse multi-label örneklerle tamamla
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
            f"Bu konu için seed bulunamadı: {subject_name}"
        )


    unique_seed_ids.update(
        selected_ids
    )


    rows = [
        int(reference_row_map[article_id])
        for article_id in selected_ids
    ]

    vectors = embeddings[rows]


    # Konu centroidi
    centroid = vectors.mean(
        axis=0
    )


    # Cosine için normalize et
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
# TEST EMBEDDINGLERİNİ HAZIRLA
# ============================================================

test_rows = (
    test["embedding_row"]
    .astype(int)
    .to_numpy()
)

X_test = embeddings[
    test_rows
]


# Test embeddinglerini normalize et
test_norms = np.linalg.norm(
    X_test,
    axis=1,
    keepdims=True
)

test_norms[
    test_norms == 0
] = 1

X_test_normalized = (
    X_test / test_norms
)


# ============================================================
# COSINE SIMILARITY
# ============================================================
#
# 5571 x 768
#       @
# 768 x 195
#
# Sonuç:
# 5571 x 195 similarity matrix
# ============================================================

print("\n" + "=" * 100)
print("COSINE SIMILARITY HESAPLANIYOR")
print("=" * 100)

similarities = (
    X_test_normalized
    @
    centroid_matrix.T
)

print(
    "Similarity matrix:",
    similarities.shape
)


# ============================================================
# TOP-5 TAHMİNLER
# ============================================================

top5_indices = np.argsort(
    -similarities,
    axis=1
)[:, :5]


top5_scores = np.take_along_axis(
    similarities,
    top5_indices,
    axis=1
)


# ============================================================
# GERÇEK ETİKETLER
# ============================================================

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


# ============================================================
# TOP-1 / TOP-3 / TOP-5 DEĞERLENDİRME
# ============================================================

top1_correct = 0
top3_correct = 0
top5_correct = 0

prediction_rows = []


for i, row in test.reset_index(drop=True).iterrows():

    external_id = str(
        row["external_id"]
    )

    real_topics = true_subjects.get(
        external_id,
        set()
    )


    predicted_topics = [
        centroid_subjects[index]
        for index in top5_indices[i]
    ]


    predicted_scores = [
        float(score)
        for score in top5_scores[i]
    ]


    # TOP-1
    top1_match = any(
        topic in real_topics
        for topic in predicted_topics[:1]
    )


    # TOP-3
    top3_match = any(
        topic in real_topics
        for topic in predicted_topics[:3]
    )


    # TOP-5
    top5_match = any(
        topic in real_topics
        for topic in predicted_topics[:5]
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


# ============================================================
# SONUÇ
# ============================================================

total = len(test)

top1_rate = (
    top1_correct / total
)

top3_rate = (
    top3_correct / total
)

top5_rate = (
    top5_correct / total
)


print("\n" + "=" * 100)
print("FIXED CENTROID + COSINE SONUCU")
print("=" * 100)

print(
    "Test makale:",
    total
)

print(
    "Top-1 doğru:",
    top1_correct
)

print(
    "Top-1 konu uyumu:",
    f"{top1_rate * 100:.2f}%"
)

print()

print(
    "Top-3 doğru:",
    top3_correct
)

print(
    "Top-3 konu uyumu:",
    f"{top3_rate * 100:.2f}%"
)

print()

print(
    "Top-5 doğru:",
    top5_correct
)

print(
    "Top-5 konu uyumu:",
    f"{top5_rate * 100:.2f}%"
)


# ============================================================
# TOP-1 CONFIDENCE
# ============================================================

top1_similarity = (
    top5_scores[:, 0]
)

print("\n" + "=" * 100)
print("TOP-1 COSINE BENZERLİK İSTATİSTİKLERİ")
print("=" * 100)

print(
    "Ortalama:",
    round(
        float(
            top1_similarity.mean()
        ),
        4
    )
)

print(
    "Medyan:",
    round(
        float(
            np.median(
                top1_similarity
            )
        ),
        4
    )
)

print(
    "Minimum:",
    round(
        float(
            top1_similarity.min()
        ),
        4
    )
)

print(
    "Maximum:",
    round(
        float(
            top1_similarity.max()
        ),
        4
    )
)


# ============================================================
# KAYDET
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


predictions_df = pd.DataFrame(
    prediction_rows
)


predictions_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "fixed_centroid_cosine_predictions.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


summary = pd.DataFrame(
    [
        {
            "Method":
                "Fixed Centroid + Cosine",

            "Seeds_Per_Subject":
                SEEDS_PER_SUBJECT,

            "Reference_Articles":
                len(reference),

            "Test_Articles":
                total,

            "Unique_Seed_Articles":
                len(unique_seed_ids),

            "Top1_Correct":
                top1_correct,

            "Top1_Rate":
                top1_rate,

            "Top3_Correct":
                top3_correct,

            "Top3_Rate":
                top3_rate,

            "Top5_Correct":
                top5_correct,

            "Top5_Rate":
                top5_rate
        }
    ]
)


summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "fixed_centroid_cosine_summary.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


np.save(
    os.path.join(
        OUTPUT_DIR,
        "fixed_centroid_cosine_centroids.npy"
    ),
    centroid_matrix
)


print("\nDosyalar oluşturuldu:")

print(
    "results/kmeans/holdout/fixed_centroid_cosine_predictions.csv"
)

print(
    "results/kmeans/holdout/fixed_centroid_cosine_summary.csv"
)