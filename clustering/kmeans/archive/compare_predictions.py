import os
import pandas as pd


# ============================================================
# DOSYALAR
# ============================================================

ADAPTIVE_FILE = (
    "results/kmeans/holdout/"
    "adaptive_kmeans_predictions.csv"
)

COSINE_FILE = (
    "results/kmeans/holdout/"
    "fixed_centroid_cosine_predictions.csv"
)

SEEDED_FILE = (
    "results/kmeans/holdout/"
    "seeded_holdout_predictions.csv"
)

SUBJECT_FILE = (
    "data/article_subjects.csv"
)

OUTPUT_FILE = (
    "results/kmeans/holdout/"
    "prediction_comparison.csv"
)


# ============================================================
# VERİLERİ OKU
# ============================================================

print("=" * 110)
print("TAHMİN KARŞILAŞTIRMA")
print("=" * 110)

adaptive = pd.read_csv(
    ADAPTIVE_FILE,
    encoding="utf-8-sig"
)

cosine = pd.read_csv(
    COSINE_FILE,
    encoding="utf-8-sig"
)

seeded = pd.read_csv(
    SEEDED_FILE,
    encoding="utf-8-sig"
)

subjects = pd.read_csv(
    SUBJECT_FILE,
    encoding="utf-8-sig"
)


for df in [
    adaptive,
    cosine,
    seeded,
    subjects
]:
    df["external_id"] = (
        df["external_id"]
        .astype(str)
    )


print(
    "Adaptive:",
    len(adaptive)
)

print(
    "Cosine:",
    len(cosine)
)

print(
    "Seeded:",
    len(seeded)
)


# ============================================================
# GERÇEK KONULAR
# ============================================================

true_subjects = (
    subjects
    .groupby(
        "external_id"
    )["subject_fullname"]
    .apply(
        lambda x:
            sorted(
                set(
                    x.dropna()
                    .astype(str)
                )
            )
    )
    .to_dict()
)


# ============================================================
# FIXED COSINE:
# TOP-3 TAHMİNİ TEK METİN HALİNE GETİR
# ============================================================

cosine["cosine_top3"] = (
    cosine[
        [
            "top1_subject",
            "top2_subject",
            "top3_subject"
        ]
    ]
    .fillna("")
    .apply(
        lambda row:
            " || ".join(
                [
                    value
                    for value
                    in row.tolist()
                    if value
                ]
            ),
        axis=1
    )
)


# ============================================================
# SEEDED:
# TEK KONU TAHMİNİ
# ============================================================

seeded_small = (
    seeded[
        [
            "external_id",
            "predicted_subject"
        ]
    ]
    .rename(
        columns={
            "predicted_subject":
                "seeded_prediction"
        }
    )
)


# ============================================================
# ADAPTIVE:
# ZATEN MULTI-LABEL ÇIKTI VAR
# ============================================================

adaptive_small = (
    adaptive[
        [
            "external_id",
            "predicted_label_count",
            "main_level_1_predictions",
            "main_level_2_predictions",
            "leaf_predictions",
            "full_topic_predictions"
        ]
    ]
    .copy()
)


# ============================================================
# COSINE:
# TOP-1 VE TOP-3
# ============================================================

cosine_small = (
    cosine[
        [
            "external_id",
            "top1_subject",
            "cosine_top3"
        ]
    ]
    .rename(
        columns={
            "top1_subject":
                "cosine_top1"
        }
    )
)


# ============================================================
# TÜMÜNÜ BİRLEŞTİR
# ============================================================

comparison = (
    adaptive_small
    .merge(
        cosine_small,
        on="external_id",
        how="left"
    )
    .merge(
        seeded_small,
        on="external_id",
        how="left"
    )
)


# ============================================================
# GERÇEK ETİKETLERİ EKLE
# ============================================================

comparison[
    "true_subjects"
] = (
    comparison[
        "external_id"
    ]
    .map(
        lambda external_id:
            " || ".join(
                true_subjects.get(
                    external_id,
                    []
                )
            )
    )
)


# ============================================================
# YARDIMCI FONKSİYON
# ============================================================

def split_labels(value):

    if pd.isna(value):
        return set()

    return set(
        [
            item.strip()
            for item
            in str(value).split("||")
            if item.strip()
        ]
    )


# ============================================================
# ADAPTIVE BAŞARI
# ============================================================

adaptive_correct_counts = []
adaptive_precision = []
adaptive_recall = []
adaptive_f1 = []


for _, row in comparison.iterrows():

    true_set = split_labels(
        row["true_subjects"]
    )

    predicted_set = split_labels(
        row[
            "full_topic_predictions"
        ]
    )

    intersection = (
        true_set
        &
        predicted_set
    )

    correct_count = len(
        intersection
    )


    if len(predicted_set) > 0:

        precision = (
            correct_count
            /
            len(predicted_set)
        )

    else:

        precision = 0


    if len(true_set) > 0:

        recall = (
            correct_count
            /
            len(true_set)
        )

    else:

        recall = 0


    if (
        precision
        +
        recall
        >
        0
    ):

        f1 = (
            2
            *
            precision
            *
            recall
            /
            (
                precision
                +
                recall
            )
        )

    else:

        f1 = 0


    adaptive_correct_counts.append(
        correct_count
    )

    adaptive_precision.append(
        precision
    )

    adaptive_recall.append(
        recall
    )

    adaptive_f1.append(
        f1
    )


comparison[
    "adaptive_correct_labels"
] = adaptive_correct_counts

comparison[
    "adaptive_precision"
] = adaptive_precision

comparison[
    "adaptive_recall"
] = adaptive_recall

comparison[
    "adaptive_f1"
] = adaptive_f1


# ============================================================
# COSINE TOP-3:
# KAÇ GERÇEK ETİKETİ YAKALADI?
# ============================================================

cosine_correct_counts = []


for _, row in comparison.iterrows():

    true_set = split_labels(
        row["true_subjects"]
    )

    cosine_set = split_labels(
        row["cosine_top3"]
    )

    cosine_correct_counts.append(
        len(
            true_set
            &
            cosine_set
        )
    )


comparison[
    "cosine_top3_correct_labels"
] = cosine_correct_counts


# ============================================================
# ÖZET
# ============================================================

print()
print("=" * 110)
print("GENEL ÖZET")
print("=" * 110)

print(
    "Karşılaştırılan makale:",
    len(comparison)
)

print(
    "Adaptive ortalama F1:",
    round(
        comparison[
            "adaptive_f1"
        ].mean(),
        4
    )
)

print(
    "Adaptive en az 1 doğru:",
    int(
        (
            comparison[
                "adaptive_correct_labels"
            ]
            >
            0
        ).sum()
    )
)

print(
    "Cosine Top-3 en az 1 doğru:",
    int(
        (
            comparison[
                "cosine_top3_correct_labels"
            ]
            >
            0
        ).sum()
    )
)


# ============================================================
# EN İYİ 10 ADAPTIVE TAHMİN
# ============================================================

print()
print("=" * 110)
print("EN İYİ 10 ADAPTIVE TAHMİN")
print("=" * 110)

best_examples = (
    comparison
    .sort_values(
        [
            "adaptive_f1",
            "adaptive_correct_labels"
        ],
        ascending=[
            False,
            False
        ]
    )
    .head(10)
)


print(
    best_examples[
        [
            "external_id",
            "true_subjects",
            "full_topic_predictions",
            "adaptive_f1"
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# EN KÖTÜ 10 ADAPTIVE TAHMİN
# ============================================================

print()
print("=" * 110)
print("EN KÖTÜ 10 ADAPTIVE TAHMİN")
print("=" * 110)

worst_examples = (
    comparison
    .sort_values(
        [
            "adaptive_f1",
            "adaptive_correct_labels"
        ],
        ascending=[
            True,
            True
        ]
    )
    .head(10)
)


print(
    worst_examples[
        [
            "external_id",
            "true_subjects",
            "full_topic_predictions",
            "adaptive_f1"
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# YÖNTEMLERİN FARKLI DÜŞÜNDÜĞÜ 10 ÖRNEK
# ============================================================

different = comparison[
    comparison[
        "seeded_prediction"
    ]
    !=
    comparison[
        "cosine_top1"
    ]
]


print()
print("=" * 110)
print("SEEDED VE COSINE FARKLI KARAR VEREN 10 MAKALE")
print("=" * 110)


print(
    different[
        [
            "external_id",
            "true_subjects",
            "seeded_prediction",
            "cosine_top1",
            "full_topic_predictions"
        ]
    ]
    .head(10)
    .to_string(
        index=False
    )
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


comparison.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


print()
print("=" * 110)
print("TAMAMLANDI")
print("=" * 110)

print(
    OUTPUT_FILE
)