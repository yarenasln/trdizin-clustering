import os
import pandas as pd


# ============================================================
# DOSYALAR
# ============================================================

SEEDED_FILE = (
    "results/kmeans/holdout/"
    "seeded_holdout_predictions.csv"
)

COSINE_FILE = (
    "results/kmeans/holdout/"
    "fixed_centroid_cosine_predictions.csv"
)

SUBJECT_FILE = "data/article_subjects.csv"

OUTPUT_FILE = (
    "results/kmeans/holdout/"
    "seeded_cosine_error_analysis.csv"
)


# ============================================================
# VERİLERİ OKU
# ============================================================

print("=" * 100)
print("SEEDED K-MEANS vs FIXED COSINE - HATA ANALİZİ")
print("=" * 100)

seeded = pd.read_csv(
    SEEDED_FILE,
    encoding="utf-8-sig"
)

cosine = pd.read_csv(
    COSINE_FILE,
    encoding="utf-8-sig"
)

subjects = pd.read_csv(
    SUBJECT_FILE,
    encoding="utf-8-sig"
)


seeded["external_id"] = (
    seeded["external_id"].astype(str)
)

cosine["external_id"] = (
    cosine["external_id"].astype(str)
)

subjects["external_id"] = (
    subjects["external_id"].astype(str)
)


print("Seeded tahmin:", len(seeded))
print("Cosine tahmin:", len(cosine))


# ============================================================
# GERÇEK KONULAR
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
# İKİ TAHMİNİ BİRLEŞTİR
# ============================================================

comparison = (
    seeded[
        [
            "external_id",
            "predicted_subject"
        ]
    ]
    .rename(
        columns={
            "predicted_subject":
                "seeded_subject"
        }
    )
    .merge(
        cosine[
            [
                "external_id",
                "top1_subject",
                "top1_score",
                "top2_subject",
                "top2_score",
                "top3_subject",
                "top3_score"
            ]
        ],
        on="external_id",
        how="inner"
    )
)


print(
    "Karşılaştırılan makale:",
    len(comparison)
)


# ============================================================
# DOĞRU / YANLIŞ KONTROLÜ
# ============================================================

def is_correct(
    external_id,
    predicted_subject
):

    real_topics = true_subjects.get(
        str(external_id),
        set()
    )

    return (
        predicted_subject
        in
        real_topics
    )


comparison["seeded_correct"] = (
    comparison.apply(
        lambda row:
            is_correct(
                row["external_id"],
                row["seeded_subject"]
            ),
        axis=1
    )
)


comparison["cosine_correct"] = (
    comparison.apply(
        lambda row:
            is_correct(
                row["external_id"],
                row["top1_subject"]
            ),
        axis=1
    )
)


# ============================================================
# KARAR GRUPLARI
# ============================================================

def get_group(row):

    if (
        row["seeded_correct"]
        and
        row["cosine_correct"]
    ):
        return "BOTH_CORRECT"

    if (
        row["seeded_correct"]
        and
        not row["cosine_correct"]
    ):
        return "SEEDED_ONLY_CORRECT"

    if (
        not row["seeded_correct"]
        and
        row["cosine_correct"]
    ):
        return "COSINE_ONLY_CORRECT"

    return "BOTH_WRONG"


comparison["result_group"] = (
    comparison.apply(
        get_group,
        axis=1
    )
)


# ============================================================
# COSINE CONFIDENCE / MARGIN
# ============================================================
#
# top1_score:
# En iyi konunun benzerliği.
#
# margin:
# 1. konu ile 2. konu arasındaki fark.
#
# Fark büyükse cosine kararından
# daha emin olabiliriz.
# ============================================================

comparison[
    "cosine_margin"
] = (
    comparison["top1_score"]
    -
    comparison["top2_score"]
)


# ============================================================
# SONUÇLAR
# ============================================================

counts = (
    comparison[
        "result_group"
    ]
    .value_counts()
)


total = len(
    comparison
)


both_correct = int(
    counts.get(
        "BOTH_CORRECT",
        0
    )
)

seeded_only = int(
    counts.get(
        "SEEDED_ONLY_CORRECT",
        0
    )
)

cosine_only = int(
    counts.get(
        "COSINE_ONLY_CORRECT",
        0
    )
)

both_wrong = int(
    counts.get(
        "BOTH_WRONG",
        0
    )
)


print("\n" + "=" * 100)
print("HATA DAĞILIMI")
print("=" * 100)

print(
    "İkisi de doğru:",
    both_correct,
    f"({both_correct / total * 100:.2f}%)"
)

print(
    "Sadece Seeded doğru:",
    seeded_only,
    f"({seeded_only / total * 100:.2f}%)"
)

print(
    "Sadece Cosine doğru:",
    cosine_only,
    f"({cosine_only / total * 100:.2f}%)"
)

print(
    "İkisi de yanlış:",
    both_wrong,
    f"({both_wrong / total * 100:.2f}%)"
)


# ============================================================
# TEORİK ORACLE
# ============================================================
#
# Eğer her makalede Seeded veya Cosine'dan
# doğru olanı kusursuz seçebilseydik
# ulaşabileceğimiz üst sınır.
# ============================================================

oracle_correct = (
    both_correct
    +
    seeded_only
    +
    cosine_only
)

oracle_rate = (
    oracle_correct
    /
    total
)


print("\n" + "=" * 100)
print("TEORİK HİBRİT ÜST SINIRI")
print("=" * 100)

print(
    "En az biri doğru:",
    oracle_correct
)

print(
    "Teorik maksimum Top-1:",
    f"{oracle_rate * 100:.2f}%"
)


# ============================================================
# COSINE CONFIDENCE ANALİZİ
# ============================================================

print("\n" + "=" * 100)
print("COSINE CONFIDENCE ANALİZİ")
print("=" * 100)


confidence_summary = (
    comparison
    .groupby(
        "result_group"
    )
    .agg(
        article_count=(
            "external_id",
            "count"
        ),

        mean_top1_score=(
            "top1_score",
            "mean"
        ),

        mean_margin=(
            "cosine_margin",
            "mean"
        ),

        median_margin=(
            "cosine_margin",
            "median"
        )
    )
    .reset_index()
)


print(
    confidence_summary
    .to_string(
        index=False
    )
)


# ============================================================
# TAHMİNLER AYNI MI?
# ============================================================

comparison[
    "same_prediction"
] = (
    comparison[
        "seeded_subject"
    ]
    ==
    comparison[
        "top1_subject"
    ]
)


same_count = int(
    comparison[
        "same_prediction"
    ].sum()
)


print("\n" + "=" * 100)
print("TAHMİN FARKLILIĞI")
print("=" * 100)

print(
    "Aynı konuyu seçtikleri:",
    same_count,
    f"({same_count / total * 100:.2f}%)"
)

print(
    "Farklı konu seçtikleri:",
    total - same_count,
    f"({(total - same_count) / total * 100:.2f}%)"
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


confidence_summary.to_csv(
    "results/kmeans/holdout/"
    "seeded_cosine_confidence_summary.csv",
    index=False,
    encoding="utf-8-sig"
)


print("\nDosyalar oluşturuldu:")

print(
    OUTPUT_FILE
)

print(
    "results/kmeans/holdout/"
    "seeded_cosine_confidence_summary.csv"
)