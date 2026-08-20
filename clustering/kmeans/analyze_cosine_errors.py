import os
import pandas as pd


# ============================================================
# DOSYALAR
# ============================================================

PREDICTION_FILE = (
    "results/kmeans/holdout/"
    "fixed_centroid_cosine_predictions.csv"
)

SUBJECT_FILE = "data/article_subjects.csv"

OUTPUT_DIR = "results/kmeans/holdout"


# ============================================================
# VERİLERİ OKU
# ============================================================

print("=" * 110)
print("FIXED CENTROID + COSINE - HATA ANALİZİ")
print("=" * 110)

predictions = pd.read_csv(
    PREDICTION_FILE,
    encoding="utf-8-sig"
)

subjects = pd.read_csv(
    SUBJECT_FILE,
    encoding="utf-8-sig"
)

predictions["external_id"] = (
    predictions["external_id"].astype(str)
)

subjects["external_id"] = (
    subjects["external_id"].astype(str)
)

print("Tahmin edilen makale:", len(predictions))


# ============================================================
# TEST MAKALELERİNİN GERÇEK ETİKETLERİ
# ============================================================

test_ids = set(
    predictions["external_id"]
)

test_subjects = subjects[
    subjects["external_id"].isin(test_ids)
].copy()


# ============================================================
# HER GERÇEK KONU İÇİN BAŞARI
# ============================================================
#
# Multi-label olduğu için:
# Bir makale birden fazla gerçek konuya sahip olabilir.
#
# Her gerçek konu için:
# "Bu konuyu taşıyan makalelerin kaçında Top-1
# tahmin gerçekten bu konu oldu?"
# ============================================================

merged = test_subjects.merge(
    predictions[
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


merged["exact_subject_match"] = (
    merged["subject_fullname"]
    ==
    merged["top1_subject"]
)


subject_performance = (
    merged
    .groupby("subject_fullname")
    .agg(
        article_count=(
            "external_id",
            "nunique"
        ),

        correct_count=(
            "exact_subject_match",
            "sum"
        ),

        mean_top1_score=(
            "top1_score",
            "mean"
        )
    )
    .reset_index()
)


subject_performance["accuracy"] = (
    subject_performance["correct_count"]
    /
    subject_performance["article_count"]
)


subject_performance[
    "accuracy_percent"
] = (
    subject_performance["accuracy"]
    *
    100
)


# ============================================================
# EN İYİ / EN KÖTÜ KONULAR
# ============================================================

# Çok az örnekli konular yanıltıcı olabileceği için
# en az 10 test makalesi olan konulara ayrıca bakıyoruz.

reliable_subjects = (
    subject_performance[
        subject_performance[
            "article_count"
        ] >= 10
    ]
    .copy()
)


best_subjects = (
    reliable_subjects
    .sort_values(
        "accuracy",
        ascending=False
    )
    .head(15)
)


worst_subjects = (
    reliable_subjects
    .sort_values(
        [
            "accuracy",
            "article_count"
        ],
        ascending=[
            True,
            False
        ]
    )
    .head(15)
)


print("\n" + "=" * 110)
print("EN BAŞARILI 15 KONU (EN AZ 10 MAKALE)")
print("=" * 110)

print(
    best_subjects[
        [
            "subject_fullname",
            "article_count",
            "correct_count",
            "accuracy_percent"
        ]
    ].to_string(
        index=False
    )
)


print("\n" + "=" * 110)
print("EN BAŞARISIZ 15 KONU (EN AZ 10 MAKALE)")
print("=" * 110)

print(
    worst_subjects[
        [
            "subject_fullname",
            "article_count",
            "correct_count",
            "accuracy_percent"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# YANLIŞ TAHMİNLERDE KARIŞAN KONU ÇİFTLERİ
# ============================================================

wrong = merged[
    ~merged["exact_subject_match"]
].copy()


confusions = (
    wrong
    .groupby(
        [
            "subject_fullname",
            "top1_subject"
        ]
    )
    .size()
    .reset_index(
        name="confusion_count"
    )
)


confusions = (
    confusions
    .sort_values(
        "confusion_count",
        ascending=False
    )
)


print("\n" + "=" * 130)
print("EN ÇOK KARIŞAN 30 KONU ÇİFTİ")
print("=" * 130)

print(
    confusions
    .head(30)
    .to_string(
        index=False
    )
)


# ============================================================
# TOP-1 YANLIŞ AMA TOP-3'TE DOĞRU MU?
# ============================================================

merged["in_top3"] = (
    (merged["subject_fullname"] == merged["top1_subject"])
    |
    (merged["subject_fullname"] == merged["top2_subject"])
    |
    (merged["subject_fullname"] == merged["top3_subject"])
)


top1_wrong = merged[
    ~merged["exact_subject_match"]
]


recovered_top3 = int(
    top1_wrong["in_top3"].sum()
)


print("\n" + "=" * 110)
print("TOP-3 KURTARMA ANALİZİ")
print("=" * 110)

print(
    "Top-1'de yanlış gerçek-konu ilişkisi:",
    len(top1_wrong)
)

print(
    "Bunlardan Top-3 içinde bulunan:",
    recovered_top3
)

if len(top1_wrong) > 0:

    print(
        "Top-3 kurtarma oranı:",
        f"{recovered_top3 / len(top1_wrong) * 100:.2f}%"
    )


# ============================================================
# KAYDET
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


subject_performance.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "cosine_subject_performance.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


confusions.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "cosine_subject_confusions.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


print("\nDosyalar oluşturuldu:")

print(
    "results/kmeans/holdout/"
    "cosine_subject_performance.csv"
)

print(
    "results/kmeans/holdout/"
    "cosine_subject_confusions.csv"
)