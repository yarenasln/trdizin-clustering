import pandas as pd

# ============================================================
# DOSYALAR
# ============================================================

ASSIGNMENT_FILE = "results/kmeans/baseline_cluster_assignments.csv"
SUBJECT_FILE = "data/article_subjects.csv"
OUTPUT_FILE = "results/kmeans/baseline_cluster_topic_evaluation.csv"


# ============================================================
# VERİLER
# ============================================================

results = pd.read_csv(ASSIGNMENT_FILE, encoding="utf-8-sig")
subjects = pd.read_csv(SUBJECT_FILE, encoding="utf-8-sig")

results["external_id"] = results["external_id"].astype(str)
subjects["external_id"] = subjects["external_id"].astype(str)


print("=" * 100)
print("K-MEANS++ KONU UYUMU")
print("=" * 100)

print("Makale sayısı:", len(results))


# ============================================================
# CLUSTER + GERÇEK ETİKETLER
# ============================================================

merged = results[
    ["external_id", "baseline_cluster_id"]
].merge(
    subjects[
        ["external_id", "subject_fullname"]
    ],
    on="external_id",
    how="left"
)


# ============================================================
# HER CLUSTER'IN EN ÇOK GÖRÜLEN KONUSUNU BUL
# ============================================================

topic_counts = (
    merged
    .groupby(
        ["baseline_cluster_id", "subject_fullname"]
    )
    .size()
    .reset_index(name="count")
)


dominant_topics = (
    topic_counts
    .sort_values(
        ["baseline_cluster_id", "count"],
        ascending=[True, False]
    )
    .drop_duplicates("baseline_cluster_id")
    .rename(
        columns={
            "subject_fullname": "assigned_subject"
        }
    )
)


cluster_to_subject = (
    dominant_topics
    .set_index("baseline_cluster_id")["assigned_subject"]
    .to_dict()
)


# ============================================================
# HER MAKALE İÇİN CLUSTER KONUSU
# ============================================================

results["assigned_subject"] = (
    results["baseline_cluster_id"]
    .map(cluster_to_subject)
)


# ============================================================
# GERÇEK ETİKETLERDEN BİRİYLE EŞLEŞİYOR MU?
# ============================================================

check = results[
    [
        "external_id",
        "baseline_cluster_id",
        "assigned_subject"
    ]
].merge(
    subjects[
        ["external_id", "subject_fullname"]
    ],
    on="external_id",
    how="left"
)


check["subject_match"] = (
    check["assigned_subject"]
    ==
    check["subject_fullname"]
)


article_match = (
    check
    .groupby("external_id")["subject_match"]
    .any()
)


matched = int(article_match.sum())
total = int(article_match.size)

match_rate = matched / total


# ============================================================
# SONUÇ
# ============================================================

print("\n" + "=" * 100)
print("KONU UYUM SONUCU")
print("=" * 100)

print("Toplam makale:", total)
print("Eşleşen makale:", matched)

print(
    "Konu uyum yüzdesi:",
    f"{match_rate * 100:.2f}%"
)


# ============================================================
# CLUSTER BAZLI SONUÇ
# ============================================================

article_result = results[
    [
        "external_id",
        "baseline_cluster_id",
        "assigned_subject"
    ]
].merge(
    article_match
    .rename("subject_match")
    .reset_index(),
    on="external_id",
    how="left"
)


cluster_eval = (
    article_result
    .groupby(
        [
            "baseline_cluster_id",
            "assigned_subject"
        ]
    )
    .agg(
        article_count=("external_id", "nunique"),
        matched_articles=("subject_match", "sum")
    )
    .reset_index()
)


cluster_eval["match_rate"] = (
    cluster_eval["matched_articles"]
    /
    cluster_eval["article_count"]
)


print("\n" + "=" * 100)
print("EN YÜKSEK UYUMLU 10 CLUSTER")
print("=" * 100)

print(
    cluster_eval
    .sort_values(
        ["match_rate", "article_count"],
        ascending=[False, False]
    )
    .head(10)
    .to_string(index=False)
)


# ============================================================
# KAYDET
# ============================================================

cluster_eval.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\nDosya oluşturuldu:")
print(OUTPUT_FILE)