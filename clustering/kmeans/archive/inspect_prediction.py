import pandas as pd


FILE = "results/kmeans/holdout/prediction_comparison.csv"


# ============================================================
# VERİYİ OKU
# ============================================================

df = pd.read_csv(
    FILE,
    encoding="utf-8-sig"
)

df["external_id"] = (
    df["external_id"]
    .astype(str)
)


# ============================================================
# ETİKET AYIR
# ============================================================

def split_labels(value):

    if pd.isna(value):
        return set()

    return {
        item.strip()
        for item in str(value).split("||")
        if item.strip()
    }


# ============================================================
# GÜZEL YAZDIR
# ============================================================

def print_labels(title, labels):

    print()
    print(title)
    print("-" * 90)

    if not labels:
        print("Yok")
        return

    for label in sorted(labels):
        print("•", label)


# ============================================================
# KARŞILAŞTIRMA
# ============================================================

def compare_prediction(
    name,
    true_labels,
    predicted_labels
):

    correct = (
        true_labels
        &
        predicted_labels
    )

    wrong = (
        predicted_labels
        -
        true_labels
    )

    missed = (
        true_labels
        -
        predicted_labels
    )


    if predicted_labels:

        precision = (
            len(correct)
            /
            len(predicted_labels)
        )

    else:
        precision = 0


    if true_labels:

        recall = (
            len(correct)
            /
            len(true_labels)
        )

    else:
        recall = 0


    if precision + recall > 0:

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


    print()
    print("=" * 100)
    print(name)
    print("=" * 100)

    print_labels(
        "TAHMİN",
        predicted_labels
    )

    print_labels(
        "DOĞRU YAKALANAN",
        correct
    )

    print_labels(
        "YANLIŞ EKLENEN",
        wrong
    )

    print_labels(
        "KAÇIRILAN",
        missed
    )

    print()

    print(
        f"Doğru: {len(correct)}"
    )

    print(
        f"Yanlış: {len(wrong)}"
    )

    print(
        f"Kaçırılan: {len(missed)}"
    )

    print(
        f"Precision: {precision * 100:.2f}%"
    )

    print(
        f"Recall: {recall * 100:.2f}%"
    )

    print(
        f"F1: {f1 * 100:.2f}%"
    )


# ============================================================
# ID AL
# ============================================================

print("=" * 100)
print("K-MEANS MAKALE TAHMİN İNCELEME")
print("=" * 100)

article_id = input(
    "İncelenecek external_id: "
).strip()


result = df[
    df["external_id"]
    ==
    article_id
]


if result.empty:

    print()
    print(
        "Bu external_id comparison dosyasında bulunamadı."
    )

    raise SystemExit


row = result.iloc[0]


# ============================================================
# GERÇEK ETİKETLER
# ============================================================

true_labels = split_labels(
    row["true_subjects"]
)


print()
print("=" * 100)
print("MAKALE:", article_id)
print("=" * 100)

print_labels(
    "GERÇEK TR DİZİN ETİKETLERİ",
    true_labels
)


# ============================================================
# SEEDED K-MEANS
# ============================================================

seeded_labels = split_labels(
    row["seeded_prediction"]
)

compare_prediction(
    "SEEDED K-MEANS",
    true_labels,
    seeded_labels
)


# ============================================================
# FIXED CENTROID + COSINE TOP-1
# ============================================================

cosine_top1 = split_labels(
    row["cosine_top1"]
)

compare_prediction(
    "FIXED CENTROID + COSINE TOP-1",
    true_labels,
    cosine_top1
)


# ============================================================
# FIXED CENTROID + COSINE TOP-3
# ============================================================

cosine_top3 = split_labels(
    row["cosine_top3"]
)

compare_prediction(
    "FIXED CENTROID + COSINE TOP-3",
    true_labels,
    cosine_top3
)


# ============================================================
# ADAPTIVE K-MEANS
# ============================================================

adaptive_labels = split_labels(
    row["full_topic_predictions"]
)

compare_prediction(
    "ADAPTIVE MULTI-LABEL K-MEANS",
    true_labels,
    adaptive_labels
)


# ============================================================
# ANA / ALT BAŞLIKLAR
# ============================================================

print()
print("=" * 100)
print("ADAPTIVE K-MEANS BAŞLIK YAPISI")
print("=" * 100)

print()
print(
    "ANA BAŞLIK TAHMİNLERİ:"
)

print(
    row[
        "main_level_2_predictions"
    ]
)

print()

print(
    "ALT KONU TAHMİNLERİ:"
)

print(
    row[
        "leaf_predictions"
    ]
)


# ============================================================
# KISA KARŞILAŞTIRMA
# ============================================================

methods = {
    "Seeded K-Means":
        seeded_labels,

    "Cosine Top-1":
        cosine_top1,

    "Cosine Top-3":
        cosine_top3,

    "Adaptive K-Means":
        adaptive_labels
}


scores = []


for method_name, predictions in methods.items():

    correct = len(
        true_labels
        &
        predictions
    )

    if predictions:

        precision = (
            correct
            /
            len(predictions)
        )

    else:

        precision = 0


    if true_labels:

        recall = (
            correct
            /
            len(true_labels)
        )

    else:

        recall = 0


    if precision + recall > 0:

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


    scores.append(
        (
            method_name,
            correct,
            precision,
            recall,
            f1
        )
    )


scores = sorted(
    scores,
    key=lambda x: x[4],
    reverse=True
)


print()
print("=" * 100)
print("GENEL KARŞILAŞTIRMA")
print("=" * 100)

for (
    name,
    correct,
    precision,
    recall,
    f1
) in scores:

    print(
        f"{name:25}"
        f" | Doğru: {correct}"
        f" | P: {precision * 100:6.2f}%"
        f" | R: {recall * 100:6.2f}%"
        f" | F1: {f1 * 100:6.2f}%"
    )


print()
print(
    "Bu makalede en yüksek F1:",
    scores[0][0]
)