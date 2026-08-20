import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# AYARLAR
# ============================================================

ERROR_FILE = (
    "results/kmeans/holdout/"
    "seeded_cosine_error_analysis.csv"
)

SUBJECT_FILE = "data/article_subjects.csv"

OUTPUT_DIR = "results/kmeans/holdout"

RANDOM_STATE = 42
VALIDATION_RATIO = 0.50


# ============================================================
# VERİLERİ OKU
# ============================================================

print("=" * 100)
print("HYBRID K-MEANS + COSINE - VALIDATION DENEYİ")
print("=" * 100)

data = pd.read_csv(
    ERROR_FILE,
    encoding="utf-8-sig"
)

subjects = pd.read_csv(
    SUBJECT_FILE,
    encoding="utf-8-sig"
)

data["external_id"] = data["external_id"].astype(str)
subjects["external_id"] = subjects["external_id"].astype(str)

print("Toplam makale:", len(data))


# ============================================================
# GERÇEK ETİKETLER
# ============================================================

true_subjects = (
    subjects
    .groupby("external_id")["subject_fullname"]
    .apply(
        lambda x:
            set(
                x.dropna().astype(str)
            )
    )
    .to_dict()
)


def is_correct(external_id, predicted_subject):

    real_topics = true_subjects.get(
        str(external_id),
        set()
    )

    return predicted_subject in real_topics


# ============================================================
# VALIDATION / FINAL TEST AYRIMI
# ============================================================
#
# Validation:
# threshold seçmek için.
#
# Final test:
# threshold seçildikten sonra sadece performans ölçmek için.
# ============================================================

validation, final_test = train_test_split(
    data,
    test_size=(1 - VALIDATION_RATIO),
    random_state=RANDOM_STATE,
    shuffle=True
)

validation = validation.reset_index(drop=True)
final_test = final_test.reset_index(drop=True)


print("\n" + "=" * 100)
print("VERİ AYRIMI")
print("=" * 100)

print(
    "Validation:",
    len(validation)
)

print(
    "Final test:",
    len(final_test)
)


# ============================================================
# HİBRİT KARAR FONKSİYONU
# ============================================================

def hybrid_predict(
    df,
    margin_threshold,
    score_threshold
):

    predictions = []

    for _, row in df.iterrows():

        seeded_subject = row[
            "seeded_subject"
        ]

        cosine_subject = row[
            "top1_subject"
        ]

        cosine_score = float(
            row["top1_score"]
        )

        cosine_margin = float(
            row["cosine_margin"]
        )


        # ----------------------------------------------------
        # İki yöntem zaten aynı şeyi söylüyorsa
        # ----------------------------------------------------

        if seeded_subject == cosine_subject:

            predictions.append(
                seeded_subject
            )

            continue


        # ----------------------------------------------------
        # Farklı söylüyorlarsa:
        #
        # Cosine yeterince eminse cosine seç.
        # Değilse Seeded seç.
        # ----------------------------------------------------

        if (
            cosine_margin
            >=
            margin_threshold
            and
            cosine_score
            >=
            score_threshold
        ):

            predictions.append(
                cosine_subject
            )

        else:

            predictions.append(
                seeded_subject
            )


    return predictions


# ============================================================
# VALIDATION ÜZERİNDE THRESHOLD ARA
# ============================================================

print("\n" + "=" * 100)
print("VALIDATION - THRESHOLD ARAMASI")
print("=" * 100)


# Margin için çeşitli eşikler
margin_thresholds = np.arange(
    0.0,
    0.061,
    0.005
)

# Cosine Top-1 score için çeşitli eşikler
score_thresholds = np.arange(
    0.50,
    0.751,
    0.025
)


search_results = []


for margin_threshold in margin_thresholds:

    for score_threshold in score_thresholds:

        predictions = hybrid_predict(
            validation,
            margin_threshold,
            score_threshold
        )

        correct = 0

        for external_id, prediction in zip(
            validation["external_id"],
            predictions
        ):

            if is_correct(
                external_id,
                prediction
            ):

                correct += 1


        accuracy = (
            correct /
            len(validation)
        )


        search_results.append(
            {
                "margin_threshold":
                    float(
                        margin_threshold
                    ),

                "score_threshold":
                    float(
                        score_threshold
                    ),

                "correct":
                    correct,

                "accuracy":
                    accuracy
            }
        )


search_df = pd.DataFrame(
    search_results
)


# ============================================================
# EN İYİ THRESHOLD
# ============================================================

best_row = (
    search_df
    .sort_values(
        [
            "accuracy",
            "margin_threshold",
            "score_threshold"
        ],
        ascending=[
            False,
            True,
            True
        ]
    )
    .iloc[0]
)


best_margin = float(
    best_row[
        "margin_threshold"
    ]
)

best_score = float(
    best_row[
        "score_threshold"
    ]
)

best_validation_accuracy = float(
    best_row[
        "accuracy"
    ]
)


print(
    "En iyi margin threshold:",
    round(
        best_margin,
        4
    )
)

print(
    "En iyi score threshold:",
    round(
        best_score,
        4
    )
)

print(
    "Validation hybrid başarısı:",
    f"{best_validation_accuracy * 100:.2f}%"
)


# ============================================================
# FINAL TEST
# ============================================================
#
# Burada threshold değiştirmek YOK.
#
# Validation'da seçtiğimiz değerleri aynen kullanıyoruz.
# ============================================================

print("\n" + "=" * 100)
print("FINAL TEST")
print("=" * 100)


hybrid_predictions = hybrid_predict(
    final_test,
    best_margin,
    best_score
)


seeded_correct = 0
cosine_correct = 0
hybrid_correct = 0


for i, row in final_test.iterrows():

    external_id = row[
        "external_id"
    ]


    # Seeded
    if is_correct(
        external_id,
        row["seeded_subject"]
    ):

        seeded_correct += 1


    # Cosine
    if is_correct(
        external_id,
        row["top1_subject"]
    ):

        cosine_correct += 1


    # Hybrid
    if is_correct(
        external_id,
        hybrid_predictions[i]
    ):

        hybrid_correct += 1


total = len(
    final_test
)


seeded_rate = (
    seeded_correct /
    total
)

cosine_rate = (
    cosine_correct /
    total
)

hybrid_rate = (
    hybrid_correct /
    total
)


print(
    "Final test makale:",
    total
)

print()

print(
    "Seeded doğru:",
    seeded_correct
)

print(
    "Seeded başarı:",
    f"{seeded_rate * 100:.2f}%"
)

print()

print(
    "Cosine doğru:",
    cosine_correct
)

print(
    "Cosine başarı:",
    f"{cosine_rate * 100:.2f}%"
)

print()

print(
    "Hybrid doğru:",
    hybrid_correct
)

print(
    "Hybrid başarı:",
    f"{hybrid_rate * 100:.2f}%"
)


# ============================================================
# HİBRİT NE KADAR COSINE / SEEDED SEÇTİ?
# ============================================================

cosine_selected = 0
seeded_selected = 0
same_prediction = 0


for _, row in final_test.iterrows():

    if (
        row["seeded_subject"]
        ==
        row["top1_subject"]
    ):

        same_prediction += 1

        continue


    if (
        float(row["cosine_margin"])
        >=
        best_margin
        and
        float(row["top1_score"])
        >=
        best_score
    ):

        cosine_selected += 1

    else:

        seeded_selected += 1


print("\n" + "=" * 100)
print("HİBRİT KARAR DAĞILIMI")
print("=" * 100)

print(
    "İki yöntem aynı:",
    same_prediction
)

print(
    "Farklıyken Cosine seçildi:",
    cosine_selected
)

print(
    "Farklıyken Seeded seçildi:",
    seeded_selected
)


# ============================================================
# FINAL TEST ORACLE
# ============================================================
#
# Bu yalnızca analiz.
# Karar verirken kullanılmıyor.
# ============================================================

oracle_correct = 0


for _, row in final_test.iterrows():

    seeded_ok = is_correct(
        row["external_id"],
        row["seeded_subject"]
    )

    cosine_ok = is_correct(
        row["external_id"],
        row["top1_subject"]
    )

    if seeded_ok or cosine_ok:
        oracle_correct += 1


oracle_rate = (
    oracle_correct /
    total
)


print("\n" + "=" * 100)
print("FINAL TEST TEORİK ÜST SINIR")
print("=" * 100)

print(
    "En az biri doğru:",
    oracle_correct
)

print(
    "Oracle başarı:",
    f"{oracle_rate * 100:.2f}%"
)


# ============================================================
# SONUÇ TABLOSU
# ============================================================

summary = pd.DataFrame(
    [
        {
            "Method":
                "Seeded K-Means",

            "Correct":
                seeded_correct,

            "Accuracy":
                seeded_rate
        },

        {
            "Method":
                "Fixed Centroid + Cosine",

            "Correct":
                cosine_correct,

            "Accuracy":
                cosine_rate
        },

        {
            "Method":
                "Hybrid",

            "Correct":
                hybrid_correct,

            "Accuracy":
                hybrid_rate
        }
    ]
)


print("\n" + "=" * 100)
print("FINAL KARŞILAŞTIRMA")
print("=" * 100)

print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# KAYDET
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


search_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "hybrid_threshold_search.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


final_output = final_test.copy()

final_output[
    "hybrid_subject"
] = hybrid_predictions


final_output.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "hybrid_final_predictions.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "hybrid_final_summary.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


threshold_summary = pd.DataFrame(
    [
        {
            "Best_Margin_Threshold":
                best_margin,

            "Best_Score_Threshold":
                best_score,

            "Validation_Accuracy":
                best_validation_accuracy,

            "Final_Hybrid_Accuracy":
                hybrid_rate
        }
    ]
)


threshold_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "hybrid_best_thresholds.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


print("\nDosyalar oluşturuldu:")

print(
    "results/kmeans/holdout/"
    "hybrid_threshold_search.csv"
)

print(
    "results/kmeans/holdout/"
    "hybrid_final_predictions.csv"
)

print(
    "results/kmeans/holdout/"
    "hybrid_final_summary.csv"
)

print(
    "results/kmeans/holdout/"
    "hybrid_best_thresholds.csv"
)