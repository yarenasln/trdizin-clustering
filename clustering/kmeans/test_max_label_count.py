import runpy
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

BASE_SCRIPT = "clustering/kmeans/adaptive_label_count_v2_20k.py"
OUTPUT_FILE = "results/kmeans/holdout/max_label_count_comparison.csv"

FOURTH_RATIOS = np.round(np.arange(1.00, 1.101, 0.01), 2)
FIFTH_RATIOS = np.round(np.arange(1.00, 1.101, 0.01), 2)

print("=" * 110)
print("MAX ETIKET SAYISI TESTI - 1 / 2 / 3 / 4 / 5")
print("=" * 110)

g = runpy.run_path(BASE_SCRIPT, run_name="__max_label_test__")

validation_distances = g["validation_distances"]
final_distances = g["final_distances"]
y_validation_true = g["y_validation_true"]
y_final_true = g["y_final_true"]
SECOND_RATIO = float(g["best_second_ratio"])
THIRD_RATIO = float(g["best_third_ratio"])

def adaptive_max_predictions(distances, max_labels, second_ratio=1.19, third_ratio=1.03, fourth_ratio=1.0, fifth_ratio=1.0):
    predictions = np.zeros(distances.shape, dtype=np.int8)
    nearest5 = np.argsort(distances, axis=1)[:, :5]

    for i in range(distances.shape[0]):
        idx = nearest5[i]

        d1 = float(distances[i, idx[0]])
        predictions[i, idx[0]] = 1
        if max_labels == 1:
            continue

        d2 = float(distances[i, idx[1]])
        if d2 / max(d1, 1e-12) > second_ratio:
            continue
        predictions[i, idx[1]] = 1
        if max_labels == 2:
            continue

        d3 = float(distances[i, idx[2]])
        if d3 / max(d2, 1e-12) > third_ratio:
            continue
        predictions[i, idx[2]] = 1
        if max_labels == 3:
            continue

        d4 = float(distances[i, idx[3]])
        if d4 / max(d3, 1e-12) > fourth_ratio:
            continue
        predictions[i, idx[3]] = 1
        if max_labels == 4:
            continue

        d5 = float(distances[i, idx[4]])
        if d5 / max(d4, 1e-12) > fifth_ratio:
            continue
        predictions[i, idx[4]] = 1

    return predictions

def calc(y_true, y_pred):
    return {
        "Micro_Precision": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "Micro_Recall": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "Micro_F1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "Macro_F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Sample_F1": f1_score(y_true, y_pred, average="samples", zero_division=0),
        "Exact_Match": float(np.all(y_true == y_pred, axis=1).mean()),
        "At_Least_One": float((((y_true & y_pred).sum(axis=1)) > 0).mean()),
        "Average_Predicted_Labels": float(y_pred.sum(axis=1).mean()),
    }

results = []

for max_labels in [1, 2, 3]:
    pred = adaptive_max_predictions(
        final_distances,
        max_labels=max_labels,
        second_ratio=SECOND_RATIO,
        third_ratio=THIRD_RATIO
    )
    results.append({
        "Max_Labels": max_labels,
        "Second_Ratio": SECOND_RATIO if max_labels >= 2 else np.nan,
        "Third_Ratio": THIRD_RATIO if max_labels >= 3 else np.nan,
        "Fourth_Ratio": np.nan,
        "Fifth_Ratio": np.nan,
        **calc(y_final_true, pred)
    })

best_fourth = None
best_fourth_f1 = -1.0
for fourth_ratio in FOURTH_RATIOS:
    pred = adaptive_max_predictions(
        validation_distances,
        max_labels=4,
        second_ratio=SECOND_RATIO,
        third_ratio=THIRD_RATIO,
        fourth_ratio=float(fourth_ratio)
    )
    score = f1_score(y_validation_true, pred, average="micro", zero_division=0)
    if score > best_fourth_f1:
        best_fourth_f1 = float(score)
        best_fourth = float(fourth_ratio)

pred4 = adaptive_max_predictions(
    final_distances,
    max_labels=4,
    second_ratio=SECOND_RATIO,
    third_ratio=THIRD_RATIO,
    fourth_ratio=best_fourth
)
results.append({
    "Max_Labels": 4,
    "Second_Ratio": SECOND_RATIO,
    "Third_Ratio": THIRD_RATIO,
    "Fourth_Ratio": best_fourth,
    "Fifth_Ratio": np.nan,
    **calc(y_final_true, pred4)
})

best_4_for_5 = None
best_fifth = None
best_five_f1 = -1.0
for fourth_ratio in FOURTH_RATIOS:
    for fifth_ratio in FIFTH_RATIOS:
        pred = adaptive_max_predictions(
            validation_distances,
            max_labels=5,
            second_ratio=SECOND_RATIO,
            third_ratio=THIRD_RATIO,
            fourth_ratio=float(fourth_ratio),
            fifth_ratio=float(fifth_ratio)
        )
        score = f1_score(y_validation_true, pred, average="micro", zero_division=0)
        if score > best_five_f1:
            best_five_f1 = float(score)
            best_4_for_5 = float(fourth_ratio)
            best_fifth = float(fifth_ratio)

pred5 = adaptive_max_predictions(
    final_distances,
    max_labels=5,
    second_ratio=SECOND_RATIO,
    third_ratio=THIRD_RATIO,
    fourth_ratio=best_4_for_5,
    fifth_ratio=best_fifth
)
results.append({
    "Max_Labels": 5,
    "Second_Ratio": SECOND_RATIO,
    "Third_Ratio": THIRD_RATIO,
    "Fourth_Ratio": best_4_for_5,
    "Fifth_Ratio": best_fifth,
    **calc(y_final_true, pred5)
})

df = pd.DataFrame(results).sort_values("Max_Labels").reset_index(drop=True)
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print()
print("=" * 110)
print("8.361 FINAL TEST - MAX ETIKET KARSILASTIRMASI")
print("=" * 110)

pretty = df.copy()
for col in ["Micro_Precision","Micro_Recall","Micro_F1","Macro_F1","Sample_F1","Exact_Match","At_Least_One"]:
    pretty[col] = (pretty[col] * 100).round(2)
pretty["Average_Predicted_Labels"] = pretty["Average_Predicted_Labels"].round(2)

print(pretty[[
    "Max_Labels","Micro_Precision","Micro_Recall","Micro_F1",
    "Macro_F1","Sample_F1","Exact_Match","At_Least_One",
    "Average_Predicted_Labels","Fourth_Ratio","Fifth_Ratio"
]].to_string(index=False))

winner = df.loc[df["Micro_F1"].idxmax()]

print()
print("En iyi max etiket:", int(winner["Max_Labels"]))
print("Micro F1:", f"{winner['Micro_F1'] * 100:.2f}%")
print("Precision:", f"{winner['Micro_Precision'] * 100:.2f}%")
print("Recall:", f"{winner['Micro_Recall'] * 100:.2f}%")
print("En az 1 dogru:", f"{winner['At_Least_One'] * 100:.2f}%")
print("Ortalama tahmin:", round(float(winner["Average_Predicted_Labels"]), 2))
print()
print("Dosya:", OUTPUT_FILE)