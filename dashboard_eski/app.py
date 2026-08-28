from flask import Flask, render_template, request, jsonify
from pathlib import Path
import pandas as pd

app = Flask(__name__)

BASE = Path(__file__).resolve().parent.parent
ARTICLES = BASE / "data" / "balanced_articles.csv"
PREDICTIONS = BASE / "results" / "kmeans" / "holdout" / "relabel_kmeans_predictions.csv"
COMPARISON = BASE / "results" / "kmeans" / "holdout" / "fixed_vs_relabel_comparison.csv"

def txt(v):
    if pd.isna(v):
        return ""
    return str(v).strip()

def topics(v):
    s = txt(v)
    return [x.strip() for x in s.split("||") if x.strip()] if s else []

articles = pd.read_csv(ARTICLES, dtype={"external_id": str}, low_memory=False)
preds = pd.read_csv(PREDICTIONS, dtype={"external_id": str}, low_memory=False)

articles["external_id"] = articles["external_id"].astype(str).str.strip()
preds["external_id"] = preds["external_id"].astype(str).str.strip()

df = preds.merge(
    articles[["external_id","doi","publication_year","language","title","abstract","keywords"]],
    on="external_id", how="left"
)

records = []
for _, r in df.iterrows():
    predicted = topics(r.get("full_topic_predictions"))
    true = topics(r.get("true_subjects"))
    pred_set, true_set = set(predicted), set(true)
    matched = sorted(pred_set & true_set)
    wrong = sorted(pred_set - true_set)
    missed = sorted(true_set - pred_set)

    tp, fp, fn = len(matched), len(wrong), len(missed)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    match_rate = tp / len(true_set) if true_set else 0

    records.append({
        "external_id": txt(r.get("external_id")),
        "title": txt(r.get("title")),
        "doi": txt(r.get("doi")),
        "year": txt(r.get("publication_year")),
        "language": txt(r.get("language")),
        "abstract": txt(r.get("abstract")),
        "keywords": txt(r.get("keywords")),
        "main_topics": topics(r.get("main_level_2_predictions")),
        "leaf_topics": topics(r.get("leaf_predictions")),
        "predicted_topics": predicted,
        "true_topics": true,
        "matched": matched,
        "wrong": wrong,
        "missed": missed,
        "matched_count": tp,
        "true_count": len(true_set),
        "predicted_count": len(pred_set),
        "match_rate": match_rate,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    })

record_map = {r["external_id"]: r for r in records}

def summary():
    # Final seçilen modelin bilinen final skorları.
    result = {
        "micro_precision": 0.3204,
        "micro_recall": 0.3969,
        "micro_f1": 0.3546,
        "at_least_one": 0.7333,
    }
    if COMPARISON.exists():
        try:
            c = pd.read_csv(COMPARISON)
            rel = c[c["Model"].astype(str).str.contains("Relabel", case=False, na=False)]
            if not rel.empty:
                x = rel.iloc[0]
                result["micro_precision"] = float(x.get("Micro_Precision", result["micro_precision"]))
                result["micro_recall"] = float(x.get("Micro_Recall", result["micro_recall"]))
                result["micro_f1"] = float(x.get("Micro_F1", result["micro_f1"]))
                result["at_least_one"] = float(x.get("At_Least_One_Match_Rate", result["at_least_one"]))
        except Exception:
            pass
    return result

@app.route("/")
def home():
    return render_template("index.html", summary=summary())

@app.route("/kmeans")
def kmeans():
    return render_template("kmeans.html", summary=summary())

@app.route("/hdbscan")
def hdbscan():
    return render_template("hdbscan.html")

@app.route("/api/kmeans/articles")
def api_articles():
    q = request.args.get("q","").strip().lower()
    match = request.args.get("match","all")
    pred_count = request.args.get("pred_count","all")
    sort = request.args.get("sort","match_desc")

    out = records
    if q:
        out = [r for r in out if (
            q in r["external_id"].lower()
            or q in r["title"].lower()
            or q in r["doi"].lower()
            or any(q in x.lower() for x in r["true_topics"])
            or any(q in x.lower() for x in r["predicted_topics"])
        )]

    if match == "0":
        out = [r for r in out if r["matched_count"] == 0]
    elif match == "1":
        out = [r for r in out if r["matched_count"] == 1]
    elif match == "2":
        out = [r for r in out if r["matched_count"] == 2]
    elif match == "3plus":
        out = [r for r in out if r["matched_count"] >= 3]
    elif match == "100":
        out = [r for r in out if r["true_count"] > 0 and r["match_rate"] == 1]
    elif match == "50plus":
        out = [r for r in out if .5 <= r["match_rate"] < 1]

    if pred_count in {"1","2","3"}:
        out = [r for r in out if r["predicted_count"] == int(pred_count)]

    if sort == "match_desc":
        out = sorted(out, key=lambda r:(r["match_rate"], r["f1"]), reverse=True)
    elif sort == "match_asc":
        out = sorted(out, key=lambda r:(r["match_rate"], r["f1"]))
    elif sort == "f1_desc":
        out = sorted(out, key=lambda r:r["f1"], reverse=True)
    elif sort == "f1_asc":
        out = sorted(out, key=lambda r:r["f1"])

    return jsonify({
        "total": len(out),
        "articles": [{
            "external_id": r["external_id"],
            "title": r["title"],
            "year": r["year"],
            "matched_count": r["matched_count"],
            "true_count": r["true_count"],
            "match_rate": r["match_rate"],
            "f1": r["f1"],
            "predicted_count": r["predicted_count"]
        } for r in out[:500]]
    })

@app.route("/api/kmeans/article/<external_id>")
def api_article(external_id):
    r = record_map.get(str(external_id).strip())
    if not r:
        return jsonify({"error":"Makale bulunamadı"}), 404
    return jsonify(r)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
