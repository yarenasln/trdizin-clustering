from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import umap
from config.paths import EMBEDDING_FILE, INDEX_FILE
# ============================================================
# K-MEANS V2 - GERÇEK UMAP KÜME GÖRSELLEŞTİRMESİ
# ============================================================
#
# Kullanılan gerçek veriler:
#   embeddings/mpnet_multilingual_embeddings.npy
#   embeddings/article_embedding_index.csv
#   results/kmeans/holdout/adaptive_v2_relabel_predictions.csv
#
# Çıktılar:
#   results/kmeans/kmeans_v2_umap_coordinates.csv
#   results/kmeans/kmeans_v2_umap_interactive.html
#
# Her nokta = final testteki bir makale
# Renk = makalenin ilk ana konu tahmini
# Hover = external_id, başlık, DOI, ana konu ve alt konular
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent
EMBEDDING_PATH = EMBEDDING_FILE
INDEX_PATH = INDEX_FILE

PREDICTION_PATH = (
    BASE_DIR
    / "results"
    / "kmeans"
    / "holdout"
    / "adaptive_v2_relabel_predictions.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "kmeans"
)

COORDINATE_OUTPUT = (
    OUTPUT_DIR
    / "kmeans_v2_umap_coordinates.csv"
)

HTML_OUTPUT = (
    OUTPUT_DIR
    / "kmeans_v2_umap_interactive.html"
)


def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def first_topic(value):
    text = clean_text(value)

    if not text:
        return "Konu Yok"

    return text.split("||")[0].strip()


print("=" * 110)
print("K-MEANS V2 - GERÇEK UMAP KÜME GÖRSELLEŞTİRMESİ")
print("=" * 110)

# ------------------------------------------------------------
# 1. VERİLERİ YÜKLE
# ------------------------------------------------------------

embeddings = np.load(
    EMBEDDING_PATH
)

index_df = pd.read_csv(
    INDEX_PATH,
    dtype={"external_id": str}
)

prediction_df = pd.read_csv(
    PREDICTION_PATH,
    dtype={"external_id": str}
)

index_df["external_id"] = (
    index_df["external_id"]
    .astype(str)
    .str.strip()
)

prediction_df["external_id"] = (
    prediction_df["external_id"]
    .astype(str)
    .str.strip()
)

print(
    "Tüm embedding shape:",
    embeddings.shape
)

print(
    "Final V2 tahmin makalesi:",
    len(prediction_df)
)


# ------------------------------------------------------------
# 2. FINAL TEST MAKALELERİNİ EMBEDDING INDEX İLE EŞLEŞTİR
# ------------------------------------------------------------

plot_df = prediction_df.merge(
    index_df[
        [
            "embedding_row",
            "external_id",
            "doi",
            "language",
            "title"
        ]
    ],
    on="external_id",
    how="inner"
)

plot_df = plot_df.drop_duplicates(
    subset=["external_id"]
).reset_index(drop=True)

embedding_rows = (
    plot_df["embedding_row"]
    .astype(int)
    .to_numpy()
)

X = embeddings[
    embedding_rows
]

print(
    "UMAP'a girecek makale:",
    len(plot_df)
)

print(
    "UMAP input shape:",
    X.shape
)

if len(plot_df) != len(prediction_df):
    print(
        "UYARI:",
        len(prediction_df) - len(plot_df),
        "tahmin makalesi embedding index ile eşleşmedi."
    )


# ------------------------------------------------------------
# 3. UMAP - 768 BOYUT -> 2 BOYUT
# ------------------------------------------------------------

print()
print("=" * 110)
print("UMAP ÇALIŞIYOR")
print("=" * 110)

reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.10,
    metric="cosine",
    random_state=42
)

coordinates = reducer.fit_transform(
    X
)

plot_df["UMAP_1"] = coordinates[:, 0]
plot_df["UMAP_2"] = coordinates[:, 1]

print(
    "UMAP output shape:",
    coordinates.shape
)


# ------------------------------------------------------------
# 4. GRAFİKTE KULLANILACAK ALANLAR
# ------------------------------------------------------------

plot_df["Ana_Konu"] = (
    plot_df[
        "main_level_2_predictions"
    ]
    .apply(first_topic)
)

plot_df["Alt_Konular"] = (
    plot_df[
        "leaf_predictions"
    ]
    .apply(clean_text)
)

plot_df["Gercek_Konular"] = (
    plot_df[
        "true_subjects"
    ]
    .apply(clean_text)
)

plot_df["title"] = (
    plot_df["title"]
    .fillna("")
)

plot_df["doi"] = (
    plot_df["doi"]
    .fillna("")
)

plot_df["language"] = (
    plot_df["language"]
    .fillna("")
)


# ------------------------------------------------------------
# 5. KOORDİNATLARI KAYDET
# ------------------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

plot_df[
    [
        "external_id",
        "embedding_row",
        "UMAP_1",
        "UMAP_2",
        "Ana_Konu",
        "main_level_2_predictions",
        "leaf_predictions",
        "true_subjects",
        "title",
        "doi",
        "language"
    ]
].to_csv(
    COORDINATE_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

print()
print(
    "Koordinat CSV:",
    COORDINATE_OUTPUT
)


# ------------------------------------------------------------
# 6. INTERAKTİF PLOTLY GRAFİĞİ
# ------------------------------------------------------------

fig = px.scatter(
    plot_df,
    x="UMAP_1",
    y="UMAP_2",
    color="Ana_Konu",
    hover_name="title",
    hover_data={
        "external_id": True,
        "doi": True,
        "language": True,
        "Ana_Konu": True,
        "Alt_Konular": True,
        "UMAP_1": False,
        "UMAP_2": False
    },
    title="K-Means 2D Konu Kümeleme Uzayı (UMAP)"
)

fig.update_traces(
    marker={
        "size": 7,
        "opacity": 0.72
    }
)

fig.update_layout(
    template="plotly_white",
    height=720,
    legend_title_text="K-Means Ana Konusu",
    margin={
        "l": 35,
        "r": 35,
        "t": 75,
        "b": 35
    }
)

fig.update_xaxes(
    title="UMAP 1"
)

fig.update_yaxes(
    title="UMAP 2"
)

fig.write_html(
    HTML_OUTPUT,
    include_plotlyjs="cdn",
    full_html=True
)

print(
    "Interaktif HTML:",
    HTML_OUTPUT
)

print()
print("=" * 110)
print("TAMAMLANDI")
print("=" * 110)
print(
    "Grafikteki nokta sayısı:",
    len(plot_df)
)
print(
    "Benzersiz ana konu:",
    plot_df["Ana_Konu"].nunique()
)