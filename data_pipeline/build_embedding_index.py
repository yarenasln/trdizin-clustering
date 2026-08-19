import os
import numpy as np
import pandas as pd


# ============================================================
# AYARLAR
# ============================================================

DATA_FILE = "data/balanced_articles.csv"

EMBEDDING_FILE = (
    "embeddings/mpnet_multilingual_embeddings.npy"
)

OUTPUT_FILE = (
    "embeddings/article_embedding_index.csv"
)


# ============================================================
# VERİYİ OKU
# ============================================================

df = pd.read_csv(
    DATA_FILE,
    encoding="utf-8-sig"
)


df["embedding_text"] = (
    df["embedding_text"]
    .fillna("")
    .astype(str)
    .str.strip()
)


# Embedding oluştururken kullandığımız koşulla
# aynı filtreyi uyguluyoruz.
df = df[
    df["embedding_text"] != ""
].copy()


df = df.reset_index(
    drop=True
)


# ============================================================
# EMBEDDING DOSYASINI OKU
# ============================================================

embeddings = np.load(
    EMBEDDING_FILE,
    mmap_mode="r"
)


print("=" * 100)
print("ORTAK EMBEDDING INDEX KONTROLÜ")
print("=" * 100)

print(
    "Makale sayısı:",
    len(df)
)

print(
    "Embedding satırı:",
    embeddings.shape[0]
)

print(
    "Embedding dimension:",
    embeddings.shape[1]
)


# ============================================================
# KRİTİK KONTROL
# ============================================================

if len(df) != embeddings.shape[0]:

    raise ValueError(
        "Makale sayısı ile embedding satır sayısı eşleşmiyor! "
        f"Makale={len(df)}, "
        f"Embedding={embeddings.shape[0]}"
    )


# ============================================================
# INDEX DOSYASI
# ============================================================

index_df = pd.DataFrame(
    {
        "embedding_row":
            range(len(df)),

        "external_id":
            df["external_id"],

        "doi":
            df["doi"],

        "language":
            df["language"],

        "title":
            df["title"]
    }
)


# ============================================================
# DUPLICATE KONTROL
# ============================================================

duplicate_external_ids = (
    index_df[
        "external_id"
    ]
    .duplicated()
    .sum()
)


print(
    "Tekrarlanan external_id:",
    duplicate_external_ids
)


# ============================================================
# ÖRNEK
# ============================================================

print("\n" + "=" * 100)
print("İLK 10 EMBEDDING ↔ MAKALE EŞLEŞMESİ")
print("=" * 100)

print(
    index_df
    .head(10)
    .to_string(
        index=False
    )
)


# ============================================================
# KAYDET
# ============================================================

os.makedirs(
    "embeddings",
    exist_ok=True
)


index_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


print("\n" + "=" * 100)
print("TAMAMLANDI")
print("=" * 100)

print(
    "Dosya oluşturuldu:",
    OUTPUT_FILE
)