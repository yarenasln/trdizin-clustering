import os
import time
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# ============================================================
# AYARLAR
# ============================================================

DATA_FILE = "data/balanced_articles.csv"

OUTPUT_FILE = (
    "embeddings/mpnet_multilingual_embeddings.npy"
)

MODEL_NAME = (
    "paraphrase-multilingual-mpnet-base-v2"
)

BATCH_SIZE = 32

# ============================================================
# CİHAZ
# ============================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 110)
print("MPNET MULTILINGUAL EMBEDDING ÜRETİMİ")
print("=" * 110)

print("Kullanılan cihaz:", DEVICE)

if DEVICE == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )
    print(
        "PyTorch CUDA:",
        torch.version.cuda
    )
    print(
        "GPU VRAM:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / 1024**3,
            2
        ),
        "GB"
    )
else:
    print(
        "UYARI: CUDA bulunamadı. "
        "Embedding CPU ile üretilecek."
    )

# ============================================================
# VERİYİ OKU
# ============================================================

df = pd.read_csv(
    DATA_FILE,
    encoding="utf-8-sig"
)

if "embedding_text" not in df.columns:
    raise ValueError(
        "balanced_articles.csv içinde "
        "'embedding_text' sütunu bulunamadı."
    )

df["embedding_text"] = (
    df["embedding_text"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df = df[
    df["embedding_text"] != ""
].copy()

df = df.reset_index(drop=True)

texts = df["embedding_text"].tolist()

print()
print("Makale sayısı:", len(df))

# ============================================================
# MODEL
# ============================================================

print()
print("Model yükleniyor:", MODEL_NAME)

model_load_start = time.time()

model = SentenceTransformer(
    MODEL_NAME,
    device=DEVICE
)

print(
    "Model yükleme süresi:",
    round(
        time.time() - model_load_start,
        2
    ),
    "sn"
)

if DEVICE == "cuda":
    torch.cuda.empty_cache()

# ============================================================
# EMBEDDING
# ============================================================

print()
print("Embedding üretiliyor...")

embedding_start = time.time()

embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True
)

embedding_time = (
    time.time()
    -
    embedding_start
)

# ============================================================
# KONTROLLER
# ============================================================

print()
print("=" * 110)
print("EMBEDDING KONTROLÜ")
print("=" * 110)

print(
    "Embedding shape:",
    embeddings.shape
)

if embeddings.shape[0] != len(df):
    raise RuntimeError(
        "Makale sayısı ile embedding satır sayısı eşleşmiyor."
    )

if embeddings.shape[1] != 768:
    raise RuntimeError(
        f"Beklenen embedding boyutu 768, "
        f"gelen: {embeddings.shape[1]}"
    )

print(
    "Toplam embedding süresi:",
    round(
        embedding_time / 60,
        2
    ),
    "dk"
)

# ============================================================
# KAYDET
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

np.save(
    OUTPUT_FILE,
    embeddings
)

print()
print("=" * 110)
print("TAMAMLANDI")
print("=" * 110)

print(
    "Dosya:",
    OUTPUT_FILE
)

print(
    "Final shape:",
    embeddings.shape
)

print(
    "dtype:",
    embeddings.dtype
)