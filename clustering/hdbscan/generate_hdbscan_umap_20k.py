from pathlib import Path
import numpy as np
import pandas as pd
import umap

from config.paths import EMBEDDING_FILE, INDEX_FILE, UMAP_FILE

# ============================================================
# HDBSCAN DASHBOARD İÇİN GERÇEK 2D UMAP KOORDİNATLARI
# ============================================================

OUTPUT_FILE = UMAP_FILE

print("=" * 100)
print("HDBSCAN 20K - GERÇEK 2D UMAP KOORDİNATLARI")
print("=" * 100)

embeddings = np.load(
    EMBEDDING_FILE,
    mmap_mode="r"
)

index_df = pd.read_csv(
    INDEX_FILE,
    dtype={"external_id": str}
)

index_df["external_id"] = (
    index_df["external_id"]
    .astype(str)
    .str.strip()
)

print("Embedding shape:", embeddings.shape)
print("Index satırı:", len(index_df))

if embeddings.shape[0] != len(index_df):
    raise RuntimeError(
        "Embedding satır sayısı ile index satır sayısı eşleşmiyor."
    )

print()
print("[*] 2D UMAP çalışıyor...")

reducer = umap.UMAP(
    n_neighbors=15,
    n_components=2,
    min_dist=0.10,
    metric="cosine",
    random_state=42,
    low_memory=True
)

coords = reducer.fit_transform(
    np.asarray(
        embeddings,
        dtype=np.float32
    )
)

print("[*] UMAP tamamlandı:", coords.shape)

out_df = pd.DataFrame({
    "external_id": index_df["external_id"],
    "umap_x": coords[:, 0],
    "umap_y": coords[:, 1],
})

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

out_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print()
print("=" * 100)
print("TAMAMLANDI")
print("=" * 100)
print("Dosya:", OUTPUT_FILE)
print("Satır:", len(out_df))