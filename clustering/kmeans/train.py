import numpy as np
import pandas as pd

EMBEDDING_FILE = "embeddings/mpnet_multilingual_embeddings.npy"
INDEX_FILE = "embeddings/article_embedding_index.csv"

print("=" * 80)
print("K-MEANS - VERİ KONTROLÜ")
print("=" * 80)

# Ortak MPNet embeddinglerini yükle
X = np.load(EMBEDDING_FILE)

# Embedding-makale eşleşmesini yükle
articles = pd.read_csv(INDEX_FILE)

print("Embedding shape:", X.shape)
print("Makale sayısı:", len(articles))

if len(X) != len(articles):
    raise ValueError("Embedding ve makale sayıları eşleşmiyor!")

print("Embedding dimension:", X.shape[1])
print("Kontrol: BAŞARILI")