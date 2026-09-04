import os

EMBEDDINGS_DIR = os.getenv(
    "EMBEDDINGS_DIR",
    "/data/app_embeddings"
)

EMBEDDING_FILE = os.path.join(
    EMBEDDINGS_DIR,
    "mpnet_multilingual_embeddings.npy"
)

INDEX_FILE = os.path.join(
    EMBEDDINGS_DIR,
    "article_embedding_index.csv"
)

UMAP_FILE = os.path.join(
    EMBEDDINGS_DIR,
    "umap_2d_coordinates.csv"
)