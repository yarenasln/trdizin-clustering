import os
import time
import gc

import numpy as np
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)


# ============================================================
# AYARLAR
# ============================================================

DATA_FILE = "data/balanced_articles.csv"

OUTPUT_FILE = (
    "results/qwen3_embedding_evaluation.csv"
)

K = 195
RANDOM_STATE = 42

BATCH_SIZE = 8


# ============================================================
# SADECE QWEN3
# ============================================================

MODELS = {
    "Qwen3":
        "Qwen/Qwen3-Embedding-0.6B"
}


# ============================================================
# DEVICE
# ============================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 110)
print("DONANIM KONTROLÜ")
print("=" * 110)

print(
    "Kullanılan cihaz:",
    DEVICE
)


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
            torch.cuda.get_device_properties(0)
            .total_memory
            /
            1024**3,
            2
        ),
        "GB"
    )

else:

    print(
        "UYARI: CUDA bulunamadı."
    )

    print(
        "Embedding CPU ile üretilecek."
    )


# ============================================================
# VERİ SETİNİ OKU
# ============================================================

print("\n" + "=" * 110)
print("VERİ SETİ")
print("=" * 110)


df = pd.read_csv(
    DATA_FILE,
    encoding="utf-8-sig"
)


if "embedding_text" not in df.columns:

    raise ValueError(
        "balanced_articles.csv içinde "
        "'embedding_text' kolonu bulunamadı."
    )


df["embedding_text"] = (
    df["embedding_text"]
    .fillna("")
    .astype(str)
    .str.strip()
)


# Boş metinleri çıkar
df = df[
    df["embedding_text"] != ""
].copy()


df = df.reset_index(
    drop=True
)


texts = (
    df["embedding_text"]
    .tolist()
)


print(
    "Makale sayısı:",
    len(texts)
)

print(
    "Test K:",
    K
)

print(
    "Batch size:",
    BATCH_SIZE
)


# ============================================================
# SONUÇLAR
# ============================================================

results = []


# ============================================================
# MODEL TESTİ
# ============================================================

for model_name, model_path in MODELS.items():

    print("\n" + "=" * 110)
    print("MODEL:", model_name)
    print("=" * 110)

    print(
        "Model yolu:",
        model_path
    )


    try:

        # ----------------------------------------------------
        # MODEL YÜKLE
        # ----------------------------------------------------

        model_load_start = time.time()


        model = SentenceTransformer(
            model_path,
            device=DEVICE
        )


        model_load_time = (
            time.time()
            -
            model_load_start
        )


        print(
            "Model yükleme süresi:",
            round(
                model_load_time,
                2
            ),
            "sn"
        )


        # ----------------------------------------------------
        # GPU CACHE TEMİZLE
        # ----------------------------------------------------

        if DEVICE == "cuda":

            torch.cuda.empty_cache()


        # ----------------------------------------------------
        # EMBEDDING ÜRET
        # ----------------------------------------------------

        print(
            "\nEmbedding üretiliyor..."
        )


        embedding_start = (
            time.time()
        )


        embeddings = model.encode(

            texts,

            batch_size=
                BATCH_SIZE,

            show_progress_bar=
                True,

            normalize_embeddings=
                True,

            convert_to_numpy=
                True
        )


        embedding_time = (
            time.time()
            -
            embedding_start
        )


        embeddings = np.asarray(
            embeddings,
            dtype=np.float32
        )


        print(
            "\nEmbedding shape:",
            embeddings.shape
        )

        print(
            "Embedding dimension:",
            embeddings.shape[1]
        )

        print(
            "Embedding süresi:",
            round(
                embedding_time,
                2
            ),
            "sn"
        )


        # ----------------------------------------------------
        # EMBEDDING KAYDET
        # ----------------------------------------------------

        os.makedirs(
            "embeddings",
            exist_ok=True
        )


        embedding_output = (
            "embeddings/"
            "qwen3_embeddings.npy"
        )


        np.save(
            embedding_output,
            embeddings
        )


        print(
            "Embedding dosyası:",
            embedding_output
        )


        # ====================================================
        # K-MEANS
        # ====================================================

        print(
            "\nK-Means çalıştırılıyor..."
        )


        clustering_start = (
            time.time()
        )


        kmeans = MiniBatchKMeans(

            n_clusters=K,

            random_state=
                RANDOM_STATE,

            batch_size=1024,

            n_init=10
        )


        labels = (
            kmeans.fit_predict(
                embeddings
            )
        )


        clustering_time = (
            time.time()
            -
            clustering_start
        )


        print(
            "K-Means süresi:",
            round(
                clustering_time,
                2
            ),
            "sn"
        )


        # ====================================================
        # CLUSTER SAYILARI
        # ====================================================

        cluster_sizes = (
            pd.Series(labels)
            .value_counts()
        )


        singleton_clusters = int(
            (
                cluster_sizes
                ==
                1
            ).sum()
        )


        clusters_le_5 = int(
            (
                cluster_sizes
                <=
                5
            ).sum()
        )


        clusters_le_10 = int(
            (
                cluster_sizes
                <=
                10
            ).sum()
        )


        smallest_cluster = int(
            cluster_sizes.min()
        )


        largest_cluster = int(
            cluster_sizes.max()
        )


        average_cluster_size = float(
            cluster_sizes.mean()
        )


        median_cluster_size = float(
            cluster_sizes.median()
        )


        # ====================================================
        # SILHOUETTE
        # ====================================================

        print(
            "\nSilhouette hesaplanıyor..."
        )


        silhouette = (
            silhouette_score(

                embeddings,

                labels,

                metric="cosine"
            )
        )


        # ====================================================
        # DAVIES-BOULDIN
        # ====================================================

        print(
            "Davies-Bouldin hesaplanıyor..."
        )


        davies_bouldin = (
            davies_bouldin_score(
                embeddings,
                labels
            )
        )


        # ====================================================
        # CALINSKI-HARABASZ
        # ====================================================

        print(
            "Calinski-Harabasz hesaplanıyor..."
        )


        calinski_harabasz = (
            calinski_harabasz_score(
                embeddings,
                labels
            )
        )


        # ====================================================
        # SONUÇ
        # ====================================================

        result = {

            "Model":
                model_name,

            "Embedding_Dim":
                embeddings.shape[1],

            "Articles":
                len(embeddings),

            "K":
                K,

            "Silhouette":
                silhouette,

            "Davies_Bouldin":
                davies_bouldin,

            "Calinski_Harabasz":
                calinski_harabasz,

            "Singleton_Clusters":
                singleton_clusters,

            "Clusters_LE_5":
                clusters_le_5,

            "Clusters_LE_10":
                clusters_le_10,

            "Smallest_Cluster":
                smallest_cluster,

            "Largest_Cluster":
                largest_cluster,

            "Average_Cluster_Size":
                average_cluster_size,

            "Median_Cluster_Size":
                median_cluster_size,

            "Embedding_Time_Seconds":
                embedding_time,

            "Clustering_Time_Seconds":
                clustering_time
        }


        results.append(
            result
        )


        # ====================================================
        # EKRANA BAS
        # ====================================================

        print("\n" + "=" * 110)
        print("QWEN3 SONUCU")
        print("=" * 110)


        print(
            "Silhouette:",
            round(
                silhouette,
                6
            )
        )


        print(
            "Davies-Bouldin:",
            round(
                davies_bouldin,
                6
            )
        )


        print(
            "Calinski-Harabasz:",
            round(
                calinski_harabasz,
                6
            )
        )


        print(
            "Singleton cluster:",
            singleton_clusters
        )


        print(
            "Cluster <= 5:",
            clusters_le_5
        )


        print(
            "Cluster <= 10:",
            clusters_le_10
        )


        print(
            "En küçük cluster:",
            smallest_cluster
        )


        print(
            "En büyük cluster:",
            largest_cluster
        )


        print(
            "Ortalama cluster:",
            round(
                average_cluster_size,
                2
            )
        )


        print(
            "Medyan cluster:",
            round(
                median_cluster_size,
                2
            )
        )


        # ====================================================
        # GPU TEMİZLİĞİ
        # ====================================================

        del model

        gc.collect()


        if DEVICE == "cuda":

            torch.cuda.empty_cache()


    except Exception as error:

        print(
            "\nMODEL ÇALIŞTIRILAMADI:"
        )

        print(
            type(error).__name__
        )

        print(
            error
        )


        if DEVICE == "cuda":

            torch.cuda.empty_cache()


# ============================================================
# DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# KAYDET
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)


results_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# GENEL TABLO
# ============================================================

print("\n" + "=" * 150)
print("QWEN3 GENEL DEĞERLENDİRME")
print("=" * 150)


if not results_df.empty:

    print(
        results_df[
            [
                "Model",
                "Embedding_Dim",
                "Articles",
                "K",
                "Silhouette",
                "Davies_Bouldin",
                "Calinski_Harabasz",
                "Singleton_Clusters",
                "Clusters_LE_5",
                "Clusters_LE_10",
                "Smallest_Cluster",
                "Largest_Cluster",
                "Average_Cluster_Size",
                "Embedding_Time_Seconds"
            ]
        ]
        .to_string(
            index=False
        )
    )


# ============================================================
# MPNET REFERANSI
# ============================================================

print("\n" + "=" * 110)
print("MPNET REFERANS SONUCU")
print("=" * 110)

print(
    "Silhouette:        0.004896"
)

print(
    "Davies-Bouldin:    2.891395"
)

print(
    "Calinski-Harabasz: 19.991684"
)

print(
    "Singleton Cluster: 18"
)

print(
    "Clusters <= 10:    50"
)


# ============================================================
# QWEN3 VS MPNET
# ============================================================

if not results_df.empty:

    qwen = (
        results_df.iloc[0]
    )


    print("\n" + "=" * 110)
    print("QWEN3 vs MPNET")
    print("=" * 110)


    if (
        qwen["Silhouette"]
        >
        0.004896
    ):

        silhouette_winner = (
            "Qwen3"
        )

    else:

        silhouette_winner = (
            "MPNet"
        )


    if (
        qwen["Davies_Bouldin"]
        <
        2.891395
    ):

        db_winner = (
            "Qwen3"
        )

    else:

        db_winner = (
            "MPNet"
        )


    if (
        qwen["Calinski_Harabasz"]
        >
        19.991684
    ):

        ch_winner = (
            "Qwen3"
        )

    else:

        ch_winner = (
            "MPNet"
        )


    print(
        "Silhouette kazanan:",
        silhouette_winner
    )

    print(
        "Davies-Bouldin kazanan:",
        db_winner
    )

    print(
        "Calinski-Harabasz kazanan:",
        ch_winner
    )


print("\n" + "=" * 110)
print("TAMAMLANDI")
print("=" * 110)

print(
    "Dosya oluşturuldu:",
    OUTPUT_FILE
)