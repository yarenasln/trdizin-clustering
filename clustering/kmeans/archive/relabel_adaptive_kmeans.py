import os
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans


# ============================================================
# AYARLAR
# ============================================================

EMBEDDING_FILE = "embeddings/mpnet_multilingual_embeddings.npy"
SUBJECT_FILE = "data/article_subjects.csv"

REFERENCE_FILE = "results/kmeans/holdout/reference.csv"

SEEDS_PER_SUBJECT = 10
RANDOM_STATE = 42


# ============================================================
# VERİLERİ OKU
# ============================================================

print("=" * 110)
print("SEEDED K-MEANS - FINAL CLUSTER RELABEL ANALİZİ")
print("=" * 110)

embeddings = np.load(
    EMBEDDING_FILE
).astype(np.float32)

reference = pd.read_csv(
    REFERENCE_FILE,
    encoding="utf-8-sig"
)

subjects = pd.read_csv(
    SUBJECT_FILE,
    encoding="utf-8-sig"
)

reference["external_id"] = (
    reference["external_id"]
    .astype(str)
)

subjects["external_id"] = (
    subjects["external_id"]
    .astype(str)
)

print("Referans makale:", len(reference))
print("Embedding shape:", embeddings.shape)


# ============================================================
# HAZIRLIK
# ============================================================

reference_ids = set(
    reference["external_id"]
)

reference_row_map = (
    reference
    .set_index("external_id")["embedding_row"]
    .to_dict()
)

label_counts = (
    subjects
    .groupby("external_id")["subject_fullname"]
    .nunique()
    .to_dict()
)

leaf_subjects = sorted(
    subjects["subject_fullname"]
    .dropna()
    .astype(str)
    .unique()
)

true_subjects = (
    subjects
    .groupby("external_id")["subject_fullname"]
    .apply(
        lambda values: sorted(
            set(
                values
                .dropna()
                .astype(str)
            )
        )
    )
    .to_dict()
)

print("Leaf konu sayısı:", len(leaf_subjects))


# ============================================================
# TEMSİL EDİCİ SEED SEÇ
# ============================================================

def select_representative(candidate_ids, count):

    valid_ids = [
        article_id
        for article_id in candidate_ids
        if article_id in reference_row_map
    ]

    if not valid_ids or count <= 0:
        return []

    rows = [
        int(reference_row_map[article_id])
        for article_id in valid_ids
    ]

    vectors = embeddings[rows]

    center = vectors.mean(axis=0)

    distances = np.linalg.norm(
        vectors - center,
        axis=1
    )

    order = np.argsort(distances)

    selected_count = min(
        count,
        len(valid_ids)
    )

    return [
        valid_ids[i]
        for i in order[:selected_count]
    ]


# ============================================================
# BAŞLANGIÇ CENTROIDLERİNİ OLUŞTUR
# ============================================================

print()
print("=" * 110)
print("BAŞLANGIÇ CENTROIDLERİ OLUŞTURULUYOR")
print("=" * 110)

initial_centroids = []
centroid_subjects = []

unique_seed_ids = set()


for subject_name in leaf_subjects:

    candidates = (
        subjects[
            (
                subjects["subject_fullname"]
                ==
                subject_name
            )
            &
            (
                subjects["external_id"]
                .isin(reference_ids)
            )
        ]["external_id"]
        .drop_duplicates()
        .tolist()
    )

    # Önce tek etiketli makaleleri tercih ediyoruz.
    single_label_candidates = [
        article_id
        for article_id in candidates
        if label_counts.get(article_id, 0) == 1
    ]

    selected_ids = []

    if len(single_label_candidates) >= SEEDS_PER_SUBJECT:

        selected_ids = select_representative(
            single_label_candidates,
            SEEDS_PER_SUBJECT
        )

    else:

        selected_ids.extend(
            select_representative(
                single_label_candidates,
                len(single_label_candidates)
            )
        )

        needed = (
            SEEDS_PER_SUBJECT
            -
            len(selected_ids)
        )

        remaining = [
            article_id
            for article_id in candidates
            if article_id not in selected_ids
        ]

        selected_ids.extend(
            select_representative(
                remaining,
                needed
            )
        )

    if not selected_ids:
        raise ValueError(
            f"Seed bulunamadı: {subject_name}"
        )

    unique_seed_ids.update(selected_ids)

    rows = [
        int(reference_row_map[article_id])
        for article_id in selected_ids
    ]

    vectors = embeddings[rows]

    centroid = vectors.mean(axis=0)

    initial_centroids.append(centroid)
    centroid_subjects.append(subject_name)


initial_centroids = (
    np.vstack(initial_centroids)
    .astype(np.float32)
)

print(
    "Initial centroid matrix:",
    initial_centroids.shape
)

print(
    "Benzersiz seed makale:",
    len(unique_seed_ids)
)


# ============================================================
# SEEDED K-MEANS
# ============================================================

print()
print("=" * 110)
print("SEEDED K-MEANS REFERENCE ÜZERİNDE ÇALIŞIYOR")
print("=" * 110)

reference_rows = (
    reference["embedding_row"]
    .astype(int)
    .to_numpy()
)

X_reference = embeddings[
    reference_rows
]

kmeans = KMeans(
    n_clusters=len(leaf_subjects),
    init=initial_centroids,
    n_init=1,
    random_state=RANDOM_STATE,
    max_iter=300,
    tol=1e-4
)

reference_labels = kmeans.fit_predict(
    X_reference
)

final_centroids = (
    kmeans.cluster_centers_
    .astype(np.float32)
)

centroid_shift = np.linalg.norm(
    final_centroids - initial_centroids,
    axis=1
)

print(
    "Final centroid matrix:",
    final_centroids.shape
)

print(
    "Kullanılan cluster:",
    len(np.unique(reference_labels))
)

print(
    "Ortalama centroid hareketi:",
    round(
        float(centroid_shift.mean()),
        4
    )
)

print(
    "Medyan centroid hareketi:",
    round(
        float(np.median(centroid_shift)),
        4
    )
)

print(
    "En fazla centroid hareketi:",
    round(
        float(centroid_shift.max()),
        4
    )
)


# ============================================================
# FINAL CLUSTERLARDA GERÇEK KONULARI SAY
# ============================================================

print()
print("=" * 110)
print("FINAL CLUSTER ETİKET ANALİZİ")
print("=" * 110)

cluster_subject_counts = {
    cluster_id: {}
    for cluster_id in range(len(leaf_subjects))
}


for position, cluster_id in enumerate(reference_labels):

    external_id = str(
        reference.iloc[position]["external_id"]
    )

    article_true_subjects = true_subjects.get(
        external_id,
        []
    )

    for subject_name in article_true_subjects:

        cluster_subject_counts[
            cluster_id
        ][subject_name] = (
            cluster_subject_counts[
                cluster_id
            ].get(
                subject_name,
                0
            )
            + 1
        )


# ============================================================
# HER CLUSTERIN BASKIN GERÇEK KONUSUNU BUL
# ============================================================

relabeled_centroid_subjects = []

cluster_analysis_rows = []


for cluster_id in range(len(leaf_subjects)):

    old_subject = centroid_subjects[
        cluster_id
    ]

    counts = cluster_subject_counts[
        cluster_id
    ]

    cluster_size = int(
        (
            reference_labels
            ==
            cluster_id
        ).sum()
    )

    if counts:

        sorted_subjects = sorted(
            counts.items(),
            key=lambda item: item[1],
            reverse=True
        )

        new_subject = sorted_subjects[0][0]
        dominant_count = sorted_subjects[0][1]

        second_subject = (
            sorted_subjects[1][0]
            if len(sorted_subjects) > 1
            else ""
        )

        second_count = (
            sorted_subjects[1][1]
            if len(sorted_subjects) > 1
            else 0
        )

    else:

        new_subject = old_subject
        dominant_count = 0
        second_subject = ""
        second_count = 0


    relabeled_centroid_subjects.append(
        new_subject
    )


    # Cluster içindeki makaleler multi-label olduğu için
    # dominant_count cluster_size'dan büyük olamaz;
    # fakat burada oranı yine güvenli hesaplıyoruz.
    if cluster_size > 0:

        dominance_ratio = (
            dominant_count
            /
            cluster_size
        )

    else:

        dominance_ratio = 0


    cluster_analysis_rows.append(
        {
            "cluster_id":
                cluster_id,

            "cluster_size":
                cluster_size,

            "old_subject":
                old_subject,

            "dominant_subject":
                new_subject,

            "dominant_count":
                dominant_count,

            "dominance_ratio":
                dominance_ratio,

            "second_subject":
                second_subject,

            "second_count":
                second_count,

            "centroid_shift":
                float(
                    centroid_shift[
                        cluster_id
                    ]
                ),

            "label_changed":
                old_subject
                !=
                new_subject
        }
    )


cluster_analysis = pd.DataFrame(
    cluster_analysis_rows
)


# ============================================================
# GENEL SONUÇLAR
# ============================================================

changed_count = int(
    cluster_analysis[
        "label_changed"
    ].sum()
)

same_count = (
    len(leaf_subjects)
    -
    changed_count
)

duplicate_count = (
    len(relabeled_centroid_subjects)
    -
    len(
        set(
            relabeled_centroid_subjects
        )
    )
)


print(
    "Toplam cluster:",
    len(leaf_subjects)
)

print(
    "Etiketi değişen cluster:",
    changed_count
)

print(
    "Etiketi aynı kalan cluster:",
    same_count
)

print(
    "Aynı konuya dönüşen ekstra cluster sayısı:",
    duplicate_count
)


# ============================================================
# İLK 15 DEĞİŞİKLİK
# ============================================================

print()
print("=" * 110)
print("İLK 15 DEĞİŞİKLİK")
print("=" * 110)

changed_clusters = (
    cluster_analysis[
        cluster_analysis[
            "label_changed"
        ]
    ]
    .sort_values(
        "dominance_ratio",
        ascending=False
    )
)


if changed_clusters.empty:

    print(
        "Hiçbir cluster etiketi değişmedi."
    )

else:

    for _, row in (
        changed_clusters
        .head(15)
        .iterrows()
    ):

        print()

        print(
            "Cluster:",
            int(row["cluster_id"])
        )

        print(
            "Cluster büyüklüğü:",
            int(row["cluster_size"])
        )

        print(
            "Eski:",
            row["old_subject"]
        )

        print(
            "Yeni:",
            row["dominant_subject"]
        )

        print(
            "Yeni konunun cluster içindeki sayısı:",
            int(row["dominant_count"])
        )

        print(
            "Baskınlık oranı:",
            f"{row['dominance_ratio'] * 100:.2f}%"
        )

        print(
            "İkinci konu:",
            row["second_subject"]
        )

        print(
            "Centroid hareketi:",
            round(
                float(row["centroid_shift"]),
                4
            )
        )


# ============================================================
# EN ÇOK HAREKET EDEN CLUSTERLAR
# ============================================================

print()
print("=" * 110)
print("EN ÇOK HAREKET EDEN 10 CENTROID")
print("=" * 110)

most_moved = (
    cluster_analysis
    .sort_values(
        "centroid_shift",
        ascending=False
    )
    .head(10)
)

for _, row in most_moved.iterrows():

    print()

    print(
        f"Cluster {int(row['cluster_id'])}"
    )

    print(
        "Başlangıç konusu:",
        row["old_subject"]
    )

    print(
        "Final baskın konu:",
        row["dominant_subject"]
    )

    print(
        "Hareket:",
        round(
            float(row["centroid_shift"]),
            4
        )
    )


# ============================================================
# CSV KAYDET
# ============================================================

OUTPUT_DIR = (
    "results/kmeans/holdout"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

output_file = os.path.join(
    OUTPUT_DIR,
    "kmeans_cluster_relabel_analysis.csv"
)

cluster_analysis.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# BİTİR
# ============================================================

print()
print("=" * 110)
print("RELABEL ANALİZİ TAMAMLANDI")
print("=" * 110)

print(
    "Dosya:",
    output_file
)

print()
print(
    "NOT: Bu deney henüz tahmin sistemini değiştirmedi."
)

print(
    "Sadece K-Means final clusterlarının "
    "başlangıç konu isimleriyle ne kadar uyumlu "
    "olduğunu ölçtü."
)