import pandas as pd
import numpy as np
from collections import Counter


def clean_taxonomy_label(raw_path):
    if not isinstance(raw_path, str):
        return ""

    parts = [
        p.strip()
        for p in raw_path.split(">")
        if p.strip()
    ]

    cleaned_parts = []

    for part in parts:
        if (
            not cleaned_parts
            or cleaned_parts[-1].lower() != part.lower()
        ):
            cleaned_parts.append(part)

    return " > ".join(cleaned_parts)


def extract_unique_article_paths(raw_value):
    article_paths = set()

    if pd.isna(raw_value):
        return article_paths

    for path in str(raw_value).split("|"):
        cleaned = clean_taxonomy_label(path)
        if cleaned:
            article_paths.add(cleaned)

    return article_paths


def generate_cluster_labels(
    df,
    cluster_col="hdbscan_kume",
    x_col="umap_x",
    y_col="umap_y",
    text_col="baslik"
):
    required_cols = [
        cluster_col,
        x_col,
        y_col
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"DataFrame içinde gerekli sütun bulunamadı: {col}"
            )

    valid_df = df[
        df[cluster_col] != -1
    ].copy()

    cluster_summaries = []

    if "tam_kategori_yollari" in valid_df.columns:
        category_col = "tam_kategori_yollari"
    elif "mevcut_kategori" in valid_df.columns:
        category_col = "mevcut_kategori"
    else:
        category_col = None

    for cluster_id, group in valid_df.groupby(cluster_col):

        size = len(group)

        if size == 0:
            continue

        # A. 2D UMAP temsilci noktası
        mean_x = group[x_col].mean()
        mean_y = group[y_col].mean()

        distances = np.sqrt(
            (group[x_col] - mean_x) ** 2
            +
            (group[y_col] - mean_y) ** 2
        )

        medoid_idx = distances.idxmin()
        medoid_row = group.loc[medoid_idx]

        x_center = float(medoid_row[x_col])
        y_center = float(medoid_row[y_col])

        # B. Varsayılan değerler
        taxonomy_path = f"Küme #{cluster_id}"
        level_1 = f"Küme #{cluster_id}"
        level_2 = f"Küme #{cluster_id}"
        level_3 = f"Küme #{cluster_id}"
        dominant_category = ""
        category_purity = 0.0

        has_category_data = False

        # C. Taksonomi analizi
        if category_col:
            raw_paths = []
            for value in group[category_col]:
                article_paths = extract_unique_article_paths(value)
                raw_paths.extend(article_paths)

            if raw_paths:
                has_category_data = True
                counter = Counter(raw_paths)
                most_common_path, count = counter.most_common(1)[0]

                dominant_category = most_common_path
                taxonomy_path = most_common_path

                category_purity = min(count / size, 1.0)

                parts = [
                    p.strip()
                    for p in most_common_path.split(">")
                    if p.strip()
                ]

                # Purity Eşiği Kontrolü (< 0.40 ise karışık küme)
                if category_purity < 0.40:
                    # Karışık kümelerde Level 1 (ana dal) korunsun, alt seviyeler ID olsun
                    if len(parts) >= 1:
                        level_1 = parts[0]
                    else:
                        level_1 = f"Küme #{cluster_id}"
                    
                    level_2 = f"Küme #{cluster_id}"
                    level_3 = f"Küme #{cluster_id}"
                else:
                    # Normal, yüksek saflıklı kümeler için tam taksonomi katmanları
                    if len(parts) >= 1:
                        level_1 = parts[0]
                    if len(parts) >= 2:
                        level_2 = parts[1]
                    else:
                        level_2 = level_1

                    if len(parts) >= 3:
                        level_3 = " > ".join(parts[2:])
                    else:
                        level_3 = level_2

        # D. Fallback: Sadece veri setinde KATEGORİ HİÇ YOKSA başlık kullan
        if (
            not has_category_data
            and text_col in group.columns
        ):
            titles = (
                group[text_col]
                .dropna()
                .astype(str)
                .tolist()
            )

            if titles:
                raw_title = titles[0].strip()
                fallback_title = (
                    raw_title[:30] + "..." if len(raw_title) > 30 else raw_title
                )
                taxonomy_path = fallback_title
                level_1 = fallback_title
                level_2 = fallback_title
                level_3 = fallback_title

        # E. Cluster summary kaydı
        cluster_summaries.append({
            "cluster_id": int(cluster_id),
            "taxonomy_path": taxonomy_path,
            "display_name_level_1": level_1,
            "display_name_level_2": level_2,
            "display_name_level_3": level_3,
            "x_center": x_center,
            "y_center": y_center,
            "size": int(size),
            "dominant_category": dominant_category,
            "category_purity": round(float(category_purity), 2)
        })

    summary_df = pd.DataFrame(cluster_summaries)
    return summary_df