import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR = os.path.join(BASE_DIR, "data")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")


def load_algorithm_data(algorithm="hdbscan"):
  algo_lower = str(algorithm).lower()
  file_name = (
      "hdbscan_anomaliler.csv"
      if algo_lower == "hdbscan"
      else "kmeans_anomaliler.csv"
  )
  file_path = os.path.join(RESULTS_DIR, file_name)

  if not os.path.exists(file_path):
    return pd.DataFrame()

  df = pd.read_csv(file_path, encoding="utf-8-sig")

  # 1. Özet sütunu yoksa oluştur veya eksik olanları doldur
  if "ozet" not in df.columns:
    df["ozet"] = np.nan

  if df["ozet"].isna().any() or (df["ozet"].astype(str).str.strip() == "").any():
    raw_files = [
        os.path.join(DATA_DIR, "balanced_articles.csv"),
        os.path.join(DATA_DIR, "processed", "makaleler_temiz.csv"),
        os.path.join(DATA_DIR, "makaleler_temiz.csv"),
        os.path.join(DATA_DIR, "makaleler.csv"),
        os.path.join(DATA_DIR, "raw", "makaleler.csv"),
        os.path.join(DATA_DIR, "tr_dizin_makaleler.csv"),
    ]

    for r_path in raw_files:
      if os.path.exists(r_path):
        try:
          raw_df = pd.read_csv(r_path, encoding="utf-8-sig")
          if "abstract" in raw_df.columns and "ozet" not in raw_df.columns:
            raw_df["ozet"] = raw_df["abstract"]

          if (
              "id" in raw_df.columns
              and "external_id" not in raw_df.columns
              and "external_id" in df.columns
          ):
            raw_df["external_id"] = raw_df["id"]

          if "external_id" in raw_df.columns and "ozet" in raw_df.columns:
            df["external_id_str"] = df["external_id"].astype(str)
            raw_df["external_id_str"] = raw_df["external_id"].astype(str)

            merged = df.merge(
                raw_df[["external_id_str", "ozet"]],
                on="external_id_str",
                how="left",
                suffixes=("", "_raw"),
            )

            if "ozet_raw" in merged.columns:
              df["ozet"] = df["ozet"].fillna(merged["ozet_raw"])

            df.drop(columns=["external_id_str"], errors="ignore", inplace=True)
            break
        except Exception:
          pass

  # 2. UMAP 2D koordinatlarını ekle (yoksa)
  if "umap_x" not in df.columns or "umap_y" not in df.columns or df["umap_x"].isna().any():
    umap_file = os.path.join(EMBEDDINGS_DIR, "umap_2d_coordinates.csv")
    if os.path.exists(umap_file):
      try:
        umap_df = pd.read_csv(umap_file)
        df["external_id_str"] = df["external_id"].astype(str)
        umap_df["external_id_str"] = umap_df["external_id"].astype(str)

        merged = df.merge(
            umap_df[["external_id_str", "umap_x", "umap_y"]],
            on="external_id_str",
            how="left",
            suffixes=("", "_coords"),
        )
        if "umap_x_coords" in merged.columns:
          df["umap_x"] = merged["umap_x_coords"]
          df["umap_y"] = merged["umap_y_coords"]
        df.drop(columns=["external_id_str"], errors="ignore", inplace=True)
      except Exception:
        pass

  # 3. Küme standardizasyonu
  if "hdbscan_kume" in df.columns and "kmeans_kume" not in df.columns:
    df["kume"] = df["hdbscan_kume"]
  elif "kmeans_kume" in df.columns:
    df["kume"] = df["kmeans_kume"]
  elif "kume" not in df.columns:
    df["kume"] = -1

  # 4. Karar Tipi ve Açıklama Standartlaştırması
  if "karar_tipi" not in df.columns:
    df["karar_tipi"] = np.where(
        df["ortak_agac_derinligi"] == 0,
        "TP-1",
        np.where(df["ortak_agac_derinligi"] == 1, "TP-2", "İnceleme Gerekli")
    )

  if "duzeltme_onerisi_tp1" not in df.columns:
    df["duzeltme_onerisi_tp1"] = np.where(
        df["karar_tipi"] == "TP-1",
        df.get("oneri_kategori", ""),
        ""
    )

  if "ikincil_etiket_tp2" not in df.columns:
    df["ikincil_etiket_tp2"] = np.where(
        df["karar_tipi"] == "TP-2",
        df.get("oneri_kategori", ""),
        ""
    )

  if "filtre_aciklamasi" not in df.columns:
    df["filtre_aciklamasi"] = np.where(
        df["ortak_agac_derinligi"] == 0,
        "Farklı Ana Disiplin Uyuşmazlığı (Kritik Öncelik)",
        "Alt Alan Uyuşmazlığı / Çoklu Disiplin Zenginleştirme"
    )

  # 5. Güvenli Boşluk Doldurma
  if "baslik" in df.columns:
    df["baslik"] = df["baslik"].fillna("Başlık Belirtilmemiş")
  else:
    df["baslik"] = "Başlık Belirtilmemiş"

  df["ozet"] = df["ozet"].fillna("Özet metni veri tabanında bulunmuyor.")

  return df