import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR = os.path.join(BASE_DIR, "data")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")


def load_algorithm_data(algorithm="hdbscan"):
  algo_lower = str(algorithm).lower()
  
  # 1. Ana kaynak olarak TÜM makaleleri içeren balanced_articles.csv'yi yükleyelim
  file_path = os.path.join(DATA_DIR, "balanced_articles.csv")
  if not os.path.exists(file_path):
    file_path = os.path.join(RESULTS_DIR, "balanced_articles.csv")
    if not os.path.exists(file_path):
      return pd.DataFrame()

  df = pd.read_csv(file_path, encoding="utf-8-sig")

  # 2. Modelin anomali ve skor sonuçlarını sonuçlar klasöründen alıp ekleyelim
  result_file = "hdbscan_anomaliler.csv" if algo_lower == "hdbscan" else "kmeans_anomaliler.csv"
  result_path = os.path.join(RESULTS_DIR, result_file)

  if os.path.exists(result_path):
    try:
      res_df = pd.read_csv(result_path, encoding="utf-8-sig")
      if "external_id" in res_df.columns and "external_id" in df.columns:
        df["external_id_str"] = df["external_id"].astype(str).str.strip()
        res_df["external_id_str"] = res_df["external_id"].astype(str).str.strip()

        # Çakışacak eski boş sütunları temizleyelim ki sonuç dosyasındakiler ezilmesin
        cols_to_drop = ["risk_skoru", "glosh_skoru", "mevcut_kategori", "oneri_kategori", "oncelik", "kume", "risk", "glosh"]
        df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore", inplace=True)

        # Sonuç dosyasını sol birleşim (left join) ile ana tabloya basalım
        df = df.merge(res_df, on="external_id_str", how="left", suffixes=("", "_res"))
        df.drop(columns=["external_id_str"], errors="ignore", inplace=True)
    except Exception as e:
      print(f"Sonuç birleştirme hatası: {e}")

  # 3. Skor kolonlarını standart isimlere eşitleyelim
  if "risk_skoru" not in df.columns:
    for c in ["risk", "risk_score", "score", "anomaly_score"]:
      if c in df.columns:
        df["risk_skoru"] = df[c]
        break

  if "glosh_skoru" not in df.columns:
    for c in ["glosh", "glosh_score", "aykirilik_skoru", "outlier_score"]:
      if c in df.columns:
        df["glosh_skoru"] = df[c]
        break

  # Boş kalan skorları 0.0 ile dolduralım
  df["risk_skoru"] = df["risk_skoru"].fillna(0.0) if "risk_skoru" in df.columns else 0.0
  df["glosh_skoru"] = df["glosh_skoru"].fillna(0.0) if "glosh_skoru" in df.columns else 0.0

  # 4. Özet sütunu kontrolü
  if "ozet" not in df.columns:
    if "abstract" in df.columns:
      df["ozet"] = df["abstract"]
    else:
      df["ozet"] = np.nan

  # 5. UMAP 2D koordinatlarını ekle (yoksa)
  umap_file = os.path.join(EMBEDDINGS_DIR, "umap_2d_coordinates.csv")
  if os.path.exists(umap_file):
    try:
      umap_df = pd.read_csv(umap_file)
      
      if "external_id" not in umap_df.columns:
        for col in ["id", "ArticleID", "makale_id"]:
          if col in umap_df.columns:
            umap_df["external_id"] = umap_df[col]
            break

      if "external_id" in umap_df.columns and "umap_x" in umap_df.columns and "umap_y" in umap_df.columns:
        df["external_id_str"] = df["external_id"].astype(str).str.strip()
        umap_df["external_id_str"] = umap_df["external_id"].astype(str).str.strip()

        df.drop(columns=["umap_x", "umap_y"], errors="ignore", inplace=True)

        df = df.merge(
            umap_df[["external_id_str", "umap_x", "umap_y"]],
            on="external_id_str",
            how="left"
        )
        df.drop(columns=["external_id_str"], errors="ignore", inplace=True)
    except Exception as e:
      print(f"UMAP okuma hatası: {e}")

  # 6. Küme standardizasyonu
  if "hdbscan_kume" in df.columns and "kume" not in df.columns:
    df["kume"] = df["hdbscan_kume"]
  elif "kmeans_kume" in df.columns and "kume" not in df.columns:
    df["kume"] = df["kmeans_kume"]
  elif "kume" not in df.columns:
    df["kume"] = -1

  # 7. Karar Tipi ve Açıklama Standartlaştırması
  if "ortak_agac_derinligi" in df.columns:
    if "karar_tipi" not in df.columns:
      df["karar_tipi"] = np.where(
          df["ortak_agac_derinligi"] == 0,
          "TP-1",
          np.where(df["ortak_agac_derinligi"] == 1, "TP-2", "İnceleme Gerekli")
      )
  else:
    if "karar_tipi" not in df.columns:
      df["karar_tipi"] = "Normal"

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
        df.get("ortak_agac_derinligi", -1) == 0,
        "Farklı Ana Disiplin Uyuşmazlığı (Kritik Öncelik)",
        "Alt Alan Uyuşmazlığı / Çoklu Disiplin Zenginleştirme"
    )

  # 8. Güvenli Boşluk Doldurma
  if "baslik" in df.columns:
    df["baslik"] = df["baslik"].fillna("Başlık Belirtilmemiş")
  elif "title" in df.columns:
    df["baslik"] = df["title"].fillna("Başlık Belirtilmemiş")
  else:
    df["baslik"] = "Başlık Belirtilmemiş"

  df["ozet"] = df["ozet"].fillna("Özet metni veri tabanında bulunmuyor.")

  return df