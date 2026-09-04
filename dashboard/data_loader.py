import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR = os.path.join(BASE_DIR, "data")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")


def load_algorithm_data(algorithm="hdbscan"):
  algo_lower = str(algorithm).lower()
  
  # 1. Eğer HDBSCAN seçildiyse, yeni oluşturduğumuz TÜM makalelerin ve skorların olduğu dosyayı okuyalım
  if algo_lower == "hdbscan":
    file_name = "hdbscan_tum_makaleler.csv"
  else:
    file_name = "kmeans_anomaliler.csv"

  file_path = os.path.join(RESULTS_DIR, file_name)

  # Eğer tam liste dosyası henüz oluşturulmadıysa balanced_articles üzerindenfallback yapalım
  if not os.path.exists(file_path):
    fallback_path = os.path.join(DATA_DIR, "balanced_articles.csv")
    if os.path.exists(fallback_path):
      df = pd.read_csv(fallback_path, encoding="utf-8-sig")
    else:
      return pd.DataFrame()
  else:
    df = pd.read_csv(file_path, encoding="utf-8-sig")

  # 2. Skor kolonlarını standartlaştıralım ve eksik varsa güvenli değer atayalım
  if "risk_skoru" not in df.columns:
    df["risk_skoru"] = 0.0
  else:
    df["risk_skoru"] = df["risk_skoru"].fillna(0.0)

  if "glosh_skoru" not in df.columns:
    df["glosh_skoru"] = 0.0
  else:
    df["glosh_skoru"] = df["glosh_skoru"].fillna(0.0)

  # 3. Özet sütunu kontrolü
  if "ozet" not in df.columns:
    if "abstract" in df.columns:
      df["ozet"] = df["abstract"]
    else:
      df["ozet"] = np.nan

  # 4. UMAP 2D koordinatlarını ekle (yoksa)
  umap_file = os.path.join(EMBEDDINGS_DIR, "umap_2d_coordinates.csv")
  if os.path.exists(umap_file):
    try:
      umap_df = pd.read_csv(umap_file)
      
      # Sütun adını esnek bulalım
      id_col_umap = None
      for col in ["external_id", "id", "ArticleID", "makale_id"]:
        if col in umap_df.columns:
          id_col_umap = col
          break

      if id_col_umap and "umap_x" in umap_df.columns and "umap_y" in umap_df.columns:
        # İki taraftaki ID'leri de tertemiz string yapalım ki eşleşmeme ihtimali kalmasın
        df["clean_id"] = df["external_id"].astype(str).str.strip().str.lower()
        umap_df["clean_id"] = umap_df[id_col_umap].astype(str).str.strip().str.lower()

        # Eski koordinat sütunları varsa temizle
        df.drop(columns=[c for c in ["umap_x", "umap_y"] if c in df.columns], errors="ignore", inplace=True)

        # Left join ile birleştir
        df = df.merge(
            umap_df[["clean_id", "umap_x", "umap_y"]],
            on="clean_id",
            how="left"
        )
        df.drop(columns=["clean_id"], errors="ignore", inplace=True)
    except Exception as e:
      print(f"UMAP okuma hatası: {e}")

  # 5. Küme standardizasyonu
  if "hdbscan_kume" in df.columns and "kume" not in df.columns:
    df["kume"] = df["hdbscan_kume"]
  elif "kmeans_kume" in df.columns and "kume" not in df.columns:
    df["kume"] = df["kmeans_kume"]
  elif "kume" not in df.columns:
    df["kume"] = -1

  # 6. Karar Tipi ve Açıklama Standartlaştırması
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
        df.get("karar_tipi") == "TP-1",
        df.get("oneri_kategori", ""),
        ""
    )

  if "ikincil_etiket_tp2" not in df.columns:
    df["ikincil_etiket_tp2"] = np.where(
        df.get("karar_tipi") == "TP-2",
        df.get("oneri_kategori", ""),
        ""
    )

  if "filtre_aciklamasi" not in df.columns:
    df["filtre_aciklamasi"] = np.where(
        df.get("ortak_agac_derinligi", -1) == 0,
        "Farklı Ana Disiplin Uyuşmazlığı (Kritik Öncelik)",
        "Alt Alan Uyuşmazlığı / Çoklu Disiplin Zenginleştirme"
    )

  # 7. Güvenli Boşluk Doldurma
  if "baslik" in df.columns:
    df["baslik"] = df["baslik"].fillna("Başlık Belirtilmemiş")
  elif "title" in df.columns:
    df["baslik"] = df["title"].fillna("Başlık Belirtilmemiş")
  else:
    df["baslik"] = "Başlık Belirtilmemiş"

  if "mevcut_kategori" not in df.columns:
    df["mevcut_kategori"] = "Belirtilmemiş"
  else:
    df["mevcut_kategori"] = df["mevcut_kategori"].fillna("Belirtilmemiş")

  if "oneri_kategori" not in df.columns:
    df["oneri_kategori"] = "Uyumlu / Normal"
  else:
    df["oneri_kategori"] = df["oneri_kategori"].fillna("Uyumlu / Normal")

  df["ozet"] = df["ozet"].fillna("Özet metni veri tabanında bulunmuyor.")

  return df


def load_article_detail(external_id):
  """
  hdbscan_tum_makaleler.csv dosyasından verilen external_id'ye sahip tek makaleyi bulup detaylarını döndürür.
  Bulunamazsa None döner.
  """
  target_id = str(external_id).strip()
  if not target_id:
    return None

  file_path = os.path.join(RESULTS_DIR, "hdbscan_tum_makaleler.csv")
  if not os.path.exists(file_path):
    fallback_path = os.path.join(DATA_DIR, "balanced_articles.csv")
    if os.path.exists(fallback_path):
      file_path = fallback_path
    else:
      return None

  try:
    matched_row = None
    for chunk in pd.read_csv(
        file_path,
        dtype={"external_id": str},
        encoding="utf-8-sig",
        chunksize=5000,
    ):
      match = chunk[chunk["external_id"].astype(str).str.strip() == target_id]
      if not match.empty:
        matched_row = match.iloc[0].to_dict()
        break

    if matched_row is None:
      return None

    def safe_val(val, default=""):
      if val is None or pd.isna(val):
        return default
      return val

    ortak_derinlik = matched_row.get("ortak_agac_derinligi")
    if pd.isna(ortak_derinlik):
      ortak_derinlik = None
    else:
      try:
        ortak_derinlik = int(ortak_derinlik)
      except (ValueError, TypeError):
        pass

    if ortak_derinlik == 0:
      karar_tipi = "TP-1"
    elif ortak_derinlik == 1:
      karar_tipi = "TP-2"
    elif ortak_derinlik is not None:
      karar_tipi = "İnceleme Gerekli"
    else:
      karar_tipi = "Normal"

    oneri_kategori = safe_val(matched_row.get("oneri_kategori"), "Uyumlu / Normal")
    duzeltme_tp1 = oneri_kategori if karar_tipi == "TP-1" else ""
    ikincil_tp2 = oneri_kategori if karar_tipi == "TP-2" else ""
    filtre_aciklamasi = (
        "Farklı Ana Disiplin Uyuşmazlığı (Kritik Öncelik)"
        if ortak_derinlik == 0
        else "Alt Alan Uyuşmazlığı / Çoklu Disiplin Zenginleştirme"
    )

    kume = matched_row.get("hdbscan_kume")
    if pd.isna(kume):
      kume = -1
    else:
      try:
        kume = int(kume)
      except (ValueError, TypeError):
        pass

    detail = {
        "external_id": target_id,
        "baslik": safe_val(matched_row.get("baslik") or matched_row.get("title"), "Başlık Belirtilmemiş"),
        "ozet": safe_val(matched_row.get("ozet") or matched_row.get("abstract"), "Özet metni veri tabanında bulunmuyor."),
        "mevcut_kategori": safe_val(matched_row.get("mevcut_kategori"), "Belirtilmemiş"),
        "tam_kategori_yollari": safe_val(matched_row.get("tam_kategori_yollari"), ""),
        "glosh_skoru": float(matched_row.get("glosh_skoru", 0.0)) if not pd.isna(matched_row.get("glosh_skoru")) else 0.0,
        "hdbscan_kume": kume,
        "oneri_yol": safe_val(matched_row.get("oneri_yol"), ""),
        "oneri_kategori": oneri_kategori,
        "label_sim_fark": float(matched_row.get("label_sim_fark", 0.0)) if not pd.isna(matched_row.get("label_sim_fark")) else 0.0,
        "knn_oneri": safe_val(matched_row.get("knn_oneri"), ""),
        "knn_baskinlik": float(matched_row.get("knn_baskinlik", 0.0)) if not pd.isna(matched_row.get("knn_baskinlik")) else 0.0,
        "knn_impurity": float(matched_row.get("knn_impurity", 0.0)) if not pd.isna(matched_row.get("knn_impurity")) else 0.0,
        "ortak_agac_derinligi": ortak_derinlik if ortak_derinlik is not None else -1,
        "oncelik": safe_val(matched_row.get("oncelik"), "NORMAL"),
        "supheli_mi": int(matched_row.get("supheli_mi", 0)) if not pd.isna(matched_row.get("supheli_mi")) else 0,
        "risk_skoru": float(matched_row.get("risk_skoru", 0.0)) if not pd.isna(matched_row.get("risk_skoru")) else 0.0,
        "kume": kume,
        "karar_tipi": karar_tipi,
        "duzeltme_onerisi_tp1": duzeltme_tp1,
        "ikincil_etiket_tp2": ikincil_tp2,
        "filtre_aciklamasi": filtre_aciklamasi,
    }
    return detail
  except Exception as e:
    print(f"Makale detayı okuma hatası: {e}")
    return None