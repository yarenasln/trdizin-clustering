import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR = os.path.join(BASE_DIR, "data")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")

# In-memory lazy cache depoları (Flask debug reloader dostu)
_DATA_CACHE = {}
_ANOMALY_IDS_CACHE = None
_ARTICLE_CACHE = {}


def clear_cache():
  """Tüm in-memory önbellekleri temizler."""
  global _DATA_CACHE, _ANOMALY_IDS_CACHE, _ARTICLE_CACHE
  _DATA_CACHE.clear()
  _ANOMALY_IDS_CACHE = None
  _ARTICLE_CACHE.clear()


def load_hdbscan_anomaly_ids(reload=False):
  """HDBSCAN anomali ID kümesini (388 kayıt) in-memory önbellekle döner."""
  global _ANOMALY_IDS_CACHE
  if not reload and _ANOMALY_IDS_CACHE is not None:
    return _ANOMALY_IDS_CACHE

  anom_file = os.path.join(RESULTS_DIR, "hdbscan_anomaliler.csv")
  if os.path.exists(anom_file):
    try:
      ids = set(
          pd.read_csv(
              anom_file,
              dtype={"external_id": str},
              usecols=["external_id"],
              encoding="utf-8-sig",
          )["external_id"]
          .astype(str)
          .str.strip()
      )
    except Exception as e:
      print(f"Anomali ID okuma hatası: {e}")
      ids = set()
  else:
    ids = set()

  _ANOMALY_IDS_CACHE = ids
  return ids


def load_algorithm_data(algorithm="hdbscan", reload=False):
  algo_lower = str(algorithm).lower()

  # 0. In-memory cache kontrolü (Tekrar disk okumasını engeller)
  if not reload and algo_lower in _DATA_CACHE:
    return _DATA_CACHE[algo_lower].copy()

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

  # 4. UMAP 2D koordinatlarını ekle (config.paths veya yerel fallback üzerinden)
  umap_file = None
  try:
    from config.paths import UMAP_FILE as CONFIG_UMAP_FILE
    if CONFIG_UMAP_FILE and os.path.exists(CONFIG_UMAP_FILE):
      umap_file = CONFIG_UMAP_FILE
  except Exception:
    pass

  if not umap_file:
    candidate = os.path.join(EMBEDDINGS_DIR, "umap_2d_coordinates.csv")
    if os.path.exists(candidate):
      umap_file = candidate

  if umap_file and os.path.exists(umap_file):
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

  _DATA_CACHE[algo_lower] = df
  return df.copy()


def _build_article_cache_if_needed():
  """
  Tüm makalelerin detay sözlüğünü in-memory index olarak hazırlar.
  Böylece /api/article/<external_id> disk okuması yapmadan O(1) hızla döner.
  """
  global _ARTICLE_CACHE
  if _ARTICLE_CACHE:
    return

  df = load_algorithm_data("hdbscan")
  if df.empty:
    return

  def safe_val(val, default=""):
    if val is None or pd.isna(val):
      return default
    return val

  cache = {}
  records = df.to_dict(orient="records")
  for row in records:
    target_id = str(row.get("external_id", "")).strip()
    if not target_id:
      continue

    ortak_derinlik = row.get("ortak_agac_derinligi")
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

    oneri_kategori = safe_val(row.get("oneri_kategori"), "Uyumlu / Normal")
    kume = row.get("hdbscan_kume")
    if pd.isna(kume):
      kume = -1
    else:
      try:
        kume = int(kume)
      except (ValueError, TypeError):
        pass

    cache[target_id] = {
        "external_id": target_id,
        "baslik": safe_val(row.get("baslik") or row.get("title"), "Başlık Belirtilmemiş"),
        "ozet": safe_val(row.get("ozet") or row.get("abstract"), "Özet metni veri tabanında bulunmuyor."),
        "mevcut_kategori": safe_val(row.get("mevcut_kategori"), "Belirtilmemiş"),
        "tam_kategori_yollari": safe_val(row.get("tam_kategori_yollari"), ""),
        "glosh_skoru": float(row.get("glosh_skoru", 0.0)) if not pd.isna(row.get("glosh_skoru")) else 0.0,
        "hdbscan_kume": kume,
        "oneri_yol": safe_val(row.get("oneri_yol"), ""),
        "oneri_kategori": oneri_kategori,
        "label_sim_fark": float(row.get("label_sim_fark", 0.0)) if not pd.isna(row.get("label_sim_fark")) else 0.0,
        "knn_oneri": safe_val(row.get("knn_oneri"), ""),
        "knn_baskinlik": float(row.get("knn_baskinlik", 0.0)) if not pd.isna(row.get("knn_baskinlik")) else 0.0,
        "knn_impurity": float(row.get("knn_impurity", 0.0)) if not pd.isna(row.get("knn_impurity")) else 0.0,
        "ortak_agac_derinligi": ortak_derinlik if ortak_derinlik is not None else -1,
        "oncelik": safe_val(row.get("oncelik"), "NORMAL"),
        "supheli_mi": int(row.get("supheli_mi", 0)) if not pd.isna(row.get("supheli_mi")) else 0,
        "risk_skoru": float(row.get("risk_skoru", 0.0)) if not pd.isna(row.get("risk_skoru")) else 0.0,
        "kume": kume,
        "karar_tipi": karar_tipi,
        "duzeltme_onerisi_tp1": oneri_kategori if karar_tipi == "TP-1" else "",
        "ikincil_etiket_tp2": oneri_kategori if karar_tipi == "TP-2" else "",
        "filtre_aciklamasi": (
            "Farklı Ana Disiplin Uyuşmazlığı (Kritik Öncelik)"
            if ortak_derinlik == 0
            else "Alt Alan Uyuşmazlığı / Çoklu Disiplin Zenginleştirme"
        ),
    }

  _ARTICLE_CACHE = cache


def load_article_detail(external_id):
  """
  Verilen external_id'ye sahip tek makaleyi in-memory index üzerinden O(1) hızla döndürür.
  Bulunamazsa None döner.
  """
  target_id = str(external_id).strip()
  if not target_id:
    return None

  try:
    _build_article_cache_if_needed()
    return _ARTICLE_CACHE.get(target_id)
  except Exception as e:
    print(f"Makale detayı okuma hatası: {e}")
    return None