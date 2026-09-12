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

  # Forced HDBSCAN küme görünümü
  if "forced_cluster" not in df.columns:
      df["forced_cluster"] = df["kume"]

  if "forced_strength" not in df.columns:
      df["forced_strength"] = 0.0

  # 6. Karar Tipi ve Açıklama Standartlaştırması
  def normalize_priority(val):
    s = str(val).strip()
    if not s or s.lower() in ["nan", "none", ""]:
      return "NORMAL"
    u = s.upper()
    if "KR" in u:
      return "KRİTİK"
    elif "Y" in u and any(c in u for c in ["KSEK", "Ü", "U"]):
      return "YÜKSEK"
    elif "ORTA" in u:
      return "ORTA"
    elif "D" in u and any(c in u for c in ["K", "Ş", "S"]):
      return "DÜŞÜK"
    return s

  if "oncelik" in df.columns:
    df["oncelik"] = df["oncelik"].apply(normalize_priority)
  else:
    df["oncelik"] = "NORMAL"

  if "ortak_agac_derinligi" in df.columns:
    df["karar_tipi"] = np.where(
        df["ortak_agac_derinligi"] == 0,
        "Ana Disiplin Uyuşmazlığı Adayı",
        np.where(df["ortak_agac_derinligi"] == 1, "Alt Alan / İkincil Disiplin Adayı", "İnceleme Adayı")
    )
  else:
    if "karar_tipi" not in df.columns:
      df["karar_tipi"] = "Normal"

  df["oncelik_etiketi"] = df["oncelik"] + " · " + df["karar_tipi"]

  if "duzeltme_onerisi_tp1" not in df.columns:
    df["duzeltme_onerisi_tp1"] = np.where(
        df.get("ortak_agac_derinligi") == 0,
        df.get("oneri_kategori", ""),
        ""
    )

  if "ikincil_etiket_tp2" not in df.columns:
    df["ikincil_etiket_tp2"] = np.where(
        df.get("ortak_agac_derinligi") == 1,
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

  # HDBSCAN anomali maskesini önbelleğe al (Hızlı LOD filtreleme için)
  if algo_lower == "hdbscan":
    anom_ids = load_hdbscan_anomaly_ids()
    df["is_hdbscan_anomaly"] = df["external_id"].astype(str).str.strip().isin(anom_ids)
  else:
    df["is_hdbscan_anomaly"] = False

  _DATA_CACHE[algo_lower] = df
  return df.copy()


_TAXONOMY_PATHS = None


def get_taxonomy_paths():
  """Taksonomi yollarını tekil liste olarak yükler ve önbelleğe alır."""
  global _TAXONOMY_PATHS
  if _TAXONOMY_PATHS is not None:
    return _TAXONOMY_PATHS

  subjects_path = os.path.join(DATA_DIR, "article_subjects.csv")
  if os.path.exists(subjects_path):
    try:
      df_sub = pd.read_csv(subjects_path, usecols=["subject_fullname"])
      _TAXONOMY_PATHS = df_sub["subject_fullname"].dropna().unique().tolist()
    except Exception as e:
      print(f"Taksonomi yolları okuma hatası: {e}")
      _TAXONOMY_PATHS = []
  else:
    _TAXONOMY_PATHS = []

  return _TAXONOMY_PATHS


def calculate_knn_onay(row, taxonomy_paths=None):
  """
  Makalenin kNN komşuluk baskın disiplini ile önerilen model disiplini
  arasındaki yerel destek uzlaşısını hesaplar (knn_onayliyor_mu == 1).
  """
  # Eğer kayıt zaten kesin anomali olarak işaretlenmişse kural gereği onaylıdır
  if int(row.get("supheli_mi", 0) or 0) == 1:
    return 1

  if taxonomy_paths is None:
    taxonomy_paths = get_taxonomy_paths()

  makale_yollari = str(row.get("tam_kategori_yollari", ""))
  makale_kok = makale_yollari.split(">")[0].strip() if ">" in makale_yollari else "Bilinmeyen"

  oneri_yol = str(row.get("oneri_yol", ""))
  label_kok = oneri_yol.split(">")[0].strip() if ">" in oneri_yol else "Bilinmeyen"

  baskin_komsu_kat = str(row.get("knn_oneri", "")).strip()
  baskin_lower = baskin_komsu_kat.lower()
  en_yakin_kat = str(row.get("oneri_kategori", "")).strip()

  knn_kok = "Bilinmeyen"
  for t_yol in taxonomy_paths:
    if baskin_lower in t_yol.lower():
      knn_kok = t_yol.split(">")[0].strip()
      break

  if makale_kok != label_kok:
    if knn_kok == label_kok or baskin_lower == en_yakin_kat.lower():
      return 1
    return 0
  return 1


def build_why_flagged(row, knn_onay_val=None):
  """
  Metodolojik anomali kararı için açıklanabilirlik objesini oluşturur.
  Tüm eşikleri, mevcut değerleri ve kural geçme durumlarını döner.
  """
  if knn_onay_val is None:
    knn_onay_val = calculate_knn_onay(row)

  sim_fark_val = float(row.get("label_sim_fark", 0.0)) if not pd.isna(row.get("label_sim_fark")) else 0.0
  knn_imp_val = float(row.get("knn_impurity", 0.0)) if not pd.isna(row.get("knn_impurity")) else 0.0
  knn_bask_val = float(row.get("knn_baskinlik", 0.0)) if not pd.isna(row.get("knn_baskinlik")) else 0.0
  glosh_val = float(row.get("glosh_skoru", 0.0)) if not pd.isna(row.get("glosh_skoru")) else 0.0

  ortak_d = row.get("ortak_agac_derinligi")
  if pd.isna(ortak_d) or ortak_d is None:
    ortak_d = -1
  else:
    try:
      ortak_d = int(ortak_d)
    except (ValueError, TypeError):
      ortak_d = -1

  if ortak_d == 0:
    candidate_type = "Ana Disiplin Uyuşmazlığı Adayı"
  elif ortak_d == 1:
    candidate_type = "Alt Alan / İkincil Disiplin Adayı"
  elif ortak_d >= 0:
    candidate_type = "İnceleme Adayı"
  else:
    candidate_type = "Normal"

  oneri_kat = str(row.get("oneri_kategori", "")).strip()
  knn_oneri = str(row.get("knn_oneri", "")).strip()

  # Bileşik Risk Skoru Katkıları (Bileşik Risk / İnceleme Önceliği)
  # risk = knn_impurity * 0.40 + min(max(label_sim_fark, 0), 1) * 0.35 + glosh * 0.25
  knn_contrib = round(knn_imp_val * 0.40, 3)
  semantic_contrib = round(min(max(sim_fark_val, 0.0), 1.0) * 0.35, 3)
  glosh_contrib = round(glosh_val * 0.25, 3)
  total_risk = round(knn_contrib + semantic_contrib + glosh_contrib, 3)

  # Karar Koşulları (Okunabilir Türkçe Açıklamalarla)
  rules = [
      {
          "id": "semantic_gap",
          "label": "Semantik Kategori Farkı",
          "val_text": f"Değer: {sim_fark_val:.3f}",
          "threshold_text": "Final eşik: ≥ 0.09 (Ön aday: > 0.08)",
          "passed": bool(sim_fark_val >= 0.09),
          "description": "Makalenin mevcut kategorisi ile alternatif kategori arasındaki embedding kosinüs mesafe farkı.",
      },
      {
          "id": "knn_impurity",
          "label": "Lokal Komşuluk Uyuşmazlığı",
          "val_text": f"kNN impurity: %{knn_imp_val * 100:.0f}",
          "threshold_text": "Eşik: ≥ %50",
          "passed": bool(knn_imp_val >= 0.50),
          "description": "En yakın 10 komşu makale içerisindeki farklı kategori oranı.",
      },
      {
          "id": "knn_support",
          "label": "kNN Yerel Destek Uzlaşısı",
          "val_text": "Alternatif alan yerel komşular tarafından destekleniyor" if knn_onay_val == 1 else "Yerel komşular önerilen alanı desteklemiyor",
          "threshold_text": "Eşik: Desteklemeli (== 1)",
          "passed": bool(knn_onay_val == 1),
          "description": "En yakın komşuların kök disiplini ile modelin önerdiği kategori hiyerarşisinin uyuşması.",
      },
  ]

  # Hiyerarşik derinliğe göre kNN baskınlık maskesi kuralı
  if ortak_d == 0:
    rules.append({
        "id": "knn_dominance",
        "label": "kNN Baskınlık (Ana Disiplin Kuralı)",
        "val_text": f"Baskınlık: %{knn_bask_val * 100:.0f}",
        "threshold_text": "Ana disiplin eşiği: ≥ %30",
        "passed": bool(knn_bask_val >= 0.30),
        "description": "Farklı ana disiplin adaylığı için komşularda tek bir kategorinin en az %30 baskınlığı.",
    })
  elif ortak_d == 1:
    is_match = (oneri_kat.lower() == knn_oneri.lower()) and (oneri_kat != "")
    sub_passed = bool(is_match or (knn_bask_val >= 0.40))
    match_str = "Eşleşti" if is_match else "Eşleşmedi"
    rules.append({
        "id": "knn_dominance",
        "label": "kNN Baskınlık / Öneri Uzlaşısı (Alt Alan Kuralı)",
        "val_text": f"Baskınlık: %{knn_bask_val * 100:.0f} | Öneri: '{knn_oneri}' ({match_str})",
        "threshold_text": "Eşik: Öneri eşleşmesi VEYA Baskınlık ≥ %40",
        "passed": sub_passed,
        "description": "Alt alan adaylığı için kNN baskın önerisinin model önerisiyle birebir eşleşmesi veya en az %40 baskınlık.",
    })
  else:
    rules.append({
        "id": "knn_dominance",
        "label": "kNN Baskınlık / Aykırılık Koşulu",
        "val_text": f"Baskınlık: %{knn_bask_val * 100:.0f}",
        "threshold_text": "Eşik: ≥ %30 veya GLOSH > 0.70",
        "passed": bool(knn_bask_val >= 0.30 or glosh_val > 0.70),
        "description": "Adaylık için kNN baskınlığı veya yüksek GLOSH aykırılığı.",
    })

  # GLOSH Yoğunluk Aykırılığı Kuralı
  rules.append({
      "id": "glosh",
      "label": "Yoğunluk Tabanlı Aykırılık (GLOSH)",
      "val_text": f"GLOSH: {glosh_val:.3f}",
      "threshold_text": "Güçlü aykırılık eşiği: > 0.70",
      "passed": bool(glosh_val > 0.70),
      "description": "HDBSCAN hiyerarşik yoğunluk aykırılık skoru.",
  })

  # Taksonomi Ağaç Derinliği Kuralı
  tree_passed = bool(ortak_d in [0, 1])
  tree_text = (
      "Farklı Ana Disiplin Uyuşmazlığı Adayı"
      if ortak_d == 0
      else ("Alt Alan / İkincil Disiplin Adayı" if ortak_d == 1 else "Normal / Uyumlu")
  )
  rules.append({
      "id": "tree_depth",
      "label": "Taksonomi Hiyerarşik Derinliği",
      "val_text": f"Derinlik: {ortak_d} ({tree_text})",
      "threshold_text": "Adaylık eşiği: ≤ 1",
      "passed": tree_passed,
      "description": "Mevcut kategori ile önerilen alternatif kategori arasındaki taksonomik ortak ağaç derinliği.",
  })

  return {
      "candidate_type": candidate_type,
      "semantic_gap": round(sim_fark_val, 4),
      "knn_impurity": round(knn_imp_val, 4),
      "knn_dominance": round(knn_bask_val, 4),
      "knn_supports_alternative": bool(knn_onay_val == 1),
      "glosh": round(glosh_val, 4),
      "tree_depth": ortak_d,
      "risk_components": {
          "knn": knn_contrib,
          "semantic": semantic_contrib,
          "glosh": glosh_contrib,
          "total": total_risk,
      },
      "rules": rules,
  }


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

  taxonomy_paths = get_taxonomy_paths()

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
      karar_tipi = "Ana Disiplin Uyuşmazlığı Adayı"
    elif ortak_derinlik == 1:
      karar_tipi = "Alt Alan / İkincil Disiplin Adayı"
    elif ortak_derinlik is not None:
      karar_tipi = "İnceleme Adayı"
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

    oncelik_val = safe_val(row.get("oncelik"), "NORMAL")
    oncelik_etiketi = safe_val(row.get("oncelik_etiketi"), f"{oncelik_val} · {karar_tipi}")

    knn_onay_val = calculate_knn_onay(row, taxonomy_paths)
    why_flagged_data = build_why_flagged(row, knn_onay_val)

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
        "oncelik": oncelik_val,
        "oncelik_etiketi": oncelik_etiketi,
        "supheli_mi": int(row.get("supheli_mi", 0)) if not pd.isna(row.get("supheli_mi")) else 0,
        "risk_skoru": float(row.get("risk_skoru", 0.0)) if not pd.isna(row.get("risk_skoru")) else 0.0,
        "kume": kume,
        "karar_tipi": karar_tipi,
        "duzeltme_onerisi_tp1": oneri_kategori if ortak_derinlik == 0 else "",
        "ikincil_etiket_tp2": oneri_kategori if ortak_derinlik == 1 else "",
        "filtre_aciklamasi": (
            "Farklı Ana Disiplin Uyuşmazlığı (Kritik Öncelik)"
            if ortak_derinlik == 0
            else "Alt Alan Uyuşmazlığı / Çoklu Disiplin Zenginleştirme"
        ),
        "knn_onayliyor_mu": knn_onay_val,
        "why_flagged": why_flagged_data,
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
