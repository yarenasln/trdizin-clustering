import os
import numpy as np
import pandas as pd
from outlier_detector import OutlierDetector
from cluster_labeler import generate_cluster_labels

from config.paths import EMBEDDING_FILE, UMAP_FILE

# ==========================================
# DOSYA YOLLARI
# ==========================================
ARTICLE_FILE = "data/balanced_articles.csv"
SUBJECT_FILE = "data/article_subjects.csv"
OUTPUT_FILE = "results/hdbscan_anomaliler.csv"

def main():
    print("=" * 80)
    print("TR DİZİN HDBSCAN & MUTABAKAT TABANLI ANOMALİ TESPİTİ BAŞLATILIYOR")
    print("=" * 80)

    # 1. Dosya Varlık Kontrolleri
    for filepath in [ARTICLE_FILE, SUBJECT_FILE, EMBEDDING_FILE]:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Gerekli dosya bulunamadı: {filepath}")

    # 2. Verileri Yükle
    print("[*] Veri setleri ve embedding dosyası yükleniyor...")
    df_articles = pd.read_csv(ARTICLE_FILE, encoding="utf-8-sig")
    df_subjects = pd.read_csv(SUBJECT_FILE, encoding="utf-8-sig")
    embeddings = np.load(EMBEDDING_FILE)

    # Taksonomi yollarını CSV'den dinamik çek
    taksonomi_yollari = df_subjects["subject_fullname"].dropna().unique().tolist()
    print(f"[*] Toplam {len(taksonomi_yollari)} tekil taksonomi yolu yüklendi.")

    # 3. Makale ile Konu Yollarını Bire Çok İlişkiden Birleştir
    subject_agg = df_subjects.groupby("external_id").agg({
        "subject_name": lambda x: ", ".join(x.unique()),
        "subject_fullname": lambda x: " | ".join(x.unique())
    }).reset_index()

    df_merged = pd.merge(df_articles, subject_agg, on="external_id", how="left")
    df_merged["subject_name"] = df_merged["subject_name"].fillna("Bilinmeyen")
    df_merged["subject_fullname"] = df_merged["subject_fullname"].fillna("Bilinmeyen")

    # 4. Pipeline Formatına Dönüştür
    makale_listesi = []
    for _, row in df_merged.iterrows():
        # Özet bilgisini alalım (abstract veya ozet sütunlarına bakalım)
        ozet_metni = row.get("ozet", row.get("abstract", ""))
        if pd.isna(ozet_metni):
            ozet_metni = ""

        makale_listesi.append({
            "external_id": str(row["external_id"]),
            "baslik": str(row["title"]),
            "ozet": str(ozet_metni),
            "mevcut_kategori": str(row["subject_name"]),
            "tam_kategori_yollari": str(row["subject_fullname"])
        })

    # 5. Modeli Çalıştır
    detector = OutlierDetector(knn_k=10)
    df_anomaliler = detector.run_pipeline(makale_listesi, embeddings, taksonomi_yollari)

    # ---  Küme Etiketleri ve Merkezleri Üretme ---
    print("[*] Otomatik küme etiketleri ve merkezleri hesaplanıyor...")
    
    # UMAP koordinat dosyasını oku
    umap_df = pd.read_csv(UMAP_FILE)
    umap_df["external_id"] = umap_df["external_id"].astype(str)
    df_anomaliler["external_id"] = df_anomaliler["external_id"].astype(str)
    
    # Skorlar/kümeler ile UMAP koordinatlarını birleştir
    df_full_cluster = pd.merge(df_anomaliler, umap_df, on="external_id", how="inner")

    cluster_summary_df = generate_cluster_labels(
        df_full_cluster, 
        cluster_col='hdbscan_kume', # Doğru küme sütun adı
        x_col='umap_x',             # UMAP X sütun adı
        y_col='umap_y',             # UMAP Y sütun adı
        text_col='baslik'
    )
    
    os.makedirs("data", exist_ok=True)
    summary_path = 'data/hdbscan_cluster_summary.csv'
    cluster_summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"[*] Küme özetleri kaydedildi: {summary_path}")

   # --- CLAUDE'UN DEBUG KODU (DOĞRU DEĞİŞKEN İSMİYLE) ---
    print(f"1. Toplam benzersiz küme sayısı (noise hariç): {df_full_cluster[df_full_cluster['hdbscan_kume'] != -1]['hdbscan_kume'].nunique()}")
    print(f"2. Etiket üretilen küme sayısı: {len(cluster_summary_df)}")
    print(f"3. Saflık dağılımı:")
    print(cluster_summary_df['category_purity'].describe())
    print(f"0.3'ün altında saflığa sahip küme sayısı: {sum(1 for p in cluster_summary_df['category_purity'] if p < 0.3)}")
    # -----------------------------------------------------------

    # 6. Sonuçları Kaydet
    os.makedirs("results", exist_ok=True)
    df_anomaliler.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print(f"[*] İşlem başarıyla tamamlandı.")
    print(f"[*] Tespit edilen kesin anomali sayısı: {len(df_anomaliler)}")
    print(f"[*] Çıktı dosyası: {OUTPUT_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()
