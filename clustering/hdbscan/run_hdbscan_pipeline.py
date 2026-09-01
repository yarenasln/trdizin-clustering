import os
import numpy as np
import pandas as pd
from clustering.hdbscan.outlier_detector import OutlierDetector

# ==========================================
# DOSYA YOLLARI
# ==========================================
ARTICLE_FILE = "data/balanced_articles.csv"
SUBJECT_FILE = "data/article_subjects.csv"
EMBEDDING_FILE = "embeddings/mpnet_multilingual_embeddings.npy"
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
        makale_listesi.append({
            "external_id": str(row["external_id"]),
            "baslik": str(row["title"]),
            "mevcut_kategori": str(row["subject_name"]),
            "tam_kategori_yollari": str(row["subject_fullname"])
        })

    # 5. Modeli Çalıştır
    detector = OutlierDetector(knn_k=10)
    df_anomaliler = detector.run_pipeline(makale_listesi, embeddings, taksonomi_yollari)

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
