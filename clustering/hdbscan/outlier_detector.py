import numpy as np
import pandas as pd
import hdbscan
import umap
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from collections import Counter


class OutlierDetector:
    def __init__(self, model_name="paraphrase-multilingual-mpnet-base-v2", knn_k=10):
        self.model_name = model_name
        self.knn_k = knn_k
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def calculate_tree_depth(self, path1: str, path2: str) -> int:
        if not path1 or not path2:
            return 0
        p1 = [x.strip() for x in path1.split(">")]
        p2 = [x.strip() for x in path2.split(">")]
        common = 0
        for a, b in zip(p1, p2):
            if a.lower() == b.lower():
                common += 1
            else:
                break
        return common

    def run_pipeline(self, makaleler: list, vektorler_np: np.ndarray, taksonomi_yollari: list) -> pd.DataFrame:
        print(f"[*] Toplam {len(makaleler)} makale işleniyor...")

        # 1. 15D UMAP & HDBSCAN Kümeleme ve GLOSH Tespiti
        print("[*] 15D UMAP indirgeme ve HDBSCAN kümelemesi yapılıyor...")
        reducer_15d = umap.UMAP(
            n_neighbors=15,
            n_components=15,
            metric="cosine",
            random_state=42
        )
        umap_15d = reducer_15d.fit_transform(vektorler_np)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=10,
            min_samples=5,
            prediction_data=True
        )
        clusterer.fit(umap_15d)

        hdbscan_labels = clusterer.labels_
        glosh_scores = clusterer.outlier_scores_

        # 2. Taksonomi Vektörleştirme
        print(f"[*] {len(taksonomi_yollari)} taksonomi yolu embed ediliyor...")
        taksonomi_emb = self.model.encode(
            taksonomi_yollari,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        taksonomi_emb = np.array(taksonomi_emb)

        # 3. Benzerlik Matrisleri
        print("[*] Benzerlik matrisleri hesaplanıyor...")
        sim_matrix = cosine_similarity(vektorler_np, taksonomi_emb)
        article_sim = cosine_similarity(vektorler_np, vektorler_np)

        sonuc_listesi = []

        for i, m in enumerate(makaleler):
            mevcut_yollar = m.get("tam_kategori_yollari", "")
            mevcut_kat = m.get("mevcut_kategori", "")
            baslik = m.get("baslik", "")
            ext_id = m.get("external_id", "")

            # Taksonomi Karşılaştırması
            sim_scores = sim_matrix[i]
            best_idx = int(np.argmax(sim_scores))
            en_yakin_yol = taksonomi_yollari[best_idx]
            en_yakin_kat = en_yakin_yol.split(">")[-1].strip()

            # Mevcut Kategori ile Benzerlik
            mevcut_sim = 0.0
            mevcut_yol_listesi = [y.strip() for y in mevcut_yollar.split("|")]
            for my in mevcut_yol_listesi:
                if my in taksonomi_yollari:
                    t_idx = taksonomi_yollari.index(my)
                    mevcut_sim = max(mevcut_sim, float(sim_scores[t_idx]))

            sim_fark = float(sim_scores[best_idx] - mevcut_sim)

            # Taksonomi Ağaç Mesafesi (Ortak Derinlik)
            ortak_derinlik = 0
            for my in mevcut_yol_listesi:
                ortak_derinlik = max(ortak_derinlik, self.calculate_tree_depth(my, en_yakin_yol))

            # --- DÜZELTME 1: k-NN Yerel Komşuluk Analizi (Tekil Ana Kategori Çıkarımı) ---
            top_k_indices = np.argsort(article_sim[i])[-(self.knn_k + 1):-1][::-1]
            
            # Her komşunun ilk/ana kategorisini al
            komsu_ana_kategoriler = []
            for idx in top_k_indices:
                raw_k = makaleler[idx].get("mevcut_kategori", "")
                ilk_kat = raw_k.split(",")[0].strip() if raw_k else "Bilinmeyen"
                komsu_ana_kategoriler.append(ilk_kat)

            komsu_sayim = Counter(komsu_ana_kategoriler)
            baskin_komsu_kat, baskin_sayi = komsu_sayim.most_common(1)[0]
            knn_baskinlik = baskin_sayi / self.knn_k

            # Esnek Impurity Hesabı
            mevcut_kat_parcalari = {k.strip().lower() for k in mevcut_kat.split(",") if k.strip()}
            ayni_kat_sayisi = 0
            for idx in top_k_indices:
                raw_k = makaleler[idx].get("mevcut_kategori", "")
                kk_parcalari = {k.strip().lower() for k in raw_k.split(",") if k.strip()}
                if len(mevcut_kat_parcalari.intersection(kk_parcalari)) > 0:
                    ayni_kat_sayisi += 1

            knn_impurity = 1.0 - (ayni_kat_sayisi / self.knn_k)

            # --- DÜZELTME 2: Hubness Önleme ve Kök Kontrolü ---
            makale_kok = mevcut_yollar.split(">")[0].strip() if ">" in mevcut_yollar else "Bilinmeyen"
            label_kok = en_yakin_yol.split(">")[0].strip()

            knn_kok = "Bilinmeyen"
            baskin_lower = baskin_komsu_kat.lower()
            for t_yol in taksonomi_yollari:
                if baskin_lower in t_yol.lower():
                    knn_kok = t_yol.split(">")[0].strip()
                    break

            # Mutabakat Kriteri
            if makale_kok != label_kok:
                # Ana disiplin değişiyorsa komşuların ana kökü de öneriyi onaylamalı
                if knn_kok == label_kok or baskin_lower == en_yakin_kat.lower():
                    knn_onayliyor_mu = 1
                else:
                    knn_onayliyor_mu = 0
            else:
                knn_onayliyor_mu = 1

            # --- DÜZELTME 3: Şüpheli Kriteri (Çoğunluk ve Mutabakat Şartı) ---
            supheli_mi = 1 if (
                ortak_derinlik <= 1
                and knn_impurity >= 0.50
                and sim_fark > 0.08  # Eşik hafif sıkılaştırıldı
                and knn_onayliyor_mu == 1
                and (knn_baskinlik >= 0.30 or glosh_scores[i] > 0.70) # Gerçek çoğunluk veya güçlü outlier
            ) else 0

            # Bileşik Risk Skoru
            glosh_val = float(glosh_scores[i])
            risk_skoru = (knn_impurity * 0.40) + (min(max(sim_fark, 0), 1) * 0.35) + (glosh_val * 0.25)

            oncelik = "DÜŞÜK"
            if ortak_derinlik == 0:
                oncelik = "KRİTİK (Farklı Ana Disiplin)"
            elif ortak_derinlik == 1:
                oncelik = "YÜKSEK (Farklı Alt Alan)"

            sonuc_listesi.append({
                "external_id": ext_id,
                "baslik": baslik,
                "mevcut_kategori": mevcut_kat,
                "tam_kategori_yollari": mevcut_yollar,
                "glosh_skoru": glosh_val,
                "hdbscan_kume": int(hdbscan_labels[i]),
                "oneri_yol": en_yakin_yol,
                "oneri_kategori": en_yakin_kat,
                "label_sim_fark": sim_fark,
                "knn_oneri": baskin_komsu_kat,
                "knn_baskinlik": knn_baskinlik,
                "knn_impurity": knn_impurity,
                "ortak_agac_derinligi": ortak_derinlik,
                "oncelik": oncelik,
                "supheli_mi": supheli_mi,
                "risk_skoru": risk_skoru
            })

        df = pd.DataFrame(sonuc_listesi)

        # Nihai Filtreleme (Sabitlenmiş 2 Kademeli Model)
        mask_ana_disiplin = (df["ortak_agac_derinligi"] == 0) & (df["knn_baskinlik"] >= 0.30)

        mask_alt_alan = (df["ortak_agac_derinligi"] == 1) & (
            (df["oneri_kategori"] == df["knn_oneri"])
            | (df["knn_baskinlik"] >= 0.40)
        )

        final_df = df[
            (df["supheli_mi"] == 1)
            & (df["label_sim_fark"] >= 0.09)
            & (mask_ana_disiplin | mask_alt_alan)
            & (~df["baslik"].astype(str).str.strip().isin(["", "-", "None", "nan"]))
            & (df["baslik"].astype(str).str.strip().str.len() > 3)
        ].sort_values(by="risk_skoru", ascending=False).reset_index(drop=True)

        return final_df