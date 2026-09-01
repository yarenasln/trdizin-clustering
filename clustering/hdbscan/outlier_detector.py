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

        if len(makaleler) != len(vektorler_np):
            raise ValueError(
                f"Makale sayısı ile embedding satır sayısı eşleşmiyor: "
                f"{len(makaleler)} != {len(vektorler_np)}"
            )

        vektorler_np = np.asarray(vektorler_np, dtype=np.float32)

        print("[*] 15D UMAP indirgeme ve HDBSCAN kümelemesi yapılıyor...")
        reducer_15d = umap.UMAP(
            n_neighbors=15,
            n_components=15,
            metric="cosine",
            random_state=42,
            low_memory=True
        )
        umap_15d = reducer_15d.fit_transform(vektorler_np)
        print("[*] UMAP tamamlandı:", umap_15d.shape)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=10,
            min_samples=5,
            prediction_data=True
        )
        clusterer.fit(umap_15d)

        hdbscan_labels = clusterer.labels_
        glosh_scores = clusterer.outlier_scores_

        print("[*] HDBSCAN tamamlandı.")
        print("[*] Küme sayısı:", len(set(hdbscan_labels) - {-1}))
        print("[*] Noise / aykırı nokta:", int(np.sum(hdbscan_labels == -1)))

        print(f"[*] {len(taksonomi_yollari)} taksonomi yolu embed ediliyor...")
        taksonomi_emb = self.model.encode(
            taksonomi_yollari,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        taksonomi_emb = np.asarray(taksonomi_emb, dtype=np.float32)

        print("[*] Taksonomi benzerlik matrisi hesaplanıyor...")
        sim_matrix = cosine_similarity(
            vektorler_np,
            taksonomi_emb
        ).astype(np.float32, copy=False)
        print("[*] Taksonomi similarity shape:", sim_matrix.shape)

        print("[*] Bellek-dostu k-NN komşulukları hesaplanıyor...")

        # Tam 20K x 20K matrisi RAM'de tutulmaz.
        # Makaleler küçük batch'ler halinde tüm veriyle karşılaştırılır.
        # Böylece sonuç yine cosine benzerliğine göre exact k-NN olur.
        chunk_size = 256
        n_samples = len(vektorler_np)

        knn_indices = np.empty(
            (n_samples, self.knn_k + 1),
            dtype=np.int32
        )

        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)

            chunk_sim = cosine_similarity(
                vektorler_np[start:end],
                vektorler_np
            ).astype(
                np.float32,
                copy=False
            )

            # En yüksek (k+1) cosine similarity indekslerini bul.
            part = np.argpartition(
                chunk_sim,
                kth=chunk_sim.shape[1] - (self.knn_k + 1),
                axis=1
            )[:, -(self.knn_k + 1):]

            row_idx = np.arange(end - start)[:, None]
            part_scores = chunk_sim[row_idx, part]

            order = np.argsort(
                part_scores,
                axis=1
            )[:, ::-1]

            knn_indices[start:end] = part[row_idx, order]

            del chunk_sim, part, part_scores, order

            if start == 0 or end % 2048 == 0 or end == n_samples:
                print(
                    f"[*] k-NN ilerleme: {end}/{n_samples}"
                )

        print("[*] k-NN tamamlandı:", knn_indices.shape)

        taxonomy_index = {
            yol: idx
            for idx, yol in enumerate(taksonomi_yollari)
        }

        sonuc_listesi = []

        for i, m in enumerate(makaleler):
            if i > 0 and i % 2000 == 0:
                print(f"[*] İşlenen makale: {i}/{len(makaleler)}")

            mevcut_yollar = m.get("tam_kategori_yollari", "")
            mevcut_kat = m.get("mevcut_kategori", "")
            baslik = m.get("baslik", "")
            ext_id = m.get("external_id", "")

            sim_scores = sim_matrix[i]
            best_idx = int(np.argmax(sim_scores))
            en_yakin_yol = taksonomi_yollari[best_idx]
            en_yakin_kat = en_yakin_yol.split(">")[-1].strip()

            mevcut_sim = 0.0
            mevcut_yol_listesi = [
                y.strip()
                for y in mevcut_yollar.split("|")
                if y.strip()
            ]

            for my in mevcut_yol_listesi:
                t_idx = taxonomy_index.get(my)
                if t_idx is not None:
                    mevcut_sim = max(
                        mevcut_sim,
                        float(sim_scores[t_idx])
                    )

            sim_fark = float(sim_scores[best_idx] - mevcut_sim)

            ortak_derinlik = 0
            for my in mevcut_yol_listesi:
                ortak_derinlik = max(
                    ortak_derinlik,
                    self.calculate_tree_depth(my, en_yakin_yol)
                )

            raw_neighbors = knn_indices[i].tolist()
            top_k_indices = [
                idx for idx in raw_neighbors
                if idx != i
            ][:self.knn_k]

            if len(top_k_indices) < self.knn_k:
                top_k_indices = raw_neighbors[:self.knn_k]

            komsu_ana_kategoriler = []
            for idx in top_k_indices:
                raw_k = makaleler[idx].get("mevcut_kategori", "")
                ilk_kat = raw_k.split(",")[0].strip() if raw_k else "Bilinmeyen"
                komsu_ana_kategoriler.append(ilk_kat)

            komsu_sayim = Counter(komsu_ana_kategoriler)
            baskin_komsu_kat, baskin_sayi = komsu_sayim.most_common(1)[0]
            knn_baskinlik = baskin_sayi / self.knn_k

            mevcut_kat_parcalari = {
                k.strip().lower()
                for k in mevcut_kat.split(",")
                if k.strip()
            }

            ayni_kat_sayisi = 0
            for idx in top_k_indices:
                raw_k = makaleler[idx].get("mevcut_kategori", "")
                kk_parcalari = {
                    k.strip().lower()
                    for k in raw_k.split(",")
                    if k.strip()
                }
                if mevcut_kat_parcalari.intersection(kk_parcalari):
                    ayni_kat_sayisi += 1

            knn_impurity = 1.0 - (ayni_kat_sayisi / self.knn_k)

            makale_kok = (
                mevcut_yollar.split(">")[0].strip()
                if ">" in mevcut_yollar
                else "Bilinmeyen"
            )
            label_kok = en_yakin_yol.split(">")[0].strip()

            knn_kok = "Bilinmeyen"
            baskin_lower = baskin_komsu_kat.lower()

            for t_yol in taksonomi_yollari:
                if baskin_lower in t_yol.lower():
                    knn_kok = t_yol.split(">")[0].strip()
                    break

            if makale_kok != label_kok:
                if knn_kok == label_kok or baskin_lower == en_yakin_kat.lower():
                    knn_onayliyor_mu = 1
                else:
                    knn_onayliyor_mu = 0
            else:
                knn_onayliyor_mu = 1

            supheli_mi = 1 if (
                ortak_derinlik <= 1
                and knn_impurity >= 0.50
                and sim_fark > 0.08
                and knn_onayliyor_mu == 1
                and (knn_baskinlik >= 0.30 or glosh_scores[i] > 0.70)
            ) else 0

            glosh_val = float(glosh_scores[i])

            risk_skoru = (
                knn_impurity * 0.40
                + min(max(sim_fark, 0), 1) * 0.35
                + glosh_val * 0.25
            )

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

        mask_ana_disiplin = (
            (df["ortak_agac_derinligi"] == 0)
            & (df["knn_baskinlik"] >= 0.30)
        )

        mask_alt_alan = (
            (df["ortak_agac_derinligi"] == 1)
            & (
                (df["oneri_kategori"] == df["knn_oneri"])
                | (df["knn_baskinlik"] >= 0.40)
            )
        )

        final_df = df[
            (df["supheli_mi"] == 1)
            & (df["label_sim_fark"] >= 0.09)
            & (mask_ana_disiplin | mask_alt_alan)
            & (~df["baslik"].astype(str).str.strip().isin(["", "-", "None", "nan"]))
            & (df["baslik"].astype(str).str.strip().str.len() > 3)
        ].sort_values(
            by="risk_skoru",
            ascending=False
        ).reset_index(drop=True)

        print()
        print("=" * 80)
        print("HDBSCAN PIPELINE ÖZETİ")
        print("=" * 80)
        print("Toplam makale:", len(makaleler))
        print("HDBSCAN küme sayısı:", len(set(hdbscan_labels) - {-1}))
        print("HDBSCAN noise:", int(np.sum(hdbscan_labels == -1)))
        print("Şüpheli aday:", int(df["supheli_mi"].sum()))
        print("Final anomali:", len(final_df))

        return final_df

