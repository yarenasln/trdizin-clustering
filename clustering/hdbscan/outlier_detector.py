import os
import numpy as np
import pandas as pd
import hdbscan
import umap
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from collections import Counter
from qdrant_client import QdrantClient
from qdrant_client.models import QueryRequest, SearchParams


class OutlierDetector:
    def __init__(
        self,
        model_name="paraphrase-multilingual-mpnet-base-v2",
        knn_k=10,
        qdrant_url=None,
        qdrant_api_key=None,
        collection_name=None,
    ):
        self.model_name = model_name
        self.knn_k = knn_k
        self._model = None
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "https://qdrant.ulakbim.gov.tr")
        self.qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY", None)
        self.collection_name = collection_name or os.getenv("QDRANT_COLLECTION", "trdizin_articles")
        self.vector_name = "mpnet_v1"

    def _get_qdrant_client(self) -> QdrantClient:
        return QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            port=None,
            prefer_grpc=False,
            check_compatibility=False,
            timeout=60,
        )

    def _get_qdrant_knn_indices(
        self,
        makaleler: list,
        vektorler_np: np.ndarray,
        batch_size: int = 256
    ) -> np.ndarray:
        n_samples = len(makaleler)
        print(f"[*] Qdrant üzerinden {n_samples} makale için batch k-NN komşulukları çekiliyor (Batch: {batch_size})...")

        # 1. external_id -> 0-tabanlı satır indeksi eşleme sözlüğü
        eid_to_row = {}
        for i, m in enumerate(makaleler):
            raw_eid = m.get("external_id")
            if raw_eid is None:
                raise ValueError(f"Makale {i} için 'external_id' eksik!")
            try:
                eid_to_row[int(raw_eid)] = i
            except (ValueError, TypeError) as e:
                raise ValueError(f"Geçersiz external_id: {raw_eid} (Satır: {i})") from e

        client = self._get_qdrant_client()
        knn_indices = np.empty((n_samples, self.knn_k + 1), dtype=np.int32)

        num_batches = (n_samples + batch_size - 1) // batch_size

        for b_idx in range(num_batches):
            start = b_idx * batch_size
            end = min(start + batch_size, n_samples)
            batch_slice = range(start, end)

            requests = []
            for i in batch_slice:
                vec = vektorler_np[i].tolist()
                requests.append(
                    QueryRequest(
                        query=vec,
                        using=self.vector_name,
                        params=SearchParams(exact=True),
                        limit=self.knn_k + 1,
                        with_payload=["subject_names", "root_names", "external_id"],
                        with_vector=False,
                    )
                )

            batch_responses = client.query_batch_points(
                collection_name=self.collection_name,
                requests=requests,
                timeout=60,
            )

            if len(batch_responses) != len(requests):
                raise RuntimeError(
                    f"Qdrant batch response sayısı istek sayısıyla eşleşmiyor! "
                    f"Beklenen: {len(requests)}, Dönen: {len(batch_responses)} (Batch {b_idx + 1}/{num_batches})"
                )

            for offset, resp in enumerate(batch_responses):
                row_idx = start + offset
                curr_eid = int(makaleler[row_idx]["external_id"])
                points = resp.points

                if len(points) < (self.knn_k + 1):
                    raise RuntimeError(
                        f"Qdrant makale {curr_eid} (satır {row_idx}) için yeterli sonuç dönmedi! "
                        f"Beklenen en az {self.knn_k + 1}, dönen: {len(points)}"
                    )

                # Self çıkarıldıktan sonra en az knn_k komşu kalmalı
                filtered_neighbors = [p for p in points if int(p.id) != curr_eid]
                if len(filtered_neighbors) < self.knn_k:
                    raise RuntimeError(
                        f"Makale {curr_eid} (satır {row_idx}) için self çıkarıldıktan sonra "
                        f"{self.knn_k} komşu kalmıyor! Kalan komşu sayısı: {len(filtered_neighbors)}"
                    )

                row_indices_11 = []
                for p in points[:self.knn_k + 1]:
                    p_id = int(p.id)
                    if p_id not in eid_to_row:
                        raise KeyError(
                            f"Qdrant'tan dönen point ID ({p_id}) makale listesinde (eid_to_row) bulunamadı! "
                            f"(Sorgulanan makale: {curr_eid}, satır: {row_idx})"
                        )
                    row_indices_11.append(eid_to_row[p_id])

                knn_indices[row_idx] = row_indices_11

            if b_idx == 0 or end % 2048 == 0 or end == n_samples:
                print(f"[*] Qdrant k-NN ilerleme: {end}/{n_samples}")

        print("[*] Qdrant k-NN tamamlandı:", knn_indices.shape)
        return knn_indices

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

        knn_indices = self._get_qdrant_knn_indices(makaleler, vektorler_np, batch_size=256)

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
            # Öncelik belirleme (Hem hiyerarşik derinlik hem de bileşik risk skoruna göre)
            if ortak_derinlik == 0 and risk_skoru >= 0.70:
                oncelik = "KRİTİK (Farklı Ana Disiplin)"
            elif risk_skoru >= 0.70:
                oncelik = "YÜKSEK"
            elif risk_skoru >= 0.40:
                oncelik = "ORTA"
            else:
                oncelik = "DÜŞÜK"

            sonuc_listesi.append({
                "external_id": ext_id,
                "baslik": baslik,
                "ozet": m.get("ozet", ""),  # <-- Özeti buraya ekliyoruz 
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

        # --- TÜM MAKALELERİN SKORLARINI DOĞRU KÖK DİZİNE KAYDETME ---
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        results_dir = os.path.join(base_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        tum_makaleler_path = os.path.join(results_dir, "hdbscan_tum_makaleler.csv")
        df.to_csv(tum_makaleler_path, index=False, encoding="utf-8-sig")
        print(f"[*] Tüm makalelerin skorları kaydedildi: {tum_makaleler_path}")
        # ------------------------------------------------------------

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