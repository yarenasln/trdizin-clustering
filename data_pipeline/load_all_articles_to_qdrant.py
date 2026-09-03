import os
import sys
import csv
import json
import time
from collections import defaultdict
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, UpdateStatus

# ============================================================
# 1. YAPILANDIRMA
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Qdrant Bağlantı Bilgileri (REST üzerinden çalışan ayarlar)
QDRANT_URL = os.getenv("QDRANT_URL", "https://qdrant.ulakbim.gov.tr")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "trdizin_articles")
VECTOR_NAME = "mpnet_v1"

# Ingestion Ayarları
BATCH_SIZE = 512
EXPECTED_TOTAL_ARTICLES = 20902
TOLERANCE_ATOL = 1e-5

# Dosya Yolları
ARTICLE_FILE = os.path.join(BASE_DIR, "data", "balanced_articles.csv")
SUBJECT_FILE = os.path.join(BASE_DIR, "data", "article_subjects.csv")
INDEX_FILE = os.path.join(BASE_DIR, "embeddings", "article_embedding_index.csv")
EMBEDDING_FILE = os.path.join(BASE_DIR, "embeddings", "mpnet_multilingual_embeddings.npy")


def parse_keywords(raw_keywords: str) -> list[str]:
    """Keywords metnini güvenli şekilde listeye ayrıştırır."""
    if not raw_keywords:
        return []
    raw = raw_keywords.strip()
    try:
        parsed = json.loads(raw.replace("'", '"'))
        if isinstance(parsed, list):
            return [str(k).strip() for k in parsed if str(k).strip()]
    except Exception:
        pass
    cleaned = raw.strip("[]").split(",")
    return [k.strip().strip("'\"") for k in cleaned if k.strip().strip("'\"")]


def main():
    start_total_time = time.time()
    print("=" * 90)
    print("TR DİZİN - TÜM MAKALELERİ QDRANT'A YÜKLEME (PRODUCTION INGESTION)")
    print(f"Hedef Collection : {COLLECTION_NAME}")
    print(f"Named Vector      : {VECTOR_NAME} (768-dim)")
    print(f"Batch Size        : {BATCH_SIZE}")
    print(f"Qdrant URL        : {QDRANT_URL}")
    print("=" * 90)

    # ============================================================
    # 2. KAYNAK DOSYALARIN VE İNDEKSİN HAZIRLANMASI
    # ============================================================
    print("\n[1/5] Kaynak dosyalar ve önbellekler yükleniyor...")

    # A) article_embedding_index.csv -> {external_id: embedding_row}
    print("  -> article_embedding_index.csv okunuyor...")
    index_map = {}
    with open(INDEX_FILE, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            eid = row.get("external_id", "").strip()
            if eid:
                index_map[eid] = int(row["embedding_row"].strip())
    print(f"     Toplam {len(index_map):,} makale index kaydı yüklendi.")

    # B) article_subjects.csv -> {external_id: subject_data}
    print("  -> article_subjects.csv gruplanıyor...")
    subjects_map = defaultdict(lambda: {
        "subject_ids": [],
        "subject_names": [],
        "subject_fullnames": [],
        "root_names": set(),
    })
    with open(SUBJECT_FILE, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            eid = row.get("external_id", "").strip()
            if eid:
                s = subjects_map[eid]
                if row.get("subject_id"):
                    s["subject_ids"].append(int(row["subject_id"].strip()))
                if row.get("subject_name"):
                    s["subject_names"].append(row["subject_name"].strip())
                if row.get("subject_fullname"):
                    s["subject_fullnames"].append(row["subject_fullname"].strip())
                if row.get("root_name"):
                    s["root_names"].add(row["root_name"].strip())
    print(f"     Toplam {len(subjects_map):,} makaleye ait subject kayıtları hazırlandı.")

    # C) balanced_articles.csv okunuyor
    print("  -> balanced_articles.csv okunuyor...")
    all_articles = []
    with open(ARTICLE_FILE, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            eid = row.get("external_id", "").strip()
            if eid:
                all_articles.append(row)

    total_articles = len(all_articles)
    print(f"     Toplam {total_articles:,} makale bulundu.")
    if total_articles != EXPECTED_TOTAL_ARTICLES:
        print(f"     [UYARI] Beklenen makale sayısı: {EXPECTED_TOTAL_ARTICLES}, okunan: {total_articles}")

    # D) .npy dosyasını bellek dostu mmap olarak açma (RAM'i gereksiz şişirmez)
    print("  -> mpnet_multilingual_embeddings.npy mmap modunda açılıyor...")
    embeddings_mmap = np.load(EMBEDDING_FILE, mmap_mode="r")
    print(f"     Embedding matris boyutu: {embeddings_mmap.shape}, dtype: {embeddings_mmap.dtype}")

    # ============================================================
    # 3. QDRANT BAĞLANTISI
    # ============================================================
    print("\n[2/5] Qdrant istemcisi başlatılıyor...")
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        port=None,
        prefer_grpc=False,
        check_compatibility=False,
        timeout=120,
    )

    # ============================================================
    # 4. BATCH'LER HALİNDE UPSERT İŞLEMİ
    # ============================================================
    num_batches = (total_articles + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"\n[3/5] Qdrant'a yükleme başlıyor ({num_batches} batch)...")

    for batch_idx in range(num_batches):
        batch_start_time = time.time()
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, total_articles)
        batch_articles = all_articles[start_idx:end_idx]

        points_to_upsert = []
        for art in batch_articles:
            eid = art["external_id"].strip()
            if eid not in index_map:
                print(f"\n[HATA] external_id = {eid} article_embedding_index.csv içinde bulunamadı!")
                sys.exit(1)

            row_idx = index_map[eid]
            vector = embeddings_mmap[row_idx].tolist()

            subjs = subjects_map[eid]
            year_val = art.get("publication_year", "").strip()
            pub_year = int(year_val) if year_val.isdigit() else None

            # Payload (embedding_text ve UMAP koordinatları kesinlikle konulmaz)
            payload = {
                "external_id": int(eid),
                "doi": art.get("doi", "").strip(),
                "publication_year": pub_year,
                "publication_type": art.get("publication_type", "").strip(),
                "language": art.get("language", "").strip(),
                "title": art.get("title", "").strip(),
                "abstract": art.get("abstract", "").strip(),
                "keywords": parse_keywords(art.get("keywords", "")),
                "subject_ids": subjs["subject_ids"],
                "subject_names": subjs["subject_names"],
                "subject_fullnames": subjs["subject_fullnames"],
                "root_names": sorted(list(subjs["root_names"])),
            }

            points_to_upsert.append(
                PointStruct(
                    id=int(eid),
                    vector={VECTOR_NAME: vector},
                    payload=payload,
                )
            )

        # Qdrant'a upsert çağrısı
        try:
            upsert_result = client.upsert(
                collection_name=COLLECTION_NAME,
                points=points_to_upsert,
                wait=True,
            )

            if upsert_result.status != UpdateStatus.COMPLETED:
                print(
                    f"\n[HATA] Batch {batch_idx + 1}/{num_batches} beklenen COMPLETED durumunu dönmedi! "
                    f"Durum: {upsert_result.status}"
                )
                sys.exit(1)

            elapsed_batch = time.time() - batch_start_time
            print(
                f"Batch {batch_idx + 1:>2}/{num_batches} | "
                f"Articles {start_idx + 1:>5}-{end_idx:<5} | "
                f"Upsert: COMPLETED ({elapsed_batch:.2f}s)"
            )

        except Exception as e:
            print(f"\n[KRİTİK HATA] Batch {batch_idx + 1}/{num_batches} yüklenirken istisna oluştu!")
            print(f"  -> Makale Aralığı: {start_idx + 1} - {end_idx}")
            print(f"  -> External ID Aralığı: {batch_articles[0]['external_id']} - {batch_articles[-1]['external_id']}")
            print(f"  -> Hata Detayı: {e}")
            sys.exit(1)

    print(f"\n-> Tüm {total_articles:,} makale başarıyla Qdrant'a gönderildi.")

    # ============================================================
    # 5. YÜKLEME SONRASI DOĞRULAMA (POST-INGESTION VERIFICATION)
    # ============================================================
    print("\n[4/5] Yükleme sonrası kapsamlı doğrulama yapılıyor...")
    all_verification_passed = True

    # A) Point Count Kontrolü
    print("  -> Collection point sayısı denetleniyor...")
    collection_info = client.get_collection(COLLECTION_NAME)
    actual_count = collection_info.points_count
    print(f"     Qdrant Point Sayısı: {actual_count:,} (Beklenen: {total_articles:,})")

    if actual_count != total_articles:
        print(f"     [HATA] Point sayısı uyuşmuyor! Beklenen: {total_articles}, Qdrant: {actual_count}")
        all_verification_passed = False
    else:
        print("     [OK] Collection point sayısı tam olarak 20.902 ile eşleşti.")

    # B) Kaynak ve Qdrant ID Seti Karşılaştırması (Scroll üzerinden)
    print("  -> Qdrant'taki tüm Point ID'leri scroll ile taranıyor...")
    source_id_set = {int(art["external_id"].strip()) for art in all_articles}
    qdrant_id_set = set()
    next_page_offset = None

    while True:
        records, next_page_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=2000,
            offset=next_page_offset,
            with_payload=False,
            with_vectors=False,
        )
        for rec in records:
            qdrant_id_set.add(rec.id)
        if next_page_offset is None:
            break

    print(f"     Scroll ile taranan benzersiz ID sayısı: {len(qdrant_id_set):,}")

    missing_ids = source_id_set - qdrant_id_set
    extra_ids = qdrant_id_set - source_id_set

    if missing_ids:
        print(f"     [HATA] Qdrant'ta eksik {len(missing_ids)} ID tespit edildi: {list(missing_ids)[:10]}...")
        all_verification_passed = False
    if extra_ids:
        print(f"     [HATA] Qdrant'ta fazla {len(extra_ids)} ID tespit edildi: {list(extra_ids)[:10]}...")
        all_verification_passed = False

    if not missing_ids and not extra_ids:
        print("     [OK] Kaynak external_id seti ile Qdrant point ID seti BİREBİR AYNIDIR.")

    # C) 10 Farklı Makale İçin Vektör ve Payload Bütünlüğü Testi
    print("\n[5/5] 10 Farklı Makale İçin Vektör ve Payload Bütünlüğü Test Ediliyor...")
    # Veri setine dengeli yayılmış 10 makale indeksi
    sample_indices = [
        0,
        2000,
        4000,
        6000,
        8000,
        10000,
        12000,
        14000,
        17000,
        total_articles - 1,
    ]

    sample_eids = [int(all_articles[i]["external_id"].strip()) for i in sample_indices]
    sample_retrieved = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=sample_eids,
        with_payload=True,
        with_vectors=True,
    )
    sample_retrieved_map = {p.id: p for p in sample_retrieved}

    print("-" * 105)
    print(f"{'Idx':<6} | {'external_id':<12} | {'Shape':<8} | {'Dtype':<8} | {'max_abs_diff':<15} | {'array_equal':<12} | {'Cosine Sim':<14} | {'Payload':<8} | {'Durum'}")
    print("-" * 105)

    for idx in sample_indices:
        eid_str = all_articles[idx]["external_id"].strip()
        eid_int = int(eid_str)
        row_idx = index_map[eid_str]

        # Yerel Vektör
        vec_local = np.array(embeddings_mmap[row_idx], dtype=np.float32)

        point = sample_retrieved_map.get(eid_int)
        if not point or not point.vector:
            print(f"{idx:<6} | {eid_str:<12} | {'HATA':<8} | {'HATA':<8} | {'KAYIP':<15} | {'False':<12} | {'0.0':<14} | {'HATA':<8} | BAŞARISIZ")
            all_verification_passed = False
            continue

        raw_vec = point.vector
        vec_qdrant_list = raw_vec.get(VECTOR_NAME) if isinstance(raw_vec, dict) else raw_vec
        vec_qdrant = np.array(vec_qdrant_list, dtype=np.float32)

        # Vektör Metrikleri
        abs_diff = np.abs(vec_local - vec_qdrant)
        max_abs = float(np.max(abs_diff))
        is_eq = bool(np.array_equal(vec_local, vec_qdrant))
        cos_sim = float(np.dot(vec_local, vec_qdrant) / (np.linalg.norm(vec_local) * np.linalg.norm(vec_qdrant)))

        # Payload Doğrulama
        payload = point.payload or {}
        has_required_fields = all(
            k in payload
            for k in [
                "external_id", "doi", "title", "abstract", "publication_year",
                "language", "subject_ids", "subject_names", "subject_fullnames", "root_names"
            ]
        )
        expected_subjs_count = len(subjects_map[eid_str]["subject_ids"])
        actual_subjs_count = len(payload.get("subject_ids", []))
        payload_ok = has_required_fields and (expected_subjs_count == actual_subjs_count)

        status = "BAŞARILI" if ((is_eq or max_abs <= TOLERANCE_ATOL) and payload_ok) else "BAŞARISIZ"
        if status == "BAŞARISIZ":
            all_verification_passed = False

        print(
            f"{idx:<6} | {eid_str:<12} | {str(vec_qdrant.shape):<8} | {str(vec_qdrant.dtype):<8} | "
            f"{max_abs:<15.2e} | {str(is_eq):<12} | {cos_sim:<14.8f} | {'OK' if payload_ok else 'HATA':<8} | {status}"
        )

    print("-" * 105)

    # ============================================================
    # NİHAİ RAPOR
    # ============================================================
    total_elapsed = time.time() - start_total_time
    print("\n" + "=" * 90)
    if all_verification_passed:
        print(">>> TÜM VERİ YÜKLEME BAŞARILI <<<")
        print(f"20.902 makalenin tamamı başarıyla yüklendi ve doğrulandı. (Toplam Süre: {total_elapsed / 60:.2f} dakika)")
    else:
        print(">>> TÜM VERİ YÜKLEME BAŞARISIZ <<<")
        print("Doğrulama kontrollerinde tutarsızlık tespit edildi. Yukarıdaki hata çıktılarını inceleyiniz.")
    print("=" * 90)


if __name__ == "__main__":
    main()
