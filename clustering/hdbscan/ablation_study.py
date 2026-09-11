#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TR Dizin Konu Anomalisi Tespiti - Ablation Çalışması (Ablation Study)
====================================================================
Bu script, üretim kodlarına (outlier_detector.py, run_hdbscan_pipeline.py, app.py)
dokunmadan, mevcut "results/hdbscan_tum_makaleler.csv" (20.902 makale) önbelleğini
kullanarak aşağıdaki bileşen kombinasyonlarını izole eder ve karşılaştırır:

- Varyant A: Taksonomi Benzerliği Baseline (k-NN ve GLOSH yok)
- Varyant B: Taksonomi Benzerliği + GLOSH (k-NN yok)
- Varyant C: Taksonomi Benzerliği + k-NN (GLOSH yok)
- Varyant D: Tam Sistem (Mevcut Taksonomi + GLOSH + k-NN)

Hiçbir eşik değeri optimize edilmez (tüm eşikler dondurulmuştur).
Hiçbir varyantta ara karar sütunu (df["supheli_mi"]) kullanılmaz; her varyant
kendi karar kuralını sıfırdan ham öznitelik sütunlarından üretir.
"""

import os
import sys
import pandas as pd
import numpy as np

# Konsol utf-8 desteği
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def calculate_knn_onay(row, taxonomy_paths):
    """
    outlier_detector.py içerisindeki knn_onayliyor_mu karar mantığının
    özgün kuralına birebir sadık kalarak sıfırdan türetilmesi.
    """
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
        else:
            return 0
    else:
        return 1


def run_ablation_study():
    # 1. Dizin ve Dosya Yolları
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    tum_makaleler_path = os.path.join(base_dir, "results", "hdbscan_tum_makaleler.csv")
    subjects_path = os.path.join(base_dir, "data", "article_subjects.csv")
    audit_path = os.path.join(base_dir, "data", "manuel_dogrulama_referans.csv")
    out_dir = os.path.join(base_dir, "results", "ablation")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 85)
    print("TR DİZİN KONU ANOMALİSİ TESPİTİ - BİLEŞEN ABLATION ÇALIŞMASI")
    print("=" * 85)
    print(f"[*] Tüm Makaleler Verisi   : {tum_makaleler_path}")
    print(f"[*] Taksonomi Yolları      : {subjects_path}")
    print(f"[*] Manuel Denetim Havuzu  : {audit_path}")
    print(f"[*] Çıktı Dizini           : {out_dir}")

    # 2. Verileri Yükle
    df = pd.read_csv(tum_makaleler_path)
    df_subjects = pd.read_csv(subjects_path)
    taxonomy_paths = df_subjects["subject_fullname"].dropna().unique().tolist()
    print(f"[*] Toplam {len(df):,} makale ve {len(taxonomy_paths)} tekil taksonomi yolu yüklendi.")

    # Manuel denetim referansını yükle
    if os.path.exists(audit_path):
        df_audit = pd.read_csv(audit_path)
    else:
        fallback_p = "C:/Users/pc_X/Downloads/manuel_dogrulama_master_389_TEMIZ (1).xlsx"
        df_audit = pd.read_excel(fallback_p)
        df_audit.to_csv(audit_path, index=False, encoding="utf-8-sig")

    # Manuel karar sözlüğü
    # Karar: 'TP-1', 'TP-2', 'FP-1' (FP-1 -> FP)
    audit_dict = dict(zip(df_audit["external_id"].astype(str), df_audit["Karar"]))
    audit_gerekce = dict(zip(df_audit["external_id"].astype(str), df_audit.get("Gerekce", "")))
    audit_ids = set(df_audit["external_id"].astype(str))

    total_tp1 = sum(1 for k in audit_dict.values() if k == "TP-1")
    total_tp2 = sum(1 for k in audit_dict.values() if k == "TP-2")
    total_fp = sum(1 for k in audit_dict.values() if k == "FP-1")
    print(f"[*] Denetim Havuzu Boyutu: {len(df_audit)} (TP-1: {total_tp1}, TP-2: {total_tp2}, FP: {total_fp})")

    # 3. knn_onayliyor_mu Koşulunu Sıfırdan Hesapla
    print("[*] 'knn_onayliyor_mu' koşulu taksonomi hiyerarşisi üzerinden türetiliyor...")
    df["knn_onay"] = df.apply(lambda r: calculate_knn_onay(r, taxonomy_paths), axis=1)

    # 4. Ortak Filtre ve Maskeleri Tanımla
    valid_baslik = (
        (~df["baslik"].astype(str).str.strip().isin(["", "-", "None", "nan"]))
        & (df["baslik"].astype(str).str.strip().str.len() > 3)
    )

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
    mask_knn = mask_ana_disiplin | mask_alt_alan

    # 5. A, B, C, D Varyantlarını Sıfırdan Hesapla
    print("[*] Varyant kararları temel özniteliklerden sıfırdan hesaplanıyor...")

    # Varyant A: Taksonomi Benzerliği Baseline (k-NN ve GLOSH yok)
    supheli_A = (df["ortak_agac_derinligi"] <= 1) & (df["label_sim_fark"] > 0.08)
    final_A = supheli_A & (df["label_sim_fark"] >= 0.09) & valid_baslik

    # Varyant B: Taksonomi Benzerliği + GLOSH (k-NN yok)
    supheli_B = (df["ortak_agac_derinligi"] <= 1) & (df["label_sim_fark"] > 0.08) & (df["glosh_skoru"] > 0.70)
    final_B = supheli_B & (df["label_sim_fark"] >= 0.09) & valid_baslik

    # Varyant C: Taksonomi Benzerliği + k-NN (GLOSH yok)
    supheli_C = (
        (df["ortak_agac_derinligi"] <= 1)
        & (df["knn_impurity"] >= 0.50)
        & (df["label_sim_fark"] > 0.08)
        & (df["knn_onay"] == 1)
        & (df["knn_baskinlik"] >= 0.30)
    )
    final_C = supheli_C & (df["label_sim_fark"] >= 0.09) & mask_knn & valid_baslik

    # Varyant D: Tam Sistem (Mevcut Sistem: Taksonomi + GLOSH + k-NN)
    supheli_D = (
        (df["ortak_agac_derinligi"] <= 1)
        & (df["knn_impurity"] >= 0.50)
        & (df["label_sim_fark"] > 0.08)
        & (df["knn_onay"] == 1)
        & ((df["knn_baskinlik"] >= 0.30) | (df["glosh_skoru"] > 0.70))
    )
    final_D = supheli_D & (df["label_sim_fark"] >= 0.09) & mask_knn & valid_baslik

    # D varyantının mevcut hdbscan_anomaliler.csv ile tam örtüşmesini doğrula
    d_flagged_ids = set(df[final_D]["external_id"].astype(str))
    anom_file = os.path.join(base_dir, "results", "hdbscan_anomaliler.csv")
    if os.path.exists(anom_file):
        df_existing_anom = pd.read_csv(anom_file)
        existing_anom_ids = set(df_existing_anom["external_id"].astype(str))
        diff_d = d_flagged_ids ^ existing_anom_ids
        print(f"[*] Doğrulama: Sıfırdan türetilen D ile hdbscan_anomaliler.csv uyuşmazlık sayısı: {len(diff_d)}")
        assert len(diff_d) == 0, f"D varyantı hdbscan_anomaliler.csv ile uyuşmuyor! Fark: {diff_d}"

    # 6. Metrikleri Hesapla
    variants = [
        ("A", "Taxonomy Baseline (No k-NN, No GLOSH)", supheli_A, final_A),
        ("B", "Taxonomy + GLOSH (No k-NN)", supheli_B, final_B),
        ("C", "Taxonomy + k-NN (No GLOSH)", supheli_C, final_C),
        ("D", "Full System (Taxonomy + GLOSH + k-NN)", supheli_D, final_D),
    ]

    summary_rows = []
    variant_id_sets = {}

    for name, desc, s_mask, f_mask in variants:
        f_df = df[f_mask]
        flagged_ids = set(f_df["external_id"].astype(str))
        variant_id_sets[name] = flagged_ids

        n_flagged = len(flagged_ids)
        intersect_ids = flagged_ids & audit_ids
        n_intersect = len(intersect_ids)
        n_unaudited = n_flagged - n_intersect  # N_un-audited = N_flagged - N_intersect

        labels = [audit_dict.get(ext_id) for ext_id in intersect_ids]
        tp1 = sum(1 for l in labels if l == "TP-1")
        tp2 = sum(1 for l in labels if l == "TP-2")
        tp = tp1 + tp2
        fp = sum(1 for l in labels if l == "FP-1")

        prec_audit = tp / n_intersect if n_intersect > 0 else 0.0
        tp1_cov = tp1 / total_tp1 if total_tp1 > 0 else 0.0
        tp2_cov = tp2 / total_tp2 if total_tp2 > 0 else 0.0

        jaccard_d = len(flagged_ids & d_flagged_ids) / len(flagged_ids | d_flagged_ids)
        lost_d = len(d_flagged_ids - flagged_ids)
        new_d = len(flagged_ids - d_flagged_ids)

        summary_rows.append({
            "Variant": name,
            "Description": desc,
            "N_flagged": n_flagged,
            "N_intersect": n_intersect,
            "N_un-audited": n_unaudited,
            "TP_detected": tp,
            "TP1_detected": tp1,
            "TP2_detected": tp2,
            "FP_detected": fp,
            "Precision_Audit": round(prec_audit, 4),
            "TP1_coverage": round(tp1_cov, 4),
            "TP2_coverage": round(tp2_cov, 4),
            "Jaccard_with_D": round(jaccard_d, 4),
            "Lost_from_D": lost_d,
            "New_vs_D": new_d
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_dir, "ablation_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n[*] Ablation özeti kaydedildi: {summary_path}")

    # 7. D \ C ve C \ D Kümelerini Analiz Et ve Kaydet
    c_flagged_ids = variant_id_sets["C"]
    d_minus_c_ids = d_flagged_ids - c_flagged_ids
    c_minus_d_ids = c_flagged_ids - d_flagged_ids

    print(f"\n[*] Küme Karşılaştırması:")
    print(f"    |D \\ C| (GLOSH'un C'ye göre eklediği adaylar) : {len(d_minus_c_ids)}")
    print(f"    |C \\ D| (C'de olup D'de olmayan adaylar)       : {len(c_minus_d_ids)}")

    # D \ C detay dosyası
    d_minus_c_df = df[df["external_id"].astype(str).isin(d_minus_c_ids)].copy()
    d_minus_c_df["Manuel_Karar"] = d_minus_c_df["external_id"].astype(str).map(audit_dict)
    d_minus_c_df["Manuel_Gerekce"] = d_minus_c_df["external_id"].astype(str).map(audit_gerekce)

    d_minus_c_path = os.path.join(out_dir, "glosh_added_candidates_D_minus_C.csv")
    d_minus_c_cols = [
        "external_id", "baslik", "mevcut_kategori", "oneri_kategori",
        "knn_oneri", "knn_baskinlik", "glosh_skoru", "ortak_agac_derinligi",
        "Manuel_Karar", "Manuel_Gerekce"
    ]
    d_minus_c_df[d_minus_c_cols].to_csv(d_minus_c_path, index=False, encoding="utf-8-sig")
    print(f"[*] D \\ C (GLOSH eklenen adaylar) kaydedildi: {d_minus_c_path}")

    # 8. 389 Denetim Makalesinin Varyantlar Bazında Tespiti
    audit_eval_df = df_audit[["external_id", "baslik", "Karar", "mevcut_kategori", "oneri_kategori", "ortak_agac_derinligi"]].copy()
    audit_eval_df["external_id"] = audit_eval_df["external_id"].astype(str)
    audit_eval_df["flagged_in_A"] = audit_eval_df["external_id"].isin(variant_id_sets["A"])
    audit_eval_df["flagged_in_B"] = audit_eval_df["external_id"].isin(variant_id_sets["B"])
    audit_eval_df["flagged_in_C"] = audit_eval_df["external_id"].isin(variant_id_sets["C"])
    audit_eval_df["flagged_in_D"] = audit_eval_df["external_id"].isin(variant_id_sets["D"])

    audit_eval_path = os.path.join(out_dir, "ablation_audit_intersections.csv")
    audit_eval_df.to_csv(audit_eval_path, index=False, encoding="utf-8-sig")
    print(f"[*] Denetim havuzu varyant kesişimleri kaydedildi: {audit_eval_path}")

    # 9. 20.902 Makalenin Tüm Varyant Bayrakları
    full_export_df = df[["external_id", "baslik", "mevcut_kategori", "oneri_kategori", "ortak_agac_derinligi", "label_sim_fark", "glosh_skoru", "knn_baskinlik", "knn_impurity"]].copy()
    full_export_df["external_id"] = full_export_df["external_id"].astype(str)
    full_export_df["flagged_A"] = final_A.astype(int)
    full_export_df["flagged_B"] = final_B.astype(int)
    full_export_df["flagged_C"] = final_C.astype(int)
    full_export_df["flagged_D"] = final_D.astype(int)
    full_export_df["is_in_audit"] = full_export_df["external_id"].isin(audit_ids).astype(int)
    full_export_df["audit_karar"] = full_export_df["external_id"].map(audit_dict).fillna("")

    all_anomalies_path = os.path.join(out_dir, "ablation_anomalies_all_variants.csv")
    full_export_df.to_csv(all_anomalies_path, index=False, encoding="utf-8-sig")
    print(f"[*] 20,902 makalenin tüm varyant etiketleri kaydedildi: {all_anomalies_path}")

    # 10. Sonuçları Konsola Yazdır
    print("\n" + "=" * 85)
    print("ABLATION STUDY SONUÇLARI")
    print("=" * 85)
    print(summary_df.to_string(index=False))
    print("=" * 85)

    print("\n[*] GLOSH'un mevcut 388 adaylık referans kümesine D tarafından eklenen 4 aday:")
    for _, r in d_minus_c_df.iterrows():
        print(f"  - ID: {r['external_id']} | Karar: {r['Manuel_Karar']} | Başlık: {r['baslik'][:65]}...")
        print(f"    Mevcut: {r['mevcut_kategori']} -> Öneri: {r['oneri_kategori']}")
        print(f"    GLOSH: {r['glosh_skoru']:.3f} | kNN Baskınlık: {r['knn_baskinlik']} | Ortak Derinlik: {r['ortak_agac_derinligi']}")
        print(f"    Gerekçe: {str(r['Manuel_Gerekce'])[:100]}...\n")


if __name__ == "__main__":
    run_ablation_study()
