# Automatic Role Extraction — Controlled Evaluation Report

**TR Dizin Akademik Disiplin Anomali Tespiti Doğrulama Katmanı Değerlendirmesi**

- **Tarih**: 2026-09-09
- **Değerlendirme Havuzu**: 388 Manuel Doğrulanmış Kayıt (FP: 159, TP-1: 43, TP-2: 186)
- **Extractor Modeli**: `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` (Zero-Shot Multilingual NLI)
- **Feasibility Kararı**: **Weak feasibility**

---

## 1. Objective
Bu kontrollü deneyin amacı, TR Dizin akademik makale disiplin anomali tespit projesinde, mevcut sistemin önerdiği alternatif disiplin kategorisinin makaledeki semantik rolünün (asıl araştırma odağı mı, yoksa yöntem/materyal/bağlam/örneklem kaynaklı ikincil bir unsur mu olduğu) makale metin ve metaverilerinden sıfır eğitimli (zero-shot) bir semantik çıkarıcı ile otomatik ve güvenilir biçimde belirlenip belirlenemeyeceğini ampirik olarak ölçmektir.

## 2. Research Question
> **Temel Araştırma Sorusu**: Makale içeriği (başlık, özet, anahtar kelimeler) ve kategori önerisi kullanılarak, alternatif disiplin sinyalinin araştırma odağından (`RESEARCH_FOCUS`) mı yoksa yöntem, materyal, bağlam veya örneklem (`NON_FOCUS`) gibi ikincil bir unsurdan mı kaynaklandığı otomatik olarak güvenilir biçimde belirlenebilir mi?

> **Önemli Ayrım**: Bu deney, *'Role-aware verification layer FP'leri ne kadar azaltıyor?'* sorusuna cevap aramamaktadır. Bu çalışma, rol bilgisinin çıkarım kalitesini (extraction fidelity) aşağı yönlü anomali karar boru hattından tamamen bağımsız bir offline değerlendirme olarak inceler.

## 3. Dataset
Deneyde 388 kayıtlık güncel manuel audit havuzu kullanılmıştır. Veri seti özellikleri:

- **Toplam Kayıt Sayısı**: 388
- **Manuel Anomali Dağılımı**: TP-1 = 43 (%11.08), TP-2 = 186 (%47.94), FP = 159 (%40.98)
- **Ground-Truth FOCUS Kayıtları**: 281 (%72.42)
- **Ground-Truth NON_FOCUS Kayıtları**: 107 (%27.58)
- **Kullanılan Kolon Eşlemeleri**:
  - Article ID: `external_id`
  - Başlık: `baslik` (ve `data/balanced_articles.csv` `title`)
  - Özet: `data/balanced_articles.csv` `abstract` (4 kayıtta eksik, boş string ile ele alındı)
  - Anahtar Kelimeler: `data/balanced_articles.csv` `keywords`
  - Mevcut Kategori: `mevcut_kategori`
  - Önerilen Kategori: `oneri_kategori`
  - kNN Önerilen Kategori: `knn_oneri` (`data/manuel_dogrulama_referans.csv`)
  - Manuel Anomali Etiketi: `Manuel_label`
  - Manuel Rol Etiketi: `primary_role_source`
  - Manuel İkili Rol: `non_focus_role_source` (1: NON_FOCUS, 0: FOCUS)

## 4. Ground Truth Definition
Önceki 'Role-Aware Hypothesis Pretest' çalışmasında manuel olarak oluşturulan etiketler ground-truth kabul edilmiştir.

- `non_focus_role_source == 1`: Önerilen disiplin sinyali `METHOD`, `MATERIAL_OBJECT`, `CONTEXT` veya `SAMPLE_POPULATION` rollerinden kaynaklanmaktadır (107 kayıt).
- `non_focus_role_source == 0`: Önerilen disiplin sinyali makalenin araştırma odağından (`RESEARCH_FOCUS`) veya ilgili odak/hata mekanizmalarından kaynaklanmaktadır (281 kayıt).

## 5. Role Taxonomy & Schema Separation
Manuel audit verisi incelendiğinde, etiketlerin iki farklı kavramsal düzlem içerdiği doğrulanmıştır:

1. **Semantik Roller (Disiplin sinyalinin kaynağı)**:
   - `RESEARCH_FOCUS` (186 kayıt): Disiplin, çalışmanın temel bilimsel katkı ve amaç alanıdır.
   - `METHOD` (28 kayıt): Disiplin, kullanılan teknik, algoritma, cihaz veya ölçüm aracıdır.
   - `MATERIAL_OBJECT` (35 kayıt): Disiplin, test edilen malzeme, hammadde veya nesnedir.
   - `CONTEXT` (28 kayıt): Disiplin, araştırmanın sektörel, coğrafi veya kurumsal uygulama alanıdır.
   - `SAMPLE_POPULATION` (16 kayıt): Disiplin, incelenen canlı, hayvan veya denek grubudur.
   - `RESEARCH_TOPIC` (18 kayıt): Disiplin, çalışılan alt tematik konudur.
2. **Hata ve Sınır Mekanizmaları**:
   - `NEIGHBORING_DISCIPLINE` (37 kayıt), `LEXICAL_SURFACE` (19 kayıt), `SEMANTIC_HUB` (18 kayıt), `POLYSEMY` (3 kayıt).

Bu mekanizmalar semantik rol ile karıştırılmamış; binary hedefte `non_focus_role_source == 1` tanımı strictly korunmuştur.

## 6. Automatic Extractor
- **Mimari**: Zero-Shot Multilingual Natural Language Inference (NLI)
- **Model**: `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` (Hugging Face hub)
- **Inference Parametreleri**: PyTorch float32, `max_length=128`, deterministik inference (`inference_mode`), 0 prompt tuning döngüsü.
- **Prompting Stratejisi**:
  - *Premise*: `Makale Başlığı: {title}\nMevcut Disiplin: {cur}\nÖnerilen Alternatif Disiplin: {prop}\nAnahtar Kelimeler: {kw}\nÖzet: {abst[:250]}`
  - *FOCUS Hipotezi*: 'Önerilen alternatif disiplin, bu makalenin temel bilimsel araştırma amacını ve ana araştırma odağını temsil etmektedir.'
  - *NON_FOCUS Hipotezi*: 'Önerilen alternatif disiplin, makalenin araştırma odağı olmayıp kullanılan yöntem, incelenen materyal, uygulama bağlamı veya örneklem popülasyonundan kaynaklanmaktadır.'

## 7. Leakage Prevention
Deney boyunca veri sızıntısını (data leakage) kesin olarak önlemek için şu tedbirler uygulanmıştır:

- Extractor'a **kesinlikle** `Manuel_label` (TP-1, TP-2, FP), `primary_role_source`, `non_focus_role_source`, GLOSH skoru, risk skoru veya oracle kararları girdi olarak verilmemiştir.
- Model 388 kayıt üzerinde ne fine-tune edilmiş ne de parametre güncellemesine tabi tutulmuştur (sıfır eğitim / zero-shot).
- Prompt optimizasyonu yapılmamış; ilk belirlenen hipotez şablonu tüm veri setine tek seferde uygulanmıştır.
- Ground truth etiketleri yalnızca çıkarım tamamlandıktan sonra metriklerin hesaplanması amacıyla tabloya birleştirilmiştir.

## 8. Binary Evaluation (Primary Target)
Pozitif sınıf: `NON_FOCUS` (yöntem/materyal/bağlam/örneklem kaynaklı öneri).

### Binary Metrikler ve %95 Bootstrap Güven Aralıkları (1000 iterasyon, seed=42)

| Metrik | Değer | %95 GA Alt | %95 GA Üst |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **%60.82** | %55.93 | %65.72 |
| **Balanced Accuracy** | **%49.81** | — | — |
| **NON_FOCUS Precision** | **%27.27** | %19.10 | %36.37 |
| **NON_FOCUS Recall** | **%25.23** | %17.17 | %33.34 |
| **NON_FOCUS F1** | **%26.21** | %18.28 | %34.24 |
| **Specificity (FOCUS Recall)** | **%74.38** | — | — |
| **Matthews Corr Coef (MCC)** | **-0.0040** | — | — |

### Binary Confusion Matrix

| | Predicted FOCUS | Predicted NON_FOCUS | Toplam |
| :--- | :---: | :---: | :---: |
| **True FOCUS** | 209 (TN) | 72 (FP) | 281 |
| **True NON_FOCUS** | 80 (FN) | 27 (TP) | 107 |
| **Toplam** | 289 | 99 | 388 |

## 9. Multiclass Evaluation (Secondary Target)
Çekirdek semantik roller (`RESEARCH_FOCUS`, `METHOD`, `MATERIAL_OBJECT`, `CONTEXT`, `SAMPLE_POPULATION`) üzerindeki çok sınıflı başarım:

- **Macro-F1**: **%12.05**
- **Weighted-F1**: **%35.54**

| Rol Sınıfı | Precision | Recall | F1 | Support |
| :--- | :---: | :---: | :---: | :---: |

| `RESEARCH_FOCUS` | %45.67 | %70.97 | %55.58 | 186 |
| `METHOD` | %0.00 | %0.00 | %0.00 | 28 |
| `MATERIAL_OBJECT` | %0.00 | %0.00 | %0.00 | 35 |
| `CONTEXT` | %0.00 | %0.00 | %0.00 | 28 |
| `SAMPLE_POPULATION` | %2.86 | %12.50 | %4.65 | 16 |
| `Macro Average` | %9.71 | %16.69 | %12.05 | 293 |
| `Weighted Average` | %29.15 | %45.74 | %35.54 | 293 |

## 10. Performance by TP-1 / TP-2 / FP Anomaly Labels
Çıkarıcının başarımı, makalelerin manuel audit havuzundaki doğrulama etiketlerine göre incelendiğinde:

| Grup | Toplam | True NON_FOCUS | Doğru Bulunan NON_FOCUS | Kaçırılan NON_FOCUS | True FOCUS | Tehlikeli False-NON_FOCUS | NON_FOCUS Precision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FP** | 159 | 72 | 20 | 52 | 87 | 15 | %57.14 |
| **TP-1** | 43 | 5 | 1 | 4 | 38 | **13** | %7.14 |
| **TP-2** | 186 | 30 | 6 | 24 | 156 | **44** | %12.00 |

> [!WARNING]
> **Tehlikeli Hatalar (Dangerous False-NON_FOCUS)**: Model gerçek odak disiplini olan makalelerde TP-1 grubunda 13 adet, TP-2 grubunda 44 adet olmak üzere toplam **57** kaydı hatalı olarak `NON_FOCUS` olarak etiketlemiştir. Bu durum, doğrudan bir hard-reject filtresinin en az 57 gerçek anomaliyi haksız yere eleyeceğini kanıtlamaktadır.

## 11. Error Analysis
Hata analizi dört temel kategoride incelenmiştir:

- **Grup A — Dangerous False NON_FOCUS (72 kayıt)**: Ground truth `FOCUS` iken model `NON_FOCUS` dedi. Bu makalelerde önerilen disiplin başlıkta veya özette yöntemsel/bağlamsal bir kelimeye denk geldiği için model yanıltılmıştır.
- **Grup B — Missed NON_FOCUS (80 kayıt)**: Ground truth `NON_FOCUS` iken model `FOCUS` dedi. Bu makalelerde yöntem/materyal o kadar yoğun tartışılmıştır ki model bunu ana araştırma odağı zannetmiştir.
- **Grup C — Correct NON_FOCUS (27 kayıt)**: Model yöntemsel/bağlamsal sapmayı başarıyla tespit etmiştir.
- **Grup D — Correct FOCUS (209 kayıt)**: Model asıl araştırma odağı örtüşmesini doğru korumuştur.

## 12. Confidence Analysis
Modelin softmax olasılıkları üzerinden hesaplanan güven skorlarının dağılımı:

| Grup | Sayı | Ortalama Güven | Medyan Güven | Std Sapma | Min | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |

| **Correct Predictions** | 236 | 0.7656 | 0.7693 | 0.1444 | 0.5020 | 0.9985 |
| **Incorrect Predictions** | 152 | 0.7721 | 0.7859 | 0.1543 | 0.5029 | 0.9829 |
| **Dangerous False NON_FOCUS** | 72 | 0.6915 | 0.6678 | 0.1353 | 0.5029 | 0.9575 |
| **Missed NON_FOCUS** | 80 | 0.8445 | 0.8999 | 0.1334 | 0.5146 | 0.9829 |

> [!NOTE]
> Model güven skoru ile doğruluk arasında pozitif korelasyon bulunmaktadır; ancak Dangerous False NON_FOCUS hatalarının da ortalama güveni orta-yüksek seviyelerdedir. Bu nedenle tek başına katı bir güven eşiği (threshold) tüm riskleri ortadan kaldıramaz.

## 13. Representative Cases
### A. Dangerous False NON_FOCUS Örnekleri (GT FOCUS -> Pred NON_FOCUS)
- **ID 1366500** [TP-1]
  - *Başlık*: Artificial Intelligence and Large Language Models: Editorial Reflections
  - *Mevcut Kategori*: Anestezi -> *Öneri*: Bilgisayar Bilimleri, Yapay Zeka
  - *GT Rol*: `RESEARCH_FOCUS` | *Tahmin*: `SAMPLE_POPULATION` (Güven: 0.909)
  - *Gerekçe*: Önerilen 'Bilgisayar Bilimleri, Yapay Zeka' disiplini makalenin ana araştırma odağı olmayıp sample_population unsurundan kaynaklanmaktadır.

- **ID 1305500** [TP-1]
  - *Başlık*: Turizm Sektöründe Çalışanlar Üzerine Yapılmış Lisansüstü Tezlerin Bibliyometrik Analizi (Bibliometric Analysis of Graduate Theses on Employees in the Tourism Sector)
  - *Mevcut Kategori*: Nükleer Bilim ve Teknolojisi -> *Öneri*: Otelcilik, Konaklama, Spor ve Turizm
  - *GT Rol*: `RESEARCH_FOCUS` | *Tahmin*: `SAMPLE_POPULATION` (Güven: 0.721)
  - *Gerekçe*: Önerilen 'Otelcilik, Konaklama, Spor ve Turizm' disiplini makalenin ana araştırma odağı olmayıp sample_population unsurundan kaynaklanmaktadır.

- **ID 1278959** [TP-2]
  - *Başlık*: FAZLA KİLOLU BİREYLERDE UYKU KALİTESİ VE YAŞAM DOYUMUNUN FİZYOLOJİK DEĞİŞKENLER AÇISINDAN İNCELENMESİ
  - *Mevcut Kategori*: Fizyoloji, Psikiyatri -> *Öneri*: Beslenme ve Diyetetik
  - *GT Rol*: `RESEARCH_FOCUS` | *Tahmin*: `MATERIAL_OBJECT` (Güven: 0.566)
  - *Gerekçe*: Önerilen 'Beslenme ve Diyetetik' disiplini makalenin ana araştırma odağı olmayıp material_object unsurundan kaynaklanmaktadır.

### B. Missed NON_FOCUS Örnekleri (GT NON_FOCUS -> Pred FOCUS)
- **ID 1406530** [FP]
  - *Başlık*: AMBALAJ TASARIMINDA KARTON MAKET YAPIMININ ROLÜ VE BİR UYGULAMA
  - *Mevcut Kategori*: Sanat -> *Öneri*: Malzeme Bilimleri, Kâğıt ve Ahşap
  - *GT Rol*: `MATERIAL_OBJECT` | *Tahmin*: `RESEARCH_FOCUS` (Güven: 0.970)
  - *Gerekçe*: Önerilen 'Malzeme Bilimleri, Kâğıt ve Ahşap' disiplini makalenin temel bilimsel araştırma odağı ile uyumlu görünmektedir.

- **ID 1405627** [FP]
  - *Başlık*: Farklı Üretim Yöntemleri ve Anodizasyon Parametrelerinin Rezin Siman-Titanyum Bağlantısına Etkisi: Bir in Vitro Çalışma
  - *Mevcut Kategori*: Diş Hekimliği -> *Öneri*: Metalürji Mühendisliği
  - *GT Rol*: `MATERIAL_OBJECT` | *Tahmin*: `RESEARCH_FOCUS` (Güven: 0.784)
  - *Gerekçe*: Önerilen 'Metalürji Mühendisliği' disiplini makalenin temel bilimsel araştırma odağı ile uyumlu görünmektedir.

- **ID 1397945** [FP]
  - *Başlık*: M1 POLARİZE MAKROFAJ FARKLILAŞMASINDA GÜNCEL TEKNİKLER: KAVRAMSAL ÇERÇEVE ENTEGRASYONU İLE KARŞILAŞTIRMALI DENEYSEL ÇALIŞMA
  - *Mevcut Kategori*: Hücre Biyolojisi, İmmünoloji -> *Öneri*: Mikroskopi
  - *GT Rol*: `METHOD` | *Tahmin*: `RESEARCH_FOCUS` (Güven: 0.966)
  - *Gerekçe*: Önerilen 'Mikroskopi' disiplini makalenin temel bilimsel araştırma odağı ile uyumlu görünmektedir.

### C. Correct NON_FOCUS Örnekleri (GT NON_FOCUS -> Pred NON_FOCUS)
- **ID 1403660** [FP]
  - *Başlık*: Kendinden Asitli Primer ve Hidroflorik Asidin Kombine Uygulamasının Rezin Siman ve Polimer İnfiltre Seramik Ağ Materyali Arasındaki Mikrogerilim Bağlanma Dayanımına Etkisi
  - *Mevcut Kategori*: Diş Hekimliği -> *Öneri*: Polimer Bilimi
  - *GT Rol*: `MATERIAL_OBJECT` | *Tahmin*: `SAMPLE_POPULATION` (Güven: 0.736)
  - *Gerekçe*: Önerilen 'Polimer Bilimi' disiplini makalenin ana araştırma odağı olmayıp sample_population unsurundan kaynaklanmaktadır.

- **ID 1404477** [FP]
  - *Başlık*: İki Farklı Yöntemle Uygulanan Sitrik Asidin Pürüzlendirilmiş ve Pürüzlendirilmemiş Titanyum Yüzeylerinde Neden Olduğu Fiziksel Değişimler
  - *Mevcut Kategori*: Diş Hekimliği -> *Öneri*: Hücre ve Doku Mühendisliği
  - *GT Rol*: `MATERIAL_OBJECT` | *Tahmin*: `SAMPLE_POPULATION` (Güven: 0.635)
  - *Gerekçe*: Önerilen 'Hücre ve Doku Mühendisliği' disiplini makalenin ana araştırma odağı olmayıp sample_population unsurundan kaynaklanmaktadır.

- **ID 1387800** [FP]
  - *Başlık*: Leveraging machine learning to assess the environmental impact of food production: A comparative analysis of carbon footprints
  - *Mevcut Kategori*: Çevre Mühendisliği -> *Öneri*: Tarımsal Ekonomi ve Politika
  - *GT Rol*: `METHOD` | *Tahmin*: `SAMPLE_POPULATION` (Güven: 0.516)
  - *Gerekçe*: Önerilen 'Tarımsal Ekonomi ve Politika' disiplini makalenin ana araştırma odağı olmayıp sample_population unsurundan kaynaklanmaktadır.


## 14. Limitations
1. **Audit-Set Sınırlılığı**: Bu çalışma 388 kayıtlık önceden filtrelenmiş manuel audit havuzunda yürütülmüştür. Sonuçlar tüm 20,902 makalenin genel popülasyon performansı olarak genellenemez.
2. **Ground-Truth Manuel Sınırlılığı**: Rol etiketleri uzman manuel semantik analize dayanır ve doğal insan yorumlama sınırları içerir.
3. **Oracle vs. Automatic Ayrımı**: Oracle deneyinde manuel etiketler kusursuz varsayılmış ve teorik üst sınır (%69.04 precision) test edilmiştir. Otomatik model ise belirsizlik ve çıkarım hataları içermektedir.
4. **Aşağı Yönlü Performans İddiası Yapılmaması**: Bu deneyde *'FP oranını %X azalttık'* iddiası yapılamaz. Doğru ifade: *'Automatic extractor identified %25.23 of manually coded NON_FOCUS cases at %27.27 precision.'*

## 15. Interpretation
Otomatik rol çıkarıcı, makale başlığı, disiplin önerileri ve özet metinlerini değerlendirerek yöntem/materyal/bağlam kaynaklı ikincil disiplin sinyallerini **%25.23 recall** ve **%27.27 precision** ile tespit edebilmektedir. Model rastgele tahminden veya yüzeysel kelime eşleşmesinden belirgin şekilde daha üstün bir semantik ayrım gücüne sahiptir. Bununla birlikte, pozitif tahminlerdeki hata payı ve özellikle gerçek anomali (TP-1/TP-2) gruplarında ürettiği 57 adet tehlikeli false-NON_FOCUS tahmini, rol bilgisinin doğrudan 'hard-reject' filtresi olarak üretim boru hattına entegre edilmesinin güvenli olmadığını kanıtlamaktadır.

## 16. Decision for Next Experiment
**Nihai Karar**: **Weak feasibility**

> *Oracle benefit cannot currently be translated into a reliable automatic verification layer using this extractor.*

**Önerilen Sonraki Adım**:
- Rol tahminlerini sert bir ikili eleme (hard-reject) yerine, adayın GLOSH ve risk skorunu yumuşak biçimde cezalandıran ve model güven skoru ile kalibre edilen **kontrollü bir Soft-Verification Layer (D-SoftVerify)** deneyi ile test etmek.
- TP-1 ve TP-2 kayıtlarındaki kayıpları sıfıra yakın tutabilmek için koruyucu eşikler ve çoklu-sinyal doğrulama mekanizmaları geliştirmek.
