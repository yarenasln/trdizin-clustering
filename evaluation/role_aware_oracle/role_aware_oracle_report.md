# Role-Aware Verification Layer — Controlled Oracle Experiment Report

**Proje:** TR Dizin Disiplin Sınıflandırma ve Hata Denetimi  
**Deney Adı:** Role-Aware Verification Layer Controlled Oracle Experiment  
**Değerlendirme Havuzu:** 388 Manuel Doğrulanmış Kayıt (TP-1 = 43, TP-2 = 186, FP = 159)  
**Tarih:** 2026-09-09  
**Durum:** Tamamlandı — Çevrimdışı Kontrollü Oracle Fizibilite Deneyi  

---

## 1. Objective (Amaç)

Bu deneyin amacı, mevcut TR Dizin anomali tespit sisteminin (Baseline D) ürettiği 388 kayıtlık manuel doğrulama havuzu üzerinde, aday disiplin sinyalinin makaledeki **yöntem (method), materyal (material), bağlam (context) veya örneklem (sample)** bileşenlerinden kaynaklanıp kaynaklanmadığına ilişkin manuel olarak doğrulanmış rol bilgisini bir "oracle" (mükemmel bilgi) kabul ederek filtreleme yapılması durumunda:
1. Yanlış Pozitif (FP) oranının ne ölçüde düşürülebileceğini,
2. Gerçek kategori hatalarını düzelten TP-1 ve tamamlayıcı disiplinleri yakalayan TP-2 kayıtlarının ne ölçüde korunabileceğini (retention),
3. Doğrudan eleme (hard reject) yaklaşımının taşıdığı aşırı filtreleme (over-filtering) riskini ve teorik üst sınırını

ölçmektir.

Bu aşamada **otomatik rol sınıflandırıcısı geliştirilmemiştir**. Deney, rol-farkındalıklı bir doğrulama katmanının (Role-Aware Verification Layer) potansiyel tavan performansını ortaya koyan bir **fizibilite ve üst sınır (upper-bound)** çalışmasıdır.

---

## 2. Baseline (Referans Sistem)

Referans sistem olarak 388 kayıtlık güncel audit havuzundaki ham üretim çıktısı (Baseline D) esas alınmıştır:

- **Toplam Aday Sayısı (Candidates):** 388
- **TP-1 (Kategori Düzeltmeleri):** 43 (%11.08)
- **TP-2 (Disiplinlerarası Eklemeler):** 186 (%47.94)
- **TP-all (Toplam Doğru Pozitif):** 229 (%59.02)
- **FP (Yanlış Pozitif):** 159 (%40.98)
- **Baseline Precision:** **%59.02** (229 / 388)
- **Baseline TP Retention:** **%100.00** (229 / 229)

---

## 3. Oracle Experiment Definition (Oracle Deneyi Tanımı)

Oracle deneyi, önceki aşamada (`results/role_aware_pretest/role_assignment_388.csv`) her kayıt için bağımsız olarak doğrulanmış semantik rol etiketlerini kusursuz bir kural filtresi olarak simüle eder.

Sistem bir makale için alternatif bir disiplin önerdiğinde (`oneri_kategori`):
- Eğer oracle rol bilgisi bu önerinin makalenin epistemik araştırma odağından (`RESEARCH_FOCUS`) değil; kullanılan yöntem (`METHOD`), incelenen fiziksel malzeme (`MATERIAL_OBJECT`), sektörel bağlam (`CONTEXT`) veya canlı denek popülasyonundan (`SAMPLE_POPULATION`) kaynaklandığını söylüyorsa (`non_focus_role_source == 1`),
- Bu aday sistem tarafından "şüpheli disiplin kayması değil, makale içi periferik bileşen yansımasıdır" gerekçesiyle elenir (reject edilir).

---

## 4. Variants (Test Edilen Varyantlar)

Kullanıcı şartnamesi doğrultusunda aşağıdaki varyantlar test edilmiştir:

1. **D Baseline:** Mevcut üretim sistemi çıktısı (hiçbir filtre uygulanmamış 388 aday).
2. **D-Oracle-1:** `non_focus_role_source == 1` olan tüm adayların doğrudan elendiği temel oracle varyantı:
   ```python
   oracle_reject = df["non_focus_role_source"] == 1
   oracle_D1 = baseline_D & (~oracle_reject)
   ```
3. **D-Oracle-2:** `primary_role_source` kolonu üzerinden güçlü non-focus rollerinin elendiği varyant:
   ```python
   strong_roles = {"METHOD", "MATERIAL_OBJECT", "CONTEXT", "SAMPLE_POPULATION"}
   oracle_D2 = baseline_D & (~df["primary_role_source"].isin(strong_roles))
   ```
   *(Not: Veri setimizde `non_focus_role_source == 1` tanımı tam olarak bu 4 güçlü rolden türetildiği için D-Oracle-2, D-Oracle-1 ile sayısal olarak birebir aynı sonuçları üretmiştir).*
4. **D-Oracle-3:** Alternatif mekanizmanın "role-based" olduğu bağımsız bir kolon üzerinden filtreleme varyantı:
   - **Durum:** **Uygulanamadı**. Veri setinde 388 kaydın tamamını kapsayan bağımsız bir `alternative_mechanism` veya `is_role_based` kolonu mevcut değildir. Şartnamenin *"Eğer böyle bir alan mevcut değilse bu variantı uygulama ve raporda 'uygulanamadı' diye belirt"* talimatına uyulmuştur.
5. **D-Oracle-Exploratory-HighConfidence (Keşifsel Varyant):** Yalnızca sınır vakası olmayan, net ve yüksek güvenli non-focus rollerin elendiği kontrollü varyant:
   ```python
   oracle_reject_exp = (df["non_focus_role_source"] == 1) & (~df["is_boundary_case"])
   oracle_D_exp = baseline_D & (~oracle_reject_exp)
   ```

---

## 5. Results (Genel Performans Karşılaştırma Tablosu)

| Variant | Candidates | TP-1 Retained | TP-2 Retained | TP-all Retained | FP Retained | Precision (%) | FP Reduction (%) | TP-1 Retention (%) | TP-2 Retention (%) | TP-all Retention (%) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **D Baseline** | 388 | 43 | 186 | 229 | 159 | **%59.02** | %0.00 | %100.00 | %100.00 | %100.00 |
| **D-Oracle-1** | 281 | 38 | 156 | 194 | 87 | **%69.04** | **%45.28** | **%88.37** | **%83.87** | **%84.72** |
| **D-Oracle-2** | 281 | 38 | 156 | 194 | 87 | **%69.04** | **%45.28** | **%88.37** | **%83.87** | **%84.72** |
| **D-Oracle-3** | — | — | — | — | — | — | — | — | — | — |
| **D-Oracle-Exp-HighConf** | 341 | 43 | 186 | 229 | 112 | **%67.16** | **%29.56** | **%100.00** | **%100.00** | **%100.00** |

*(Not: D-Oracle-3 alanı veri setinde bulunmadığından uygulanamamıştır).*

---

## 6. TP-1 Retention (Kategori Düzeltmeleri Korunumu)

TP-1 havuzu, TR Dizin'deki mevcut kategorinin ağır biçimde hatalı olduğu ve modelin önerdiği disiplinin bu hatayı düzelttiği 43 kritik kayıttan oluşmaktadır.

- **D-Oracle-1 / D-Oracle-2 Sonucu:** 43 TP-1 kaydının **38'i korunmuş (%88.37)**, **5'i yanlışlıkla elenmiştir (%11.63 kayıp)**.
- **Elenen 5 TP-1 Kaydının Analizi:**
  1. `ID: 1333454` (Akış Destekli MRG ile gıdalarda in vitro sindirim -> `CONTEXT` / in vitro gastrointestinal sindirim ortamı klinik Gastroenteroloji sanılmış; asıl alan Gıda Bilimi).
  2. `ID: 1181471` (Kesirli mertebeden dif. denklemlerin YSA ile çözümü -> `METHOD` / Diferansiyel denklem makalesinde YSA hesaplama yöntemi olarak kullanılmış; model YZ önermiş).
  3. `ID: 1254169` (Türkiye Florası için yeni kayıt: Kafkas Filburnu -> `CONTEXT` / Yabani botanik taksonomisi ziraat/bahçe bağlamı sanılmış).
  4. `ID: 1232595` (Çam ağaçlarında mantar İHA izleme -> `CONTEXT` / Çam ormanı ve fotogrametri ziraat/bahçe bağlamı sanılmış).
  5. `ID: 1196614` (Türkiye briyofit vejetasyonu kontrol listesi -> `CONTEXT` / Karayosunları floristiği ziraat/bahçe bağlamı sanılmış).

**Kritik Çıkarım:** Bu 5 kayıtta modelin önerdiği kategori zaten makalenin gerçek ana disiplini değil, bağlam veya yöntem tuzağıdır; ancak denetçi makalenin mevcut etiketinin çöp olması nedeniyle bu kaydı "TP-1 kategori temizliği/düzeltmesi" olarak işaretlemiştir. Dolayısıyla oracle filtresi bu adayları elediğinde sistemin hatalı kategoriyi düzeltme fırsatı kaçırılmaktadır.

---

## 7. TP-2 Retention (Disiplinlerarası Eklemeler Korunumu)

TP-2 havuzu, makalenin mevcut ana disiplininin doğru olduğu, ancak modelin önerdiği alternatif disiplinin makaleye geçerli bir ikincil/disiplinlerarası etiket sağladığı 186 kayıttan oluşmaktadır.

- **D-Oracle-1 / D-Oracle-2 Sonucu:** 186 TP-2 kaydının **156'sı korunmuş (%83.87)**, **30'u elenmiştir (%16.13 kayıp)**.
- **Elenen 30 TP-2 Kaydının Rol Dağılımı:**
  - `METHOD`: **10 kayıt** (ör. ID `1379499` tekstil kalite kontrolünde istatistiksel analiz; ID `1391339` trafik kazası tahmininde Random Forest; ID `1402334` inşaat harcında optik ışık kürleme).
  - `MATERIAL_OBJECT`: **9 kayıt** (ör. ID `1395166` ve ID `1319566` folklorik halı/dokuma kumaş malzemesi; ID `1395268` arkeolojik seramik kaplar; ID `1404129` diş dolgu rezini).
  - `CONTEXT`: **6 kayıt** (ör. ID `1331610` güneş lekeleri ve orman yangınları; ID `1287758` kent ağaçları ekosistem hizmeti; ID `1385427` temiz enerji yatırımı).
  - `SAMPLE_POPULATION`: **5 kayıt** (ör. ID `1394234` çocukluk aşı tereddütü / Pediatri; ID `1322426` yaşlı bireyler / Geriatri; ID `1432781` afazili hasta yakınları / Aile Çalışmaları).

> [!WARNING]
> **Aşırı Filtreleme (Over-Filtering) Riski:** Non-focus rollerin doğrudan katı bir filtre (hard reject) ile elenmesi, **her 6 geçerli disiplinlerarası öneriden 1'inin (%16.1) kaybolmasına** yol açmaktadır. Bu bulgu, yöntemin veya materyalin makaleye özgün bir disiplinlerarası boyut kattığı durumlarda katı elemenin tehlikeli olduğunu kanıtlar.

---

## 8. FP Reduction (Yanlış Pozitif Azaltımı)

Baseline D sisteminde denetçilerin reddettiği 159 FP adayının:
- **72 adedi (%45.28)** D-Oracle-1 tarafından başarıyla elenmiştir.
- Geriye kalan **87 FP adedi (%54.72)** sistemde kalmaya devam etmiştir.

### Filtre Sonrası Kalan 87 FP Kaydının Neden Elenemediği:
Bu 87 kayıt non-focus rollerden (yöntem, materyal, bağlam, örneklem) kaynaklanmamaktadır:
- `NEIGHBORING_DISCIPLINE` (Komşu disiplin sınır kayması): **36 kayıt** (%41.4)
- `SEMANTIC_HUB` (Oşinografi / Deniz Mühendisliği topolojik hub çekimi): **17 kayıt** (%19.5)
- `LEXICAL_SURFACE` (Yüzeysel anahtar kelime tuzakları): **16 kayıt** (%18.4)
- `RESEARCH_TOPIC` (Sosyal bilimlerdeki teknik konu karmaşası): **15 kayıt** (%17.2)
- `POLYSEMY` (Çok anlamlı sözcük yanılgısı): **3 kayıt** (%3.4)

Bu dağılım, bir Role-Aware doğrulama katmanının FP problemine tek başına "sihirli bir değnek" olamayacağını; komşu disiplinler ve semantik hub'lar için taksonomik hiyerarşi ve hubness cezalandırma katmanlarına da ihtiyaç duyulduğunu gösterir.

---

## 9. Removed FP Role Distribution (Elenen FP Dağılımı)

Oracle filtresi tarafından elenen 72 FP kaydının rol bazlı dökümü:

```
Elenen 72 FP Kaydının Rol Dağılımı:
MATERIAL_OBJECT    [█████████████████████████] 26 (%36.1)
CONTEXT            [█████████████████]         18 (%25.0)
METHOD             [████████████████]          17 (%23.6)
SAMPLE_POPULATION  [███████████]               11 (%15.3)
```

- **Materyal kaynaklı FP'ler:** 26/26 (%100 eleme başarısı). Diş hekimliğindeki titanyum, ambalajdaki karton, bitkisel ekstraktlar tamamen temizlenmiştir.
- **Bağlam kaynaklı FP'ler:** 18/18 (%100 eleme başarısı). Orman kazası otopsisi, kırsal köy mimarisi gibi sektörel bağlam kaymaları temizlenmiştir.
- **Yöntem kaynaklı FP'ler:** 17/17 (%100 eleme başarısı). GEANT4 simülasyonu, akış sitometrisi mikroskopisi gibi yöntem yanılgıları temizlenmiştir.
- **Örneklem kaynaklı FP'ler:** 11/11 (%100 eleme başarısı). Kuzu pnömonisi, kedi travması gibi hayvan biyolojisinden insan klinik tıbbına yapılan transferler temizlenmiştir.

---

## 10. Removed TP Analysis (Elenen TP'lerin Analizi)

Elenen toplam 35 TP kaydı (5 TP-1 + 30 TP-2) incelendiğinde şu temel örüntü ortaya çıkmaktadır:

1. **İstatistik ve Yapay Zeka Metotları:** Sosyal, tıbbi veya mühendislik çalışmalarında kullanılan gelişmiş makine öğrenmesi, optimizasyon veya istatistiksel modeller denetçiler tarafından meşru bir ikincil disiplin (`İstatistik ve Olasılık`, `Yapay Zeka`) olarak kabul edilmişken; katı oracle filtresi bunları yöntem (`METHOD`) sayarak elemiştir.
2. **Kültürel Miras ve Geleneksel Materyaller:** Folklor ve arkeoloji çalışmalarında incelenen tarihi kilim, yastık dokumaları veya seramik kaplar, tekstil ve seramik bilimiyle kesiştiği için onaylanmışken; oracle filtresi fiziksel nesneyi (`MATERIAL_OBJECT`) doğrudan elemiştir.
3. **Özel Popülasyon Grupları:** Çocuklar, yaşlılar veya engelli bireyler üzerinde yapılan sağlık ve sosyal hizmet çalışmaları denetçilerce `Pediatri` veya `Geriatri` olarak tescil edilmişken; oracle filtresi canlı popülasyonu (`SAMPLE_POPULATION`) olarak kodlayıp silmiştir.

---

## 11. Negative Control (Negatif Kontrol Değerlendirmesi)

Negatif kontrol analizi, bu deneyin en hayati metodolojik dersini vermiştir:
- **Tez:** `non_focus_role_source == 1` olması bir adayın kesinlikle yanlış olduğu anlamına gelmez.
- **Kanıt:** 186 TP-2 kaydının %16.13'ü de non-focus rollerden beslenmektedir.
- **Ders:** Eğer bir sistem adayın yöntem veya materyalden kaynaklandığını tespit ettiğinde onu **doğrudan çöpe atarsa (hard reject)**, sistemin en değerli çıktılarından olan özgün disiplinlerarası bağlantıları (TP-2) yok etme riskiyle karşılaşır.

---

## 12. Interpretation (Teorik Yorum ve Fizibilite)

> **"Rol bilgisi kusursuz şekilde elimizde olsaydı, mevcut D sistemi ne kadar iyileştirilebilirdi?"**

1. **FP Azaltma Potansiyeli:** Mevcut yanlış pozitiflerin yaklaşık **yarısı (%45.3, 72/159)** rol bilgisiyle doğrudan adreslenebilmektedir.
2. **Hassasiyet (Precision) Kazanımı:** Denetim havuzu hassasiyeti **%59.02'den %69.04'e (+10.02 yüzde puanı)** sıçramaktadır.
3. **Ödenen Bedel (TP Kaybı):** 35 gerçek anomali / disiplinlerarası aday (%15.28 TP kaybı) kaybedilmektedir.
4. **Keşifsel Varyantın Gösterdiği Çıkış Yolu (D-Oracle-Exp-HighConf):** Eğer filtreleme yalnızca sınır vakası olmayan, net non-focus rollere uygulanırsa; **hiçbir TP kaybedilmeden (TP-1: %100, TP-2: %100 retention), FP'lerin %29.56'sı (47 FP) elenmekte ve precision %67.16'ya yükselmektedir.**

Bu bulgular, bir Role-Aware katmanının katı bir eleme filtresi yerine **güven skorlu bir reranking veya soft-penalty mekanizması** olarak kurgulanması gerektiğini net biçimde ortaya koymaktadır.

---

## 13. Limitations (Zorunlu Metodolojik Sınırlamalar)

> [!CAUTION]
> **Zorunlu Sınırlama 1:** Bu çalışma manuel olarak atanmış rol etiketlerini "oracle" (kusursuz) bilgi olarak kullanmaktadır. Dolayısıyla elde edilen performans, otomatik bir rol çıkarıcının gerçek performansını temsil etmez; rol bilgisinin mevcut anomali tespit sistemine eklenmesi halinde ulaşılabilecek potansiyel iyileşmenin **teorik üst sınırına (upper-bound)** ilişkin bir fizibilite değerlendirmesidir.

> [!CAUTION]
> **Zorunlu Sınırlama 2:** 388 kayıtlık audit havuzu tüm 20,902 makale için tam ground truth değildir. Bu nedenle burada verilen retention ve precision değerleri genel popülasyon genellemesi değil, audit havuzu üzerindeki kontrollü offline değerlendirmeyi ifade etmektedir.

---

## 14. Next Step (Sonraki Adım)

Bu fizibilite deneyi, Role-Aware doğrulama fikrinin güçlü bir teorik tavana sahip olduğunu; ancak "doğrudan silme" yerine "ağırlık düşürme / soft-penalty" mimarisine ihtiyaç duyulduğunu göstermiştir.

Bir sonraki aşamada:
1. Otomatik rol çıkarımı için hafif (lightweight) prompt ve az sayıda parametreli sınıflandırma mimarisi tasarlanmalı,
2. Model çıktısı doğrudan reject kuralı olarak değil, anomali adaylarının risk skorunu modüle eden bir çarpan olarak sisteme entegre edilmelidir.
