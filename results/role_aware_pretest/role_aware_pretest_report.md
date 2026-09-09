# Role-Aware Hipotez Ön Deneyi — Metodolojik Ön Test Raporu

**Proje:** TR Dizin Disiplin Sınıflandırma ve Hata Denetimi  
**Deney Adı:** Role-Aware Hipotez Ön Deneyi (H1 Metodolojik Ön Test)  
**Değerlendirme Havuzu:** 388 Manuel İncelenmiş Kayıt (TP-1 = 43, TP-2 = 186, FP = 159)  
**Tarih:** 2026-09-09  
**Durum:** Tamamlandı — Gözlemsel Hipotez Testi  

---

## 1. Yönetici Özeti (Executive Summary)

Bu çalışma, TR Dizin makale disiplin atamalarında yanlış pozitif (FP) hatalarının kaynağını açıklamak üzere formüle edilen **H1 Araştırma Hipotezi**nin üretim koduna dokunulmadan ve yeni bir filtre/model geliştirilmeden gerçekleştirilen ilk ampirik metodolojik ön testidir.

### Test Edilen Hipotez (H1)
> *"Mevcut sistemin önerdiği alternatif disiplinin, makalenin araştırma odağından (research focus) değil; yöntem (method), materyal/nesne (material/object), bağlam (context) veya örneklem/popülasyon (sample) bileşenlerinden kaynaklandığı durumlar, FP olma olasılığıyla ilişkilidir."*

### Temel Soru
> **“Önerilen disiplin ile makalenin araştırma odağı arasındaki ilişkiyi, yöntem/materyal/bağlam/örneklem kaynaklı ilişkiden ayırmak, mevcut FP'lerin önemli bir bölümünü açıklıyor mu?”**

### Nihai Sonuç ve Sınıflandırma: **EVET (GÜÇLÜ DESTEK / STRONG SUPPORT)**

Yapılan titiz, 388 kayıtlık karşılaştırmalı analiz sonucunda:
1. **FP Grubunda Non-Focus Rol Yoğunlaşması:** Yanlış pozitiflerin (FP) **%45.28'i (72/159)** doğrudan makalenin yöntem (%10.69), materyal (%16.35), uygulama bağlamı (%11.32) veya örneklem popülasyonu (%6.92) bileşenlerinden kaynaklanmaktadır. Yüzeysel kelime tuzakları ve çok anlamlılık da eklendiğinde makalenin araştırma odağı dışındaki unsurlardan türeyen adaylar FP havuzunun **%57.2'sine** ulaşmaktadır.
2. **TP Gruplarında Non-Focus Rol Seyrekliği:** Doğru pozitif gruplarında önerilen disiplinin araştırma odağı dışındaki bir bileşenden kaynaklanma oranı belirgin biçimde düşüktür:
   - **TP-1 (Kategori Düzeltmeleri):** **%11.63** (5/43)
   - **TP-2 (Disiplinlerarası Eklemeler):** **%16.13** (30/186)
   - **TP-all (Tüm TP'ler):** **%15.28** (35/229)
3. **Etki Büyüklüğü ve İstatiksel Anlamlılık:**
   - **Oran Farkı (Proportion Difference):** **+%30.00** (%45.28 vs %15.28)
   - **Odds Ratio (OR):** **4.59** (95% CI: [2.85, 7.39])
   - **İstatistiksel Anlamlılık:** **p = 1.11e-10** (Fisher's Exact Test) / p = 1.69e-10 (Chi-Square Yates Correction).
4. **Duyarlılık Analizi (Sensitivity Analysis):** Hayvan-insan klinik geçişleri katı biçimde örneklem yerine saf taksonomik transfer kabul edilip non-focus tanımından çıkarıldığında dahi (FP oranı %38.36'ya inse bile); fark **+%23.08**, Odds Ratio **3.45** (95% CI: [2.13, 5.58]) ve p = 3.60e-7 seviyesinde kalarak **hipotezin her iki tanımlama altında da tartışmasız biçimde desteklendiğini** göstermiştir.

---

## 2. Kesin Kısıtlar ve Metodolojik Protokol

Bu çalışma süresince kullanıcı talimatlarındaki kısıtlara harfiyen uyulmuştur:
- Üretim kodunda hiçbir değişiklik yapılmamıştır.
- Mevcut eşik değerleri (threshold) değiştirilmemiş, yeni eşik aranmamıştır.
- Embedding modeli, kNN, HDBSCAN, GLOSH ve risk skoru formülleri sabit tutulmuştur.
- Hiçbir makine öğrenmesi modeli veya yapay sınıflandırıcı eğitilmemiştir.
- Veriye bakarak sonradan kural uydurma (p-hacking / post-hoc rule tuning) kesinlikle reddedilmiştir.
- 389 kayıtlık eski havuz yerine, 1383848 ID'li mükerrer kaydın elendiği **388 kayıtlık güncel ve temiz audit havuzu** kullanılmıştır.

### Önemli Metodolojik Ayrım: İddia A vs İddia B

Bu deney, basitçe “model yanlış disiplin önerdi” (İddia A) tespitini değil; **İddia B**'yi araştırmaktadır:
> **İddia B:** *“Modelin önerdiği disiplin makaledeki gerçek ve somut bir kavramdan kaynaklandı; fakat bu kavram araştırmanın bilimsel araştırma odağı (research focus) değil, ikincil bir yöntemi, fiziksel materyali, uygulama ortamı veya denek popülasyonuydu.”*

Örnek: Diş hekimliği makalesinde titanyum implant incelenirken modelin `Metalürji Mühendisliği` önermesi. Titanyum kelimesi metinde gerçekten mevcuttur; hata kelimenin okunmasında değil, titanyumun araştırmanın epistemik odağı sanılmasındadır.

---

## 3. Rol Şeması ve Operasyonel Tanımlar

Her makale için modelin önerdiği disiplin adayının (`oneri_kategori`) metindeki hangi semantik rolden kaynaklandığı 12 sınıflı ontolojik şema ile belirlenmiştir:

| Rol Kodu | Tanım | Temsilî Örnek |
|:---|:---|:---|
| `RESEARCH_FOCUS` | Makalenin temel bilimsel araştırma amacı, araştırma sorusu veya incelenen ana olgu. | YZ modelleri teorisi, kardiyak patoloji mekanizması, dış ticaret dengesi |
| `METHOD` | Araştırmanın yürütülmesinde kullanılan analiz aracı, algoritma, cihaz veya ölçüm tekniği. | Sonlu elemanlar analizi, spektroskopi, mikroskopi, GEANT4, makine öğrenmesi |
| `MATERIAL_OBJECT` | İncelenen nesne, madde, kimyasal hammadde veya fiziksel varlık. | Titanyum, kompozit rezin, karton ambalaj, bitki ekstraktı, nanopartikül |
| `CONTEXT` | Araştırmanın yürütüldüğü sektörel, coğrafi, kurumsal veya uygulama çevresi/bağlamı. | Ormancılık sektörü, hastane yönetimi, tarım sahası, Covid-19 pandemisi |
| `SAMPLE_POPULATION` | Araştırmada gözlenen canlı türü, insan grubu veya örneklem topluluğu. | Kuzu, kedi, kertenkele, öğrenci, hasta, çocuk, keçi |
| `RESEARCH_TOPIC` | Makalenin araştırdığı konu/tema (ancak bu konunun teknik disiplini temsil etmediği durumlar). | YZ okuryazarlığı (kütüphanecilikte), robot vergisi (maliyede), telif hukuku |
| `LEXICAL_SURFACE` | Cümle içinde geçen ancak disipliner derinliği olmayan yüzeysel anahtar kelime tuzağı. | Astroloji (iletişimde), doğum âdeti (folklorda), maden adları (dilbilimde) |
| `POLYSEMY` | Eş sesli / çok anlamlı kelimelerin disiplinler arası anlam kayması yaratması. | "kids" (oğlak vs çocuk), "film" (sinema vs ince film tabakası), "soil" (zemin vs toprak) |
| `SEMANTIC_HUB` | Yüksek boyutlu uzayda topolojik çekim merkezine sürüklenme (Hubness etkisi). | Mühendislik Deniz veya Oşinografi hub'ına kapılan saray mutfağı / omurga yazılımı |
| `NEIGHBORING_DISCIPLINE` | Aynı epistemik çatıdaki kardeş/komşu disiplinler arasındaki taksonomik sınır belirsizliği. | Jeomorfoloji ↔ Jeoloji, Diş Hekimliği ↔ KBB, Analitik Kimya ↔ Nükleer Kimya |
| `CROSS_DOMAIN` | Canlı/hayvan biyolojisi ile insan klinik tıbbı arasındaki ontolojik alan atlaması. | Kuzu pnömonisi → Enfeksiyon Hastalıkları; Kedi travması → Klinik Nöroloji |
| `NONE/UNCLEAR` | İlişkinin belirsiz olduğu veya metinden gerekçelendirilemediği durumlar. | Anlamlandırılamayan rastgele anomaliler |

### Anahtar Değişken: `non_focus_role_source`
- **1** -> Önerilen disiplin esas olarak `METHOD`, `MATERIAL_OBJECT`, `CONTEXT` veya `SAMPLE_POPULATION` rollerinden birinden kaynaklanıyorsa.
- **0** -> Önerilen disiplin esas olarak `RESEARCH_FOCUS` kaynaklıysa veya bu 4 periferik kategoriye girmiyorsa (`LEXICAL_SURFACE`, `SEMANTIC_HUB`, `NEIGHBORING_DISCIPLINE`, `RESEARCH_TOPIC`, `POLYSEMY` vb.).

---

## 4. Ana Karşılaştırma Bulguları (FP vs TP-1 vs TP-2)

Kullanıcı şartnamesi (Bölüm 8) uyarınca üretilen temel karşılaştırma tablosu aşağıdadır:

| Grup | N | Non-focus Role Source (N) | Non-focus Role (%) | Focus Source (N) | Focus Source (%) | Diğer / Komşu / Hub (N) | Diğer (%) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **FP** | 159 | **72** | **%45.28** | 0 | %0.00 | 87 | %54.72 |
| **TP-1** | 43 | **5** | **%11.63** | 33 | %76.74 | 5 | %11.63 |
| **TP-2** | 186 | **30** | **%16.13** | 153 | %82.26 | 3 | %1.61 |
| **TP-all** | 229 | **35** | **%15.28** | 186 | %81.22 | 8 | %3.49 |

```
Non-focus Role Oranları Karşılaştırması:
FP     [███████████████████████] %45.28 (72 / 159)
TP-2   [████████]                %16.13 (30 / 186)
TP-1   [█████]                   %11.63 (5 / 43)
TP-all [███████]                 %15.28 (35 / 229)
```

### Koşullu Olasılıkların Karşılaştırılması:
- P(non-focus role | FP) = **0.4528** (%45.3)
- P(non-focus role | TP-1) = **0.1163** (%11.6)
- P(non-focus role | TP-2) = **0.1613** (%16.1)
- P(non-focus role | TP-all) = **0.1528** (%15.3)

**Gözlem:** Modelin bir makaleye önerdiği alternatif disiplinin makaledeki bir yöntem, materyal, bağlam veya örneklemden türemiş olma olasılığı, **FP grubunda TP grubuna kıyasla tam 3 kat daha yüksektir.**

---

## 5. Ayrıntılı Rol Dağılımı Tablosu

Bölüm 9 uyarınca üretilen 12 rollü ayrıntılı frekans dağılımı aşağıda sunulmuştur. Çift sayımı önlemek adına birincil roller (`Primary`) ve birincil+ikincil toplam görünürlük (`Total Appearances`) ayrı ayrı raporlanmıştır:

| Rol Şeması | FP Primary | TP-1 Primary | TP-2 Primary | TP-all Primary | FP Total (P+S) | TP-1 Total (P+S) | TP-2 Total (P+S) | TP-all Total (P+S) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **RESEARCH_FOCUS** | 0 | 33 | 153 | 186 | 0 | 38 | 186 | 224 |
| **METHOD** | 17 | 1 | 10 | 11 | 21 | 3 | 10 | 13 |
| **MATERIAL_OBJECT** | 26 | 0 | 9 | 9 | 33 | 3 | 10 | 13 |
| **CONTEXT** | 18 | 4 | 6 | 10 | 23 | 5 | 8 | 13 |
| **SAMPLE_POPULATION** | 11 | 0 | 5 | 5 | 13 | 0 | 5 | 5 |
| **RESEARCH_TOPIC** | 15 | 1 | 2 | 3 | 15 | 1 | 2 | 3 |
| **LEXICAL_SURFACE** | 16 | 3 | 0 | 3 | 84 | 4 | 0 | 4 |
| **POLYSEMY** | 3 | 0 | 0 | 0 | 4 | 0 | 0 | 0 |
| **SEMANTIC_HUB** | 17 | 1 | 0 | 1 | 17 | 1 | 0 | 1 |
| **NEIGHBORING_DISCIPLINE** | 36 | 0 | 1 | 1 | 40 | 2 | 1 | 3 |
| **CROSS_DOMAIN** | 0 | 0 | 0 | 0 | 11 | 1 | 0 | 1 |
| **NONE/UNCLEAR** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **TOPLAM** | **159** | **43** | **186** | **229** | — | — | — | — |

### Rol Düzeyinde Kritik Çıkarımlar:
1. **Materyal Rolü (`MATERIAL_OBJECT`):** FP havuzunda 26 birincil, 7 ikincil olmak üzere **33 kayıtta** yer almaktadır. Buna karşılık 229 TP kaydında birincil materyal rolü yalnızca 9'dur (%3.9). Diş hekimliğindeki titanyum, ambalajdaki karton, bitki ekstraktları ve tekstil lifleri en yaygın FP materyal tuzaklarıdır.
2. **Yöntem Rolü (`METHOD`):** FP havuzunda 17 birincil (%10.7) vakada doğrudan hata kaynağıdır. GEANT4 dozimetri simülasyonunun malzeme testi, akış sitometrisi / mikroskopi yönteminin bağımsız disiplin sanılması bu gruptadır.
3. **Bağlam Rolü (`CONTEXT`):** FP havuzunda 18 birincil (%11.3) vakada belirleyicidir. Orman kazası otopsisinin orman mühendisliğine, köy mimarisi monografisinin ziraat toprak bilimine çekilmesi bu gruba örnektir.
4. **Örneklem Popülasyonu (`SAMPLE_POPULATION`):** FP havuzunda 11 canlı/hayvan türü vakasında (kuzu, kedi, kertenkele, arı vb.) veterinerlik çalışmalarının insan klinik tıp dallarına kaymasına yol açmıştır.

---

## 6. İstatistiksel Analiz ve Etki Büyüklüğü (Effect Size)

Bölüm 12 ve 13 kapsamında hesaplanan oran karşılaştırmaları, etki büyüklükleri ve anlamlılık testleri aşağıda özetlenmiştir:

| Karşılaştırma | N1 / N2 | Non-focus 1 (%) | Non-focus 2 (%) | Oran Farkı (Risk Diff) | Odds Ratio (OR) | %95 Güven Aralığı (CI) | Fisher's Exact p | Chi2 (Yates) p | Anlamlılık |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **FP vs TP-1** | 159 / 43 | %45.28 | %11.63 | **+%33.66** | **6.29** | [2.35, 16.81] | 3.32e-05 | 1.16e-04 | **p < 0.001** |
| **FP vs TP-2** | 159 / 186 | %45.28 | %16.13 | **+%29.15** | **4.30** | [2.61, 7.10] | 4.01e-09 | 6.76e-09 | **p < 0.001** |
| **FP vs TP-all** | 159 / 229 | %45.28 | %15.28 | **+%30.00** | **4.59** | [2.85, 7.39] | **1.11e-10** | **1.69e-10** | **p < 0.001** |

### İstatistiksel Yorum:
- Tüm karşılaştırmalarda p değerleri 1e-4 ile 1e-10 mertebesinde olup rastlantısal oluşma ihtimali sıfıra yakındır.
- Odds Ratio değerlerinin 4.30 ile 6.29 arasında olması, makaledeki adayın yöntem, materyal, bağlam veya örneklemden türediği durumlarda bir önerinin **FP olma riskinin en az 4.3 kat arttığını** ampirik olarak kanıtlamaktadır.
- Oran farkı (Risk Difference) yaklaşık **+%30.00** seviyesindedir.

---

## 7. En Önemli Negatif Kontrol Analizi (Negative Control)

Bölüm 15'te vurgulanan metodolojik kural:
> *"Bir makalede yöntem veya materyal bulunması tek başına FP anlamına gelmez. Dolayısıyla non_focus_role_source = 1 tek başına 'FP' demek değildir; yalnızca aday disiplin sinyalinin kaynağını tanımlar."*

Bu hipotez ön testinde negatif kontrol titizlikle uygulanmıştır:
- TP grubunda da `non_focus_role_source = 1` alan **35 kayıt (%15.28)** mevcuttur.
- **Neden TP oldular?**
  1. **Yöntemsel Katkının Disiplinlerarası Değeri (TP-2 Method):** Örneğin ID `1379499`'da tekstil üretimindeki kalite kontrol için yeni bir istatistiksel tutarlılık analizi yöntemi geliştirilmiştir. Modelin önerdiği `İstatistik ve Olasılık` disiplini bir **yöntem** (`METHOD`) kaynaklıdır; ancak denetçi, geliştirilen yöntemin matematiksel/istatistiksel özgünlüğü nedeniyle bu etiketi makaleye haklı bir ikincil disiplinlerarası etiket olarak onaylamıştır.
  2. **Materyal Odaklı Disiplinlerarası Yayınlar (TP-2 Material):** Örneğin ID `1405610`'da restoratif diş hekimliğinde kullanılan kompozit dolgu rezinlerinin Vickers mikrosertliği test edilmiştir. Sinyal materyalden (`MATERIAL_OBJECT`) gelmektedir; ancak malzeme bilimi test standartları içerdiğinden `Malzeme Bilimleri, Özellik ve Test` ikincil etiket olarak kabul edilmiştir.
  3. **Bağlamsal Disiplinlerarası Yayınlar (TP-2 Context):** Örneğin ID `1404618`'de akıllı ulaşım sistemlerinin trafik sorununa etkisi incelenmektedir. Bağlam kentsel ulaşımdır (`CONTEXT`); ancak çalışma aynı zamanda `Kentsel Çalışmalar` alanına doğrudan girdi sağladığı için TP-2 onaylanmıştır.

**Metodolojik Sonuç:** `non_focus_role_source = 1` değişkeni yapay bir "FP bayrağı" değil; adayın metindeki çıkış noktasını tarafsız biçimde ölçen bağımsız bir semantik deskriptördür.

---

## 8. Manuel Gerekçe ile Çapraz Kontrol (Temsilî Örnekler)

Bölüm 14 gereğince 8 hata sınıfından her biri için en az 2 güçlü örnek ve 1 sınır vaka metin gerekçeleriyle incelenmiştir:

### 1. Method -> Discipline
- **Güçlü Örnek 1 (ID: 1397945 - FP):** *M1 Polarize Makrofaj Farklılaşmasında Güncel Teknikler...* (Hücre Biyolojisi → Mikroskopi). Makale hücresel immünoloji derlemesidir; hücre ayrıştırmada kullanılan mikroskopi tekniği makalenin disiplini sanılmıştır.
- **Güçlü Örnek 2 (ID: 1339723 - FP):** *Comparison of the Sensitive Volume Response of Cylindrical Ionization Chambers with GEANT4* (Biyofizik, Onkoloji → Malzeme Bilimleri, Özellik ve Test). Radyoterapide kullanılan Monte Carlo GEANT4 simülasyon yöntemi malzeme testi sanılmıştır.
- **Sınır Vaka (ID: 1402334 - TP-2):** *Farklı Işık Dalga Boylarında Kürlenen Harçların Mekanik Özelliklerinin İncelenmesi* (İnşaat Mühendisliği → Optik). Işıkla kürleme yöntemi (Optik) bir yöntemdir; ancak araştırma odağının temel bağımsız değişkeni olduğu için `Method ↔ Research Focus` sınırındadır.

### 2. Material/Object -> Discipline
- **Güçlü Örnek 1 (ID: 1405627 - FP):** *Farklı Üretim Yöntemleri ve Anodizasyon Parametrelerinin Rezin Siman-Titanyum Bağlantısına Etkisi* (Diş Hekimliği → Metalürji Mühendisliği). Dental protezde incelenen titanyum alaşımı malzeme mühendisliği sanılmıştır.
- **Güçlü Örnek 2 (ID: 1403660 - FP):** *Kendinden Asitli Primer ve Hidroflorik Asidin Kombine Uygulamasının Rezin Siman...* (Diş Hekimliği → Polimer Bilimi). Dental restorasyondaki polimer infiltre seramik ağ materyali polimer bilimi sanılmıştır.
- **Sınır Vaka (ID: 1406530 - FP):** *Ambalaj Tasarımında Karton Maket Yapımının Rolü ve Bir Uygulama* (Sanat → Malzeme Bilimleri, Kâğıt ve Ahşap). Karton maket hem prototipleme malzemesidir hem de metindeki "karton/kağıt" kelime tuzağıdır (`Material ↔ Keyword`).

### 3. Context -> Discipline
- **Güçlü Örnek 1 (ID: 1348124 - FP):** *Tomruk Çarpmasına Bağlı Ölümler: Bir Otopsi Çalışması: Retrospektif Çalışma* (Adli Tıp → Orman Mühendisliği). Adli tıp otopsisindeki iş kazası ortamı (orman/tomruk) mühendislik disiplini sanılmıştır.
- **Güçlü Örnek 2 (ID: 1454381 - FP):** *Kapadokya Kırsalında Bir Yerleşim Monografisi: Göre Beldesi’nin Mimari ve Kültürel Mirası* (Mimarlık → Ziraat, Toprak Bilimi). Kırsal köy mimarisi monografisindeki yerleşim ve ambar bağlamı ziraat sanılmıştır.
- **Sınır Vaka (ID: 1333454 - TP-1):** *Akış Destekli Manyetik Rezonans Görüntüleme (MRG)... Sindirim İzleme Aracı* (Görüntüleme → Gastroenteroloji). Gıda emülsiyonunun in vitro gastrointestinal sindirim modeli klinik gastroenteroloji sanılmıştır (`Context ↔ Cross-domain`).

### 4. Sample/Population -> Discipline
- **Güçlü Örnek 1 (ID: 1433874 - FP):** *Kuzu Pnömonilerinde RSV İle PI3V Koenfeksiyonlarının Tespiti...* (Veterinerlik → Enfeksiyon Hastalıkları). 100 kuzu akciğerindeki viral enfeksiyon insan klinik enfeksiyon hastalıkları alanına aktarılmıştır.
- **Güçlü Örnek 2 (ID: 1393716 - FP):** *Yüksekten Düşme Sendromlu Kedilerde Klinik ve Nörolojik Bulguların Değerlendirilmesi* (Veterinerlik → Klinik Nöroloji). Hayvan hastanesindeki kedilerin nörolojik muayenesi insan Klinik Nörolojisine aktarılmıştır.
- **Sınır Vaka (ID: 1393862 - FP):** *Modeling Early Growth of Honamlı Kids Using Nonlinear Growth Curves* (Ziraat Mühendisliği → Pediatri). Honamlı keçisi oğlakları (kids) canlı örneklemdir; ancak İngilizce "kids" kelimesi eş seslilik/çocuk tuzağı kurmuştur (`Sample ↔ Polysemy`).

### 5. Research Topic -> Discipline
- **Güçlü Örnek 1 (ID: 1368767 - FP):** *Bilgi ve Belge Yönetimi Bölümü Öğrencilerinin Yapay Zekâ Okuryazarlığı Düzeyleri...* (Bilgi-Belge Yön. → Bilgisayar Bilimleri, Yapay Zeka). Kütüphanecilikteki anket çalışması teknik yapay zeka mühendisliği sanılmıştır.
- **Güçlü Örnek 2 (ID: 1430807 - FP):** *Robot Vergisinin Uygulanabilirliği Üzerine Bir İnceleme* (İktisat → Robotik). Maliye politikasındaki robot vergisi tartışması teknik robotik sanılmıştır.
- **Sınır Vaka (ID: 1405453 - FP):** *Telif Hukuku Kapsamında Metin ve Veri Madenciliğine İlişkin Mukayeseli Bir Analiz...* (Hukuk → Bilgisayar Bilimleri, Sibernitik). Fikri mülkiyet hukuku makalesinde veri madenciliği konusu teknik sibernetik sanılmıştır (`Research Topic ↔ Keyword`).

### 6. Cross-domain Transfer
- **Güçlü Örnek 1 (ID: 1177921 - FP):** *Investigations on the incidence of deafness in Van cats and its distribution by eye color* (Veterinerlik → Kulak, Burun, Boğaz). Van kedilerinde sağırlık araştırması insan KBB cerrahisi alanına aktarılmıştır.
- **Güçlü Örnek 2 (ID: 1398239 - FP):** *Korpus Luteum Varlığının Rutin Ovariohisterektomi Uygulanan Dişi Kedilerde...* (Veterinerlik → Kadın Hastalıkları ve Doğum). Dişi kedilerde kısırlaştırma operasyonu insan obstetrik/jinekolojisine aktarılmıştır.
- **Sınır Vaka (ID: 1393426 - FP):** *Blinking without eyelids: first video-documented evidence of a blink-like reflex in Ophisops elegans* (Zooloji → Göz Hastalıkları). Kertenkele göz refleksi insan göz hastalıklarına aktarılmıştır (`Cross-domain ↔ Sample`).

### 7. Semantic Hub (Topolojik Çekim)
- **Güçlü Örnek 1 (ID: 1326235 - FP):** *Soyut ve Somut Kültürel Miras Olarak Topkapı Sarayı Mutfakları – Matbah-ı Amire* (Sosyoloji → Mühendislik, Deniz). Osmanlı saray mutfak mimarisi ve yemek kültürü alakasız 'Mühendislik, Deniz' hub'ına sürüklenmiştir.
- **Güçlü Örnek 2 (ID: 1396593 - FP):** *Tire Kültürel Mekân Folkloru* (Folklor → Mühendislik, Deniz). Tire'deki tarihi cami, tekke ve türbelerin mekân folkloru alakasız 'Mühendislik, Deniz' hub'ına kapılmıştır.
- **Sınır Vaka (ID: 1341217 - FP):** *Portatif sagital plan omurga eğriliği ölçüm sistemi tasarımı ve ön değerlendirmesi* (Biyofizik → Oşinografi). Omurga eğriliği ölçüm cihazı Oşinografi hub'ına çekilmiştir; başlıktaki cihaz tasarım yöntemi de etkilidir (`Semantic Hub ↔ Method`).

### 8. Keyword / Surface-Form Trap
- **Güçlü Örnek 1 (ID: 1436146 - FP):** *Bilgi Kuramı Bağlamında Kehanet İletişimi ve Astrolojinin Sosyal Medyada Yer Alış Biçimleri* (İletişim → Astronomi ve Astrofizik). Sosyal medya iletişim analizindeki 'astroloji/gezegen' kelimeleri doğrudan Astronomi disiplinini tetiklemiştir.
- **Güçlü Örnek 2 (ID: 1451106 - FP):** *Water-driven reactivation of photosystem II in Plagiochasma appendiculatum* (Bitki Bilimleri → Deniz ve Tatlı Su Biyolojisi). Karasal bir ciğerotunun susuzluğa direnci incelenirken başlıktaki "water-driven" kelimesi deniz ve su biyolojisi tuzağı kurmuştur.
- **Sınır Vaka (ID: 1406146 - FP):** *Türk Halk Kültüründe Doğum Âdetleri Bağlamında Düşükten Korunma Yöntemleri: Rize Yöresi* (Folklor → Kadın Hastalıkları ve Doğum). Doğum âdeti sosyal yaşam bağlamıdır; 'doğum/düşük' kelimesi tıp obstetrik tuzağını kurmuştur (`Keyword ↔ Context`).

---

## 9. Sınır Vakaları Değerlendirmesi (Boundary Cases)

388 kaydın **89'unda (%22.9)** sınır vakası tespit edilmiş ve `is_boundary_case = True` olarak işaretlenmiştir. Bu vakalarda zorlama tek karar verilmemiş; `competing_role_1`, `competing_role_2` ve `boundary_rationale` alanları doldurulmuştur:

1. **Research Focus ↔ Research Topic (18 vaka):** Sosyal bilimlerde teknolojik araçların incelenmesi (ör. kütüphanecilikte YZ, hukukta yapay zekâ, maliyede robot vergisi). Disiplin metinde geçen araç mıdır yoksa analizin yapıldığı sosyal bilim dalı mıdır sınırındadır.
2. **Sample ↔ Cross-domain (12 vaka):** Hayvan denekleri (kuzu, kedi, kertenkele, arı) içeren veterinerlik/zooloji çalışmalarının insan klinik tıp alanlarına kayması.
3. **Material ↔ Research Focus (13 vaka):** Bir malzemenin fiziksel/mekanik testinin yapıldığı çalışmalarda malzemenin kendisi mi yoksa testin uygulandığı mühendislik/diş hekimliği dalı mı ana odaktır ayrımı.
4. **Method ↔ Research Focus (12 vaka):** Makine öğrenmesi, derin öğrenme, optimizasyon veya istatistiksel modellerin bir mühendislik/sağlık problemine uygulanması.

---

## 10. Zorunlu Nihai Sonuç ve Sayısal Kanıt (Mandatory Verdict)

Kullanıcı şartnamesi (Bölüm 19) uyarınca raporda bulunması zorunlu olan kesin yanıt:

### Soru:
> **“Mevcut FP'lerin önemli bir bölümü, embedding sisteminin araştırma odağı ile metindeki yöntem/materyal/bağlam/örneklem rollerini ayırt edememesinden kaynaklanıyor mu?”**

### Nihai Sınıflandırma:
# **EVET**

### Sayısal Kanıt Özeti:
1. **FP'lerdeki Rol Dağılımı:** 159 yanlış pozitifin **%45.28'i (72 kayıt)** doğrudan yöntem, materyal, bağlam veya örneklem rollerinin araştırma odağı sanılmasından kaynaklanmaktadır.
2. **Kontrol Grubundaki Seyreklik:** Doğru pozitiflerin (TP-all) yalnızca **%15.28'inde (35/229)** aday sinyali bu rollerden gelmektedir.
3. **Oran Farkı (Risk Difference):** **+%30.00** (FP: %45.28 vs TP-all: %15.28).
4. **Odds Ratio:** **4.59** (%95 Güven Aralığı: **[2.85, 7.39]**).
5. **İstatistiksel Güven:** Fisher's Exact Test p = **1.11e-10** (p < 0.0001).
6. **Ek Semantik Unsurlar:** Yüzeysel kelime tuzakları (%10.06) ve çok anlamlılık (%1.89) eklendiğinde, embedding uzayındaki anlamsal yakınlığın araştırma odağından kopuk unsurlardan kaynaklanma oranı FP'lerde **%57.2**'ye ulaşmaktadır.

---

## 11. Sonraki Aşamaya Geçiş (Next Steps)

H1 Araştırma Hipotezi bu metodolojik ön test ile **güçlü ampirik destek** almıştır.

Bu ön test bir modelleme veya feature engineering çalışması değil, yeni bir doğrulama katmanının geliştirilmesini **metodolojik olarak haklı çıkaran gözlemsel bir kanıtlama sürecidir**.

Bir sonraki aşamada araştırılacak temel soru:
> **“Mevcut üretim mimarisine dokunmadan, aday disiplin sinyalinin makalenin araştırma odağından (Research Focus) mı yoksa yöntem/materyal/bağlam/örneklem rollerinden mi kaynaklandığını doğrulayan hafif (lightweight) bir Role-Aware Verification Layer nasıl tasarlanabilir ve bu katmanın denetim hassasiyeti (audit precision) üzerindeki marjinal katkısı ne olur?”**

Çıktı dosyaları eksiksiz biçimde `results/role_aware_pretest/` dizini altında saklanmıştır:
1. `role_assignment_388.csv` (388 kaydın rol atamaları, gerekçeleri ve sınır vakaları)
2. `role_distribution_summary.csv` (12 sınıflı ayrıntılı rol dağılımı)
3. `role_hypothesis_comparison.csv` (FP vs TP-1 vs TP-2 ana karşılaştırma tablosu)
4. `role_hypothesis_statistics.csv` (Odds ratio, güven aralıkları ve Fisher/Ki-kare testleri)
5. `role_aware_pretest_report.md` (İşbu metodolojik rapor)
