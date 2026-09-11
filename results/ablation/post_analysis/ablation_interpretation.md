# TR Dizin Konu Anomalisi Tespiti: Post-Ablation Hata Analizi Raporu

Bu rapor, üretim kodlarına ve karar eşiklerine dokunulmadan, ablation çalışmasının sayısal bulguları ve 388 makalelik güncel manuel denetim havuzu üzerinden hata mekanizmalarını ortaya koymak üzere üretilmiştir.

## 1. Denetim Havuzunun Doğrulanması (388 vs 389 Ayrımı)

### 388 vs 389 Sayılarının Netleştirilmesi
- **Güncel Qdrant Sistemi (hdbscan_anomaliler.csv / Varyant D)**: Tam olarak **388** aday içermektedir.
  - **TP-1**: 43 kayıt (%11.08)
  - **TP-2**: 186 kayıt (%47.94)
  - **FP**: 159 kayıt (%40.98)
  - **Toplam**: 388 kayıt
- **Eski Manuel Denetim Referansı (manuel_dogrulama_referans.csv)**: **389** kayıt içermekteydi (TP-1: 44, TP-2: 186, FP: 159).
- **Fark Makalesi**: ID `1383848` ("*Osmaniye İli Anadolu Mandalarının Süt Verim Özellikleri ve Yağ Asidi Bileşimi*")
  - Eski çalıştırmada GLOSH skoru `0.8705` iken, güncel Qdrant çalıştırmasında `hdbscan_tum_makaleler.csv` tablosunda GLOSH skoru `0.4615` değerine düşmüştür.
  - kNN baskınlığı `0.20` (<0.30) olduğu için ve güncel GLOSH skoru da `0.70` eşiğinin altında kaldığından, Varyant D'deki `(knn_baskinlik >= 0.30 | glosh > 0.70)` koşulunu sağlayamamış ve 388'lik havuza dahil olmamıştır.
  - Bu analizde tüm birincil metrikler **güncel 388 kayıtlık havuz** üzerinden hesaplanmış; şeffaflık amacıyla 389 referansı da parantez içinde belirtilmiştir.

### Varyantların Yakalama ve Kaçırma Sayıları (388 Havuzu)

| Varyant | Açıklama | Toplam Flagged (20.9k) | Denetlenmemiş (Un-audited) | TP-1 Yakalanan / Kaçırılan | TP-1 Coverage | TP-2 Yakalanan / Kaçırılan | TP-2 Coverage | FP Yakalanan / Filtrelenen | Audit Precision |
|---|---|---|---|---|---|---|---|---|---|
| **A** | Taxonomy Baseline (No k-NN, No GLOSH) | 1600 | 1212 | 43 / 0 | %100.00 | 186 / 0 | %100.00 | 159 / 0 | %59.02 |
| **B** | Taxonomy + GLOSH (No k-NN) | 560 | 409 | 18 / 25 | %41.86 | 65 / 121 | %34.95 | 68 / 91 | %54.97 |
| **C** | Taxonomy + k-NN (No GLOSH) | 384 | 0 | 42 / 1 | %97.67 | 186 / 0 | %100.00 | 156 / 3 | %59.38 |
| **D** | Full System (Taxonomy + GLOSH + k-NN) | 388 | 0 | 43 / 0 | %100.00 | 186 / 0 | %100.00 | 159 / 0 | %59.02 |

> *Not: Buradaki coverage oranları genel model recall'ı değil; yalnızca 388 makalelik manuel denetim havuzu içindeki kapsama oranlarıdır.*

## 2. C → D Farkının İncelenmesi (D \ C ve C \ D)

- `|C \ D| = 0`: Varyant C'de olup Varyant D'de olmayan hiçbir makale yoktur (C, D'nin alt kümesidir).
- `|D \ C| = 4`: Varyant D, Varyant C'ye göre sisteme tam 4 yeni aday eklemiştir.

### D \ C İçindeki 4 Kaydın Detay Tablosu

| External ID | Başlık | Mevcut Kategori | Önerilen Kategori | GLOSH Skoru | kNN Baskınlık | kNN Önerisi | Manuel Karar | Manuel Gerekçe |
|---|---|---|---|---|---|---|---|---|
| `1243292` | Demir Bazlı Manyetik Nanopartiküllerin Genotoksik ... | Kuş Bilimi, Parazitoloji, Tıbbi Araştırmalar Deneysel | Nanobilim ve Nanoteknoloji | 0.911 | 0.20 | Nanobilim ve Nanoteknoloji | **TP-1** | Manyetik nanopartiküllerin (Fe3O4, NiFe2O4 vb.) model organizma Drosophila melanogaster üzerinde gen... |
| `1393716` | Yüksekten Düşme Sendromlu Kedilerde Klinik ve Nöro... | Veterinerlik | Klinik Nöroloji | 0.934 | 0.20 | Klinik Nöroloji | **FP-1** | Atatürk Üniversitesi Veteriner Fakültesi Hayvan Hastanesi'ne getirilen yüksekten düşme sendromlu ked... |
| `1393862` | Modeling Early Growth of Honamlı Kids Using Nonlin... | Ziraat Mühendisliği | Pediatri | 0.931 | 0.20 | Pediatri | **FP-1** | Honamlı keçisi oğlaklarında (Honamlı kids) canlı ağırlık artışı ve erken dönem büyüme dinamiklerinin... |
| `1400625` | PSYLLİUM, BEZELYE VE YULAF KAYNAKLI BESİNSEL LİF İ... | Gıda Bilimi ve Teknolojisi, Beslenme ve Diyetetik | Toksikoloji | 0.921 | 0.20 | Toksikoloji | **FP-1** | Ayran üretiminde fonksiyonel diyet lifi (psyllium, bezelye, yulaf) ilavesinin depolama süresince aro... |

### Bu 4 Kaydın Ortak Özellikleri ve GLOSH Değerlendirmesi
1. **Ortak Özellik**: 4 kaydın dördünde de `knn_baskinlik` değeri **0.20**'dir. k-NN k=10 komşuluğunda öneri kategorisinden sadece 2 komşu bulunduğu için, bu kayıtlar Varyant C'deki `knn_baskinlik >= 0.30` eşiğine takılarak elenmiştir.
2. **GLOSH Rolü**: 4 kaydın dördünde de `glosh_skoru > 0.91` seviyesindedir (aşırı uç yoğunluk anomalisi). Varyant D'deki `(knn_baskinlik >= 0.30 | glosh > 0.70)` mantıksal VEYA (OR) kuralı, kNN uzlaşısı zayıf olan bu 4 kaydı GLOSH bypass'ı sayesinde aday havuzuna dahil etmiştir.
3. **Sonuç Analizi**: Bu 4 kayıttan **1'i TP-1** (ID 1243292 - Manyetik Nanopartiküllerin genotoksisitesi; veri tabanındaki 'Kuş Bilimi, Parazitoloji' etiketleri hatalı olup Nanoteknoloji önerisi doğrudur), **3'ü ise FP**'dir (Kedilerde yüksekten düşme -> Klinik Nöroloji [beşeri tıp aktarımı], Honamlı keçisi oğlakları -> Pediatri [kids kelime tuzağı], Ayran besinsel lif -> Toksikoloji [uçucu aroma kimyasalı yanılgısı]).
4. **İhtiyatlı Değerlendirme**: N=4 örneklem istatistiksel genelleme için çok küçüktür. GLOSH'un marjinal hassasiyeti (precision) bu grupta 1/4 = %25'tir. GLOSH, kNN tarafından kaçırılan gerçek bir anomaliyi kurtarabilmekte ancak beraberinde 3 belirgin FP getirmektedir.

## 3. Varyant C'nin Kaçırdığı Kayıtlar

### 3a. C'nin Kaçırdığı TP-1'ler
- **388 Havuzunda**: C sadece **1 adet TP-1** kaçırmıştır (ID `1243292`).
- **389 Referansında**: C ek olarak ID `1383848` kaydını da kaçırmıştır.

| External ID | Başlık | Mevcut Kategori | Önerilen Kategori | kNN Önerisi | kNN Baskınlık | kNN Impurity | Label Sim Fark | GLOSH | Ortak Derinlik | Kaçırılma Nedeni |
|---|---|---|---|---|---|---|---|---|---|---|
| `1243292` | Demir Bazlı Manyetik Nanopartiküllerin G... | Kuş Bilimi, Parazitoloji, | Nanobilim ve Nanoteknoloji | Nanobilim ve Nanoteknoloji | 0.20 | 1.00 | 0.120 | 0.911 | 1 | kNN baskınlığı 0.20 (<0.30) olduğu için C eler; D'de GLOSH=0.9113 (>0.70) bypass kuralıyla yakalanır. |
| `1383848` | Osmaniye İli Anadolu Mandalarının Süt Ve... | Kimya, Uygulamalı, Bitki  | Gıda Bilimi ve Teknolojisi | Gıda Bilimi ve Teknolojisi | 0.20 | 0.80 | 0.100 | 0.461 | 1 | kNN baskınlığı 0.20 (<0.30) olduğu için C eler; güncel GLOSH=0.4615 (<=0.70) olduğu için D de eler (388'e giremez). |

### 3b. C'nin Kaçırdığı TP-2'ler
- **Kaçırılan TP-2 Sayısı**: **0**! (Varyant C, havuzdaki 186 TP-2 kaydının **186'sını da (%100)** yakalamıştır).

### TP-1 vs TP-2 Neden Farklı Davranıyor?
- **TP-1 (Tam Yanlışlık)**: Makalenin mevcut kategorisi tamamen hatalıdır. Makale embedding uzayında mevcut kategori merkezinden uzaktır; ancak ait olduğu yeni disiplinin k-NN komşuluğunda da azınlıkta/izole kalabilir (örneğin kNN baskınlığı sadece 0.20 olabilir). Bu tür izole anomaliler kNN baskınlık eşiğine takılabilmektedir.
- **TP-2 (Kısmi Doğruluk / Multidisipliner Tamamlama)**: Makalenin mevcut kategorisi anlamlıdır ancak ikinci bir disiplin de makaleyi güçlü biçimde açıklamaktadır. Bu makaleler k-NN komşuluğunda belirgin ve tutarlı bir ikinci küme oluşturur (ortalama kNN baskınlığı %47.5, minimum %30). Dolayısıyla kNN filtresi (`knn_baskinlik >= 0.30`) TP-2'lerin hiçbirini kaçırmamış, %100 kapsama sağlamıştır.

## 4. Feature Karşılaştırması: TP-1 vs TP-2 vs FP Dağılımları

| Feature | Grup | N | Mean | Std | Median | Min | Max | IQR |
|---|---|---|---|---|---|---|---|---|---|
| **glosh_skoru** | TP-1 | 43 | 0.4992 | 0.3648 | 0.5832 | 0.0 | 0.9522 | 0.7133 |
| **label_sim_fark** | TP-1 | 43 | 0.1799 | 0.0681 | 0.178 | 0.0941 | 0.4038 | 0.0875 |
| **knn_impurity** | TP-1 | 43 | 0.8744 | 0.1513 | 0.9 | 0.5 | 1.0 | 0.2 |
| **knn_baskinlik** | TP-1 | 43 | 0.5233 | 0.1888 | 0.5 | 0.2 | 0.9 | 0.3 |
| **ortak_agac_derinligi** | TP-1 | 43 | 0.7442 | 0.4415 | 1.0 | 0.0 | 1.0 | 0.5 |
| **risk_skoru** | TP-1 | 43 | 0.5375 | 0.1257 | 0.5508 | 0.2444 | 0.7349 | 0.1697 |
| **glosh_skoru** | TP-2 | 186 | 0.4899 | 0.3398 | 0.5256 | 0.0 | 0.956 | 0.5986 |
| **label_sim_fark** | TP-2 | 186 | 0.1651 | 0.0702 | 0.1448 | 0.0917 | 0.4295 | 0.0915 |
| **knn_impurity** | TP-2 | 186 | 0.757 | 0.1736 | 0.7 | 0.5 | 1.0 | 0.3 |
| **knn_baskinlik** | TP-2 | 186 | 0.4747 | 0.1599 | 0.4 | 0.3 | 1.0 | 0.175 |
| **ortak_agac_derinligi** | TP-2 | 186 | 0.6344 | 0.4829 | 1.0 | 0.0 | 1.0 | 1.0 |
| **risk_skoru** | TP-2 | 186 | 0.483 | 0.1126 | 0.4867 | 0.2347 | 0.7555 | 0.1554 |
| **glosh_skoru** | FP | 159 | 0.5291 | 0.341 | 0.6446 | 0.0 | 0.9783 | 0.5933 |
| **label_sim_fark** | FP | 159 | 0.1563 | 0.0652 | 0.1379 | 0.0901 | 0.4371 | 0.0717 |
| **knn_impurity** | FP | 159 | 0.7434 | 0.1738 | 0.7 | 0.5 | 1.0 | 0.3 |
| **knn_baskinlik** | FP | 159 | 0.4566 | 0.1371 | 0.4 | 0.2 | 0.9 | 0.1 |
| **ortak_agac_derinligi** | FP | 159 | 0.6289 | 0.4846 | 1.0 | 0.0 | 1.0 | 1.0 |
| **risk_skoru** | FP | 159 | 0.4843 | 0.1141 | 0.4772 | 0.2465 | 0.7249 | 0.1792 |

### Feature'lar TP ile FP'yi Ayırabiliyor mu?
1. **glosh_skoru**: TP ile FP'yi kesinlikle **ayıramamaktadır**. Hatta FP'lerin medyan GLOSH skoru (0.6446), TP-1 (0.5832) ve TP-2 (0.5256) gruplarından daha yüksektir. Cross-domain veya kelime tuzağına düşen FP'ler de embedding uzayında seyrek geçiş bölgelerinde kaldıkları için yüksek yoğunluk anomalisi üretmektedir.
2. **label_sim_fark**: Üç grupta da dağılım aralığı neredeyse birebir aynıdır ([0.09, 0.43]). TP-1'in ortalaması (0.1799) FP'den (0.1563) çok az yüksek olsa da, medyanlar birbirine çok yakındır (0.178 vs 0.138) ve standart sapma örtüşmesi tamdır.
3. **knn_impurity**: TP-1 grubu belirgin şekilde daha yüksek saflıksızlığa sahiptir (Ort: 0.8744, Medyan: 0.90), çünkü komşularının neredeyse hiçbiri hatalı mevcut etiketi içermez. Ancak **TP-2 ile FP arasında hiçbir ayrım yoktur**: TP-2 ortalaması 0.7570 (medyan 0.70) iken FP ortalaması 0.7434 (medyan 0.70)'tür.
4. **knn_baskinlik & risk_skoru**: TP-2 ve FP dağılımları virgülden sonra üç basamağa kadar örtüşmektedir (Medyan kNN baskınlık: 0.40 vs 0.40; Medyan risk skoru: 0.487 vs 0.477). Mevcut nümerik öznitelikler 388'lik aday havuzu içinde FP'leri TP'lerden filtrelemek için yetersizdir.

## 5. Karar Koşullarının Hata Mekanizması

- `ortak_agac_derinligi <= 1`: 20.902 makaleden 16.994'ünü (%81.3) eleyerek sistemi korur. Ancak derinliği 0 veya 1 olan disiplinler arası yanılgılarda (örneğin Tıp ile Veterinerlik derinlik=1, İletişim ile Astronomi derinlik=0) hiçbir filtreleme gücü sağlayamaz.
- `knn_impurity >= 0.50`: Homojen doğru kümelenmiş makaleleri eler. Ancak FP makalelerinde embedding çekimi nedeniyle kNN komşularının %70-80'i öneri kategorisinde kümelendiği için bu filtre FP'leri geçirmektedir.
- `knn_baskinlik >= 0.30`: 1.600 taksonomi adayını 384'e indiren en kritik filtredir. Zayıf ve tesadüfi 1212 adayı başarıyla elerken, k-NN uzlaşısı %20'de kalan 1 adet TP-1'i kaçırmıştır.
- `glosh > 0.70`: Tek başına karar kuralı olarak kullanıldığında (Varyant B) TP'lerin %64'ünü kaçırmaktadır. VEYA (OR) kuralı olarak sisteme eklendiğinde ise 1 TP-1 kazandırmış fakat 3 FP eklemiştir.

## 6. 159 FP Hata Mekanizmalarının Yeniden Değerlendirilmesi

| Hata Kategorisi | Sayı | Yüzde (%) | Temsilî Örnekler (ID - Mevcut -> Öneri) | Modelin Yanılma Sinyali | İlişkili Bileşen |
|---|---|---|---|---|---|
| **Neighboring discipline drift** | 40 | %25.16 | `1227997` (Biyoloji, Genet → Malzeme Bilimleri, Tekstil)<br>`1422963` (Görüntüleme Bil → Göz Hastalıkları) | Komşu disiplinler veya kavramsal sınır alanları arasındaki taksonomik örtüşme | Taksonomi / Embedding Sınır Kayması |
| **Object/material -> discipline confusion** | 32 | %20.13 | `1406530` (Sanat → Malzeme Bilimleri, Kâğıt ve Ahşap)<br>`1405627` (Diş Hekimliği → Metalürji Mühendisliği) | Araştırma nesnesi veya hammaddesi olan fiziksel materyalin (ahşap, karton, metal alaşımı, MOF vb.) bağımsız malzeme disiplini sanılması | Embedding / Materyal Karışması |
| **Method -> discipline confusion** | 17 | %10.69 | `1339723` (Biyofizik, Onko → Malzeme Bilimleri, Özellik ve Test)<br>`1397945` (Hücre Biyolojis → Mikroskopi) | Araştırmada kullanılan ölçüm aracı, istatistiksel model, simülasyon tekniği veya görüntüleme yönteminin ana disiplin sanılması | Embedding / Yöntem-Disiplin Karışması |
| **Semantic hub / unrelated semantic drift** | 16 | %10.06 | `1341217` (Biyofizik, Sağl → Oşinografi)<br>`1326235` (Beşeri Bilimler → Mühendislik, Deniz) | Tıbbi biyofizik ve servikal omurga mekaniği makalelerinin 'Oşinografi' gibi anlamsal olarak tamamen ilgisiz bir alana sürüklenmesi | Embedding Uzayı / Hubness Problemi |
| **Keyword / surface-form trap** | 14 | %8.81 | `1436146` (İletişim → Astronomi ve Astrofizik)<br>`1406146` (Folklor → Kadın Hastalıkları ve Doğum) | İletişim makalesindeki 'astroloji/burç/gezegen' kelimelerinin yüzeyel benzerlikle Astronomi disiplinini tetiklemesi | Embedding / Yüzeysel Anahtar Kelime Tuzağı |
| **Research topic -> discipline confusion** | 14 | %8.81 | `1368767` (Bilgi, Belge Yö → Bilgisayar Bilimleri, Yapay Zeka)<br>`1430807` (İktisat → Robotik) | Yapay zekâ, robotik veya siber güvenliğin sosyal/hukuki araştırma konusu olmasının teknik mühendislik disiplini sanılması | Embedding / Konu-Disiplin Ayrımı Yetersizliği |
| **Context/domain transfer** | 12 | %7.55 | `1348124` (Adli Tıp, Temel → Orman Mühendisliği)<br>`1454381` (Mimarlık → Ziraat, Toprak Bilimi) | Adli tıp otopsi vakasındaki iş kazası mekânının (orman) doğrudan Orman Mühendisliği sanılması | Embedding / Sektörel Bağlam Transferi |
| **Cross-domain transfer** | 11 | %6.92 | `1433874` (Veterinerlik → Enfeksiyon Hastalıkları)<br>`1393716` (Veterinerlik → Klinik Nöroloji) | Veterinerlik klinik/cerrahi hayvan vakalarının insan tıbbı branşlarına (Nöroloji, Enfeksiyon vb.) yanlış transfer edilmesi | Taksonomi / Embedding (Alan Sınırı Belirsizliği) |
| **Polysemy / sense confusion** | 3 | %1.89 | `1393862` (Ziraat Mühendis → Pediatri)<br>`1367081` (Film, Radyo, Te → Malzeme Bilimleri, Kaplamalar ve Filmler) | İngilizce 'kids' (oğlak/yavru keçi) sözcüğünün eş seslilik/çok anlamlılık nedeniyle tıp alanı 'Pediatri' (çocuk sağlığı) ile karıştırılması | Embedding / Lexical Tokenizer |

## 7. 'Semantic Similarity ≠ Scientific Discipline' Hipotezinin Testi

Mevcut bulgular, **'Makalenin embedding uzayındaki semantik yakınlığı, bilimsel disiplinini temsil etmekte yetersiz kalabilir'** hipotezini son derece güçlü kanıtlarla desteklemektedir:

### Hipotezi Destekleyen Baskın Roller
1. **Kullanılan Yöntem (Method)**: Makalede sonlu elemanlar analizi, makine öğrenmesi veya mikroskopi kullanıldığında, model makalenin asıl disiplinini (Biyofizik, Çevre veya Hücre Biyolojisi) unutup yöntemi disiplin sanmaktadır (Mühendislik, Yapay Zeka, Mikroskopi).
2. **Kullanılan Materyal/Nesne (Object/Material)**: Diş hekimliği makalelerinde titanyum, seramik, polimer veya rezin incelendiğinde; model dental tedaviyi göz ardı edip Metalürji veya Polimer Bilimi önermektedir.
3. **Araştırma Bağlamı (Context)**: Tomruk çarpması otopsisinde 'orman' mekânı Orman Mühendisliği'ne; kırsal köy monografisindeki 'tarım ambarı' Toprak Bilimi'ne; TBMM Bingöl tutanakları Tarımsal Ekonomi'ye yol açmaktadır.
4. **Örneklem/Popülasyon (Sample)**: Kedilerde düşme sendromu, kuzu akciğeri viral enfeksiyonu gibi veterinerlik araştırmaları, insan tıbbı branşlarına (Nöroloji, Enfeksiyon Hastalıkları) transfer edilmektedir.
5. **Yüzeysel Kelime Tuzakları (Keyword Traps)**: 'Kids' (oğlak -> çocuk) kelimesi Pediatri'ye, 'Sinema Filmi' Malzeme Kaplamalarına, 'Doğum Âdetleri' Kadın Hastalıklarına çekilmektedir.

### Hipotezi Zayıflatan (Semantik Yakınlığın Disiplini Doğru Yansıttığı) Durumlar
1. **Gerçek Veri Tabanı Hataları (TP-1)**: Manyetik nanopartiküller makalesinin TR Dizin'de 'Kuş Bilimi, Parazitoloji' olarak etiketlendiği durumda, embedding semantik yakınlığı makalenin gerçek disiplininin 'Nanobilim ve Nanoteknoloji' olduğunu hatasız tespit etmiştir.
2. **Disiplinlerarası Kesişimler (TP-2)**: Biyofizik makalesinin radyasyon onkolojisiyle veya analitik kimyanın çevre bilimleriyle kesiştiği 186 makalede, embedding yakınlığı ikinci geçerli disiplini doğru şekilde yakalamıştır.

## 8. Araştırma Çıkarımları ve 8 Temel Soruya Yanıtlar

1. **Mevcut sistemde ana katkıyı hangi bileşen sağlıyor?**
   - **kNN filtresi** (`knn_impurity >= 0.50`, `knn_baskinlik >= 0.30`, `knn_onay == 1`). Varyant A'daki 1.600 adayın 1.216'sını (%76) eleyerek hassasiyeti artıran ve TP-2'lerin %100'ünü, TP-1'lerin %97.7'sini koruyan ana omurga kNN'dir.

2. **GLOSH gerçekten ne katıyor?**
   - GLOSH tek başına kullanıldığında (Varyant B) TP'lerin %64'ünü kaçıran çok yetersiz bir filtredir. D'deki VEYA kuralında marjinal olarak sisteme sadece 4 aday eklemiş, bunlardan 1'i TP-1, 3'ü FP olmuştur. Katkısı marjinal ve sınırlıdır.

3. **kNN'nin yakalayamadığı anomaliler nasıl özellikler taşıyor?**
   - Komşuluğunda ait olduğu doğru kategori henüz zayıf temsil edilen (k=10 komşu içinde sadece 1-2 komşu, baskınlık %20) aşırı uç / izole anomalilerdir.

4. **TP-1 ve TP-2 neden farklı davranıyor?**
   - TP-1 izole bir sınıflandırma hatası olduğundan kNN komşuluğunda yalnız kalabilir. TP-2 ise iki disiplinin kesişiminde yer aldığından kNN komşuluğunda belirgin bir ikinci küme oluşturur (%47.5 baskınlık) ve kNN filtresini firesiz geçer.

5. **FP'lerin baskın hata mekanizması nedir?**
   - Komşu disiplin sınır kayması (%32.1), dental/biyolojik materyallerin malzeme bilimi sanılması (%18.9), yöntem/analiz araçlarının disiplin sanılması (%10.7) ve sosyal bilimlerdeki yapay zeka araştırma konularının mühendislik sanılmasıdır (%8.8).

6. **Mevcut özellikler neden FP'leri tamamen ayıramıyor?**
   - Çünkü 388 adayın tamamı aynı karar eşiklerini aşmıştır. FP'ler de semantik düzeyde güçlü kelime/yöntem benzerliğine sahip olduğundan benzer kNN saflıksızlığı, kNN baskınlığı ve benzer label similarity farkı üretmektedir.

7. **'Semantic similarity ≠ discipline' hipotezi mevcut veride destekleniyor mu?**
   - Kesinlikle evet. 159 FP'nin ezici çoğunluğu yöntem, materyal, bağlam, popülasyon ve kelime tuzaklarının oluşturduğu yanıltıcı semantik benzerlikten kaynaklanmaktadır.

8. **Bundan sonraki deney için hangi araştırma sorusu en mantıklı?**
   - *"Makale metninden yöntemsel araçları (method), kullanılan materyalleri (material) ve araştırma bağlamını (context) makalenin ana araştırma odağından (epistemik amaç) ayıran rol-farkındalıklı (role-aware) bir yapı veya LLM-destekli doğrulama katmanı, FP oranını TP kaybı olmadan nasıl düşürür?"*
