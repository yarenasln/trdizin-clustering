# TR Dizin Yanlış Pozitif (FP) Hata Taksonomisi Metodolojik Doğrulama Raporu

Bu rapor, ablation analizi sonrasında tespit edilen 159 Yanlış Pozitif (FP) adayın hata taksonomisinin metodolojik sağlamlığını, çoklu hata mekanizmalarını (primary / secondary), sınır vakalarını ve 'semantic similarity ≠ scientific discipline' hipotezinin veriye dayalı geçerliliğini doğrulamaktadır.

## 1. D \ C Kümesinin Kesin Doğrulanması

- **Kaynak Dosya**: `results/ablation/glosh_added_candidates_D_minus_C.csv`
- **|D \ C| Büyüklüğü**: **4** kayıt
- **Benzersiz External ID Sayısı**: **4**
- **Benzersiz External ID Listesi**: `1243292`, `1393716`, `1393862`, `1400625`

### Manuel Karar ve Dağılım
- **TP-1**: 1 kayıt (%25.0)
- **TP-2**: 0 kayıt (%0.0)
- **FP (FP-1)**: 3 kayıt (%75.0)
- **Toplam**: 4 kayıt (%100.0)

### Kayıt Bazında Doğrulama Listesi

| External ID | Başlık | Mevcut Kategori | Önerilen Kategori | GLOSH | kNN Bask. | Karar | Manuel Gerekçe Özeti |
|:---|:---|:---|:---|:---:|:---:|:---:|:---|
| `1243292` | Demir Bazlı Manyetik Nanopartiküllerin Genoto... | Kuş Bilimi, Parazitoloji, | Nanobilim ve Nanoteknoloji | 0.911 | 0.20 | **TP-1** | Manyetik nanopartiküllerin (Fe3O4, NiFe2O4 vb.) model organizma Drosophila melanogaster üz... |
| `1393716` | Yüksekten Düşme Sendromlu Kedilerde Klinik ve... | Veterinerlik | Klinik Nöroloji | 0.934 | 0.20 | **FP-1** | Atatürk Üniversitesi Veteriner Fakültesi Hayvan Hastanesi'ne getirilen yüksekten düşme sen... |
| `1393862` | Modeling Early Growth of Honamlı Kids Using N... | Ziraat Mühendisliği | Pediatri | 0.931 | 0.20 | **FP-1** | Honamlı keçisi oğlaklarında (Honamlı kids) canlı ağırlık artışı ve erken dönem büyüme dina... |
| `1400625` | PSYLLİUM, BEZELYE VE YULAF KAYNAKLI BESİNSEL ... | Gıda Bilimi ve Teknolojis | Toksikoloji | 0.921 | 0.20 | **FP-1** | Ayran üretiminde fonksiyonel diyet lifi (psyllium, bezelye, yulaf) ilavesinin depolama sür... |

### Çift Görünme / Tablo Bozulması Kontrolü
- **Veri Kümesi Düzeyinde Kontrol**: `glosh_added_candidates_D_minus_C.csv` dosyasında hiçbir External ID mükerrer (duplicate) değildir; satır sayısı 4, tekil ID sayısı 4'tür.
- **Raporlama Düzeyinde Kontrol**: Önceki raporda ID `1243292` hem 'D \ C Kümesi Tablosu'nda (Tablo 2) hem de 'C'nin Kaçırdığı TP-1'ler Tablosu'nda (Tablo 3a) listelenmiştir. Bu bir tablo veya script hatası **değildir**; bilakis matematiksel bir gerekliliktir: D'nin C'ye GLOSH ile eklediği tek TP-1 bu kayıt olduğu için ($D \setminus C$), C'nin 388 havuzunda kaçırdığı tek TP-1 de zorunlu olarak aynı ID `1243292` kaydıdır. Tablolarda boru karakteri (`|`) veya metin kırılmasına bağlı hiçbir Markdown tablo bozulması bulunmamaktadır.

## 2. 159 FP Hata Taksonomisinin Metodolojik İncelemesi

Mevcut 9 hata sınıfı, 159 FP kaydının başlık, kategori ve manuel denetim gerekçeleri (`Manuel_Gerekce`) taranarak metodolojik teste tabi tutulmuştur:

### Soru 1: Bir FP birden fazla hata mekanizmasına aynı anda sahip olabilir mi?
**Evet, kesinlikle.** 159 FP'nin **98'inde (%61.6)** birden fazla hata mekanizması iç içe geçmiştir. Örneğin, bir makale sektörel bağlam nedeniyle (*Context transfer*) yanlış disipline yönelirken, başlığındaki spesifik bir kelime (*Keyword trap*) bu kaymayı tetikleyen araç olmaktadır. Benzer şekilde, bir yöntem (*Method confusion*) tıp alanından hayvan araştırmasına uygulanırken (*Cross-domain transfer*) ikili hata üretmektedir.

### Soru 2: Bazı sınıflar birbirine fazla mı yakın?
- *Keyword/surface-form trap* ile *Polysemy/sense confusion* kavramsal olarak yakın görünse de ontolojik olarak ayrılır: Polysemy'de aynı sözcük biçimi iki farklı anlam taşır (ör. 'kids' = oğlak vs çocuk; 'film' = sinema vs ince film kaplama). Keyword trap'te ise sözcük anlamı değişmez (ör. 'astroloji', 'doğum', 'su'); ancak makalenin ana disiplinini değil ikincil bir ögesini temsil ettiği halde modeli saptırır.
- *Object/material* ile *Method*: İncelenen fiziksel madde (titanyum, seramik, polimer, ekstrakt) ile onu incelemek için uygulanan prosedür (sonlu elemanlar, spektroskopi, mikroskopi, simülasyon) ayrımı nettir.

### Soru 3: 'Neighboring discipline' ile 'Cross-domain transfer' arasında tutarlı bir ayrım var mı?
**Evet.** Bu iki sınıf arasındaki sınır son derece belirgin ve tutarlıdır:
- **Neighboring discipline drift**: Aynı epistemik üst dalda veya kardeş disiplinler arasındaki taksonomik sınır belirsizliğidir (ör. Jeomorfoloji vs. Jeoloji Mühendisliği; Diş Hekimliği vs. KBB; Analitik Kimya vs. Nükleer Kimya). İki alan da insan veya doğa bilimlerinde aynı nesneyi benzer araçlarla inceler.
- **Cross-domain transfer**: Ontolojik sınırın aşılmasıdır. Özellikle **Veterinerlik/Zooloji (hayvan) çalışmalarının insan Klinik Tıbbı branşlarına transfer edilmesi** (kedide nörolojik travma -> Klinik Nöroloji; kuzu pnömonisi -> Enfeksiyon Hastalıkları; kertenkele gözü -> Oftalmoloji) belirgin bir ontolojik alan transferidir.

### Soru 4: 'Object/material → discipline' ile 'Method → discipline' sınır vakaları var mı?
Evet, sınır vakalar mevcuttur. Örneğin ID `1403661`'de hem dental adeziv biyomalzeme (materyal) hem de sonik aktivasyon cihazı (yöntem) incelenmektedir. Kuralımız: Modelin önerdiği kategori bir analiz aracı/tekniğiyse (*Akustik, Mikroskopi, İstatistik, YZ*) bu **Method**; incelenen hammaddenin/maddenin bilimi ise (*Polimer Bilimi, Metalürji Mühendisliği*) bu **Object/material** olarak belirlenmiştir.

### Soru 5: 'Semantic hub' gerçekten ayrı bir hata mekanizması mı, yoksa diğer hataların sonucu mu?
- 'Mühendislik, Deniz' ve 'Oşinografi' kategorilerine çekilen makalelerin bir kısmında zayıf sözcük çağrışımları ('Van Gölü', 'dalga/sıvı') bulunsa da; Topkapı Sarayı mutfakları, mezar taşları, Tire folkloru, omurga biyofiziği ve felsefi film analizlerinin topluca aynı etiketlere yığılması tekil kelime tuzaklarıyla açıklanamaz.
- Bu durum, yüksek boyutlu uzaylarda bilinen **Hubness Problemi**nin (Radovanović et al., 2010) doğrudan bir sonucudur. Dolayısıyla bu sınıf, modelin ve embedding uzayının geometrik/topolojik kusurunu temsil eden bağımsız bir hata mekanizmasıdır.

## 3. Primary ve Secondary Hata Mekanizması Dağılımı

159 FP kaydı için gerekçe metninden açıkça desteklenen birincil (*primary_error*) ve ikincil (*secondary_error*) sınıflar atanmıştır. Hiçbir yapay etiket uydurulmamış; gerekçede ikincil kanıt bulunmayan 61 kayıtta ikincil alan boş bırakılmıştır.

### Doğrulanmış Taksonomi Özet Tablosu

| Hata Sınıfı | Primary Sayı | Primary Yüzde (%) | Secondary Sayı | Secondary Yüzde (159 içinde %) |
|:---|:---:|:---:|:---:|:---:|
| **Neighboring discipline drift** | 36 | %22.64 | 4 | %2.52 |
| **Object/material -> discipline confusion** | 26 | %16.35 | 7 | %4.40 |
| **Context/domain transfer** | 18 | %11.32 | 6 | %3.77 |
| **Method -> discipline confusion** | 17 | %10.69 | 4 | %2.52 |
| **Semantic hub / unrelated semantic drift** | 17 | %10.69 | 0 | %0.00 |
| **Keyword / surface-form trap** | 16 | %10.06 | 74 | %46.54 |
| **Research topic -> discipline confusion** | 15 | %9.43 | 0 | %0.00 |
| **Cross-domain transfer** | 11 | %6.92 | 2 | %1.26 |
| **Polysemy / sense confusion** | 3 | %1.89 | 1 | %0.63 |
| **TOPLAM** | **159** | **%100.00** | **98** | **%61.64** |

### Secondary Mekanizmaların Önemi
- İkincil mekanizmalarda en büyük payı **Keyword / surface-form trap (74 kayıt, %46.5)** almaktadır.
- Bu bulgu, modelin materyal, yöntem veya bağlam karışıklığı yaşarken, tetikleyici kıvılcımın metindeki yüzeysel bir sözcükten ('titanyum', 'orman', 'kurtarma', 'pestisit', 'makine öğrenmesi') geldiğini somutlaştırmaktadır.

## 4. Her Primary Sınıf İçin Güçlü ve Sınır Vakalar

### Neighboring discipline drift
**Güçlü / Temsilî Örnekler:**
- `ID: 1389941` | *Türkiye Üniversitelerinin Jeomorfoloji Alanındaki Lisansüstü Tezlerinin Bibliyometrik Analizi* (Coğrafya → Mühendislik, Jeoloji): Jeomorfoloji fiziki coğrafya ile jeoloji mühendisliğinin doğrudan ortak sınırındadır.
- `ID: 1404126` | *Predictive Value of Panoramic Referral Signs in Third Molar Surgery* (Diş Hekimliği → Kulak, Burun, Boğaz): Maksillofasiyal diş cerrahisi ile KBB anatomik olarak doğrudan komşudur.
**Sınır Vaka:**
- `ID: 1372974` | *TBDY-2018 Esaslı Zemin Sıvılaşma Potansiyelinin Çözümlenmesinde Enerji Oranı Düzeltme Katsayısının Etkisi* (İnşaat Müh. → Su Kaynakları): **İnşaat geoteknik zemin mekaniği ile hidroloji komşudur; ancak 'sıvılaşma/su' sözcüğü nedeniyle Keyword Trap ile sınırdadır.**

### Object/material -> discipline confusion
**Güçlü / Temsilî Örnekler:**
- `ID: 1405627` | *Farklı Üretim Yöntemleri ve Anodizasyon Parametrelerinin Rezin Siman-Titanyum Bağlantısına Etkisi* (Diş Hekimliği → Metalürji Mühendisliği): Protetik diş tedavisinde kullanılan titanyum altyapı bağımsız metalürji mühendisliği sanılmıştır.
- `ID: 1403660` | *Kendinden Asitli Primer ve Hidroflorik Asidin Kombine Uygulamasının Rezin Siman...* (Diş Hekimliği → Polimer Bilimi): Dental restorasyonda kullanılan polimer infiltre seramik ağ materyali polimer mühendisliği sanılmıştır.
**Sınır Vaka:**
- `ID: 1406530` | *AMBALAJ TASARIMINDA KARTON MAKET YAPIMININ ROLÜ VE BİR UYGULAMA* (Sanat → Malzeme Bilimleri, Kâğıt ve Ahşap): **Tasarım prototiplemesinde karton kullanımı incelenmektedir; hem incelenen materyal hem de 'karton/kağıt' kelime tuzağıdır.**

### Context/domain transfer
**Güçlü / Temsilî Örnekler:**
- `ID: 1348124` | *Tomruk Çarpmasına Bağlı Ölümler: Bir Otopsi Çalışması: Retrospektif Çalışma* (Adli Tıp → Orman Mühendisliği): Adli tıp otopsisindeki iş kazası mekânı (orman) doğrudan mühendislik disiplini sanılmıştır.
- `ID: 1454381` | *KAPADOKYA KIRSALINDA BİR YERLEŞİM MONOGRAFİSİ: GÖRE BELDESİ’NİN MİMARİ VE KÜLTÜREL MİRASI* (Mimarlık → Ziraat, Toprak Bilimi): Kırsal köy mimarisi monografisindeki yerleşim ve ambar bağlamı ziraat sanılmıştır.
**Sınır Vaka:**
- `ID: 1445833` | *International Carriage of Goods Contracts Defeated by Coronavirus...* (Taşınım Bilimi, Hukuk → Viroloji): **Uluslararası eşya taşıma hukukunda Covid-19 pandemisi olgusal bağlamdır; metindeki 'Coronavirus' kelimesi de Viroloji tuzağını kurmuştur.**

### Method -> discipline confusion
**Güçlü / Temsilî Örnekler:**
- `ID: 1339723` | *Comparison of the Sensitive Volume Response of Cylindrical Ionization Chambers with GEANT4* (Biyofizik, Onkoloji → Malzeme Bilimleri, Özellik ve Test): Radyoterapide iyon odası dozimetrik yanıt simülasyonu bağımsız malzeme testi sanılmıştır.
- `ID: 1397945` | *M1 POLARİZE MAKROFAJ FARKLILAŞMASINDA GÜNCEL TEKNİKLER...* (Hücre Biyolojisi → Mikroskopi): Hücresel immünolojide fenotipik analiz aracı olan mikroskopi/sitometri yöntemi makalenin disiplini sanılmıştır.
**Sınır Vaka:**
- `ID: 1403661` | *Sonik Uygulama Modunun İki Farklı Adeziv Sistemin Mikrogerilim Bağlanma Dayanımına Etkisi* (Diş Hekimliği → Akustik): **Diş hekimliğinde sonik aktivasyon el aletinin (yöntem) kullanımı ile dental adeziv biyomalzeme (materyal) arasında sınır vakadır.**

### Semantic hub / unrelated semantic drift
**Güçlü / Temsilî Örnekler:**
- `ID: 1341217` | *Portatif sagital plan omurga eğriliği ölçüm sistemi tasarımı ve ön değerlendirmesi* (Biyofizik → Oşinografi): Omurga eğriliği biyomedikal cihaz prototipi alakasız biçimde Oşinografi (okyanus bilimi) hub'ına çekilmiştir.
- `ID: 1326235` | *Soyut ve Somut Kültürel Miras Olarak Topkapı Sarayı Mutfakları – Matbah-ı Amire* (Sosyoloji → Mühendislik, Deniz): Osmanlı saray mutfak mimarisi ve yemek kültürü alakasız 'Mühendislik, Deniz' hub'ına sürüklenmiştir.
**Sınır Vaka:**
- `ID: 1396886` | *Van Gölü Havzası Seramiklerinde Kullanılan Motifler ve Boyalar* (Arkeoloji → Mühendislik, Deniz): **Seramik arkeolojisi 'Mühendislik, Deniz' hub'ına çekilmiştir; ancak başlıktaki 'Van Gölü' göl/su çağrışımı da tetikleyicidir.**

### Keyword / surface-form trap
**Güçlü / Temsilî Örnekler:**
- `ID: 1436146` | *BİLGİ KURAMI BAĞLAMINDA KEHANET İLETİŞİMİ VE ASTROLOJİNİN SOSYAL MEDYADA YER ALIŞ BİÇİMLERİ* (İletişim → Astronomi ve Astrofizik): Sosyal medya analizi makalesindeki 'astroloji/burç/gezegen' kelimeleri doğrudan Astronomiyi tetiklemiştir.
- `ID: 1406146` | *TÜRK HALK KÜLTÜRÜNDE DOĞUM ÂDETLERİ BAĞLAMINDA DÜŞÜKTEN KORUNMA YÖNTEMLERİ...* (Folklor → Kadın Hastalıkları ve Doğum): Folklor saha derlemesindeki 'doğum/düşük' kelimeleri klinik obstetriği tetiklemiştir.
**Sınır Vaka:**
- `ID: 1382585` | *ANALYSIS OF LIQUEFACTION POTENTIAL OF SOILS BY OBJECT ORIENTED COMPUTER SOFTWARE...* (İnşaat Müh. → Ziraat, Toprak Bilimi): **Zemin mekaniğindeki 'soil' kelimesi zemin yerine tarımsal toprak sanılmıştır; hem kelime tuzağı hem eş seslilik taşır.**

### Research topic -> discipline confusion
**Güçlü / Temsilî Örnekler:**
- `ID: 1368767` | *Bilgi ve Belge Yönetimi Bölümü Öğrencilerinin Yapay Zekâ Okuryazarlığı Düzeyleri...* (Bilgi-Belge Yön. → Bilgisayar Bilimleri, Yapay Zeka): Kütüphanecilikte YZ farkındalık anketi teknik yapay zeka mühendisliği sanılmıştır.
- `ID: 1430807` | *ROBOT VERGİSİNİN UYGULANABİLİRLİĞİ ÜZERİNE BİR İNCELEME* (İktisat → Robotik): Maliye politikasındaki robot vergisi tartışması teknik robotik mühendisliği sanılmıştır.
**Sınır Vaka:**
- `ID: 1405453` | *TELIF HUKUKU KAPSAMINDA METIN VE VERI MADENCILIĞINE ILIŞKIN MUKAYESELI BIR ANALIZ...* (Hukuk → Bilgisayar Bilimleri, Sibernitik): **Fikri mülkiyet hukukunda veri madenciliği ve YZ konusu teknik sibernetik sanılmıştır; konu-disiplin ve kelime tuzağı sınırındadır.**

### Cross-domain transfer
**Güçlü / Temsilî Örnekler:**
- `ID: 1433874` | *Kuzu Pnömonilerinde RSV İle PI3V Koenfeksiyonlarının Tespiti...* (Veterinerlik → Enfeksiyon Hastalıkları): Kuzularda viral pnömoni patolojisi insan tıp disiplini olan Enfeksiyon Hastalıklarına aktarılmıştır.
- `ID: 1393716` | *Yüksekten Düşme Sendromlu Kedilerde Klinik ve Nörolojik Bulguların Değerlendirilmesi* (Veterinerlik → Klinik Nöroloji): Hayvan hastanesine getirilen kedilerin nörolojik muayenesi insan Klinik Nörolojisine aktarılmıştır.
**Sınır Vaka:**
- `ID: 1393426` | *Blinking without eyelids: first video-documented evidence of a blink-like reflex in Ophisops elegans* (Zooloji → Göz Hastalıkları): **Kertenkele göz refleksi çalışması insan Göz Hastalıklarına aktarılmıştır; zoolojik organ adı kelime tuzağı da oluşturmuştur.**

### Polysemy / sense confusion
**Güçlü / Temsilî Örnekler:**
- `ID: 1393862` | *Modeling Early Growth of Honamlı Kids Using Nonlinear Growth Curves* (Ziraat Mühendisliği → Pediatri): İngilizce 'kids' (keçi oğlağı) sözcüğü tıp alanı Pediatri (çocuk sağlığı) ile karıştırılmıştır.
- `ID: 1367081` | *GÖRSEL KÜLTÜRÜN İZİNDE: SİNEMA ENDÜSTRİSİ ALANINDAKİ LİTERATÜRÜN ANALİZİ* (Film, TV → Malzeme Bilimleri, Kaplamalar ve Filmler): Sinema 'film' sözcüğü katı hal fiziğindeki ince film/kaplama ile eş seslilik yanılgısına yol açmıştır.
**Sınır Vaka:**
- `ID: 1374526` | *Sinema ve Estetik Alanında Bibliyometrik Analiz* (Film, Sanat → Malzeme Bilimleri, Kaplamalar ve Filmler): **Sinema estetiğindeki 'film' kelimesi eş seslidir; ancak bibliyometrik haritalama yöntemi nedeniyle Method ile de temas eder.**

## 5. 'Semantic Similarity ≠ Scientific Discipline' Hipotezinin Titiz Değerlendirilmesi

### Değerlendirme Düzeyi: **GÜÇLÜ DESTEK (STRONG SUPPORT)**

Bu değerlendirme toptancı bir abartı ('159 FP'nin tamamı kusursuzca böyledir') üzerinden değil, verinin sunduğu somut kanıtlar üzerinden yapılmıştır:

1. **Yöntem, Materyal, Bağlam, Konu ve Örneklem Rolleri**: 159 FP'nin **%54.7'si (87 kayıt)** doğrudan yöntem (17), materyal (26), sektörel bağlam (18), sosyal bilimlerdeki araştırma konusu (15) veya canlı popülasyonu/örneklem (11) rollerinin araştırma amacıyla karıştırılmasından doğmaktadır.
2. **Yüzeysel Sözcük ve Çok Anlamlılık**: 159 FP'nin **%12.0'si (19 kayıt)** birincil olarak yüzeysel sözcük tuzakları (16) ve leksikal çok anlamlılıktan (3) kaynaklanmaktadır. Ayrıca 74 kayıtta ikincil tetikleyici olarak yüzeyel sözcük tuzağı rol oynamaktadır (toplamda 90/159, %56.6).
3. **Semantik Hub / Geometrik Çekim**: 159 FP'nin **%10.7'si (17 kayıt)** embedding uzayındaki hubness (Oşinografi, Mühendislik Deniz) çekiminden doğmaktadır.
4. **Komşu Disiplin Sınır Kayması**: Geriye kalan **%22.6'lık kısım (36 kayıt)** kardeş disiplinler arasındaki taksonomik sınır belirsizliğini yansıtmaktadır.
5. **Sonuç**: Veri kümesi, embedding tabanlı semantik yakınlığın bir makalenin *'ne hakkında olduğunu'* (araştırma amacı / epistemik katkı) değil, metinde *'hangi sözcüklerin, materyallerin, araçların ve mekânların geçtiğini'* ölçtüğünü tartışmasız şekilde kanıtlamaktadır.

## 6. Yeni Yöntem Önermeden Sonraki Deney İçin Araştırma Hipotezi

Mevcut bulgular ışığında, bir sonraki deney için en sağlam, doğrudan veriden türetilmiş araştırma sorusu ve hipotezi şudur:

### Temel Araştırma Sorusu (Research Question)
> *"Bir akademik makalenin metnindeki kavramlar epistemik araştırma odağı (objective/focus), uygulanan yöntem (method), incelenen nesne/materyal (material), sektörel bağlam (context) ve örneklem popülasyonu (sample) rollerine ayrıştırıldığında; bu rol-farkındalıklı (role-aware) filtreleme, mevcut sistemin yakaladığı gerçek anomalileri (TP-1 ve TP-2) koruyarak yanlış pozitifleri (FP) istatistiksel olarak anlamlı düzeyde azaltabilir mi?"*

### Test Edilecek Temel Hipotez (Hypothesis H1)
> **Hipotez (H1)**: *"Mevcut sistemin tespit ettiği adaylara yönelik rol-farkındalıklı bir doğrulama katmanı uygulandığında, önerilen disiplinin makalenin 'araştırma odağı'ndan değil de yalnızca 'yöntem', 'materyal', 'bağlam' veya 'örneklem' bileşenlerinden kaynaklandığı durumlar elenerek, sistemin denetim hassasiyeti (audit precision) %59.0 seviyesinden anlamlı ölçüde yukarı çekilebilir; bu sırada gerçek disiplin kaymaları olan TP-1 ve TP-2'ler üzerindeki kayıp %5'in altında tutulabilir."*

