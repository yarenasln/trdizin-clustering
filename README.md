# TR Dizin Makale Kümeleme ve Anomali Tespiti

    

    TÜBİTAK ULAKBİM TR Dizin makalelerinin çok dilli semantik vektör temsilleri (*embeddings*) kullanılarak yoğunluk

  tabanlı (**HDBSCAN**) ve yarı denetimli (**Seeded & Adaptive K-Means**) algoritmalarla kümelenmesi, taksonomi

  uyuşmazlıklarının (anomali) tespiti ve sonuçların etkileşimli bir web panelinde sunulması için geliştirilmiş makine

  öğrenmesi ve görselleştirme aracıdır.

    

    Docker Compose ile Flask tabanlı Dashboard uygulaması (GPU destekli PyTorch/ML ortamı) ve PostgreSQL veritabanı

  servisi çalıştırılır.

    

    ---

    

    ## Hazırlık

    

    ```bash

    # 1. Depoyu klonlayın ve proje dizinine geçin

    git clone  https://github.com/yarenasln/trdizin-clustering

    cd trdizin-clustering

    

    # 2. Gerekli veri ve sonuç dizinlerinin varlığını doğrulayın

    mkdir -p data embeddings results/kmeans/holdout models/kmeans

  │ GPU / CUDA Gereksinimi: Embedding üretimi ve UMAP boyut indirgeme adımlarında donanım hızlandırması için sunucuda

  │ nvidia-container-toolkit kurulu olmalı ve Docker'ın GPU'ya erişebildiğinden emin olunmalıdır (docker-compose.yml

  │ içinde gpus: all tanımlıdır).

  ──────

  ## İlk Kurulum

  Uygulamayı ve servisleri arka planda ayağa kaldırmak için:



    docker compose up -d --build



  Container'lar ayağa kalktığında:



  • app servisi Python 3.12, PyTorch (CUDA 12.6) ve kütüphane bağımlılıklarını yükleyerek Flask web sunucusunu 5001

  portunda başlatır.

  • postgres servisi 5434 (host) portunu dinler ve veritabanı sağlığını (healthcheck) denetler.



  Servislerin durumunu ve sağlık kontrolünü izlemek için:



    docker compose ps



  Uygulama loglarını anlık takip etmek için:

    docker compose logs -f app

  ──────

  ## Web Arayüzü (Dashboard) Erişimi

  Uygulama ayağa kalktığında tarayıcınızdan aşağıdaki adresler üzerinden erişilebilir:



  • HDBSCAN Anomali & 2D UMAP Haritası: http://localhost:5001/ (veya http://<sunucu-ip>:5001/)

      • WebGL tabanlı 2D UMAP dağılım grafiği, interaktif nokta seçimi, GLOSH aykırılık skorları ve TP-1 (Farklı Ana

      Disiplin) / TP-2 (Alt Alan Uyuşmazlığı) anomali filtreleme tablosu.

  • K-Means Çoklu Konu Tahmin Analizi: http://localhost:5001/kmeans

      • Seeded K-Means modelinin gerçek konu etiketleri ile tahmin edilen etiketlerinin karşılaştırması, F1 skorları,

      Exact Match oranları ve makale detay modalı.



  ──────

  ## Veri ve Model Boru Hattı (Pipeline) Çalıştırma



  Önceden üretilmiş veri ve modeller data/, embeddings/ ve results/ dizinlerinde mevcuttur. Pipeline'ı sıfırdan veya yeni

  verilerle baştan çalıştırmak isterseniz container içinde sırasıyla şu adımları izleyebilirsiniz:



    # 1. Container içine interaktif kabuk açın

    docker exec -it trdizin_clustering_app bash

    

    # 2. TR Dizin API'sinden dengeli veri çekimi

    python data_pipeline/fetch_balanced_trdizin.py

    

    # 3. MPNet ile 768-D çok dilli embedding üretimi

    python data_pipeline/generate_mpnet_embeddings_20k.py

    

    # 4. Vektör-Makale eşleme indeksini doğrula

    python data_pipeline/build_embedding_index.py

    

    # 5. HDBSCAN & Üçlü Mutabakat Anomali Pipeline'ını çalıştır

    python clustering/hdbscan/run_hdbscan_pipeline.py

    

    # 6. Seeded & Adaptive K-Means modelini eğit ve test et

    python clustering/kmeans/train.py

    python clustering/kmeans/adaptive_label_count_v2.py

  ──────

  ## Servisler



  • app: Web paneli ve API servislerini sunan Python 3.12 Flask uygulaması.

      • Port: 5001

      • Donanım: NVIDIA GPU (CUDA) geçişi aktif.

      • Görev: dashboard/app.py üzerinden Plotly grafiklerini ve anomali analizlerini sunar.

  • postgres: PostgreSQL 16 veritabanı.

      • Port: 5434 (Host) → 5432 (Container içi).

      • DB Adı: trdizin_clustering_db

      • Volume: postgres_data altında kalıcı olarak saklanır.



  ──────

  ## Dizin ve Veri Yapısı



    ├── data/               # Ham ve dengelenmiş makale üst verileri (balanced_articles.csv, article_subjects.csv)

    ├── embeddings/         # 768-D vektör matrisleri (.npy) ve 2D UMAP koordinatları (.csv)

    ├── clustering/         # HDBSCAN ve K-Means modelleme, anomali tespit scriptleri

    ├── dashboard/          # Flask web uygulaması, REST API uç noktaları, statik varlıklar ve şablonlar

    ├── results/            # Anomali listeleri, küme özetleri ve model değerlendirme metrikleri

    ├── Dockerfile          # PyTorch CUDA destekli imaj yapılandırması

    └── docker-compose.yml  # Çoklu servis (App + Postgres) orkestrasyonu

  ──────

  ## Yedekleme ve Veri Kalıcılığı (Öneri)



  Üretilen modeller, embedding matrisleri ve analiz sonuçları dosya sisteminde tutulmaktadır. Düzenli yedek almak için:



    # Veri, embedding ve sonuç çıktılarını arşivle

    tar -czvf trdizin_clustering_backup_$(date +%F).tar.gz data/ embeddings/ results/


    # PostgreSQL veritabanı yedeği (ileride aktif veri yazılması durumunda)

    docker exec trdizin_clustering_postgres pg_dump -U postgres trdizin_clustering_db > trdizin_db_$(date +%F).sql