\# TR Dizin Makale Kümeleme Projesi



Bu projede TR Dizin makalelerinin ortak embedding temsilleri kullanılarak

farklı kümeleme algoritmaları ile analiz edilmesi amaçlanmaktadır.



\## Kullanılan Kümeleme Yöntemleri



\- K-Means

\- HDBSCAN



\## Proje Yapısı



\- `data/`: Ortak veri seti

\- `embeddings/`: Ortak makale embeddingleri

\- `clustering/kmeans/`: K-Means çalışmaları

\- `clustering/hdbscan/`: HDBSCAN çalışmaları

\- `evaluation/`: Algoritmaların karşılaştırılması

\- `dashboard/`: Ortak web arayüzü

\- `results/`: Deney sonuçları



\## Ortak Deney Yapısı



K-Means ve HDBSCAN aynı veri seti ve aynı embeddingler üzerinde

çalıştırılacaktır. Böylece iki kümeleme algoritmasının sonuçları

adil bir şekilde karşılaştırılabilecektir.

