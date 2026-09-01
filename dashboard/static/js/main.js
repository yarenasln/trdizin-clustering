let currentAnomalies = [];
let detailModalInstance = null;
let evalModalInstance = null;
let currentAlgo = 'hdbscan';

document.addEventListener("DOMContentLoaded", () => {
    const modalEl = document.getElementById('detailModal');
    if (modalEl) {
        detailModalInstance = new bootstrap.Modal(modalEl);
    }

    const evalModalEl = document.getElementById('evalModal');
    if (evalModalEl) {
        evalModalInstance = new bootstrap.Modal(evalModalEl);
        document.getElementById('btnMetrics').addEventListener('click', loadEvaluationMetrics);
    }

    document.getElementById("algoSelect").addEventListener("change", (e) => {
        currentAlgo = e.target.value;
        if (currentAlgo === "kmeans") {
            window.location.href = "/kmeans";
            return;
        }
        loadDashboard();
    });
    document.getElementById("sortSelect").addEventListener("change", loadDashboard);
    document.getElementById("prioritySelect").addEventListener("change", loadDashboard);
    document.getElementById("searchInput").addEventListener("input", loadDashboard);

    loadDashboard();
});

async function loadDashboard() {
    const algo = document.getElementById("algoSelect").value;
    currentAlgo = algo;
    const sort = document.getElementById("sortSelect").value;
    const priority = document.getElementById("prioritySelect").value;
    const search = document.getElementById("searchInput").value;

    // Badge güncelle
    const badgeEl = document.getElementById("algoBadge");
    if (badgeEl) {
        badgeEl.innerText = algo === "hdbscan" ? "HDBSCAN" : "K-MEANS";
        badgeEl.className = algo === "hdbscan" ? "badge bg-primary" : "badge bg-success";
    }

    const poolTitle = document.getElementById("poolTitle");
    if (poolTitle) {
        poolTitle.innerText = `${algo.toUpperCase()} Anomali ve Uyuşmazlık Havuzu (Detay için karta tıklayın)`;
    }

    // 1. PLOTLY KÜME GRAFİĞİNİ YÜKLE
    try {
        const plotRes = await fetch(`/api/plot?algorithm=${algo}`);
        const plotObj = await plotRes.json();

        Plotly.newPlot('clusterPlot', plotObj.data, plotObj.layout, { 
            responsive: true, 
            displayModeBar: 'hover',
            displaylogo: false,
            scrollZoom: true
        });

        // NOMIC ATLAS TARZI: Grafikteki noktaya tıklama olayı (Güncellendi)
        const plotElement = document.getElementById('clusterPlot');
        
        // Eski dinleyicileri temizle
        plotElement.removeAllListeners?.('plotly_click');
        plotElement.removeAllListeners?.('plotly_hover');
        plotElement.removeAllListeners?.('plotly_unhover');
        
        // Tıklama olayı hem sol paneli açar hem de haritada noktayı büyütüp parletir
        plotElement.on('plotly_click', function(data){
            if(data.points && data.points.length > 0) {
                const point = data.points.find(p => p.curveNumber === 0);
                if(point && point.customdata) {
                    // 1. Sol paneli aç
                    openSidePanel(point.customdata);
                    
                    // 2. Haritada tıklanan noktayı sabit renkli büyük katmana taşı
                    Plotly.restyle(
                        plotElement,
                        {
                            x: [[point.x]],
                            y: [[point.y]]
                        },
                        [1]
                    );
                }
            }
        });

    } catch (err) {
        console.error("Grafik çizilirken hata oluştu:", err);
    }

    // 2. ANOMALİ KARTLARINI YÜKLE
    try {
        const res = await fetch(`/api/anomalies?algorithm=${algo}&sort=${sort}&priority=${priority}&search=${encodeURIComponent(search)}`);
        const json = await res.json();
        currentAnomalies = json.data || [];

        // İstatistik Sayaçları
        document.getElementById("statTotal").innerText = json.stats.total_anomalies || 0;
        document.getElementById("statRisk").innerText = (json.stats.avg_risk || 0).toFixed(3);
        document.getElementById("statCritical").innerText = json.stats.critical_count || 0;

        const infoText = document.getElementById("systemInfoText");
        if (infoText) {
            infoText.innerText = json.stats.system_info || `${algo.toUpperCase()} Modülü`;
        }

        const container = document.getElementById("cardContainer");
        container.innerHTML = "";

        if (currentAnomalies.length === 0) {
            container.innerHTML = `<div class="alert alert-light text-center border p-3">Filtrelere uygun anomali kaydı bulunamadı.</div>`;
            return;
        }

        const scoreLabel = algo === "hdbscan" ? "GLOSH" : "Aykırılık";

        // Kartları listeye ekle
        currentAnomalies.forEach((item) => {
            const isCritical = item.oncelik && item.oncelik.includes("KRİTİK");
            const badgeClass = isCritical ? "badge-critical" : "badge-high";

            const card = document.createElement("div");
            card.className = "card card-custom p-3";
            card.style.cursor = "pointer";
            // Karta tıklandığında hem yan paneli hem modalı açabilirsin
            card.onclick = () => openSidePanel(item);

            const scoreVal = Number(item.glosh_skoru || item.aykirilik_skoru || 0).toFixed(3);
            const riskVal = Number(item.risk_skoru || 0).toFixed(3);
            const kumeVal = item.kume !== undefined && item.kume !== -1 ? `#${item.kume}` : (item.kmeans_kume !== undefined && item.kmeans_kume !== -1 ? `#${item.kmeans_kume}` : 'Aykırı / -1');

            card.innerHTML = `
                <div class="d-flex justify-content-between align-items-start mb-2">
                    <span class="badge ${badgeClass} badge-risk">${item.oncelik || 'BELİRTİLMEDİ'}</span>
                    <div class="d-flex gap-2">
                        <span class="score-pill score-pill-danger">Bileşik Risk: <strong>${riskVal}</strong></span>
                        <span class="score-pill score-pill-info">${scoreLabel}: <strong>${scoreVal}</strong></span>
                        <span class="score-pill">Küme: <strong>${kumeVal}</strong></span>
                    </div>
                </div>
                <h6 class="fw-bold mb-1" style="color: var(--text-primary); font-size: 0.95rem;">${item.baslik || 'Başlık Belirtilmemiş'}</h6>
                <p class="small text-secondary mb-3">${item.ozet && item.ozet !== 'Özet metni veri tabanında bulunmuyor.' && item.ozet !== 'Özet metni bulunmuyor.' ? item.ozet.substring(0, 180) + '...' : 'Detayları ve tam analizi görmek için tıklayın.'}</p>
                
                <div class="row g-2 pt-2 border-top" style="border-color: #f1f5f9 !important;">
                    <div class="col-md-4">
                        <small class="text-muted d-block font-monospace" style="font-size: 0.72rem;">MEVCUT KATEGORİ</small>
                        <span class="small fw-semibold" style="color: var(--pastel-rose-text);">${item.mevcut_kategori || '-'}</span>
                    </div>
                    <div class="col-md-4">
                        <small class="text-muted d-block font-monospace" style="font-size: 0.72rem;">MODEL ÖNERİSİ</small>
                        <span class="small fw-semibold" style="color: var(--pastel-sage-text);">${item.oneri_kategori || '-'}</span>
                    </div>
                    <div class="col-md-4">
                        <small class="text-muted d-block font-monospace" style="font-size: 0.72rem;">k-NN YEREL ÖNERİ</small>
                        <span class="small fw-semibold" style="color: var(--pastel-amber-text);">${item.knn_oneri || '-'}</span>
                    </div>
                </div>
            `;
            container.appendChild(card);
        });
    } catch (err) {
        console.error("Anomali verisi çekilirken hata:", err);
    }
}

// NOMIC ATLAS: Soldaki Sabit Detay Panelini Dolduran Fonksiyon
function openSidePanel(item) {
    if (!item) return;

    // Rehber ekranını gizle, aktif içerik alanını aç
    const welcomeWrapper = document.getElementById('panel-content-wrapper');
    const activeContent = document.getElementById('panel-active-content');
    
    if (welcomeWrapper) welcomeWrapper.style.display = 'none';
    if (activeContent) activeContent.style.display = 'block';

    // Başlık alanını güvenli şekilde yerleştir (Önce başlık, yoksa ID)
    const titleText = item.baslik || item.title || (item.external_id ? `Makale ID: ${item.external_id}` : 'Başlık Belirtilmemiş');
    document.getElementById('panel-title').innerText = titleText;
    
    // Özet
    document.getElementById('panel-abstract').innerText = item.ozet || item.abstract || 'Özet metni bulunmuyor.';
    
    // Risk Skoru
    const risk = item.risk_skoru !== undefined ? item.risk_skoru : (item.bileşik_risk || 0);
    document.getElementById('panel-risk').innerText = `%${(Number(risk) * 100).toFixed(1)}`;
    
    // Kategori ve Öneri (Alternatif sütun adları eklenmiştir)
    document.getElementById('panel-cat').innerText = item.mevcut_kategori || item.gercek_kategori || item.kategori || '-';
    document.getElementById('panel-suggestion').innerText = item.oneri_kategori || item.model_onerisi || item.tahmin_kategori || '-';
}

// Paneli başlangıç durumuna döndüren fonksiyon
function resetSidePanel() {
    const welcomeWrapper = document.getElementById('panel-content-wrapper');
    const activeContent = document.getElementById('panel-active-content');
    
    if (welcomeWrapper) welcomeWrapper.style.display = 'block';
    if (activeContent) activeContent.style.display = 'none';
}

// Sağdaki Paneli Kapatma Fonksiyonu
function closeDetailPanel() {
    const panel = document.getElementById('article-detail-panel');
    if (panel) {
        panel.style.display = 'none';
    }
}

// Eski Modal Fonksiyonu (Yedek olarak durabilir)
function openDetailModal(item) {
    if (!item || !detailModalInstance) return;

    const isCritical = item.oncelik && item.oncelik.includes("KRİTİK");
    const pBadge = document.getElementById("modalPriority");
    if (pBadge) {
        pBadge.className = `badge badge-risk ${isCritical ? "badge-critical" : "badge-high"} mb-1`;
        pBadge.innerText = item.oncelik || 'BELİRTİLMEDİ';
    }

    if (document.getElementById("modalTitle")) document.getElementById("modalTitle").innerText = item.baslik || 'Başlık Yok';
    if (document.getElementById("modalRisk")) document.getElementById("modalRisk").innerText = Number(item.risk_skoru || 0).toFixed(3);
    
    const scoreVal = Number(item.glosh_skoru || item.aykirilik_skoru || 0).toFixed(3);
    if (document.getElementById("modalGlosh")) document.getElementById("modalGlosh").innerText = scoreVal;

    if (document.getElementById("modalMevcutKat")) document.getElementById("modalMevcutKat").innerText = item.mevcut_kategori || '-';
    if (document.getElementById("modalOneriKat")) document.getElementById("modalOneriKat").innerText = item.oneri_kategori || '-';
    if (document.getElementById("modalOzet")) document.getElementById("modalOzet").innerText = item.ozet || 'Özet metni veri kümesinde bulunamadı.';

    detailModalInstance.show();
}

// Model Değerlendirme Metriklerini Çeken Fonksiyon
async function loadEvaluationMetrics() {
    try {
        const res = await fetch('/api/evaluation');
        const json = await res.json();

        // K-Means Tablosu
        const tbodyK = document.getElementById('evalTableKmeans');
        if (tbodyK) {
            tbodyK.innerHTML = '';
            const allKmeans = [...(json.seeded_kmeans || []), ...(json.baseline_kmeans || [])];
            if (allKmeans.length === 0) {
                tbodyK.innerHTML = '<tr><td colspan="7" class="text-center text-muted">K-Means değerlendirme verisi bulunamadı.</td></tr>';
            } else {
                allKmeans.forEach(row => {
                    const matchRate = row.Topic_Match_Rate ? `%${(row.Topic_Match_Rate * 100).toFixed(2)}` : (row.Matched_Articles ? `${row.Matched_Articles} / ${row.Articles}` : '-');
                    tbodyK.innerHTML += `
                        <tr>
                            <td class="fw-bold">${row.Method || '-'}</td>
                            <td>${row.Articles || '-'}</td>
                            <td>${row.Clusters || '-'}</td>
                            <td><span class="badge bg-light text-dark border">${row.Silhouette !== undefined ? Number(row.Silhouette).toFixed(4) : '-'}</span></td>
                            <td>${row.Davies_Bouldin !== undefined ? Number(row.Davies_Bouldin).toFixed(3) : '-'}</td>
                            <td>${row.Calinski_Harabasz !== undefined ? Number(row.Calinski_Harabasz).toFixed(2) : '-'}</td>
                            <td><span class="badge bg-success">${matchRate}</span></td>
                        </tr>
                    `;
                });
            }
        }

        // Embedding Tablosu
        const tbodyE = document.getElementById('evalTableEmb');
        if (tbodyE) {
            tbodyE.innerHTML = '';
            const embs = json.embedding_comparison || [];
            if (embs.length === 0) {
                tbodyE.innerHTML = '<tr><td colspan="6" class="text-center text-muted">Embedding karşılaştırma verisi bulunamadı.</td></tr>';
            } else {
                embs.forEach(row => {
                    tbodyE.innerHTML += `
                        <tr>
                            <td class="fw-bold">${row.Model || '-'}</td>
                            <td>${row.Embedding_Dim || '-'}D</td>
                            <td><span class="badge bg-light text-dark border">${row.Silhouette !== undefined ? Number(row.Silhouette).toFixed(4) : '-'}</span></td>
                            <td>${row.Davies_Bouldin !== undefined ? Number(row.Davies_Bouldin).toFixed(3) : '-'}</td>
                            <td>${row.Calinski_Harabasz !== undefined ? Number(row.Calinski_Harabasz).toFixed(2) : '-'}</td>
                            <td>${row.Clustering_Time_Seconds !== undefined ? Number(row.Clustering_Time_Seconds).toFixed(2) + ' sn' : '-'}</td>
                        </tr>
                    `;
                });
            }
        }

    } catch (err) {
        console.error("Metrikler yüklenirken hata:", err);
    }
}