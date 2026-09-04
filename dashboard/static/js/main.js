let currentAnomalies = [];
let detailModalInstance = null;
let evalModalInstance = null;
let currentAlgo = 'hdbscan';

let currentPage = 1;
const perPage = 50;
let totalPages = 0;
let totalAnomaliesCount = 0;
let isLoadingAnomalies = false;
let searchDebounceTimer = null;

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
    document.getElementById("sortSelect").addEventListener("change", () => {
        resetAndLoadAnomalies();
    });
    document.getElementById("prioritySelect").addEventListener("change", () => {
        resetAndLoadAnomalies();
    });
    document.getElementById("searchInput").addEventListener("input", () => {
        clearTimeout(searchDebounceTimer);
        searchDebounceTimer = setTimeout(() => {
            resetAndLoadAnomalies();
        }, 300);
    });

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

        // --- SEVİYELİ ATLAS ETİKETLERİ VE DİNAMİK ZOOM ---
        fetch('/api/cluster-summaries')
            .then(response => response.json())
            .then(clusters => {
                // Kalabalığı önlemek için boyutu 3 ve üzeri olan kümeleri filtrele
                const significantClusters = clusters.filter(c => c.size >= 3);

                // Etiketleri oluşturan yardımcı fonksiyon (Zoom seviyesine göre metin seçer)
                function updateAnnotations(zoomLevel = 'level_1') {
                    const annotations = significantClusters.map(c => {
                        let displayText = c.display_name_level_1; // Varsayılan en genel

                        if (zoomLevel === 'level_3') {
                            displayText = c.display_name_level_3 || c.display_name_level_2 || c.display_name_level_1;
                        } else if (zoomLevel === 'level_2') {
                            displayText = c.display_name_level_2 || c.display_name_level_1;
                        } else {
                            displayText = c.display_name_level_1;
                        }

                        return {
                            x: c.x_center,
                            y: c.y_center,
                            text: `<b>${displayText}</b>`,
                            showarrow: false,
                            xanchor: 'center',
                            yanchor: 'middle',
                            bgcolor: 'rgba(255, 255, 255, 0.75)', 
                            bordercolor: 'rgba(203, 213, 225, 0.8)', 
                            borderwidth: 1,
                            borderpad: 4,                       
                            font: {
                                family: 'Arial, sans-serif',
                                size: zoomLevel === 'level_3' ? 10 : 11, // Yaklaştıkça fontu hafif küçültebiliriz
                                color: '#0f172a'                
                            }
                        };
                    });

                    Plotly.relayout('clusterPlot', { annotations: annotations });
                }

                // 1. İlk açılışta en genel katmanla (Level 1) başlat
                updateAnnotations('level_1');

                // 2. Kullanıcı haritada zoom yaptıkça veya kaydırdıkça tetiklenen olay
                const plotElement = document.getElementById('clusterPlot');
                if (plotElement && plotElement.on) {
                    plotElement.on('plotly_relayout', function(eventData) {
                        // Eğer olay bir zoom veya range (eksen) değişimi ise
                        if (eventData['xaxis.range[0]'] || eventData['xaxis.autorange']) {
                            let xRange, yRange;

                            if (eventData['xaxis.autorange']) {
                                // Tamamen uzaklaşma (Reset zoom)
                                updateAnnotations('level_1');
                                return;
                            }

                            xRange = eventData['xaxis.range[1]'] - eventData['xaxis.range[0]'];
                            yRange = eventData['yaxis.range[1]'] - eventData['yaxis.range[0]'];
                            
                            // Eksen aralığının büyüklüğüne göre zoom derinliğini seç
                            // (Bu eşik değerlerini haritanın boyutuna göre ufakça revize edebilirsin)
                            if (xRange < 3.0) {
                                updateAnnotations('level_3'); // Çok yakın plan -> Spesifik konular
                            } else if (xRange < 7.0) {
                                updateAnnotations('level_2'); // Orta zoom -> Alt alanlar
                            } else {
                                updateAnnotations('level_1'); // Kuşbakışı -> Ana disiplinler
                            }
                        }
                    });
                }
            })
            .catch(error => console.error('Küme etiketleri yüklenirken hata oluştu:', error));
        // -------------------------------------------------------------

        //Grafikteki noktaya tıklama olayı (Güncellendi)
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
                    // 1. Tıklanan noktadan yalnızca external_id alınır ve lazy API ile detaylar yüklenir
                    const externalId = point.customdata.external_id || (typeof point.customdata === 'string' ? point.customdata : null);
                    if (externalId) {
                        loadArticleDetails(externalId);
                    }
                    
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

    // 2. ANOMALİ KARTLARINI YÜKLE (Sayfalı / Lazy)
    await resetAndLoadAnomalies();
}

// Sayfalamayı 1'e sıfırlayıp anomali kartlarını yeniden yükleyen fonksiyon
async function resetAndLoadAnomalies() {
    currentPage = 1;
    await loadAnomalies(1);
}

// Sayfalı anomali verisini çeken ve DOM kartlarını oluşturan fonksiyon
async function loadAnomalies(page = 1) {
    if (isLoadingAnomalies) return;
    isLoadingAnomalies = true;

    const algo = document.getElementById("algoSelect").value;
    currentAlgo = algo;
    const sort = document.getElementById("sortSelect").value;
    const priority = document.getElementById("prioritySelect").value;
    const search = document.getElementById("searchInput").value;
    const container = document.getElementById("cardContainer");
    const countText = document.getElementById("poolCountText");

    if (container) {
        container.innerHTML = '<div class="text-center py-4 text-muted"><span class="spinner-border spinner-border-sm me-2"></span>Kayıtlar yükleniyor...</div>';
    }

    try {
        const res = await fetch(`/api/anomalies?algorithm=${algo}&sort=${sort}&priority=${priority}&search=${encodeURIComponent(search)}&page=${page}&per_page=${perPage}`);
        const json = await res.json();

        const items = json.items || json.data || [];
        totalAnomaliesCount = json.total !== undefined ? json.total : (json.stats?.total_anomalies || 0);
        totalPages = json.total_pages || (totalAnomaliesCount > 0 ? Math.ceil(totalAnomaliesCount / perPage) : 0);
        currentPage = page;

        // İstatistik Sayaçları (her zaman toplam filtrelenmiş veriden gelir)
        if (json.stats) {
            const statTotal = document.getElementById("statTotal");
            if (statTotal) statTotal.innerText = json.stats.total_anomalies || 0;

            const statRisk = document.getElementById("statRisk");
            if (statRisk) statRisk.innerText = (json.stats.avg_risk || 0).toFixed(3);

            const statCritical = document.getElementById("statCritical");
            if (statCritical) statCritical.innerText = json.stats.critical_count || 0;

            const infoText = document.getElementById("systemInfoText");
            if (infoText) {
                infoText.innerText = json.stats.system_info || `${algo.toUpperCase()} Modülü`;
            }
        }

        if (container) {
            container.innerHTML = "";
            container.scrollTop = 0;
        }
        currentAnomalies = [...items];

        if (currentAnomalies.length === 0) {
            if (container) {
                container.innerHTML = `<div class="alert alert-light text-center border p-3">Filtrelere uygun anomali kaydı bulunamadı.</div>`;
            }
            if (countText) {
                countText.innerText = "0 anomali";
            }
            renderPagination(1, 0);
            return;
        }

        // Sayaç metni
        if (countText) {
            countText.innerText = `Sayfa ${currentPage} / ${totalPages} (${totalAnomaliesCount} anomali)`;
        }

        // Kartları listeye ekle (yalnızca mevcut sayfanın kayıtları)
        items.forEach((item) => {
            renderAnomalyCard(item, container, algo);
        });

        // Sayfalama bileşenini render et
        renderPagination(currentPage, totalPages);

    } catch (err) {
        console.error("Anomali verisi çekilirken hata:", err);
        if (container) {
            container.innerHTML = `<div class="alert alert-danger text-center p-3">Anomali verisi yüklenirken bir hata oluştu.</div>`;
        }
        renderPagination(1, 0);
    } finally {
        isLoadingAnomalies = false;
    }
}

// Tekil anomali kartını DOM'a oluşturan fonksiyon
function renderAnomalyCard(item, container, algo) {
    const isCritical = item.oncelik && item.oncelik.includes("KRİTİK");
    const badgeClass = isCritical ? "badge-critical" : "badge-high";

    const card = document.createElement("div");
    card.className = "card card-custom p-3";
    card.style.cursor = "pointer";
    // Karta tıklandığında sol paneli aç
    card.onclick = () => {
        if (item.external_id) {
            loadArticleDetails(item.external_id);
        } else {
            openSidePanel(item);
        }
    };

    const scoreLabel = algo === "hdbscan" ? "GLOSH" : "Aykırılık";
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
}

// Bootstrap 5 Sayfalama Bileşenini Render Eden Fonksiyon
function renderPagination(page, total) {
    const nav = document.getElementById("paginationNav");
    const list = document.getElementById("paginationList");
    if (!nav || !list) return;

    if (total <= 1) {
        nav.style.display = "none";
        list.innerHTML = "";
        return;
    }

    nav.style.display = "block";
    list.innerHTML = "";

    // ← Önceki
    const prevLi = document.createElement("li");
    prevLi.className = `page-item ${page <= 1 ? "disabled" : ""}`;
    prevLi.innerHTML = `<a class="page-link" href="javascript:void(0)" ${page <= 1 ? 'tabindex="-1" aria-disabled="true"' : `onclick="goToPage(${page - 1})"`}>&larr; Önceki</a>`;
    list.appendChild(prevLi);

    // Sayfa numaraları
    const pages = getPageNumbers(page, total);
    pages.forEach((p) => {
        const li = document.createElement("li");
        if (p === "...") {
            li.className = "page-item disabled";
            li.innerHTML = `<span class="page-link">&hellip;</span>`;
        } else if (p === page) {
            li.className = "page-item active";
            li.setAttribute("aria-current", "page");
            li.innerHTML = `<span class="page-link">${p}</span>`;
        } else {
            li.className = "page-item";
            li.innerHTML = `<a class="page-link" href="javascript:void(0)" onclick="goToPage(${p})">${p}</a>`;
        }
        list.appendChild(li);
    });

    // Sonraki →
    const nextLi = document.createElement("li");
    nextLi.className = `page-item ${page >= total ? "disabled" : ""}`;
    nextLi.innerHTML = `<a class="page-link" href="javascript:void(0)" ${page >= total ? 'tabindex="-1" aria-disabled="true"' : `onclick="goToPage(${page + 1})"`}>Sonraki &rarr;</a>`;
    list.appendChild(nextLi);
}

// Sayfa numaralarını belirleyen yardımcı fonksiyon
function getPageNumbers(current, total) {
    if (total <= 8) {
        const pages = [];
        for (let i = 1; i <= total; i++) {
            pages.push(i);
        }
        return pages;
    }

    const pages = [];
    if (current <= 4) {
        for (let i = 1; i <= 5; i++) {
            pages.push(i);
        }
        pages.push("...");
        pages.push(total);
    } else if (current >= total - 3) {
        pages.push(1);
        pages.push("...");
        for (let i = total - 4; i <= total; i++) {
            pages.push(i);
        }
    } else {
        pages.push(1);
        pages.push("...");
        pages.push(current - 1);
        pages.push(current);
        pages.push(current + 1);
        pages.push("...");
        pages.push(total);
    }
    return pages;
}

// Belirtilen sayfaya geçişi sağlayan fonksiyon
function goToPage(targetPage) {
    if (targetPage < 1 || targetPage > totalPages || targetPage === currentPage || isLoadingAnomalies) {
        return;
    }
    loadAnomalies(targetPage);
}
window.goToPage = goToPage;

let currentLoadingArticleId = null;

// Tıklanan makalenin detaylarını lazy loading ile API'den çeken fonksiyon
async function loadArticleDetails(externalId) {
    if (!externalId) return;

    const targetId = String(externalId).trim();
    currentLoadingArticleId = targetId;

    // Rehber ekranını gizle, aktif içerik alanını aç
    const welcomeWrapper = document.getElementById('panel-content-wrapper');
    const activeContent = document.getElementById('panel-active-content');
    if (welcomeWrapper) welcomeWrapper.style.display = 'none';
    if (activeContent) activeContent.style.display = 'block';

    // Panelde loading durumu göster
    const extIdElem = document.getElementById('panel-id');
    if (extIdElem) extIdElem.innerText = `(ID: ${targetId})`;

    const titleElem = document.getElementById('panel-title');
    if (titleElem) {
        titleElem.innerHTML = '<span class="spinner-border spinner-border-sm text-secondary me-2" role="status"></span>Yükleniyor...';
    }

    const abstractElem = document.getElementById('panel-abstract');
    if (abstractElem) {
        abstractElem.innerText = 'Makale detayları sunucudan getiriliyor, lütfen bekleyin...';
    }

    const riskElem = document.getElementById('panel-risk');
    if (riskElem) riskElem.innerText = '...';

    const catElem = document.getElementById('panel-cat');
    if (catElem) catElem.innerText = '...';

    const sugElem = document.getElementById('panel-suggestion');
    if (sugElem) sugElem.innerText = '...';

    try {
        const res = await fetch(`/api/article/${encodeURIComponent(targetId)}`);

        // Kullanıcı başka bir noktaya tıkladıysa eski isteğin sonucunu yoksay
        if (currentLoadingArticleId !== targetId) return;

        if (res.status === 404) {
            if (titleElem) titleElem.innerText = 'Makale Bulunamadı';
            if (abstractElem) abstractElem.innerText = `ID: ${targetId} olan makalenin detay kaydı veri tabanında bulunamadı.`;
            if (riskElem) riskElem.innerText = '-';
            if (catElem) catElem.innerText = '-';
            if (sugElem) sugElem.innerText = '-';
            return;
        }

        if (!res.ok) {
            throw new Error(`HTTP ${res.status}`);
        }

        const data = await res.json();

        // Yanıt geldiğinde hala bu makale mi aktif kontrolü
        if (currentLoadingArticleId !== targetId) return;

        openSidePanel(data);
    } catch (err) {
        console.error("Makale detayı yüklenirken hata:", err);
        if (currentLoadingArticleId !== targetId) return;

        if (titleElem) titleElem.innerText = 'Yükleme Hatası';
        if (abstractElem) abstractElem.innerText = 'Makale detayları sunucudan alınırken bir hata oluştu. Lütfen tekrar deneyin.';
        if (riskElem) riskElem.innerText = '-';
        if (catElem) catElem.innerText = '-';
        if (sugElem) sugElem.innerText = '-';
    }
}

// Soldaki Sabit Detay Panelini Dolduran Fonksiyon
function openSidePanel(item) {
    if (!item) return;

    // Rehber ekranını gizle, aktif içerik alanını aç
    const welcomeWrapper = document.getElementById('panel-content-wrapper');
    const activeContent = document.getElementById('panel-active-content');
    
    if (welcomeWrapper) welcomeWrapper.style.display = 'none';
    if (activeContent) activeContent.style.display = 'block';

    // External ID bilgisini panele yazdıralım
    const extIdElem = document.getElementById('panel-id');
    if (extIdElem) {
        extIdElem.innerText = item.external_id ? `(ID: ${item.external_id})` : '';
    }

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
let currentView = 'risk';

function switchMapView(viewType) {
    currentView = viewType;
    const plotElement = document.getElementById('clusterPlot');
    if (!plotElement || !plotElement.data || !plotElement.data[0].customdata) return;

    // Buton aktiflik sınıflarını güncelle
    const btnRisk = document.getElementById('btnRiskView');
    const btnCluster = document.getElementById('btnClusterView');

    if (viewType === 'risk') {
        // Risk seçiliyken: Risk butonu kırmızı (aktif), Küme butonu sade gri (pasif)
        btnRisk.className = 'btn btn-danger btn-sm active';
        btnCluster.className = 'btn btn-sm text-secondary bg-light border'; 
    } else {
        // Küme seçiliyken: Küme butonu koyu (aktif), Risk butonu sade gri (pasif)
        btnRisk.className = 'btn btn-sm text-secondary bg-light border';
        btnCluster.className = 'btn btn-dark btn-sm active';
    }

    const records = plotElement.data[0].customdata;
    let colorData = [];
    let colorScale = '';
    let colorBarTitle = '';

    if (viewType === 'risk') {
        // Risk skorlarına göre renklendirme
        colorData = records.map(d => d.risk_skoru !== undefined ? Number(d.risk_skoru) : 0.5);
        colorScale = [
            [0, '#474747'],
            [0.5, '#BB5B5B'],
            [1, '#e60404']
        ];
        colorBarTitle = 'Risk';
    } else {
        // Tableau10 benzeri kategorik renk paleti
        const palette = [
            '#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', 
            '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#374983'
        ];

        // Her noktanın küme ID'sine göre paletten renk seçiyoruz (mod alarak döndürüyoruz)
        colorData = records.map(d => {
            const kid = d.kume !== undefined ? d.kume : (d.kmeans_kume !== undefined ? d.kmeans_kume : 0);
            if (kid === -1) return '#d3d3d3'; // Gürültü (noise) noktaları için hafif gri
            return palette[Math.abs(kid) % palette.length];
        });

        colorScale = null; // Kategorik renklendirmede colorscale kullanılmaz
        colorBarTitle = 'Küme (Kategorik)';
    }

    // Grafiği yeniden çizmeden sadece renk verilerini ve bar görünürlüğünü güncelle
    Plotly.restyle(plotElement, {
        'marker.color': [colorData],
        'marker.colorscale': [colorScale],
        'marker.showscale': [viewType === 'risk'], // Sadece risk görünümünde renk barı açık olur
        'marker.colorbar.title': colorBarTitle
    }, [0]);
}