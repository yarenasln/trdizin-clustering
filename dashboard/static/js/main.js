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

// --- LEVEL OF DETAIL (LOD) STATE DEĞİŞKENLERİ ---
let lodDebounceTimer = null;
let lodRequestId = 0;
let isLodUpdating = false;
let lastLodBBox = null;
let pendingLodBBox = null;
let lastSelectedPoint = null;
let currentClusterAnnotations = [];
let significantClustersData = [];
let currentLodMeta = null;
const LOD_DEBOUNCE_MS = 400;

// --- CUSTOM RAF WHEEL ZOOM STATE DEĞİŞKENLERİ ---
let accumulatedWheelDeltaY = 0;
let wheelRafId = null;
let lastWheelCursor = { x: 0, y: 0 };

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

    // Reset LOD & Viewport State on Dashboard Load
    if (lodDebounceTimer) {
        clearTimeout(lodDebounceTimer);
        lodDebounceTimer = null;
    }
    lodRequestId++;
    lastLodBBox = null;
    lastSelectedPoint = null;
    currentLodMeta = null;

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

        // LOD metadata rozetini göster
        updateLodMetaBadge(plotObj.lod_meta);

        // Küme görünümü aktifse ilk çizimde de renkleri uygula
        if (currentView === 'cluster' && plotObj.data && plotObj.data[0]) {
            applyClusterColoring(plotObj.data[0]);
        }

        await Plotly.newPlot('clusterPlot', plotObj.data, plotObj.layout, { 
            responsive: true, 
            displayModeBar: 'hover',
            displaylogo: false,
            scrollZoom: false
        });

        // --- SEVİYELİ ATLAS ETİKETLERİ VE DİNAMİK ZOOM ---
        fetch('/api/cluster-summaries')
            .then(response => response.json())
            .then(clusters => {
                significantClustersData = clusters.filter(c => c.size >= 3);
                updateClusterAnnotations('level_1', true);
            })
            .catch(error => console.error('Küme etiketleri yüklenirken hata oluştu:', error));

        const plotElement = document.getElementById('clusterPlot');
        
        // Eski dinleyicileri temizle
        plotElement.removeAllListeners?.('plotly_click');
        plotElement.removeAllListeners?.('plotly_relayout');
        plotElement.removeAllListeners?.('plotly_hover');
        plotElement.removeAllListeners?.('plotly_unhover');

        // Zoom / Pan sonrası BBox + LOD Dinleyicisi
        plotElement.on('plotly_relayout', function(eventData) {
            if (isLodUpdating || !eventData) {
                return;
            }

            // Reset zoom / autorange kontrolü
            const isAutorange = eventData['xaxis.autorange'] || eventData['yaxis.autorange'] || eventData['autosize'];
            if (isAutorange) {
                updateClusterAnnotations('level_1', false);
                scheduleLodUpdate(null);
                return;
            }

            // Koordinatları ayıkla
            let xmin = null, xmax = null, ymin = null, ymax = null;

            if (eventData['xaxis.range[0]'] !== undefined && eventData['xaxis.range[1]'] !== undefined) {
                xmin = Number(eventData['xaxis.range[0]']);
                xmax = Number(eventData['xaxis.range[1]']);
            } else if (Array.isArray(eventData['xaxis.range'])) {
                xmin = Number(eventData['xaxis.range'][0]);
                xmax = Number(eventData['xaxis.range'][1]);
            }

            if (eventData['yaxis.range[0]'] !== undefined && eventData['yaxis.range[1]'] !== undefined) {
                ymin = Number(eventData['yaxis.range[0]']);
                ymax = Number(eventData['yaxis.range[1]']);
            } else if (Array.isArray(eventData['yaxis.range'])) {
                ymin = Number(eventData['yaxis.range'][0]);
                ymax = Number(eventData['yaxis.range'][1]);
            }

            if (xmin === null || ymin === null) {
                const hasAxisKey = Object.keys(eventData).some(k => k.startsWith('xaxis') || k.startsWith('yaxis'));
                if (hasAxisKey && plotElement._fullLayout?.xaxis?.range && plotElement._fullLayout?.yaxis?.range) {
                    xmin = Number(plotElement._fullLayout.xaxis.range[0]);
                    xmax = Number(plotElement._fullLayout.xaxis.range[1]);
                    ymin = Number(plotElement._fullLayout.yaxis.range[0]);
                    ymax = Number(plotElement._fullLayout.yaxis.range[1]);
                }
            }

            if (xmin !== null && xmax !== null && ymin !== null && ymax !== null && !isNaN(xmin) && !isNaN(xmax) && !isNaN(ymin) && !isNaN(ymax)) {
                const b_xmin = Math.min(xmin, xmax);
                const b_xmax = Math.max(xmin, xmax);
                const b_ymin = Math.min(ymin, ymax);
                const b_ymax = Math.max(ymin, ymax);

                // Zoom seviyesine göre küme etiket derinliği
                const xRange = b_xmax - b_xmin;
                if (xRange < 3.0) {
                    updateClusterAnnotations('level_3', false);
                } else if (xRange < 7.0) {
                    updateClusterAnnotations('level_2', false);
                } else {
                    updateClusterAnnotations('level_1', false);
                }

                // Debounced LOD güncellemesi
                scheduleLodUpdate({ xmin: b_xmin, xmax: b_xmax, ymin: b_ymin, ymax: b_ymax });
            }
        });

        // Custom rAF Wheel Zoom Entegrasyonu
        initCustomWheelZoom(plotElement);
        
        // Tıklama olayı: Seçilen noktayı hatırla, detayları yükle, parlat
        plotElement.on('plotly_click', function(data){
            if(data.points && data.points.length > 0) {
                const point = data.points.find(p => p.curveNumber === 0);
                if(point && point.customdata) {
                    const externalId = point.customdata.external_id || (typeof point.customdata === 'string' ? point.customdata : null);
                    if (externalId) {
                        lastSelectedPoint = { x: point.x, y: point.y, externalId: externalId };
                        loadArticleDetails(externalId);
                    }
                    
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

// --- LEVEL OF DETAIL (LOD) & VIEWPORT YARDIMCI FONKSİYONLARI ---

// ==============================================================================
// CUSTOM REQUESTANIMATIONFRAME WHEEL ZOOM ENTEGRASYONU
// ==============================================================================
function initCustomWheelZoom(plotElement) {
    if (!plotElement || plotElement._customWheelAttached) return;
    plotElement._customWheelAttached = true;

    function processCustomWheelFrame() {
        wheelRafId = null;
        if (accumulatedWheelDeltaY === 0) return;

        // Devam eden bir Plotly.react çizimi varsa çakışmayı önlemek için sonraki frame'e ertele
        if (isLodUpdating) {
            wheelRafId = requestAnimationFrame(processCustomWheelFrame);
            return;
        }

        const delta = accumulatedWheelDeltaY;
        accumulatedWheelDeltaY = 0;

        if (!plotElement._fullLayout) return;
        const xa = plotElement._fullLayout.xaxis;
        const ya = plotElement._fullLayout.yaxis;
        if (!xa || !ya || !xa.range || !ya.range) return;

        // Plotly benzeri üstel ölçek katsayısı (delta < 0: zoom in, delta > 0: zoom out)
        const zoomFactor = Math.exp(delta * 0.0015);

        // Veri koordinatlarını içeren çizim alanını (nsewdrag) tespit et
        const dragBox = plotElement.querySelector('.nsewdrag');
        const rect = dragBox ? dragBox.getBoundingClientRect() : plotElement.getBoundingClientRect();
        if (!rect || rect.width <= 0 || rect.height <= 0) return;

        // İmlecin grafik alanındaki oransal konumu [0, 1]
        const fracX = Math.max(0, Math.min(1, (lastWheelCursor.x - rect.left) / rect.width));
        const fracY = Math.max(0, Math.min(1, (rect.bottom - lastWheelCursor.y) / rect.height));

        const xRange = xa.range[1] - xa.range[0];
        const yRange = ya.range[1] - ya.range[0];
        const xCenter = xa.range[0] + fracX * xRange;
        const yCenter = ya.range[0] + fracY * yRange;

        const newXmin = xCenter - fracX * (xRange * zoomFactor);
        const newXmax = xCenter + (1 - fracX) * (xRange * zoomFactor);
        const newYmin = yCenter - fracY * (yRange * zoomFactor);
        const newYmax = yCenter + (1 - fracY) * (yRange * zoomFactor);

        // İmleç merkezli tek birleştirilmiş relayout işlet
        Plotly.relayout(plotElement, {
            'xaxis.range': [newXmin, newXmax],
            'yaxis.range': [newYmin, newYmax]
        });
    }

    plotElement.addEventListener('wheel', function(e) {
        // Native sayfa kaydırmasını ve Plotly dahili unthrottled wheel dinleyicisini durdur
        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation();

        // deltaMode normalizasyonu (0: pixel, 1: line, 2: page)
        let rawDelta = e.deltaY;
        if (e.deltaMode === 1) {
            rawDelta *= 20;
        } else if (e.deltaMode === 2) {
            rawDelta *= 300;
        }

        // Aşırı ani sıçramaları önlemek için tekil event delta'sını sınırla [-300, 300]
        const normalizedDelta = Math.max(-300, Math.min(300, rawDelta));
        accumulatedWheelDeltaY += normalizedDelta;
        lastWheelCursor = { x: e.clientX, y: e.clientY };

        // Kullanıcı tekerleği çevirdiği sürece LOD debounce timer'ını sıfırla
        if (lodDebounceTimer) {
            clearTimeout(lodDebounceTimer);
            lodDebounceTimer = setTimeout(() => {
                lodDebounceTimer = null;
                fetchAndUpdateLodPlot(pendingLodBBox);
            }, LOD_DEBOUNCE_MS);
        }

        // Frame başına en fazla 1 zoom güncellemesi için rAF zamanla
        if (!wheelRafId) {
            wheelRafId = requestAnimationFrame(processCustomWheelFrame);
        }
    }, { passive: false });
}

function updateLodMetaBadge(lodMeta) {
    currentLodMeta = lodMeta;
    const infoText = document.getElementById("systemInfoText");
    if (!infoText) return;
    if (!lodMeta) {
        infoText.innerText = `${currentAlgo.toUpperCase()} Modülü`;
        return;
    }

    const disp = lodMeta.displayed_count !== undefined ? lodMeta.displayed_count.toLocaleString('tr-TR') : '0';
    const bboxTotal = lodMeta.bbox_count !== undefined ? lodMeta.bbox_count.toLocaleString('tr-TR') : '0';
    const level = lodMeta.lod_level !== undefined ? lodMeta.lod_level : 0;

    if (level === 0) {
        infoText.innerText = `${disp} makale (LOD 0 - Tam Detay)`;
    } else {
        infoText.innerText = `${disp} / ${bboxTotal} makale (LOD ${level})`;
    }
}

function updateClusterAnnotations(zoomLevel = 'level_1', triggerRelayout = true) {
    if (!significantClustersData || significantClustersData.length === 0) return;

    currentClusterAnnotations = significantClustersData.map(c => {
        let displayText = c.display_name_level_1;
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
                size: zoomLevel === 'level_3' ? 10 : 11,
                color: '#0f172a'
            }
        };
    });

    if (triggerRelayout && !isLodUpdating) {
        Plotly.relayout('clusterPlot', { annotations: currentClusterAnnotations });
    }
}

function applyClusterColoring(trace) {
    if (!trace || !trace.customdata) return;
    const palette = [
        '#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', 
        '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#374983'
    ];
    const records = trace.customdata;
    const colorData = records.map(d => {
        const kid = d.kume !== undefined ? d.kume : (d.kmeans_kume !== undefined ? d.kmeans_kume : 0);
        if (kid === -1) return '#d3d3d3';
        return palette[Math.abs(kid) % palette.length];
    });
    if (!trace.marker) trace.marker = {};
    trace.marker.color = colorData;
    trace.marker.colorscale = null;
    trace.marker.showscale = false;
}

function scheduleLodUpdate(bbox) {
    // Devam eden önceki istekleri ve timer'ı geçersiz kıl
    lodRequestId++;
    if (lodDebounceTimer) {
        clearTimeout(lodDebounceTimer);
        lodDebounceTimer = null;
    }

    pendingLodBBox = bbox;

    // Duplicate BBox kontrolü: Viewport değişmediyse fazladan istek atma
    if (bbox && lastLodBBox &&
        Math.abs(bbox.xmin - lastLodBBox.xmin) < 1e-4 &&
        Math.abs(bbox.xmax - lastLodBBox.xmax) < 1e-4 &&
        Math.abs(bbox.ymin - lastLodBBox.ymin) < 1e-4 &&
        Math.abs(bbox.ymax - lastLodBBox.ymax) < 1e-4) {
        return;
    }
    if (!bbox && lastLodBBox === null) {
        return;
    }

    const infoText = document.getElementById("systemInfoText");
    if (infoText) {
        infoText.innerText = "Güncelleniyor...";
    }

    // 400 ms trailing idle debounce süresi: Kullanıcı zoom yapmayı bıraktıktan 400 ms sonra tek istek
    lodDebounceTimer = setTimeout(() => {
        lodDebounceTimer = null;
        fetchAndUpdateLodPlot(pendingLodBBox);
    }, LOD_DEBOUNCE_MS);
}

async function fetchAndUpdateLodPlot(bbox) {
    const reqId = lodRequestId;
    const plotElement = document.getElementById('clusterPlot');
    if (!plotElement) return;

    let url = `/api/plot?algorithm=${currentAlgo}`;
    if (bbox) {
        url += `&xmin=${bbox.xmin.toFixed(6)}&xmax=${bbox.xmax.toFixed(6)}&ymin=${bbox.ymin.toFixed(6)}&ymax=${bbox.ymax.toFixed(6)}`;
    }

    try {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const plotObj = await res.json();

        // Race condition: Yeni bir zoom/pan geldiyse bu eski cevabı çöpe at
        if (reqId !== lodRequestId) {
            return;
        }

        lastLodBBox = bbox ? { ...bbox } : null;

        // Küme Görünümü aktifse kategorik renkleri uygula
        if (currentView === 'cluster' && plotObj.data && plotObj.data[0]) {
            applyClusterColoring(plotObj.data[0]);
        }

        // Tıklanan seçili makale noktasını koru (trace 1)
        if (lastSelectedPoint && plotObj.data && plotObj.data.length > 1) {
            plotObj.data[1].x = [lastSelectedPoint.x];
            plotObj.data[1].y = [lastSelectedPoint.y];
        }

        // Mevcut görünüm sınırlarını ve annotation'ları koru
        const targetLayout = {
            ...plotObj.layout,
            annotations: currentClusterAnnotations || [],
            xaxis: {
                ...plotObj.layout.xaxis,
                ...(bbox ? { range: [bbox.xmin, bbox.xmax], autorange: false } : { autorange: true })
            },
            yaxis: {
                ...plotObj.layout.yaxis,
                ...(bbox ? { range: [bbox.ymin, bbox.ymax], autorange: false } : { autorange: true })
            }
        };

        const plotConfig = {
            ...(plotObj.config || {}),
            responsive: true,
            displayModeBar: 'hover',
            displaylogo: false,
            scrollZoom: false
        };

        // Relayout döngüsünü önlemek için kilit bayrağı
        isLodUpdating = true;
        try {
            await Plotly.react('clusterPlot', plotObj.data, targetLayout, plotConfig);
        } finally {
            setTimeout(() => {
                isLodUpdating = false;
            }, 60);
        }

        // LOD metadata rozetini güncelle
        updateLodMetaBadge(plotObj.lod_meta);

    } catch (err) {
        console.error("LOD verisi güncellenirken hata:", err);
        const infoText = document.getElementById("systemInfoText");
        if (infoText) {
            infoText.innerText = `${currentAlgo.toUpperCase()} Modülü`;
        }
    }
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
            if (infoText && !currentLodMeta) {
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

// Adaylık Tipi Belirleme Yardımcısı (Metodolojik Adaylık Sınıflandırması)
function getCandidateType(item) {
    if (!item) return 'Normal';
    if (item.karar_tipi && !item.karar_tipi.includes('TP-')) {
        return item.karar_tipi;
    }
    const d = item.ortak_agac_derinligi;
    if (d === 0 || d === '0') return 'Ana Disiplin Uyuşmazlığı Adayı';
    if (d === 1 || d === '1') return 'Alt Alan / İkincil Disiplin Adayı';
    if (d !== undefined && d !== null && d !== -1 && d !== '-1') return 'İnceleme Adayı';
    return 'Normal';
}

// Öncelik Metni Standardizasyonu (Karakter bozulmalarını düzeltir)
function getCleanPriority(item) {
    if (!item) return 'NORMAL';
    const raw = String(item.oncelik || '').toUpperCase();
    if (raw.includes('KR')) return 'KRİTİK';
    if (raw.includes('Y') || raw.includes('Ü')) return 'YÜKSEK';
    if (raw.includes('ORTA')) return 'ORTA';
    if (raw.includes('D')) return 'DÜŞÜK';
    return raw && raw !== 'NAN' && raw !== 'NONE' ? raw : 'NORMAL';
}

// "Neden Şüpheli?" Açıklanabilirlik Paneli ve Bileşik Risk Katkı Çubuğunu Render Eden Fonksiyon
function renderWhyFlagged(whyFlagged, fallbackItem, prefix = 'hdbscan') {
    const isModal = prefix === 'modal';
    const totalBadge = document.getElementById(isModal ? 'modalRiskTotalBadge' : 'hdbscanDetailRiskTotalBadge');
    const barKnn = document.getElementById(isModal ? 'modalRiskBarKnn' : 'riskBarKnn');
    const barSemantic = document.getElementById(isModal ? 'modalRiskBarSemantic' : 'riskBarSemantic');
    const barGlosh = document.getElementById(isModal ? 'modalRiskBarGlosh' : 'riskBarGlosh');
    const valKnn = document.getElementById(isModal ? 'modalRiskValKnn' : 'riskValKnn');
    const valSemantic = document.getElementById(isModal ? 'modalRiskValSemantic' : 'riskValSemantic');
    const valGlosh = document.getElementById(isModal ? 'modalRiskValGlosh' : 'riskValGlosh');
    const rulesList = document.getElementById(isModal ? 'modalRulesList' : 'hdbscanRulesList');

    if (!rulesList) return;

    const simFark = Number(whyFlagged?.semantic_gap ?? fallbackItem?.label_sim_fark ?? 0);
    const knnImp = Number(whyFlagged?.knn_impurity ?? fallbackItem?.knn_impurity ?? 0);
    const knnBask = Number(whyFlagged?.knn_dominance ?? fallbackItem?.knn_baskinlik ?? 0);
    const glosh = Number(whyFlagged?.glosh ?? fallbackItem?.glosh_skoru ?? 0);
    const candidateType = whyFlagged?.candidate_type || getCandidateType(fallbackItem);

    // Bileşik Risk Katkıları (%40 kNN, %35 Semantik Fark, %25 GLOSH)
    let knnComp = whyFlagged?.risk_components?.knn;
    let semComp = whyFlagged?.risk_components?.semantic;
    let gloshComp = whyFlagged?.risk_components?.glosh;
    let totalRisk = whyFlagged?.risk_components?.total;

    if (knnComp === undefined) {
        knnComp = Number((knnImp * 0.40).toFixed(3));
        semComp = Number((Math.min(Math.max(simFark, 0), 1) * 0.35).toFixed(3));
        gloshComp = Number((glosh * 0.25).toFixed(3));
        totalRisk = Number((knnComp + semComp + gloshComp).toFixed(3));
    }

    if (totalBadge) totalBadge.innerText = `Skor: ${Number(totalRisk).toFixed(3)}`;
    if (barKnn) barKnn.style.width = `${Math.min(Math.max(knnComp * 100, 0), 100)}%`;
    if (barSemantic) barSemantic.style.width = `${Math.min(Math.max(semComp * 100, 0), 100)}%`;
    if (barGlosh) barGlosh.style.width = `${Math.min(Math.max(gloshComp * 100, 0), 100)}%`;

    if (valKnn) valKnn.innerText = `+${Number(knnComp).toFixed(3)}`;
    if (valSemantic) valSemantic.innerText = `+${Number(semComp).toFixed(3)}`;
    if (valGlosh) valGlosh.innerText = `+${Number(gloshComp).toFixed(3)}`;

    // Karar Koşulları (Rules) Listesi
    let rules = whyFlagged?.rules;
    if (!rules || !Array.isArray(rules) || rules.length === 0) {
        const d = fallbackItem?.ortak_agac_derinligi;
        const oneriKat = String(fallbackItem?.oneri_kategori || '').trim();
        const knnOneri = String(fallbackItem?.knn_oneri || '').trim();
        const knnOnay = fallbackItem?.knn_onayliyor_mu ?? (fallbackItem?.supheli_mi === 1 ? 1 : 0);

        rules = [
            {
                label: "Semantik Kategori Farkı",
                val_text: `Değer: ${simFark.toFixed(3)}`,
                threshold_text: "Final eşik: ≥ 0.09 (Ön aday: > 0.08)",
                passed: simFark >= 0.09
            },
            {
                label: "Lokal Komşuluk Uyuşmazlığı",
                val_text: `kNN impurity: %${(knnImp * 100).toFixed(0)}`,
                threshold_text: "Eşik: ≥ %50",
                passed: knnImp >= 0.50
            },
            {
                label: "kNN Yerel Destek Uzlaşısı",
                val_text: knnOnay === 1 ? "Alternatif alan yerel komşular tarafından destekleniyor" : "Yerel komşular önerilen alanı desteklemiyor",
                threshold_text: "Eşik: Desteklemeli (== 1)",
                passed: knnOnay === 1
            },
            {
                label: d === 0 ? "kNN Baskınlık (Ana Disiplin Kuralı)" : "kNN Baskınlık / Öneri Uzlaşısı (Alt Alan Kuralı)",
                val_text: d === 0 ? `Baskınlık: %${(knnBask * 100).toFixed(0)}` : `Baskınlık: %${(knnBask * 100).toFixed(0)} | Öneri: '${knnOneri}'`,
                threshold_text: d === 0 ? "Ana disiplin eşiği: ≥ %30" : "Eşik: Öneri eşleşmesi VEYA Baskınlık ≥ %40",
                passed: d === 0 ? (knnBask >= 0.30) : ((oneriKat && oneriKat.toLowerCase() === knnOneri.toLowerCase()) || knnBask >= 0.40)
            },
            {
                label: "Yoğunluk Tabanlı Aykırılık (GLOSH)",
                val_text: `GLOSH: ${glosh.toFixed(3)}`,
                threshold_text: "Güçlü aykırılık eşiği: > 0.70",
                passed: glosh > 0.70
            },
            {
                label: "Taksonomi Hiyerarşik Derinliği",
                val_text: `Derinlik: ${d ?? '-1'} (${candidateType})`,
                threshold_text: "Adaylık eşiği: ≤ 1",
                passed: (d === 0 || d === 1)
            }
        ];
    }

    rulesList.innerHTML = rules.map(r => `
        <div class="p-2 px-3 border rounded bg-white shadow-sm d-flex align-items-center justify-content-between">
            <div class="d-flex align-items-center gap-2">
                <span class="fs-6 ${r.passed ? 'text-success' : 'text-muted'} fw-bold" style="min-width: 22px;">${r.passed ? '✓' : '✗'}</span>
                <div>
                    <div class="fw-semibold text-dark" style="font-size: 0.82rem;">${r.label}</div>
                    <div class="text-secondary" style="font-size: 0.74rem;">
                        <span class="fw-medium text-dark">${r.val_text}</span> &bull; <span class="text-muted">${r.threshold_text}</span>
                    </div>
                </div>
            </div>
            <span class="badge ${r.passed ? 'bg-success-subtle text-success border border-success-subtle' : 'bg-light text-muted border'} text-nowrap" style="font-size: 0.68rem;">
                ${r.passed ? 'Koşul Sağlandı' : 'Eşik Altı'}
            </span>
        </div>
    `).join('');
}

// Tekil anomali kartını DOM'a oluşturan fonksiyon
function renderAnomalyCard(item, container, algo) {
    const candidateType = getCandidateType(item);
    const cleanPriority = getCleanPriority(item);
    const isCritical = candidateType.includes("Ana Disiplin") || cleanPriority === "KRİTİK";
    const badgeClass = isCritical ? "badge-critical" : (candidateType.includes("Alt Alan") || cleanPriority === "YÜKSEK" ? "badge-high" : "bg-secondary text-white");
    const badgeLabel = `${cleanPriority} · ${candidateType}`;

    const card = document.createElement("div");
    card.className = "card card-custom p-3 anomaly-card-item";
    card.style.cursor = "pointer";
    // Karta tıklandığında sağdaki anomali detay panelini aç
    card.onclick = () => {
        document.querySelectorAll('.anomaly-card-item').forEach(el => {
            el.classList.remove('card-anomaly-active');
        });
        card.classList.add('card-anomaly-active');
        showHdbscanCardDetail(item);
    };

    const scoreLabel = algo === "hdbscan" ? "GLOSH" : "Aykırılık";
    const scoreVal = Number(item.glosh_skoru || item.aykirilik_skoru || 0).toFixed(3);
    const riskVal = Number(item.risk_skoru || 0).toFixed(3);
    const kumeVal = item.kume !== undefined && item.kume !== -1 ? `#${item.kume}` : (item.kmeans_kume !== undefined && item.kmeans_kume !== -1 ? `#${item.kmeans_kume}` : 'Aykırı / -1');

    card.innerHTML = `
        <div class="d-flex justify-content-between align-items-start mb-2">
            <span class="badge ${badgeClass} badge-risk">${badgeLabel}</span>
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

// Sağdaki Anomali Detay Panelini Dolduran Fonksiyon
function showHdbscanCardDetail(item) {
    if (!item) return;

    const emptyEl = document.getElementById('hdbscanDetailEmpty');
    const contentEl = document.getElementById('hdbscanDetailContent');
    if (emptyEl) emptyEl.style.display = 'none';
    if (contentEl) contentEl.style.display = 'block';

    const candidateType = getCandidateType(item);
    const cleanPriority = getCleanPriority(item);
    const isCritical = candidateType.includes("Ana Disiplin") || cleanPriority === "KRİTİK";

    const pBadge = document.getElementById("hdbscanDetailPriority");
    if (pBadge) {
        pBadge.className = `badge badge-risk ${isCritical ? "badge-critical" : (candidateType.includes("Alt Alan") || cleanPriority === "YÜKSEK" ? "badge-high" : "bg-secondary text-white")} mb-1`;
        pBadge.innerText = `${cleanPriority} · ${candidateType}`;
    }

    const idElem = document.getElementById("hdbscanDetailId");
    if (idElem) {
        idElem.innerText = item.external_id ? `(ID: ${item.external_id})` : '';
    }

    const titleElem = document.getElementById("hdbscanDetailTitle");
    if (titleElem) {
        titleElem.innerText = item.baslik || item.title || (item.external_id ? `Makale ID: ${item.external_id}` : 'Başlık Belirtilmemiş');
    }

    // 1. Metrik Kutusu: Bileşik Risk
    const riskVal = item.risk_skoru !== undefined ? Number(item.risk_skoru).toFixed(3) : '-';
    const riskElem = document.getElementById("hdbscanDetailRisk");
    if (riskElem) riskElem.innerText = riskVal;

    // 2. Metrik Kutusu: Semantik Fark
    const simFarkVal = item.label_sim_fark !== undefined ? Number(item.label_sim_fark).toFixed(3) : '-';
    const simFarkElem = document.getElementById("hdbscanDetailSimFark");
    if (simFarkElem) simFarkElem.innerText = simFarkVal;

    // 3. Metrik Kutusu: k-NN Impurity
    const knnImpurityVal = item.knn_impurity !== undefined 
        ? `%${(Number(item.knn_impurity) * 100).toFixed(0)}`
        : '-';
    const knnImpurityElem = document.getElementById("hdbscanDetailKnnImpurity");
    if (knnImpurityElem) knnImpurityElem.innerText = knnImpurityVal;

    // 4. Metrik Kutusu: GLOSH Skoru
    const scoreLabel = currentAlgo === "hdbscan" ? "GLOSH SKORU" : "AYKIRILIK SKORU";
    const lblElem = document.getElementById("hdbscanDetailScoreLabel");
    if (lblElem) lblElem.innerText = scoreLabel;

    const scoreVal = item.glosh_skoru !== undefined || item.aykirilik_skoru !== undefined 
        ? Number(item.glosh_skoru || item.aykirilik_skoru || 0).toFixed(3) 
        : '-';
    const gloshElem = document.getElementById("hdbscanDetailGlosh");
    if (gloshElem) gloshElem.innerText = scoreVal;

    // Sınıflandırma ve Öneri Analizi Tablosu
    const mevcutKatElem = document.getElementById("hdbscanDetailMevcutKat");
    if (mevcutKatElem) mevcutKatElem.innerText = item.mevcut_kategori || item.gercek_kategori || item.kategori || '-';

    const oneriKatElem = document.getElementById("hdbscanDetailOneriKat");
    if (oneriKatElem) oneriKatElem.innerText = item.oneri_kategori || item.model_onerisi || '-';

    const knoneriElem = document.getElementById("hdbscanDetailKnnOneri");
    if (knoneriElem) knoneriElem.innerText = item.knn_oneri || '-';

    // Küme Bilgisi (İkincil metadata olarak tabloya taşındı)
    const kumeVal = item.kume !== undefined && item.kume !== -1 
        ? `Küme #${item.kume}` 
        : (item.hdbscan_kume !== undefined && item.hdbscan_kume !== -1 ? `Küme #${item.hdbscan_kume}` : 'Aykırı / -1');
    const kumeElem = document.getElementById("hdbscanDetailKume");
    if (kumeElem) kumeElem.innerText = kumeVal;

    const kararElem = document.getElementById("hdbscanDetailKararTipi");
    if (kararElem) {
        const badgeClass = candidateType.includes('Ana Disiplin') ? 'bg-danger' : (candidateType.includes('Alt Alan') ? 'bg-warning text-dark' : 'bg-secondary');
        const descText = item.filtre_aciklamasi || (candidateType.includes('Ana Disiplin') ? 'Farklı Ana Disiplin Uyuşmazlığı (Kritik Öncelik)' : 'Alt Alan Uyuşmazlığı / Çoklu Disiplin Zenginleştirme');
        kararElem.innerHTML = `<span class="badge ${badgeClass} me-2">${candidateType}</span> <small class="text-secondary">${descText}</small>`;
    }

    const ozetElem = document.getElementById("hdbscanDetailOzet");
    if (ozetElem) {
        ozetElem.innerText = item.ozet || item.abstract || 'Özet metni veri kümesinde bulunamadı.';
    }

    // Neden Şüpheli bölümünü render et
    renderWhyFlagged(item.why_flagged, item, 'hdbscan');

    // Lazy tam detay getirme (tam özet, why_flagged ve güncel alanlar için)
    if (item.external_id) {
        fetch(`/api/article/${encodeURIComponent(item.external_id)}`)
            .then(res => res.ok ? res.json() : null)
            .then(data => {
                if (!data) return;
                if (document.getElementById("hdbscanDetailId")?.innerText.includes(item.external_id)) {
                    if (data.ozet && ozetElem) ozetElem.innerText = data.ozet;
                    
                    const updatedCandType = getCandidateType(data);
                    if (kararElem) {
                        const badgeClass = updatedCandType.includes('Ana Disiplin') ? 'bg-danger' : (updatedCandType.includes('Alt Alan') ? 'bg-warning text-dark' : 'bg-secondary');
                        const descText = data.filtre_aciklamasi || (updatedCandType.includes('Ana Disiplin') ? 'Farklı Ana Disiplin Uyuşmazlığı (Kritik Öncelik)' : 'Alt Alan Uyuşmazlığı / Çoklu Disiplin Zenginleştirme');
                        kararElem.innerHTML = `<span class="badge ${badgeClass} me-2">${updatedCandType}</span> <small class="text-secondary">${descText}</small>`;
                    }
                    if (data.label_sim_fark !== undefined && simFarkElem) {
                        simFarkElem.innerText = Number(data.label_sim_fark).toFixed(3);
                    }
                    if (data.knn_impurity !== undefined && knnImpurityElem) {
                        knnImpurityElem.innerText = `%${(Number(data.knn_impurity) * 100).toFixed(0)}`;
                    }
                    if (data.why_flagged) {
                        renderWhyFlagged(data.why_flagged, data, 'hdbscan');
                    }
                }
            })
            .catch(() => {});
    }
}

function resetHdbscanDetail() {
    const emptyEl = document.getElementById('hdbscanDetailEmpty');
    const contentEl = document.getElementById('hdbscanDetailContent');
    if (emptyEl) emptyEl.style.display = 'block';
    if (contentEl) contentEl.style.display = 'none';

    document.querySelectorAll('.anomaly-card-item').forEach(el => {
        el.classList.remove('card-anomaly-active');
    });
}
window.resetHdbscanDetail = resetHdbscanDetail;

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
window.resetSidePanel = resetSidePanel;

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

    const candidateType = getCandidateType(item);
    const cleanPriority = getCleanPriority(item);
    const isCritical = candidateType.includes("Ana Disiplin") || cleanPriority === "KRİTİK";

    const pBadge = document.getElementById("modalPriority");
    if (pBadge) {
        pBadge.className = `badge badge-risk ${isCritical ? "badge-critical" : (candidateType.includes("Alt Alan") || cleanPriority === "YÜKSEK" ? "badge-high" : "bg-secondary text-white")} mb-1`;
        pBadge.innerText = `${cleanPriority} · ${candidateType}`;
    }

    if (document.getElementById("modalTitle")) document.getElementById("modalTitle").innerText = item.baslik || 'Başlık Yok';
    if (document.getElementById("modalRisk")) document.getElementById("modalRisk").innerText = Number(item.risk_skoru || 0).toFixed(3);
    
    if (document.getElementById("modalSimFark")) {
        document.getElementById("modalSimFark").innerText = item.label_sim_fark !== undefined ? Number(item.label_sim_fark).toFixed(3) : '-';
    }
    if (document.getElementById("modalKnnImpurity")) {
        document.getElementById("modalKnnImpurity").innerText = item.knn_impurity !== undefined ? `%${(Number(item.knn_impurity) * 100).toFixed(0)}` : '-';
    }

    const scoreVal = Number(item.glosh_skoru || item.aykirilik_skoru || 0).toFixed(3);
    if (document.getElementById("modalGlosh")) document.getElementById("modalGlosh").innerText = scoreVal;

    if (document.getElementById("modalMevcutKat")) document.getElementById("modalMevcutKat").innerText = item.mevcut_kategori || '-';
    if (document.getElementById("modalOneriKat")) document.getElementById("modalOneriKat").innerText = item.oneri_kategori || '-';
    if (document.getElementById("modalKnnOneri")) document.getElementById("modalKnnOneri").innerText = item.knn_oneri || '-';

    const kumeVal = item.kume !== undefined && item.kume !== -1 
        ? `Küme #${item.kume}` 
        : (item.hdbscan_kume !== undefined && item.hdbscan_kume !== -1 ? `Küme #${item.hdbscan_kume}` : 'Aykırı / -1');
    if (document.getElementById("modalKume")) document.getElementById("modalKume").innerText = kumeVal;

    const modalKarar = document.getElementById("modalKararTipi");
    if (modalKarar) {
        const badgeClass = candidateType.includes('Ana Disiplin') ? 'bg-danger' : (candidateType.includes('Alt Alan') ? 'bg-warning text-dark' : 'bg-secondary');
        const descText = item.filtre_aciklamasi || (candidateType.includes('Ana Disiplin') ? 'Farklı Ana Disiplin Uyuşmazlığı (Kritik Öncelik)' : 'Alt Alan Uyuşmazlığı / Çoklu Disiplin Zenginleştirme');
        modalKarar.innerHTML = `<span class="badge ${badgeClass} me-2">${candidateType}</span> <small class="text-secondary">${descText}</small>`;
    }

    if (document.getElementById("modalOzet")) document.getElementById("modalOzet").innerText = item.ozet || 'Özet metni veri kümesinde bulunamadı.';

    renderWhyFlagged(item.why_flagged, item, 'modal');

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