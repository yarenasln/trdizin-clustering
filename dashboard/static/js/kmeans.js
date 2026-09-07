const km = (id) => document.getElementById(id);
const kmPct = (v) => `%${(Number(v || 0) * 100).toFixed(2)}`;

function kmEscape(value) {
    return String(value ?? '').replace(/[&<>"']/g, (char) => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;'
    }[char]));
}

function kmTags(values, type = '') {
    if (!values || values.length === 0) return '<span class="km-tag">Yok</span>';
    return values.map((value) => `<span class="km-tag ${type}">${kmEscape(value)}</span>`).join('');
}

async function loadKmeansArticles() {
    const params = new URLSearchParams({
        search: km('kmSearch').value.trim(),
        match: km('kmMatchFilter').value,
        pred_count: km('kmPredCount').value,
        sort: km('kmSort').value,
    });

    const res = await fetch(`/api/kmeans/articles?${params.toString()}`);
    const json = await res.json();
    km('kmCountText').innerText = `${json.total || 0} makale bulundu`;

    const list = km('kmArticleList');
    list.innerHTML = '';

    if (!json.data || json.data.length === 0) {
        list.innerHTML = '<div class="alert alert-light border text-center small">Filtrelere uygun makale bulunamadı.</div>';
        return;
    }

    json.data.forEach((item) => {
        const card = document.createElement('div');
        card.className = 'km-article-item';
        card.dataset.id = item.external_id;
        const title = item.title || 'Başlık bilgisi yok';
        card.innerHTML = `
            <div class="d-flex justify-content-between gap-2 align-items-start">
                <div class="km-article-id">${kmEscape(item.external_id)}${item.year ? ` • ${kmEscape(item.year)}` : ''}</div>
                <span class="score-pill">${item.predicted_count} tahmin</span>
            </div>
            <div class="km-article-title">${kmEscape(title)}</div>
            <div class="d-flex flex-wrap gap-2 mt-2">
                <span class="score-pill score-pill-info">✓ ${item.matched_count}/${item.true_count} • ${kmPct(item.match_rate)}</span>
                <span class="score-pill score-pill-danger">✕ ${item.wrong_count} eşleşmeyen</span>
                <span class="score-pill">F1 ${kmPct(item.f1)}</span>
            </div>
            <div class="km-progress mt-2"><span style="width:${Math.min(100, Number(item.match_rate || 0) * 100)}%"></span></div>
        `;
        card.onclick = () => loadKmeansArticle(item.external_id, card);
        list.appendChild(card);
    });
}

async function loadKmeansArticle(externalId, selectedCard) {
    document.querySelectorAll('.km-article-item').forEach((el) => el.classList.remove('active'));
    if (selectedCard) selectedCard.classList.add('active');

    const res = await fetch(`/api/kmeans/article/${encodeURIComponent(externalId)}`);
    const item = await res.json();
    if (!res.ok) {
        alert(item.error || 'Makale yüklenemedi.');
        return;
    }

    const detail = km('kmDetail');
    detail.innerHTML = `
        <div class="d-flex justify-content-between align-items-start gap-3 mb-2">
            <div>
                <div class="km-kicker">MAKALE DETAYI</div>
                <h4 class="fw-bold mt-1 mb-2">${kmEscape(item.title || 'Başlık bilgisi yok')}</h4>
            </div>
            <span class="score-pill score-pill-info">${item.matched_count}/${item.true_count} doğru • ${kmPct(item.match_rate)}</span>
        </div>

        <div class="d-flex flex-wrap gap-2 mb-3">
            <span class="score-pill">External ID: <strong>${kmEscape(item.external_id)}</strong></span>
            <span class="score-pill">DOI: <strong>${kmEscape(item.doi || 'Yok')}</strong></span>
            <span class="score-pill">Yıl: <strong>${kmEscape(item.year || 'Yok')}</strong></span>
            <span class="score-pill">Dil: <strong>${kmEscape(item.language || 'Yok')}</strong></span>
        </div>

        <div class="km-section-title">📄 ÖZET</div>
        <div class="km-abstract">${kmEscape(item.abstract || 'Özet bilgisi bulunamadı.')}</div>

        <div class="km-section-title">🧭 ANA BAŞLIK TAHMİNİ</div>
        <div class="km-tags">${kmTags(item.main_topics, 'main')}</div>

        <div class="row g-3 mt-1">
            <div class="col-md-6">
                <div class="km-topic-box h-100">
                    <div class="km-section-title mt-0">GERÇEK TR DİZİN KONULARI</div>
                    <div class="km-tags">${kmTags(item.true_topics)}</div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="km-topic-box h-100">
                    <div class="km-section-title mt-0">K-MEANS TAHMİNLERİ</div>
                    <div class="km-tags">${kmTags(item.matched, 'good')}${kmTags(item.wrong, 'bad')}</div>
                </div>
            </div>
        </div>

        <div class="row g-3 mt-1">
            <div class="col-md-4"><div class="km-result-box good"><small>✓ Doğru Yakalanan</small><strong>${item.matched_count}</strong><div class="km-tags mt-2">${kmTags(item.matched, 'good')}</div></div></div>
            <div class="col-md-4"><div class="km-result-box bad"><small>✕ Eşleşmeyen Tahmin</small><strong>${item.wrong_count}</strong><div class="km-tags mt-2">${kmTags(item.wrong, 'bad')}</div></div></div>
            <div class="col-md-4"><div class="km-result-box missed"><small>! Kaçırılan Gerçek Konu</small><strong>${item.missed_count}</strong><div class="km-tags mt-2">${kmTags(item.missed, 'missed')}</div></div></div>
        </div>

        <div class="km-match-panel mt-3">
            <div class="d-flex justify-content-between align-items-end gap-3">
                <div><small class="d-block">GERÇEK KONULARIN YAKALANMA ORANI</small><strong>${item.matched_count} / ${item.true_count} gerçek konu yakalandı</strong></div>
                <div class="km-match-big">${kmPct(item.match_rate)}</div>
            </div>
            <div class="km-progress light mt-2"><span style="width:${Math.min(100, Number(item.match_rate || 0) * 100)}%"></span></div>
            <div class="row g-2 mt-2">
                <div class="col-4"><div class="km-mini-metric"><small>PRECISION</small><strong>${kmPct(item.precision)}</strong></div></div>
                <div class="col-4"><div class="km-mini-metric"><small>RECALL</small><strong>${kmPct(item.recall)}</strong></div></div>
                <div class="col-4"><div class="km-mini-metric"><small>F1</small><strong>${kmPct(item.f1)}</strong></div></div>
            </div>
        </div>
    `;
}

// UMAP noktasından tıklandığında üst sol paneli (#article-detail-panel) dolduran fonksiyon
async function loadUmapPointDetails(externalId) {
    if (!externalId) return;

    const welcomeWrapper = document.getElementById('panel-content-wrapper');
    const activeContent = document.getElementById('panel-active-content');
    const titleElem = document.getElementById('panel-title');
    const abstractElem = document.getElementById('panel-abstract');
    const riskElem = document.getElementById('panel-risk');
    const catElem = document.getElementById('panel-cat');
    const sugElem = document.getElementById('panel-suggestion');
    const idElem = document.getElementById('panel-id');

    if (welcomeWrapper) welcomeWrapper.style.display = 'none';
    if (activeContent) activeContent.style.display = 'block';

    if (idElem) idElem.innerText = `(ID: ${kmEscape(externalId)})`;
    if (titleElem) titleElem.innerHTML = '<span class="spinner-border spinner-border-sm text-secondary me-2"></span>Yükleniyor...';
    if (abstractElem) abstractElem.innerText = 'Makale detayları sunucudan getiriliyor, lütfen bekleyin...';
    if (riskElem) riskElem.innerText = '...';
    if (catElem) catElem.innerText = '...';
    if (sugElem) sugElem.innerText = '...';

    try {
        const res = await fetch(`/api/kmeans/article/${encodeURIComponent(externalId)}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const item = await res.json();

        if (titleElem) titleElem.innerText = item.title || 'Başlık bilgisi yok';
        if (abstractElem) abstractElem.innerText = item.abstract || 'Özet bilgisi bulunamadı.';
        
        const rate = item.match_rate !== undefined ? item.match_rate : (item.f1 || 0);
        if (riskElem) riskElem.innerText = `%${(Number(rate) * 100).toFixed(1)}`;
        
        if (catElem) {
            catElem.innerText = Array.isArray(item.true_topics) && item.true_topics.length > 0 
                ? item.true_topics.join(' || ') 
                : (item.true_topics || '-');
        }
        
        if (sugElem) {
            sugElem.innerText = Array.isArray(item.predicted_topics) && item.predicted_topics.length > 0 
                ? item.predicted_topics.join(' || ') 
                : (item.predicted_topics || '-');
        }
    } catch (err) {
        console.error('UMAP makale detayı yüklenirken hata:', err);
        if (titleElem) titleElem.innerText = 'Yükleme Hatası';
        if (abstractElem) abstractElem.innerText = `ID: ${externalId} olan makalenin detayları getirilemedi.`;
    }
}

// Üst sol paneli başlangıç rehber durumuna döndüren fonksiyon
function resetSidePanel() {
    const welcomeWrapper = document.getElementById('panel-content-wrapper');
    const activeContent = document.getElementById('panel-active-content');
    if (welcomeWrapper) welcomeWrapper.style.display = 'block';
    if (activeContent) activeContent.style.display = 'none';
}

// UMAP iframe'i ve postMessage ile tıklama dinleyicisini kuran fonksiyon
function setupUmapInteraction() {
    window.addEventListener('message', (event) => {
        if (event.data && event.data.type === 'KMEANS_POINT_CLICK' && event.data.externalId) {
            loadUmapPointDetails(event.data.externalId);
        }
    });

    const iframe = document.getElementById('kmeansUmapFrame') || document.querySelector('.km-umap-frame');
    if (iframe) {
        const attachDirect = () => {
            try {
                const doc = iframe.contentDocument || iframe.contentWindow?.document;
                if (!doc) return;
                const plotEl = doc.querySelector('.plotly-graph-div');
                if (plotEl && iframe.contentWindow?.Plotly) {
                    plotEl.on('plotly_click', (data) => {
                        if (data && data.points && data.points.length > 0) {
                            const pt = data.points[0];
                            const extId = (pt.customdata && pt.customdata.length > 0) ? pt.customdata[0] : null;
                            if (extId) {
                                loadUmapPointDetails(extId);
                            }
                        }
                    });
                }
            } catch (err) {
                // postMessage yedek olarak çalışır
            }
        };

        iframe.addEventListener('load', attachDirect);
        attachDirect();
    }
}

document.addEventListener('DOMContentLoaded', () => {
    km('kmeansAlgoSelect').addEventListener('change', (event) => {
        if (event.target.value === 'hdbscan') window.location.href = '/';
    });
    km('kmSearchBtn').addEventListener('click', loadKmeansArticles);
    km('kmSearch').addEventListener('keydown', (event) => {
        if (event.key === 'Enter') loadKmeansArticles();
    });
    ['kmMatchFilter', 'kmPredCount', 'kmSort'].forEach((id) => {
        km(id).addEventListener('change', loadKmeansArticles);
    });
    loadKmeansArticles();
    setupUmapInteraction();
});
