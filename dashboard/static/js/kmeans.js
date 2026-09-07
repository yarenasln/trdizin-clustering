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
    renderKmeansDetail(item);
}

function renderKmeansDetail(item) {
    const emptyEl = document.getElementById('kmDetailEmpty');
    const contentEl = document.getElementById('kmDetailContent');
    if (emptyEl) emptyEl.style.display = 'none';
    if (contentEl) contentEl.style.display = 'block';

    const matchPct = Number(item.match_rate || 0) * 100;
    let badgeClass = 'bg-danger';
    let badgeText = 'Eşleşme Yok';
    if (matchPct >= 100) {
        badgeClass = 'bg-success';
        badgeText = '%100 Tam Eşleşme';
    } else if (matchPct >= 50) {
        badgeClass = 'bg-warning text-dark';
        badgeText = `%${matchPct.toFixed(0)} Kısmi Eşleşme`;
    } else if (item.matched_count > 0) {
        badgeClass = 'bg-secondary';
        badgeText = `%${matchPct.toFixed(0)} Düşük Eşleşme`;
    }

    contentEl.innerHTML = `
        <div class="d-flex justify-content-between align-items-start mb-3">
            <div>
                <span class="badge ${badgeClass} mb-1">${badgeText}</span>
                <span class="text-muted ms-2 fw-semibold" style="font-size: 0.85rem; font-family: monospace;">(ID: ${kmEscape(item.external_id)})</span>
            </div>
            <button type="button" class="btn-close btn-sm" onclick="resetKmDetail()" title="Kapat"></button>
        </div>

        <h5 class="fw-bold text-dark mb-3 text-break" style="font-size: 1.05rem;">${kmEscape(item.title || 'Başlık bilgisi yok')}</h5>

        <!-- Metrik Kutuları (Ortak 4'lü Şema) -->
        <div class="row g-2 mb-3">
            <div class="col-md-3 col-6">
                <div class="metric-box text-center">
                    <small class="text-muted d-block font-monospace" style="font-size: 0.72rem;">YAKALANMA ORANI</small>
                    <span class="fw-bold text-success fs-5">${kmPct(item.match_rate)}</span>
                </div>
            </div>
            <div class="col-md-3 col-6">
                <div class="metric-box text-center">
                    <small class="text-muted d-block font-monospace" style="font-size: 0.72rem;">F1 SKORU</small>
                    <span class="fw-bold text-info fs-5">${Number(item.f1 || 0).toFixed(3)}</span>
                </div>
            </div>
            <div class="col-md-3 col-6">
                <div class="metric-box text-center">
                    <small class="text-muted d-block font-monospace" style="font-size: 0.72rem;">PRECISION</small>
                    <span class="fw-bold text-warning fs-5">${kmPct(item.precision)}</span>
                </div>
            </div>
            <div class="col-md-3 col-6">
                <div class="metric-box text-center">
                    <small class="text-muted d-block font-monospace" style="font-size: 0.72rem;">RECALL</small>
                    <span class="fw-bold text-primary fs-5">${kmPct(item.recall)}</span>
                </div>
            </div>
        </div>

        <!-- Sınıflandırma ve Konu Analizi Tablosu (Ortak Tablo Şeması) -->
        <h6 class="fw-bold text-dark text-uppercase small mb-2">🏷️ Konu Kümeleme ve Eşleşme Analizi</h6>
        <div class="table-responsive mb-3">
            <table class="table table-bordered table-sm align-middle mb-0">
                <tbody>
                    <tr>
                        <th class="bg-light text-secondary w-25">Ana Başlık Tahmini:</th>
                        <td>
                            <div class="d-flex flex-wrap gap-1">
                                ${kmTags(item.main_topics, 'main')}
                            </div>
                        </td>
                    </tr>
                    <tr>
                        <th class="bg-light text-secondary">Gerçek TR Dizin Konuları:</th>
                        <td>
                            <div class="d-flex flex-wrap gap-1">
                                ${kmTags(item.true_topics)}
                            </div>
                        </td>
                    </tr>
                    <tr>
                        <th class="bg-light text-secondary">K-Means Tahminleri:</th>
                        <td>
                            <div class="d-flex flex-wrap gap-1">
                                ${kmTags(item.matched, 'good')}
                                ${kmTags(item.wrong, 'bad')}
                            </div>
                        </td>
                    </tr>
                    <tr>
                        <th class="bg-light text-secondary">Eşleşme Durumu:</th>
                        <td>
                            <div class="d-flex align-items-center gap-2 mb-1">
                                <div class="progress flex-grow-1" style="height: 6px; background-color: #e2e8f0;">
                                    <div class="progress-bar bg-success" role="progressbar" style="width: ${Math.min(100, Number(item.match_rate || 0) * 100)}%"></div>
                                </div>
                                <span class="small fw-bold text-success" style="font-size: 0.78rem;">${kmPct(item.match_rate)}</span>
                            </div>
                            <div class="d-flex flex-wrap gap-1 align-items-center">
                                <span class="badge bg-success" style="font-size: 0.72rem;">✓ ${item.matched_count} Doğru Yakalanan</span>
                                <span class="badge ${item.wrong_count > 0 ? 'bg-danger' : 'bg-light text-muted border'}" style="font-size: 0.72rem;">✕ ${item.wrong_count} Eşleşmeyen</span>
                                <span class="badge ${item.missed_count > 0 ? 'bg-warning text-dark' : 'bg-light text-muted border'}" style="font-size: 0.72rem;">! ${item.missed_count} Kaçırılan</span>
                                <small class="text-muted ms-auto" style="font-size: 0.72rem;">(${item.matched_count} / ${item.true_count} gerçek konu)</small>
                            </div>
                        </td>
                    </tr>
                    ${(item.doi || item.year || item.language) ? `
                    <tr>
                        <th class="bg-light text-secondary">Yayın Bilgisi:</th>
                        <td class="small text-secondary">
                            ${item.doi ? `<span class="me-3"><strong>DOI:</strong> ${kmEscape(item.doi)}</span>` : ''}
                            ${item.year ? `<span class="me-3"><strong>Yıl:</strong> ${kmEscape(item.year)}</span>` : ''}
                            ${item.language ? `<span><strong>Dil:</strong> ${kmEscape(item.language)}</span>` : ''}
                        </td>
                    </tr>` : ''}
                </tbody>
            </table>
        </div>

        <!-- Özet Metni (Ortak Kutu Şeması) -->
        <h6 class="fw-bold text-dark text-uppercase small mb-2">📄 Özet Metni</h6>
        <div class="p-3 bg-light border rounded mb-2">
            <p class="small mb-0 text-secondary" style="line-height: 1.6; text-align: justify;">
                ${kmEscape(item.abstract || 'Özet metni bulunmuyor.')}
            </p>
        </div>
    `;
}

function resetKmDetail() {
    const emptyEl = document.getElementById('kmDetailEmpty');
    const contentEl = document.getElementById('kmDetailContent');
    if (emptyEl) emptyEl.style.display = 'block';
    if (contentEl) contentEl.style.display = 'none';

    document.querySelectorAll('.km-article-item').forEach((el) => {
        el.classList.remove('active');
    });
}
window.resetKmDetail = resetKmDetail;

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
