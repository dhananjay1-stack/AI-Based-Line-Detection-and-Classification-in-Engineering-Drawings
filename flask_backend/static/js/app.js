/**
 * Engineering Drawing — Line Detection Frontend
 */

const API_BASE = '';

// State
let currentJobId = null;
let currentResults = null;
let selectedFiles = [];
let zoomLevel = 1;
let pollInterval = null;
let classOptions = [];   // [{index, name, color, color_rgb}, …]

// DOM
const uploadArea     = document.getElementById('uploadArea');
const fileInput      = document.getElementById('fileInput');
const fileList       = document.getElementById('fileList');
const processBtn     = document.getElementById('processBtn');
const statusSection  = document.getElementById('statusSection');
const emptyState     = document.getElementById('emptyState');
const resultsContainer = document.getElementById('resultsContainer');
const resultsGrid    = document.getElementById('resultsGrid');
const detailView     = document.getElementById('detailView');
const historyModal   = document.getElementById('historyModal');

// ── Init ────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    setupUploadHandlers();
    setupViewTabs();
    setupOpacitySlider();
    setupSliders();
    loadCheckpoints();
    loadClassOptions();
});

// ── Upload ──────────────────────────────────────────────────────────
function setupUploadHandlers() {
    uploadArea.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', e => handleFiles(e.target.files));
    uploadArea.addEventListener('dragover', e => { e.preventDefault(); uploadArea.classList.add('dragover'); });
    uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
    uploadArea.addEventListener('drop', e => { e.preventDefault(); uploadArea.classList.remove('dragover'); handleFiles(e.dataTransfer.files); });
}

function handleFiles(files) {
    selectedFiles = Array.from(files);
    renderFileList();
    processBtn.disabled = selectedFiles.length === 0;
}

function renderFileList() {
    fileList.innerHTML = selectedFiles.map((f, i) => `
        <div class="file-item">
            <span><i class="fas fa-file-image"></i> ${f.name}</span>
            <span class="remove-file" onclick="removeFile(${i})"><i class="fas fa-times"></i></span>
        </div>
    `).join('');
}

function removeFile(idx) {
    selectedFiles.splice(idx, 1);
    renderFileList();
    processBtn.disabled = selectedFiles.length === 0;
}

// ── Checkpoints ─────────────────────────────────────────────────────
async function loadCheckpoints() {
    try {
        const r = await fetch(`${API_BASE}/api/checkpoints`);
        const d = await r.json();
        const sel = document.getElementById('checkpointSelect');
        sel.innerHTML = '';
        if (d.checkpoints.length === 0) {
            sel.innerHTML = '<option value="">No checkpoints found</option>';
            return;
        }
        d.checkpoints.forEach(c => {
            const opt = document.createElement('option');
            opt.value = c.path;
            opt.textContent = `${c.name}  (${c.size_mb} MB)`;
            if (c.path === d.default) opt.selected = true;
            sel.appendChild(opt);
        });
    } catch (e) {
        console.error('Failed to load checkpoints:', e);
    }
}

// ── Class Options ───────────────────────────────────────────────────
async function loadClassOptions() {
    try {
        const r = await fetch(`${API_BASE}/api/classes`);
        const d = await r.json();
        classOptions = d.classes;
        renderClassFilter();
    } catch (e) {
        console.error('Failed to load classes:', e);
    }
}

function renderClassFilter() {
    const grid = document.getElementById('classFilterGrid');
    grid.innerHTML = classOptions.map(c => `
        <label class="class-toggle" title="${formatClassName(c.name)}">
            <input type="checkbox" value="${c.index}" checked data-classname="${c.name}">
            <span class="class-chip" style="--chip-color: ${c.color};">
                <span class="class-dot" style="background: ${c.color};"></span>
                ${formatClassName(c.name)}
            </span>
        </label>
    `).join('');
}

function toggleAllClasses(state) {
    document.querySelectorAll('#classFilterGrid input[type="checkbox"]')
        .forEach(cb => cb.checked = state);
}

function getSelectedClasses() {
    const checked = document.querySelectorAll('#classFilterGrid input[type="checkbox"]:checked');
    return Array.from(checked).map(cb => parseInt(cb.value));
}

// ── Sliders ─────────────────────────────────────────────────────────
function setupSliders() {
    const confSlider = document.getElementById('confidenceSlider');
    const dilateSlider = document.getElementById('dilateSlider');

    confSlider.addEventListener('input', () => {
        document.getElementById('confValue').textContent = parseFloat(confSlider.value).toFixed(2);
    });
    dilateSlider.addEventListener('input', () => {
        document.getElementById('dilateValue').textContent = dilateSlider.value + ' px';
    });
}

// ── Process ─────────────────────────────────────────────────────────
processBtn.addEventListener('click', async () => {
    if (selectedFiles.length === 0) return;

    try {
        processBtn.disabled = true;
        processBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Uploading…';

        const formData = new FormData();
        selectedFiles.forEach(f => formData.append('files', f));

        const upRes = await fetch(`${API_BASE}/api/upload`, { method: 'POST', body: formData });
        if (!upRes.ok) throw new Error('Upload failed');

        const upData = await upRes.json();
        currentJobId = upData.job_id;

        statusSection.style.display = 'block';
        updateStatus('uploaded');

        processBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing…';

        // collect params
        const payload = {
            job_id: currentJobId,
            checkpoint: document.getElementById('checkpointSelect').value,
            classes: getSelectedClasses(),
            use_tta: document.getElementById('ttaToggle').checked,
            confidence: parseFloat(document.getElementById('confidenceSlider').value),
            dilate: parseInt(document.getElementById('dilateSlider').value),
        };

        const runRes = await fetch(`${API_BASE}/api/run`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (!runRes.ok) throw new Error('Failed to start processing');

        updateStatus('processing');
        startPolling();

    } catch (err) {
        console.error(err);
        alert('Error: ' + err.message);
        resetProcessButton();
    }
});

// ── Status / Polling ────────────────────────────────────────────────
function updateStatus(status) {
    const steps = ['uploaded', 'processing', 'done'];
    const cur = steps.indexOf(status);
    document.querySelectorAll('.status-step').forEach((el, i) => {
        el.classList.remove('active', 'completed');
        if (i < cur) el.classList.add('completed');
        else if (i === cur) el.classList.add('active');
    });
}

function startPolling() {
    if (pollInterval) clearInterval(pollInterval);
    pollInterval = setInterval(async () => {
        try {
            const r = await fetch(`${API_BASE}/api/status/${currentJobId}`);
            const d = await r.json();
            if (!r.ok) { clearInterval(pollInterval); alert('Job error'); resetProcessButton(); return; }
            if (d.progress) {
                document.getElementById('progressInfo').textContent =
                    `Processing ${d.progress.current}/${d.progress.total}: ${d.progress.filename}`;
            }
            if (d.status === 'done') { clearInterval(pollInterval); updateStatus('done'); await loadResults(); resetProcessButton(); }
            else if (d.status === 'failed') { clearInterval(pollInterval); alert('Failed: ' + (d.error || '')); resetProcessButton(); }
        } catch (e) { console.error(e); }
    }, 1000);
}

function resetProcessButton() {
    processBtn.disabled = false;
    processBtn.innerHTML = '<i class="fas fa-play"></i> Run Detection';
}

// ── Results ─────────────────────────────────────────────────────────
async function loadResults() {
    try {
        const r = await fetch(`${API_BASE}/api/result/${currentJobId}`);
        const d = await r.json();
        if (d.status !== 'done' || !d.results) throw new Error('Results not ready');
        currentResults = d.results;
        emptyState.style.display = 'none';
        resultsContainer.style.display = 'block';
        if (currentResults.length === 1) showDetailView(0);
        else renderResultsGrid();
    } catch (e) { console.error(e); alert('Error loading results'); }
}

function renderResultsGrid() {
    detailView.style.display = 'none';
    resultsGrid.style.display = 'grid';
    resultsGrid.innerHTML = currentResults.map((res, i) => {
        const conf = res.confidence_report?.overall_confidence || 0;
        const badge = conf > 0.7 ? 'badge-success' : conf > 0.5 ? 'badge-warning' : 'badge-danger';
        const img = res.detection_result_url || res.original_url;
        return `
            <div class="result-card" onclick="showDetailView(${i})">
                <div class="result-card-image">
                    <img src="${API_BASE}${img}" alt="${res.filename}">
                    <div class="result-card-overlay"><span class="badge ${badge}">${Math.round(conf*100)}%</span></div>
                </div>
                <div class="result-card-info">
                    <div class="result-card-title">${res.filename}</div>
                    <div class="result-card-stats">
                        <span><i class="fas fa-layer-group"></i> ${res.summary?.total_classes_detected||0} classes</span>
                        <span><i class="fas fa-chart-line"></i> ${(res.summary?.total_detected_lines||0).toLocaleString()} px</span>
                    </div>
                </div>
            </div>`;
    }).join('');
}

// ── Detail View ─────────────────────────────────────────────────────
function showDetailView(index) {
    const res = currentResults[index];
    if (!res) return;

    resultsGrid.style.display = 'none';
    detailView.style.display = 'flex';

    document.getElementById('detailFilename').textContent = res.filename;
    const qb = document.getElementById('qualityBadge');
    const q = res.summary?.quality_badge || 'moderate';
    qb.textContent = q.charAt(0).toUpperCase() + q.slice(1);
    qb.className = 'quality-badge ' + q;

    const mainImg = document.getElementById('mainImage');
    const compImg = document.getElementById('comparisonImage');
    mainImg.src = res.detection_result_url ? `${API_BASE}${res.detection_result_url}` : `${API_BASE}${res.original_url}`;
    compImg.src = `${API_BASE}${res.original_url}`;

    mainImg.dataset.overlayUrl  = res.detection_result_url ? `${API_BASE}${res.detection_result_url}` : '';
    mainImg.dataset.originalUrl = `${API_BASE}${res.original_url}`;

    // per-class gallery
    renderPerClassGallery(res.per_class_overlays || []);

    renderLegend(res.legend);
    renderConfidenceReport(res.confidence_report);
    renderSummary(res.summary);
    renderDownloads(res.downloads, res.per_class_overlays);

    // reset tabs
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelector('.tab-btn[data-view="overlay"]').classList.add('active');
    document.getElementById('opacityControl').style.display = 'none';
    document.getElementById('perClassGallery').style.display = 'none';
    document.getElementById('imageContainer').style.display = 'flex';
    compImg.style.display = 'none';
    zoomLevel = 1;
    updateZoom();
}

function hideDetailView() {
    if (currentResults && currentResults.length > 1) {
        detailView.style.display = 'none';
        resultsGrid.style.display = 'grid';
    }
}

function renderPerClassGallery(overlays) {
    const g = document.getElementById('perClassGallery');
    if (!overlays || overlays.length === 0) { g.innerHTML = '<p class="empty-msg">No per-class overlays</p>'; return; }
    g.innerHTML = overlays.map(o => `
        <div class="per-class-card">
            <img src="${API_BASE}${o.url}" alt="${o.class}" loading="lazy">
            <span class="per-class-label">${formatClassName(o.class)}</span>
        </div>
    `).join('');
}

// ── Legend / Confidence / Summary / Downloads ───────────────────────
function renderLegend(legend) {
    const c = document.getElementById('legendContent');
    if (!legend || legend.length === 0) { c.innerHTML = '<p class="empty-msg">No lines detected</p>'; return; }
    c.innerHTML = legend.map(it => `
        <div class="legend-item">
            <div class="legend-color" style="background:${it.color}"></div>
            <div class="legend-info">
                <span class="legend-name">${formatClassName(it.class)}</span>
                <div class="legend-stats">
                    <span>${it.count.toLocaleString()} px</span>
                    <span>${Math.round(it.avg_confidence*100)}%</span>
                </div>
            </div>
        </div>
    `).join('');
}

function renderConfidenceReport(report) {
    const c = document.getElementById('confidenceContent');
    if (!report || !report.per_class) { c.innerHTML = '<p class="empty-msg">No confidence data</p>'; return; }
    const ov = report.overall_confidence || 0;
    const sc = ov > 0.7 ? 'var(--success)' : ov > 0.5 ? 'var(--warning)' : 'var(--danger)';
    let html = `<div class="confidence-overall"><div class="confidence-score" style="color:${sc}">${Math.round(ov*100)}%</div><div style="color:var(--text-secondary)">Overall Confidence</div></div>`;
    for (const [cn, cv] of Object.entries(report.per_class)) {
        if (cv === 0) continue;
        const bc = cv > 0.7 ? 'var(--success)' : cv > 0.5 ? 'var(--warning)' : 'var(--danger)';
        html += `<div class="confidence-bar"><span class="confidence-bar-label">${formatClassName(cn)}</span><div class="confidence-bar-track"><div class="confidence-bar-fill" style="width:${cv*100}%;background:${bc}"></div></div><span class="confidence-bar-value">${Math.round(cv*100)}%</span></div>`;
    }
    if (report.low_confidence_classes?.length) {
        html += `<div class="low-conf-warning"><i class="fas fa-exclamation-triangle"></i> Low confidence: ${report.low_confidence_classes.map(formatClassName).join(', ')}</div>`;
    }
    c.innerHTML = html;
}

function renderSummary(s) {
    const c = document.getElementById('summaryContent');
    if (!s) { c.innerHTML = '<p class="empty-msg">No summary</p>'; return; }
    c.innerHTML = `
        <div class="summary-item"><span class="summary-label">Total Pixels</span><span class="summary-value">${(s.total_detected_lines||0).toLocaleString()}</span></div>
        <div class="summary-item"><span class="summary-label">Classes Detected</span><span class="summary-value">${s.total_classes_detected||0}</span></div>
        <div class="summary-item"><span class="summary-label">Strongest</span><span class="summary-value">${formatClassName(s.strongest_class)||'-'}</span></div>
        <div class="summary-item"><span class="summary-label">Weakest</span><span class="summary-value">${formatClassName(s.weakest_class)||'-'}</span></div>
    `;
}

function renderDownloads(dl, perClass) {
    const c = document.getElementById('downloadContent');
    if (!dl) { c.innerHTML = '<p class="empty-msg">No downloads</p>'; return; }
    const items = [];
    if (dl.overlay_png) items.push({url:dl.overlay_png, label:'Detection Overlay', icon:'fa-image'});
    if (dl.mask_png) items.push({url:dl.mask_png, label:'Segmentation Mask', icon:'fa-layer-group'});
    if (dl.confidence_json) items.push({url:dl.confidence_json, label:'Report (JSON)', icon:'fa-file-code'});
    if (dl.all_zip) items.push({url:dl.all_zip, label:'All Outputs (ZIP)', icon:'fa-file-archive'});
    c.innerHTML = items.map(it => `<a href="${API_BASE}${it.url}" class="download-btn" download><i class="fas ${it.icon}"></i><span>${it.label}</span></a>`).join('');
}

// ── Helpers ─────────────────────────────────────────────────────────
function formatClassName(n) {
    if (!n) return '';
    return n.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

// ── View Tabs ───────────────────────────────────────────────────────
function setupViewTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const v = btn.dataset.view;
            const mainImg = document.getElementById('mainImage');
            const compImg = document.getElementById('comparisonImage');
            const opCtrl  = document.getElementById('opacityControl');
            const pcGal   = document.getElementById('perClassGallery');
            const imgCont = document.getElementById('imageContainer');

            // Reset zoom when switching views
            zoomLevel = 1;

            pcGal.style.display = 'none';
            imgCont.style.display = 'flex';
            compImg.style.display = 'none';
            opCtrl.style.display = 'none';
            imgCont.classList.remove('compare-mode');
            mainImg.style.maxWidth = '';
            compImg.style.maxWidth = '';

            switch(v) {
                case 'overlay':
                    mainImg.src = mainImg.dataset.overlayUrl || mainImg.dataset.originalUrl;
                    break;
                case 'original':
                    mainImg.src = mainImg.dataset.originalUrl;
                    break;
                case 'per_class':
                    imgCont.style.display = 'none';
                    pcGal.style.display = 'grid';
                    break;
                case 'comparison':
                    imgCont.classList.add('compare-mode');
                    mainImg.src = mainImg.dataset.overlayUrl || mainImg.dataset.originalUrl;
                    mainImg.style.maxWidth = '50%';
                    compImg.src = mainImg.dataset.originalUrl;
                    compImg.style.maxWidth = '50%';
                    compImg.style.display = 'block';
                    opCtrl.style.display = 'flex';
                    updateOpacity();
                    break;
            }
            updateZoom();
        });
    });
}

// ── Opacity / Zoom ──────────────────────────────────────────────────
function setupOpacitySlider() {
    document.getElementById('opacitySlider').addEventListener('input', updateOpacity);
}
function updateOpacity() {
    const v = document.getElementById('opacitySlider').value;
    document.getElementById('opacityValue').textContent = v + '%';
    document.getElementById('mainImage').style.opacity = v / 100;
}
function zoomIn()   { zoomLevel = Math.min(zoomLevel * 1.25, 4); updateZoom(); }
function zoomOut()  { zoomLevel = Math.max(zoomLevel / 1.25, 0.25); updateZoom(); }
function resetZoom(){ zoomLevel = 1; updateZoom(); }
function updateZoom() {
    const img = document.getElementById('mainImage');
    const compImg = document.getElementById('comparisonImage');
    const container = document.getElementById('imageContainer');
    const isCompare = container.classList.contains('compare-mode');

    img.style.transform = `scale(${zoomLevel})`;
    compImg.style.transform = `scale(${zoomLevel})`;

    if (zoomLevel > 1) {
        container.style.justifyContent = 'flex-start';
        container.style.alignItems = 'flex-start';
        container.style.padding = '5px';
    } else {
        container.style.justifyContent = 'center';
        container.style.alignItems = 'center';
        container.style.padding = '0';
    }
}

// ── History ─────────────────────────────────────────────────────────
async function showHistory() {
    historyModal.classList.add('show');
    try {
        const r = await fetch(`${API_BASE}/api/history`);
        const h = await r.json();
        const c = document.getElementById('historyContent');
        if (h.length === 0) { c.innerHTML = '<p class="empty-msg">No history</p>'; return; }
        c.innerHTML = h.map(it => `
            <div class="history-item" onclick="loadHistoryJob('${it.job_id}')">
                <div class="history-icon"><i class="fas fa-search"></i></div>
                <div class="history-info">
                    <div class="history-title">${it.file_count} file(s)</div>
                    <div class="history-meta">${formatDate(it.created_at)}</div>
                </div>
                <span class="history-status ${it.status}">${it.status}</span>
            </div>
        `).join('');
    } catch (e) { console.error(e); }
}
function hideHistory() { historyModal.classList.remove('show'); }

async function loadHistoryJob(jobId) {
    hideHistory();
    currentJobId = jobId;
    try {
        const r = await fetch(`${API_BASE}/api/result/${jobId}`);
        const d = await r.json();
        if (d.status === 'done' && d.results) {
            currentResults = d.results;
            emptyState.style.display = 'none';
            resultsContainer.style.display = 'block';
            if (currentResults.length === 1) showDetailView(0);
            else renderResultsGrid();
        }
    } catch (e) { console.error(e); alert('Error loading job'); }
}

function formatDate(s) { if (!s) return ''; return new Date(s).toLocaleString(); }

historyModal.addEventListener('click', e => { if (e.target === historyModal) hideHistory(); });
