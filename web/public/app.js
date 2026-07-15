// ═══════════════════════════════════════════════════════════════
// Boeuf Tracker — Logique UI (Bun/Hono frontend)
// Migré depuis static/app.js, adapté pour la nouvelle UI + races + thème
// ═══════════════════════════════════════════════════════════════
const $ = (id) => document.getElementById(id);
let lastSource = '';
let switchingDevice = false;
let pushInFlight = false;

// ─── Thème dark/light ───────────────────────────────────────────
const btnTheme = $('btn-theme');
function updateThemeIcon() {
    const t = document.documentElement.getAttribute('data-theme');
    btnTheme.textContent = t === 'dark' ? 'LIGHT' : 'DARK';
}
updateThemeIcon();
btnTheme.addEventListener('click', () => {
    const cur = document.documentElement.getAttribute('data-theme');
    const next = cur === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
    updateThemeIcon();
});

// ─── Helpers ────────────────────────────────────────────────────
function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

const BEHAVIOR_BADGES = {
    'pâture': 'badge-grazing',
    'boit': 'badge-drinking',
    'couché': 'badge-lying',
    'court': 'badge-running',
    'rué': 'badge-running',
    'immobile': 'badge-neutral',
    'marche': 'badge-neutral',
};

// ─── Polling stats (1s) ─────────────────────────────────────────
async function refreshStats() {
    try {
        const res = await fetch('/api/stats');
        const data = await res.json();

        const sourceLabel = data.source_label || data.source || '--';
        $('meta-source').textContent = 'source: ' + sourceLabel;
        $('meta-device').textContent = 'device: ' + (data.device || '--');
        const fps = data.fps || 0;
        const fpsEl = $('meta-fps');
        fpsEl.textContent = 'fps: ' + fps.toFixed(1);
        fpsEl.style.color = fps >= 24 ? 'var(--accent)' : fps >= 15 ? 'var(--amber)' : 'var(--rose)';
        $('meta-frames').textContent = 'frames: ' + (data.frame_count || 0);
        $('current-source').textContent = '→ ' + sourceLabel;

        // Changements en attente
        const desired = data.desired || {};
        const pending = [];
        if (desired.imgsz != null) pending.push(`imgsz=${desired.imgsz}`);
        if (desired.embed_every != null) pending.push(`embed=${desired.embed_every}`);
        if (desired.threshold != null) pending.push(`th=${desired.threshold}`);
        if (desired.conf != null) pending.push(`conf=${desired.conf}`);
        if (desired.yolo_model != null) pending.push(`model=${desired.yolo_model}`);
        const pendingEl = $('meta-pending');
        if (pending.length > 0) {
            pendingEl.textContent = '⏳ ' + pending.join(', ');
            pendingEl.style.display = '';
        } else {
            pendingEl.style.display = 'none';
        }

        // Sync controls avec valeurs courantes
        if (data.current) {
            if (data.current.imgsz != null) {
                document.querySelectorAll('[data-imgsz]').forEach(btn => {
                    btn.classList.toggle('active', parseInt(btn.dataset.imgsz) === data.current.imgsz);
                });
            }
            if (data.current.threshold != null && !$('setting-threshold').matches(':active')) {
                $('setting-threshold').value = data.current.threshold;
                $('setting-threshold-val').textContent = data.current.threshold.toFixed(2);
            }
            if (data.current.conf != null && !$('setting-conf').matches(':active')) {
                $('setting-conf').value = data.current.conf;
                $('setting-conf-val').textContent = data.current.conf.toFixed(2);
            }
            if (data.current.embed_every != null && !$('setting-embed').matches(':active')) {
                $('setting-embed').value = data.current.embed_every;
                $('setting-embed-val').textContent = data.current.embed_every;
            }
            if (data.current.yolo_model && $('setting-model').value !== data.current.yolo_model) {
                $('setting-model').value = data.current.yolo_model;
            }
        }

        if (sourceLabel !== lastSource) { refreshStream(); lastSource = sourceLabel; }

        if (!switchingDevice && data.device && $('device-select').value !== data.device) {
            const opts = Array.from($('device-select').options).map(o => o.value);
            if (opts.includes(data.device)) $('device-select').value = data.device;
        }

        // ── Animaux (avec race) ──
        const active = data.active || [];
        $('active-count').textContent = active.length;
        $('animal-list').innerHTML = active.length === 0
            ? '<li class="list-empty">Aucun animal visible</li>'
            : active.map(a => {
                const breed = a.breed || '';
                const breedConf = a.breed_confidence ? ` (${Math.round(a.breed_confidence * 100)}%)` : '';
                return `<li class="animal-item">
                    <span class="animal-name">
                        <span>${escapeHtml(a.name)}</span>
                        ${breed ? `<span class="animal-breed">${escapeHtml(breed)}${breedConf}</span>` : ''}
                    </span>
                    <span class="animal-conf">${(a.conf * 100).toFixed(0)}%</span>
                </li>`;
            }).join('');

        // ── Événements ──
        const ev = data.events || [];
        $('event-list').innerHTML = ev.length === 0
            ? '<li class="list-empty">Aucun événement</li>'
            : ev.map(e => {
                const cls = e.startsWith('NEW') ? 'event-new'
                          : e.startsWith('MATCH') ? 'event-match'
                          : e.startsWith('CRASH') ? 'event-crash'
                          : '';
                return `<li class="event-item ${cls}">${escapeHtml(e)}</li>`;
            }).join('');

        // ── Activités ──
        const bh = data.behavior || [];
        $('behavior-list').innerHTML = bh.length === 0
            ? '<li class="list-empty">Aucune activité</li>'
            : bh.map(b => {
                const badgeCls = BEHAVIOR_BADGES[b.action] || 'badge-neutral';
                return `<li class="animal-item">
                    <span>${escapeHtml(b.name)}</span>
                    <span class="badge ${badgeCls}">${escapeHtml(b.action)}</span>
                    <span class="animal-conf">${b.speed}px/s</span>
                </li>`;
            }).join('');
    } catch (e) {}
}

// ─── Liste vidéos ───────────────────────────────────────────────
async function loadVideoList() {
    try {
        const res = await fetch('/api/videos');
        const data = await res.json();
        const sel = $('video-select');
        const current = sel.value;
        sel.innerHTML = '<option value="">— Choisir une vidéo —</option>';
        (data.videos || []).forEach(v => {
            const opt = document.createElement('option');
            opt.value = v.path;
            const tag = v.source === 'uploads' ? '[uploads]' : '[projet]';
            opt.textContent = `${tag} ${v.name} (${v.size_mb} MB)`;
            sel.appendChild(opt);
        });
        if (current) sel.value = current;
    } catch (e) {}
}
loadVideoList();

$('video-select').addEventListener('change', async (e) => {
    const path = e.target.value;
    if (!path) return;
    const status = $('upload-status');
    status.textContent = 'Chargement...';
    status.style.color = 'var(--amber)';
    try {
        await fetch('/api/source/file', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ path })
        });
        status.textContent = '';
    } catch (err) {
        status.textContent = 'Erreur: ' + err.message;
        status.style.color = 'var(--rose)';
    }
});

// ─── Upload ─────────────────────────────────────────────────────
$('btn-upload').addEventListener('click', () => $('video-input').click());
$('video-input').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const status = $('upload-status');
    status.textContent = `Envoi ${file.name}...`;
    status.style.color = 'var(--amber)';
    const fd = new FormData();
    fd.append('video', file);
    try {
        const res = await fetch('/api/upload-video', { method: 'POST', body: fd });
        const data = await res.json();
        if (data.ok) { status.textContent = `Chargé: ${data.filename}`; status.style.color = 'var(--accent)'; }
        else { status.textContent = `Erreur: ${data.error}`; status.style.color = 'var(--rose)'; }
    } catch (err) {
        status.textContent = `Réseau: ${err.message}`; status.style.color = 'var(--rose)';
    }
    e.target.value = '';
});

// ─── Webcam ─────────────────────────────────────────────────────
$('btn-webcam').addEventListener('click', async () => {
    const status = $('upload-status');
    status.textContent = 'Bascule webcam...';
    status.style.color = 'var(--amber)';
    try {
        await fetch('/api/source/webcam', { method: 'POST', headers: {'Content-Type':'application/json'}, body: '{}' });
        status.textContent = '';
    } catch (err) {
        status.textContent = `Erreur: ${err.message}`; status.style.color = 'var(--rose)';
    }
});

// ─── Devices ────────────────────────────────────────────────────
async function loadDevices() {
    try {
        const res = await fetch('/api/devices');
        const data = await res.json();
        const sel = $('device-select');
        sel.innerHTML = '<option value="auto">Auto</option>';
        (data.available || []).forEach(d => {
            const opt = document.createElement('option');
            opt.value = d;
            opt.textContent = d === 'cpu' ? 'CPU' : d.startsWith('cuda:') ? `GPU ${d.split(':')[1]}` : d;
            sel.appendChild(opt);
        });
        sel.value = data.current;
    } catch (e) {}
}
loadDevices();

$('device-select').addEventListener('change', async () => {
    if (switchingDevice) return;
    switchingDevice = true;
    const target = $('device-select').value;
    const status = $('device-status');
    status.textContent = 'bascule...';
    status.style.color = 'var(--amber)';
    try {
        const res = await fetch('/api/device', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ device: target }) });
        const data = await res.json();
        if (data.ok) { status.textContent = `→ ${data.device}`; status.style.color = 'var(--accent)'; await loadDevices(); }
        else { status.textContent = data.error || 'erreur'; status.style.color = 'var(--rose)'; }
    } catch (err) { status.textContent = `réseau: ${err.message}`; status.style.color = 'var(--rose)'; }
    setTimeout(() => { switchingDevice = false; }, 500);
});

// ─── Reset DB modal ─────────────────────────────────────────────
$('btn-reset-db').addEventListener('click', () => $('modal-reset').showModal());
$('btn-reset-cancel').addEventListener('click', () => $('modal-reset').close());
$('btn-reset-confirm').addEventListener('click', async (e) => {
    e.preventDefault();
    try {
        await fetch('/api/db/reset', { method: 'POST', headers: {'Content-Type':'application/json'}, body: '{}' });
        $('modal-reset').close();
        const status = $('upload-status');
        status.textContent = 'Base purgée'; status.style.color = 'var(--accent)';
    } catch (err) { alert('Erreur: ' + err.message); }
});

// ─── Settings live ──────────────────────────────────────────────
async function loadSettings() {
    try {
        const res = await fetch('/api/settings');
        const data = await res.json();
        const cur = data.current || {};
        const sel = $('setting-model');
        sel.innerHTML = '';
        (data.models_available || []).forEach(m => {
            const opt = document.createElement('option');
            opt.value = m; opt.textContent = m; sel.appendChild(opt);
        });
        if (cur.yolo_model) sel.value = cur.yolo_model;
        $('setting-embed').value = cur.embed_every || 10;
        $('setting-embed-val').textContent = cur.embed_every || 10;
        $('setting-threshold').value = cur.threshold || 0.65;
        $('setting-threshold-val').textContent = (cur.threshold || 0.65).toFixed(2);
        $('setting-conf').value = cur.conf || 0.40;
        $('setting-conf-val').textContent = (cur.conf || 0.40).toFixed(2);
        document.querySelectorAll('[data-imgsz]').forEach(btn => {
            btn.classList.toggle('active', parseInt(btn.dataset.imgsz) === cur.imgsz);
        });
    } catch (e) {}
}
loadSettings();

async function pushSetting(payload) {
    if (pushInFlight) return;
    pushInFlight = true;
    const status = $('settings-status');
    status.textContent = '⏳ envoi...';
    status.style.color = 'var(--amber)';
    try {
        await fetch('/api/settings', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) });
        status.textContent = '✓ appliqué'; status.style.color = 'var(--accent)';
        setTimeout(() => { status.textContent = ''; }, 2500);
    } catch (e) { status.textContent = '✗ erreur'; status.style.color = 'var(--rose)'; }
    finally { pushInFlight = false; }
}

$('setting-model').addEventListener('change', (e) => pushSetting({ yolo_model: e.target.value }));
document.querySelectorAll('[data-imgsz]').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('[data-imgsz]').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        pushSetting({ imgsz: parseInt(btn.dataset.imgsz) });
    });
});
$('setting-embed').addEventListener('input', (e) => { $('setting-embed-val').textContent = e.target.value; });
$('setting-embed').addEventListener('change', (e) => pushSetting({ embed_every: parseInt(e.target.value) }));
$('setting-threshold').addEventListener('input', (e) => { $('setting-threshold-val').textContent = parseFloat(e.target.value).toFixed(2); });
$('setting-threshold').addEventListener('change', (e) => pushSetting({ threshold: parseFloat(e.target.value) }));
$('setting-conf').addEventListener('input', (e) => { $('setting-conf-val').textContent = parseFloat(e.target.value).toFixed(2); });
$('setting-conf').addEventListener('change', (e) => pushSetting({ conf: parseFloat(e.target.value) }));
$('btn-rematch').addEventListener('click', async () => {
    const status = $('settings-status');
    status.textContent = '⏳ re-id...'; status.style.color = 'var(--amber)';
    try {
        await fetch('/api/rematch', { method: 'POST' });
        status.textContent = '✓ re-id demandée'; status.style.color = 'var(--accent)';
        setTimeout(() => { status.textContent = ''; }, 2500);
    } catch (e) { status.textContent = '✗ erreur'; status.style.color = 'var(--rose)'; }
});

// ─── Init polling ───────────────────────────────────────────────
setInterval(refreshStats, 1000);
refreshStats();

// ─── Polling JPEG du flux vidéo (~25 fps) ───────────────────────
const streamImg = $('stream');
let streamTimer = null;
function refreshStream() { streamImg.src = '/video_feed?t=' + Date.now(); }
streamImg.addEventListener('load', () => { clearTimeout(streamTimer); streamTimer = setTimeout(refreshStream, 40); });
streamImg.addEventListener('error', () => { clearTimeout(streamTimer); streamTimer = setTimeout(refreshStream, 500); });
refreshStream();

// ═══════════════════════════════════════════════════════════════
//  DASHBOARD — Charts (Chart.js) + Heatmap (Canvas) + Timeline
// ═══════════════════════════════════════════════════════════════

// ── Toggle panneau ──
const dashPanel = $('dashboard-panel');
$('btn-dashboard').addEventListener('click', () => {
    dashPanel.classList.add('open');
    refreshDashboard();
});
$('btn-dashboard-close').addEventListener('click', () => dashPanel.classList.remove('open'));

// ── Config couleurs par robe-type (mêmes swatches que Python) ──
const COAT_COLORS = {
    "Noir uni":         "#1a1a1a",
    "Blanc uni":        "#f0ebe0",
    "Pie noir":         "#2a2a2a",
    "Pie fauve":        "#c8a06a",
    "Pie rouge":        "#b85c3a",
    "Fauve uni":        "#c89858",
    "Rouge / acajou":   "#9e3d22",
    "Gris":             "#8a8a8a",
    "Bringe":           "#6b4a2a",
    "Indeterminee":     "#555555",
};

// Chart.js defaults
Chart.defaults.color = getComputedStyle(document.documentElement)
    .getPropertyValue('--text-muted').trim();
Chart.defaults.borderColor = getComputedStyle(document.documentElement)
    .getPropertyValue('--border').trim();

// ── Chart FPS ──
const fpsCtx = $('chart-fps').getContext('2d');
const fpsChart = new Chart(fpsCtx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: 'FPS',
            data: [],
            borderColor: '#16a34a',
            backgroundColor: 'rgba(22, 163, 74, 0.1)',
            fill: true,
            tension: 0.3,
            pointRadius: 0,
            borderWidth: 2,
        }]
    },
    options: {
        responsive: true, maintainAspectRatio: false,
        scales: {
            y: { beginAtZero: true, max: 40 },
        },
        plugins: { legend: { display: false } },
        animation: false,
    }
});

// ── Chart Races (doughnut) ──
const racesCtx = $('chart-races').getContext('2d');
const racesChart = new Chart(racesCtx, {
    type: 'doughnut',
    data: {
        labels: [],
        datasets: [{
            data: [],
            backgroundColor: [],
            borderWidth: 0,
        }]
    },
    options: {
        responsive: true, maintainAspectRatio: false,
        plugins: {
            legend: { display: false },
            tooltip: { callbacks: {
                label: (ctx) => `${ctx.label}: ${ctx.raw} (${ctx.dataset.data[ctx.dataIndex]}%)`
            }}
        },
        animation: false,
    }
});

// ── Chart Activities (bar) ──
const actCtx = $('chart-activities').getContext('2d');
const actChart = new Chart(actCtx, {
    type: 'bar',
    data: {
        labels: [],
        datasets: [{
            label: 'Occurrences',
            data: [],
            backgroundColor: '#3b82f6',
            borderRadius: 4,
        }]
    },
    options: {
        responsive: true, maintainAspectRatio: false,
        indexAxis: 'y',
        plugins: { legend: { display: false } },
        scales: { x: { beginAtZero: true } },
        animation: false,
    }
});

// ── Heatmap en Canvas natif ──
const heatCanvas = $('heatmap');
const heatCtx = heatCanvas.getContext('2d');
function drawHeatmap(heatmapData) {
    const size = heatmapData.grid_size;
    const cells = heatmapData.global || [];
    const total = heatmapData.total_samples || 1;
    const cellPx = heatCanvas.width / size;

    // Resize haute résolution pour écrans retina
    const dpr = window.devicePixelRatio || 1;
    heatCanvas.width = 400 * dpr;
    heatCanvas.height = 400 * dpr;
    heatCanvas.style.width = '400px';
    heatCanvas.style.height = '400px';
    heatCtx.scale(dpr, dpr);

    // Fond
    heatCtx.fillStyle = '#0f1410';
    heatCtx.fillRect(0, 0, heatCanvas.width / dpr, heatCanvas.height / dpr);

    // Cellules
    let maxCount = 0;
    cells.forEach(c => { if (c.count > maxCount) maxCount = c.count; });
    if (maxCount === 0) maxCount = 1;

    cells.forEach(c => {
        const intensity = c.count / maxCount;
        // Vert → Jaune → Rouge
        let r, g, b;
        if (intensity < 0.5) {
            // vert (0.4) → ambre (0.6)
            const t = intensity * 2;
            r = Math.round(22 + (217 - 22) * t);
            g = Math.round(163 + (119 - 163) * t);
            b = Math.round(74 + (6 - 74) * t);
        } else {
            // ambre (0.5) → rouge (1.0)
            const t = (intensity - 0.5) * 2;
            r = Math.round(217 + (239 - 217) * t);
            g = Math.round(119 + (68 - 119) * t);
            b = Math.round(6 + (68 - 6) * t);
        }
        heatCtx.fillStyle = `rgba(${r}, ${g}, ${b}, ${0.3 + intensity * 0.7})`;
        heatCtx.fillRect(c.x * cellPx, c.y * cellPx, cellPx, cellPx);
    });

    // Grille subtile
    heatCtx.strokeStyle = 'rgba(255,255,255,0.05)';
    heatCtx.lineWidth = 1;
    for (let i = 0; i <= size; i++) {
        heatCtx.beginPath();
        heatCtx.moveTo(i * cellPx, 0);
        heatCtx.lineTo(i * cellPx, 400);
        heatCtx.stroke();
        heatCtx.beginPath();
        heatCtx.moveTo(0, i * cellPx);
        heatCtx.lineTo(400, i * cellPx);
        heatCtx.stroke();
    }
}

// ── Timeline avec filtre ──
let currentTimelineFilter = 'ALL';
function fmtTime(t) {
    const m = Math.floor(t / 60);
    const s = Math.floor(t % 60);
    return `${m}:${String(s).padStart(2, '0')}`;
}
function renderTimeline(events) {
    const list = $('timeline-list');
    const filtered = currentTimelineFilter === 'ALL'
        ? events
        : events.filter(e => e.type === currentTimelineFilter);
    list.innerHTML = filtered.slice(-50).reverse().map(e => `
        <li class="timeline-item t-${e.type.toLowerCase()}">
            <span class="timeline-time">+${fmtTime(e.t)}</span>
            <span class="timeline-type">${e.type}</span>
            <span class="t-msg">${escapeHtml(e.msg)}</span>
        </li>
    `).join('') || '<li class="list-empty">Aucun evenement</li>';
}

document.querySelectorAll('.timeline-filter').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.timeline-filter').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentTimelineFilter = btn.dataset.type;
        // refresh immediat
        refreshDashboard();
    });
});

// ── Refresh dashboard (3s, plus lent que les stats) ──
async function refreshDashboard() {
    try {
        const [dResp, hResp] = await Promise.all([
            fetch('/api/dashboard'),
            fetch('/api/heatmap'),
        ]);
        const d = await dResp.json();
        const h = await hResp.json();

        // KPIs
        $('kpi-unique').textContent = d.unique_animals ?? 0;
        $('kpi-detections').textContent = d.detection_count ?? 0;
        const recentFps = d.fps_history.slice(-10).map(p => p.fps);
        const avgFps = recentFps.length
            ? (recentFps.reduce((a, b) => a + b, 0) / recentFps.length).toFixed(1)
            : '-';
        $('kpi-fps').textContent = avgFps;
        const up = d.uptime ?? 0;
        $('kpi-uptime').textContent = up > 3600
            ? `${Math.floor(up/3600)}h${Math.floor(up%3600/60)}m`
            : `${Math.floor(up/60)}m${Math.floor(up%60)}s`;

        // Chart FPS
        fpsChart.data.labels = d.fps_history.map(p => fmtTime(p.t));
        fpsChart.data.datasets[0].data = d.fps_history.map(p => p.fps);
        fpsChart.update('none');

        // Chart races (avec couleurs)
        const races = d.race_counts;
        racesChart.data.labels = Object.keys(races);
        racesChart.data.datasets[0].data = Object.values(races);
        racesChart.data.datasets[0].backgroundColor =
            Object.keys(races).map(r => COAT_COLORS[r] || '#888');
        racesChart.update('none');
        // Legend custom
        const total = Object.values(races).reduce((a, b) => a + b, 0) || 1;
        $('races-legend').innerHTML = Object.entries(races).map(([r, c]) => `
            <span class="legend-item">
                <span class="legend-swatch" style="background:${COAT_COLORS[r] || '#888'}"></span>
                ${escapeHtml(r)} (${Math.round(c/total*100)}%)
            </span>
        `).join('');

        // Chart activities
        const acts = d.activity_counts;
        const actOrder = Object.entries(acts).sort((a, b) => b[1] - a[1]);
        actChart.data.labels = actOrder.map(a => a[0]);
        actChart.data.datasets[0].data = actOrder.map(a => a[1]);
        actChart.update('none');

        // Heatmap
        drawHeatmap(h);

        // Timeline
        renderTimeline(d.timeline_events || []);
    } catch (e) {
        console.error('dashboard refresh error:', e);
    }
}
setInterval(() => { if (dashPanel.classList.contains('open')) refreshDashboard(); }, 3000);
