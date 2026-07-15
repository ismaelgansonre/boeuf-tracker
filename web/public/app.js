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
    btnTheme.textContent = t === 'dark' ? '☀️' : '🌙';
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
