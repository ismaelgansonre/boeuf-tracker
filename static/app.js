// app.js — Frontend pour Boeuf Tracker (DaisyUI)
const $ = (id) => document.getElementById(id);
let lastSource = '';
let switchingDevice = false;
let pushInFlight = false;

// === Refresh stats ===
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
        fpsEl.className = fps >= 20 ? 'text-success' : fps >= 10 ? 'text-warning' : 'text-error';
        $('meta-frames').textContent = 'frames: ' + (data.frame_count || 0);
        $('current-source').textContent = '→ ' + sourceLabel;

        // Indicateur de changements en attente
        const desired = data.desired || {};
        const pending = [];
        if (desired.imgsz != null) pending.push(`imgsz=${desired.imgsz}`);
        if (desired.embed_every != null) pending.push(`embed=${desired.embed_every}`);
        if (desired.threshold != null) pending.push(`th=${desired.threshold}`);
        if (desired.conf != null) pending.push(`conf=${desired.conf}`);
        if (desired.yolo_model != null) pending.push(`model=${desired.yolo_model}`);
        const pendingEl = $('meta-pending');
        if (pendingEl) {
            if (pending.length > 0) {
                pendingEl.textContent = '⏳ ' + pending.join(', ');
                pendingEl.className = 'text-warning';
                pendingEl.style.display = '';
            } else {
                pendingEl.style.display = 'none';
            }
        }

        // Boutons imgsz: marquer le bouton actif selon la valeur COURANTE (déjà appliquée)
        if (data.current && data.current.imgsz != null) {
            document.querySelectorAll('[data-imgsz]').forEach(btn => {
                const v = parseInt(btn.dataset.imgsz);
                if (data.current.imgsz === v) btn.classList.add('btn-active');
                else btn.classList.remove('btn-active');
            });
        }
        // Sliders: refléter la valeur COURANTE si l'utilisateur n'est pas en train d'éditer
        if (data.current) {
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

        // Refresh immédiat si source a changé (sinon le polling continue)
        if (sourceLabel !== lastSource) {
            refreshStream();
            lastSource = sourceLabel;
        }

        // Sync device select
        if (!switchingDevice && data.device && $('device-select').value !== data.device) {
            const opts = Array.from($('device-select').options).map(o => o.value);
            if (opts.includes(data.device)) $('device-select').value = data.device;
        }

        // Animaux
        const active = data.active || [];
        $('active-count').textContent = active.length;
        $('animal-list').innerHTML = active.length === 0
            ? '<li class="text-center text-base-content/40 italic py-2">Aucun animal visible</li>'
            : active.map(a => `
                <li class="flex justify-between items-center bg-base-300 px-3 py-2 rounded font-mono text-sm">
                    <span>${escapeHtml(a.name)}</span>
                    <span class="text-base-content/60">${(a.conf * 100).toFixed(0)}%</span>
                </li>`).join('');

        // Événements
        const ev = data.events || [];
        $('event-list').innerHTML = ev.length === 0
            ? '<li class="text-center text-base-content/40 italic py-2">Aucun événement</li>'
            : ev.map(e => {
                const cls = e.startsWith('NEW') ? 'text-warning border-l-2 border-warning'
                          : e.startsWith('MATCH') ? 'text-success border-l-2 border-success'
                          : 'text-base-content/60 border-l-2 border-base-content/20';
                return `<li class="px-3 py-1.5 ${cls}">${escapeHtml(e)}</li>`;
            }).join('');

        // Activités
        const bh = data.behavior || [];
        $('behavior-list').innerHTML = bh.length === 0
            ? '<li class="text-center text-base-content/40 italic py-2">Aucune activité</li>'
            : bh.map(b => {
                const colors = {
                    'immobile': 'badge-ghost',
                    'pâture': 'badge-success',
                    'boit': 'badge-info',
                    'couché': 'badge-warning',
                    'marche': 'badge-primary',
                    'court': 'badge-secondary',
                    'rué': 'badge-error',
                };
                return `<li class="grid grid-cols-[1fr_auto_auto] gap-2 items-center bg-base-300 px-3 py-2 rounded font-mono text-xs">
                    <span>${escapeHtml(b.name)}</span>
                    <span class="badge ${colors[b.action] || 'badge-ghost'} badge-sm">${escapeHtml(b.action)}</span>
                    <span class="text-base-content/60">${b.speed}px/s</span>
                </li>`;
            }).join('');
    } catch (e) {}
}

function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

// === Liste des vidéos du projet ===
async function loadVideoList() {
    try {
        const res = await fetch('/api/videos');
        const data = await res.json();
        const sel = $('video-select');
        const current = sel.value;
        sel.innerHTML = '<option value="">— Choisir une vidéo du projet —</option>';
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
    status.className = 'text-xs font-mono text-warning';
    try {
        const res = await fetch('/api/source/file', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ path })
        });
        const data = await res.json();
        if (data.ok) {
            status.textContent = '';
            status.className = 'text-xs font-mono';
        } else {
            status.textContent = 'Erreur: ' + data.error;
            status.className = 'text-xs font-mono text-error';
        }
    } catch (err) {
        status.textContent = 'Erreur: ' + err.message;
        status.className = 'text-xs font-mono text-error';
    }
});

// === Upload vidéo ===
$('btn-upload').addEventListener('click', () => $('video-input').click());

$('video-input').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const status = $('upload-status');
    const sizeMB = (file.size / 1024 / 1024).toFixed(1);
    status.textContent = `Envoi ${file.name} (${sizeMB} MB)...`;
    status.className = 'text-xs font-mono text-warning';

    const fd = new FormData();
    fd.append('video', file);
    try {
        const res = await fetch('/api/upload-video', { method: 'POST', body: fd });
        const data = await res.json();
        if (data.ok) {
            status.textContent = `Chargé: ${data.filename}`;
            status.className = 'text-xs font-mono text-success';
        } else {
            status.textContent = `Erreur: ${data.error}`;
            status.className = 'text-xs font-mono text-error';
        }
    } catch (err) {
        status.textContent = `Réseau: ${err.message}`;
        status.className = 'text-xs font-mono text-error';
    }
    e.target.value = '';
});

// === Webcam ===
$('btn-webcam').addEventListener('click', async () => {
    const status = $('upload-status');
    status.textContent = 'Bascule webcam...';
    status.className = 'text-xs font-mono text-warning';
    try {
        await fetch('/api/source/webcam', { method: 'POST', headers: {'Content-Type':'application/json'}, body: '{}' });
        status.textContent = '';
        status.className = 'text-xs font-mono';
    } catch (err) {
        status.textContent = `Erreur: ${err.message}`;
        status.className = 'text-xs font-mono text-error';
    }
});

// === Devices ===
async function loadDevices() {
    try {
        const res = await fetch('/api/devices');
        const data = await res.json();
        const sel = $('device-select');
        sel.innerHTML = '';
        const optAuto = document.createElement('option');
        optAuto.value = 'auto';
        optAuto.textContent = 'Auto';
        sel.appendChild(optAuto);
        (data.available || []).forEach(d => {
            const opt = document.createElement('option');
            opt.value = d;
            if (d === 'cpu') opt.textContent = 'CPU';
            else if (d.startsWith('cuda:')) {
                const idx = d.split(':')[1];
                const gpu = (data.gpus || []).find(g => String(g.index) === idx);
                opt.textContent = gpu ? `GPU ${idx} (${gpu.name.split(' ').slice(-2).join(' ')})` : `GPU ${idx}`;
            } else opt.textContent = d;
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
    $('device-status').textContent = 'bascule...';
    $('device-status').className = 'text-xs font-mono text-warning';
    try {
        const res = await fetch('/api/device', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ device: target })
        });
        const data = await res.json();
        if (data.ok) {
            $('device-status').textContent = `→ ${data.device}`;
            $('device-status').className = 'text-xs font-mono text-success';
            await loadDevices();
            $('device-select').value = data.device;
        } else {
            $('device-status').textContent = data.error || 'erreur';
            $('device-status').className = 'text-xs font-mono text-error';
        }
    } catch (err) {
        $('device-status').textContent = `réseau: ${err.message}`;
        $('device-status').className = 'text-xs font-mono text-error';
    }
    setTimeout(() => { switchingDevice = false; }, 500);
});

// === Reset DB ===
$('btn-reset-db').addEventListener('click', () => {
    $('modal-reset').showModal();
});

// === Recharger (rafraîchit la frame MJPEG) ===
$('btn-reload').addEventListener('click', () => {
    location.reload();
});

// === Restart serveur (utilise le watcher) ===
$('btn-restart').addEventListener('click', async () => {
    const status = $('upload-status');
    status.textContent = 'Redémarrage...';
    status.className = 'text-xs font-mono text-warning';
    try {
        await fetch('/api/restart', { method: 'POST' });
    } catch (e) {}
    // Attendre 4s puis recharger la page (le watcher aura relancé)
    setTimeout(() => {
        status.textContent = 'Redémarrage en cours...';
        location.reload();
    }, 3500);
});

$('btn-reset-confirm').addEventListener('click', async (e) => {
    e.preventDefault();
    try {
        await fetch('/api/db/reset', { method: 'POST', headers: {'Content-Type':'application/json'}, body: '{}' });
        $('modal-reset').close();
        const status = $('upload-status');
        status.textContent = 'Base purgée';
        status.className = 'text-xs font-mono text-success';
    } catch (err) {
        alert('Erreur: ' + err.message);
    }
});

// === Settings live ===
async function loadSettings() {
    try {
        const res = await fetch('/api/settings');
        const data = await res.json();
        const cur = data.current || {};
        const sel = $('setting-model');
        sel.innerHTML = '';
        (data.models_available || []).forEach(m => {
            const opt = document.createElement('option');
            opt.value = m;
            opt.textContent = m;
            sel.appendChild(opt);
        });
        if (cur.yolo_model) sel.value = cur.yolo_model;
        // Valeurs initiales (seront ensuite tenues à jour par refreshStats)
        $('setting-embed').value = cur.embed_every || 10;
        $('setting-embed-val').textContent = cur.embed_every || 10;
        $('setting-threshold').value = cur.threshold || 0.65;
        $('setting-threshold-val').textContent = (cur.threshold || 0.65).toFixed(2);
        $('setting-conf').value = cur.conf || 0.40;
        $('setting-conf-val').textContent = (cur.conf || 0.40).toFixed(2);
        document.querySelectorAll('[data-imgsz]').forEach(btn => {
            const v = parseInt(btn.dataset.imgsz);
            if (cur.imgsz === v) btn.classList.add('btn-active');
            else btn.classList.remove('btn-active');
        });
    } catch (e) {}
}
loadSettings();

// === Re-match : force la re-id de tous les tracks en cours ===
async function rematch() {
    const status = $('settings-status');
    status.textContent = '⏳ re-id de tous les tracks...';
    status.className = 'text-xs text-warning';
    try {
        const res = await fetch('/api/rematch', { method: 'POST' });
        const d = await res.json();
        if (d.ok) {
            status.textContent = '✓ re-id demandée (prochaines frames)';
            status.className = 'text-xs text-success';
            setTimeout(() => { status.textContent = ''; }, 3000);
        } else {
            status.textContent = '✗ ' + (d.error || 'erreur');
            status.className = 'text-xs text-error';
        }
    } catch (e) {
        status.textContent = '✗ ' + e.message;
        status.className = 'text-xs text-error';
    }
}

async function pushSetting(payload) {
    if (pushInFlight) return;
    pushInFlight = true;
    const status = $('settings-status');
    const summary = Object.entries(payload).map(([k,v]) => `${k}=${v}`).join(', ');
    status.textContent = '⏳ envoi: ' + summary;
    status.className = 'text-xs text-warning';
    try {
        await fetch('/api/settings', {
            method: 'POST',
            headers: {'Content-Type':'application/json'},
            body: JSON.stringify(payload)
        });
        // Vérifier ce qui a effectivement été appliqué
        const r = await fetch('/api/settings');
        const d = await r.json();
        const cur = d.current || {};
        const des = d.desired || {};
        // Pour chaque clé envoyée, vérifier current vs demandé
        const applied = [];
        const pending = [];
        for (const [k, v] of Object.entries(payload)) {
            const key = k === 'yolo_model' ? 'yolo_model'
                      : k === 'imgsz' ? 'imgsz'
                      : k === 'embed_every' ? 'embed_every'
                      : k === 'threshold' ? 'threshold'
                      : k === 'conf' ? 'conf' : k;
            const curVal = cur[key];
            const eq = (curVal === v) || (typeof curVal === 'number' && Math.abs(curVal - v) < 1e-6);
            if (eq) applied.push(`${k}=${v}`);
            else pending.push(`${k}=${v}`);
        }
        if (pending.length === 0) {
            status.textContent = '✓ appliqué: ' + applied.join(', ');
            status.className = 'text-xs text-success';
        } else {
            status.textContent = '⏳ en attente: ' + pending.join(', ');
            status.className = 'text-xs text-warning';
        }
        setTimeout(() => { status.textContent = ''; }, 3500);
    } catch (e) {
        status.textContent = '✗ erreur: ' + e.message;
        status.className = 'text-xs text-error';
    } finally {
        pushInFlight = false;
    }
}

// Model change
$('setting-model').addEventListener('change', (e) => {
    pushSetting({ yolo_model: e.target.value });
});

// imgsz buttons
document.querySelectorAll('[data-imgsz]').forEach(btn => {
    btn.addEventListener('click', () => {
        const v = parseInt(btn.dataset.imgsz);
        document.querySelectorAll('[data-imgsz]').forEach(b => b.classList.remove('btn-active'));
        btn.classList.add('btn-active');
        pushSetting({ imgsz: v });
    });
});

// Embed every
$('setting-embed').addEventListener('input', (e) => {
    $('setting-embed-val').textContent = e.target.value;
});
$('setting-embed').addEventListener('change', (e) => {
    pushSetting({ embed_every: parseInt(e.target.value) });
});

// Threshold
$('setting-threshold').addEventListener('input', (e) => {
    $('setting-threshold-val').textContent = parseFloat(e.target.value).toFixed(2);
});
$('setting-threshold').addEventListener('change', (e) => {
    pushSetting({ threshold: parseFloat(e.target.value) });
});

// Conf
$('setting-conf').addEventListener('input', (e) => {
    $('setting-conf-val').textContent = parseFloat(e.target.value).toFixed(2);
});
$('setting-conf').addEventListener('change', (e) => {
    pushSetting({ conf: parseFloat(e.target.value) });
});

// Re-match button
const rematchBtn = $('btn-rematch');
if (rematchBtn) {
    rematchBtn.addEventListener('click', () => {
        rematch();
    });
}

// === Init ===
setInterval(refreshStats, 1000);
refreshStats();

// === Polling JPEG du flux vidéo (~25 fps) ===
// Remplace le MJPEG stream qui timeout après ~100s sur Cloudflare (524)
const streamImg = $('stream');
let streamTimer = null;
function refreshStream() {
    streamImg.src = `/video_feed?t=${Date.now()}`;
}
streamImg.addEventListener('load', () => {
    clearTimeout(streamTimer);
    streamTimer = setTimeout(refreshStream, 40);   // ~25 fps
});
streamImg.addEventListener('error', () => {
    clearTimeout(streamTimer);
    streamTimer = setTimeout(refreshStream, 500);  // back-off si erreur
});
refreshStream();