// app.js — Frontend pour Boeuf Tracker (DaisyUI)
const $ = (id) => document.getElementById(id);
let lastSource = '';
let switchingDevice = false;

// === Refresh stats ===
async function refreshStats() {
    try {
        const res = await fetch('/api/stats');
        const data = await res.json();

        const sourceLabel = data.source_label || data.source || '--';
        $('meta-source').textContent = 'source: ' + sourceLabel;
        $('meta-device').textContent = 'device: ' + (data.device || '--');
        $('meta-fps').textContent = 'fps: ' + (data.fps || 0).toFixed(1);
        $('meta-frames').textContent = 'frames: ' + (data.frame_count || 0);
        $('current-source').textContent = '→ ' + sourceLabel;

        // Reconnexion MJPEG si source a changé
        if (sourceLabel !== lastSource) {
            $('stream').src = '/video_feed?t=' + Date.now();
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

// === Init ===
setInterval(refreshStats, 1000);
refreshStats();