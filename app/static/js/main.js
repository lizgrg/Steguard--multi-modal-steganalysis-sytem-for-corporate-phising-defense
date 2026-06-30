// StegGuard SOC — main.js
// CSY4022 Computing Dissertation | Liza Gurung 24812928

// ── Global threshold state ────────────────────────────────────────────────────
window.THRESHOLDS = {
  stego:  parseFloat(localStorage.getItem('thresh_stego')  || '0.65'),
  review: parseFloat(localStorage.getItem('thresh_review') || '0.35'),
};

// ── Live alerts from real scan data ───────────────────────────────────────────
function refreshAlerts() {
  fetch('/api/logs')
    .then(r => r.json())
    .then(data => {
      const logs = data.logs || [];
      const feed = document.getElementById('alerts-feed');
      if (!feed) return;

      if (logs.length === 0) {
        feed.innerHTML = `
          <div class="alert-item">
            <span class="alert-dot dot-green"></span>
            <div>
              <div class="alert-text">No scans yet — upload files to see alerts.</div>
              <div class="alert-time">—</div>
            </div>
          </div>`;
        return;
      }

      // Show most recent 5 — all verdicts, highlight STEGO and REVIEW
      const recent = logs.slice(0, 5);
      feed.innerHTML = recent.map(a => {
        const v = (a.verdict || '').toString().trim().toUpperCase();
        let dotClass, label, msg;

        if (v === 'STEGO') {
          dotClass = 'dot-red';
          msg = `<strong>CRITICAL:</strong> ${a.file_name} — LSB payload detected, prob ${a.stego_prob}.`;
        } else if (v === 'REVIEW') {
          dotClass = 'dot-amber';
          msg = `<strong>REVIEW:</strong> ${a.file_name} — borderline score ${a.stego_prob}. Analyst required.`;
        } else {
          dotClass = 'dot-green';
          msg = `CLEAN: ${a.file_name} — no payload detected, prob ${a.stego_prob}.`;
        }

        const time = a.timestamp ? a.timestamp.split(' ')[1] || a.timestamp : '—';
        return `
          <div class="alert-item">
            <span class="alert-dot ${dotClass}"></span>
            <div>
              <div class="alert-text">${msg}</div>
              <div class="alert-time">${time}</div>
            </div>
          </div>`;
      }).join('');
    })
    .catch(() => {});
}

// ── Poll stats and update topbar ──────────────────────────────────────────────
function refreshStats() {
  fetch('/api/stats')
    .then(r => r.json())
    .then(d => {
      const el = document.getElementById('threat-count');
      if (el) el.textContent = d.threats + ' Threat' + (d.threats !== 1 ? 's' : '');

      const dot = document.getElementById('sb-alert-dot');
      if (dot) dot.style.display = d.threats > 0 ? 'block' : 'none';

      const total   = document.getElementById('stat-total');
      const threats = document.getElementById('stat-threats');
      if (total)   total.textContent   = d.total.toLocaleString();
      if (threats) threats.textContent = d.threats;
    })
    .catch(() => {});
}

// ── Drag-and-drop highlight ───────────────────────────────────────────────────
document.querySelectorAll('.upload-zone').forEach(zone => {
  zone.addEventListener('dragover',  e => { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop',      e => { e.preventDefault(); zone.classList.remove('dragover'); });
});

// ── Init ──────────────────────────────────────────────────────────────────────
refreshStats();
refreshAlerts();
setInterval(refreshStats,  10000);
setInterval(refreshAlerts, 15000);