/**
 * OpenSource Clipping Studio — API Client
 * 
 * Shared module for all studio pages.
 * Backend URL is stored in localStorage and configurable via the Connect modal.
 */

const StudioAPI = (() => {
  const STORAGE_KEY = 'osc_backend_url';
  const TUNNEL_HEADERS = { 'ngrok-skip-browser-warning': 'true' };
  let connection = { status: 'unknown', ready: null };
  let connectionCheck = null;

  function setConnection(status, details = {}) {
    connection = { status, ready: null, ...details };
    window.dispatchEvent(new CustomEvent('studio:connection', { detail: connection }));
  }

  function normalizeBackendUrl(value) {
    const url = new URL(value.trim());
    if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash) {
      throw new Error('Enter an HTTP(S) backend URL without credentials, query parameters, or fragments.');
    }
    return url.href.replace(/\/+$/, '');
  }

  /** Get the saved backend URL */
  function getBackendUrl() {
    return (localStorage.getItem(STORAGE_KEY) || '').replace(/\/+$/, '');
  }

  /** Save backend URL */
  function setBackendUrl(url) {
    localStorage.setItem(STORAGE_KEY, normalizeBackendUrl(url));
    setConnection('unknown');
  }

  /** Clear backend URL */
  function clearBackendUrl() {
    localStorage.removeItem(STORAGE_KEY);
    setConnection('offline');
    window.dispatchEvent(new CustomEvent('studio:disconnected'));
  }

  /** A saved URL does not imply the backend is reachable or ready. */
  function isConfigured() {
    return !!getBackendUrl();
  }

  /** Generic fetch wrapper; candidate connections do not change the saved URL. */
  async function requestAt(base, path, options = {}) {
    if (!base) throw new Error('Backend URL not configured. Please connect first.');
    const { timeoutMs = 30000, ...fetchOptions } = options;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const res = await fetch(`${base}${path}`, {
        ...fetchOptions,
        signal: controller.signal,
        headers: {
          'Content-Type': 'application/json',
          ...TUNNEL_HEADERS,
          ...(fetchOptions.headers || {}),
        },
      });
      const body = await res.json().catch(() => null);
      if (!res.ok) {
        const detail = body?.detail;
        const error = new Error(typeof detail === 'string' ? detail : `Request failed: ${res.status}`);
        error.status = res.status;
        error.body = body;
        throw error;
      }
      if (body === null) throw new Error('Backend returned an unexpected page. Check the tunnel URL.');
      return body;
    } catch (error) {
      if (!error.status && base === getBackendUrl()) setConnection('offline');
      if (error.name === 'AbortError') throw new Error('Backend request timed out. Check the notebook and tunnel.');
      throw error;
    } finally {
      clearTimeout(timeout);
    }
  }

  async function request(path, options = {}) {
    return requestAt(getBackendUrl(), path, options);
  }

  // ---- Health ----
  async function checkHealth() {
    return request('/api/health');
  }

  async function probeConnection(base) {
    const health = await requestAt(base, '/api/health');
    if (health.status !== 'ok') throw new Error('Backend health check failed.');
    let readiness;
    try {
      readiness = await requestAt(base, '/api/ready', { timeoutMs: 150000 });
    } catch (error) {
      // Older backends have liveness only. Keep compatibility, visibly unverified.
      if (error.status === 404) return { health, ready: null, legacy: true };
      if (error.status === 503) {
        const failure = new Error('Backend is reachable, but the renderer is not ready. Check the notebook startup output.');
        failure.notReady = true;
        throw failure;
      }
      throw error;
    }
    if (readiness.ready !== true) {
      const error = new Error('Backend is reachable, but the renderer is not ready. Check the notebook startup output.');
      error.notReady = true;
      throw error;
    }
    return { health, ready: true, legacy: false };
  }

  async function checkConnection() {
    const base = getBackendUrl();
    if (!base) { setConnection('offline'); return null; }
    if (connectionCheck?.base === base) return connectionCheck.promise;
    const pending = { base };
    pending.promise = (async () => {
      try {
        const result = await probeConnection(base);
        if (base === getBackendUrl()) setConnection('online', result);
        return result;
      } catch (error) {
        if (base === getBackendUrl()) setConnection(error.notReady ? 'not_ready' : 'offline');
        throw error;
      }
    })();
    connectionCheck = pending;
    try {
      return await pending.promise;
    } finally {
      if (connectionCheck === pending) connectionCheck = null;
    }
  }

  async function connect(url) {
    const base = normalizeBackendUrl(url);
    let result;
    try {
      result = await probeConnection(base);
    } catch (error) {
      if (base === getBackendUrl()) setConnection(error.notReady ? 'not_ready' : 'offline');
      throw error;
    }
    setBackendUrl(base);
    setConnection('online', result);
    window.dispatchEvent(new CustomEvent('studio:connected', { detail: result }));
    return result;
  }

  // ---- Jobs ----
  async function fetchJobs() {
    return request('/api/jobs');
  }

  async function fetchJob(jobId) {
    return request(`/api/jobs/${encodeURIComponent(jobId)}`);
  }

  async function createJob(payload) {
    return request('/api/jobs', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  async function deleteJob(jobId) {
    return request(`/api/jobs/${encodeURIComponent(jobId)}`, { method: 'DELETE' });
  }

  // ---- Settings ----
  async function fetchSettings() {
    return request('/api/settings');
  }

  async function updateSettings(payload) {
    return request('/api/settings', {
      method: 'PUT',
      body: JSON.stringify(payload),
    });
  }

  // ---- SSE for real-time job status ----
  function createSSE(jobId, onMessage, onError = () => {}) {
    const base = getBackendUrl();
    if (!base) return null;
    let closed = false;
    let retryTimer = null;
    let controller;
    const connection = {
      close() {
        closed = true;
        clearTimeout(retryTimer);
        if (controller) controller.abort();
      },
    };

    async function stream() {
      controller = new AbortController();
      let reader;
      try {
        const response = await fetch(`${base}/api/jobs/${encodeURIComponent(jobId)}/status`, {
          headers: { Accept: 'text/event-stream', ...TUNNEL_HEADERS },
          signal: controller.signal,
        });
        if (!response.ok || !response.body || !response.headers.get('content-type')?.includes('text/event-stream')) {
          throw new Error('Live updates unavailable; polling will continue.');
        }
        reader = response.body.getReader();
        const decoder = new TextDecoder();
        let pending = '';
        while (!closed) {
          const { value, done } = await reader.read();
          if (done) break;
          pending += decoder.decode(value, { stream: true });
          let separator;
          while ((separator = /\r?\n\r?\n/.exec(pending))) {
            const frame = pending.slice(0, separator.index);
            pending = pending.slice(separator.index + separator[0].length);
            const data = frame.split(/\r?\n/).filter(line => line.startsWith('data:'))
              .map(line => line.slice(5).replace(/^ /, '')).join('\n');
            if (!data) continue;
            const event = JSON.parse(data);
            onMessage(event);
            if (['completed', 'failed', 'cancelled'].includes(event.type)) connection.close();
            if (closed) break;
          }
        }
      } catch (error) {
        if (!closed) onError(error);
      } finally {
        if (reader) await reader.cancel().catch(() => {});
        if (!closed) retryTimer = setTimeout(stream, 3000);
      }
    }
    stream();
    return connection;
  }

  // ---- Server Shutdown ----
  async function shutdownServer() {
    return request('/api/shutdown', { method: 'POST' });
  }

  // Public API
  return {
    getBackendUrl,
    setBackendUrl,
    clearBackendUrl,
    isConfigured,
    connect,
    checkConnection,
    getConnectionState: () => connection,
    checkHealth,
    fetchJobs,
    fetchJob,
    createJob,
    deleteJob,
    fetchSettings,
    updateSettings,
    createSSE,
    shutdownServer,
  };
})();


// ============================================================
// Shared UI Utilities (used across all studio pages)
// ============================================================

/** Theme toggle */
function toggleTheme() {
  if (document.documentElement.classList.contains('dark')) {
    document.documentElement.classList.remove('dark');
    localStorage.setItem('theme', 'light');
  } else {
    document.documentElement.classList.add('dark');
    localStorage.setItem('theme', 'dark');
  }
}

/** Format date string to locale */
function formatDate(dateStr) {
  if (!dateStr) return '—';
  const d = new Date(dateStr);
  return d.toLocaleString('id-ID', {
    day: '2-digit', month: 'short', year: 'numeric',
    hour: '2-digit', minute: '2-digit',
  });
}

/** Escape backend-provided text before inserting it into an HTML template. */
function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, char => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[char]));
}

/** Status labels */
const STATUS_LABELS = {
  queued: 'Queued',
  downloading: 'Downloading',
  transcribing: 'Transcribing',
  analyzing: 'Analyzing',
  rendering: 'Rendering',
  completed: 'Completed',
  failed: 'Failed',
  cancelled: 'Cancelled',
};

/** Mobile sidebar toggle */
function toggleSidebar() {
  const sidebar = document.getElementById('sidebar');
  const overlay = document.getElementById('mobile-overlay');
  sidebar.classList.toggle('open');
  if (sidebar.classList.contains('open')) {
    overlay.style.display = 'block';
  } else {
    overlay.style.display = 'none';
  }
}

function closeSidebar() {
  const sidebar = document.getElementById('sidebar');
  const overlay = document.getElementById('mobile-overlay');
  sidebar.classList.remove('open');
  overlay.style.display = 'none';
}

/** Update sidebar connection indicator */
function updateConnectIndicator() {
  const dot = document.getElementById('connect-dot');
  const label = document.getElementById('connect-label');
  const btn = document.getElementById('connect-btn');
  const stopBtn = document.getElementById('sidebar-stop-btn');
  if (!dot || !label) return;

  const connection = StudioAPI.getConnectionState();
  if (StudioAPI.isConfigured() && connection.status === 'online') {
    dot.className = 'connect-dot online';
    const url = StudioAPI.getBackendUrl();
    try {
      label.textContent = `${new URL(url).hostname}${connection.legacy ? ' (readiness unverified)' : ''}`;
    } catch {
      label.textContent = 'Connected';
    }
    if (btn) btn.textContent = 'Change';
    if (stopBtn) stopBtn.classList.remove('hidden');
  } else {
    dot.className = 'connect-dot offline';
    label.textContent = connection.status === 'not_ready' ? 'Backend not ready'
      : StudioAPI.isConfigured() ? (connection.status === 'unknown' ? 'Checking connection…' : 'Backend unreachable')
      : 'Not connected';
    if (btn) btn.textContent = StudioAPI.isConfigured() ? 'Reconnect' : 'Connect';
    if (stopBtn) stopBtn.classList.toggle('hidden', connection.status !== 'not_ready');
  }
}

/** Show connect modal */
function showConnectModal() {
  const existing = document.getElementById('connect-modal-backdrop');
  if (existing) existing.remove();

  const currentUrl = StudioAPI.getBackendUrl();

  const backdrop = document.createElement('div');
  backdrop.id = 'connect-modal-backdrop';
  backdrop.className = 'connect-modal-backdrop';
  backdrop.innerHTML = `
    <div class="connect-modal">
      <div class="flex items-center justify-between mb-4">
        <h3 class="font-display font-bold text-lg text-slate-900 dark:text-white">Connect to Backend</h3>
        <button onclick="closeConnectModal()" class="p-1 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors">
          <i data-lucide="x" class="w-5 h-5 text-slate-400"></i>
        </button>
      </div>
      <p class="text-sm text-slate-500 dark:text-slate-400 mb-4">
        Paste the tunnel URL from your Kaggle/Colab notebook. Run the notebook first to get the URL.
      </p>
      <div class="mb-4">
        <input id="connect-url-input" type="url"
          placeholder="https://xxxx-xx-xx.ngrok-free.app"
          class="w-full px-4 py-3 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500/50 focus:border-brand-500 text-slate-900 dark:text-white placeholder-slate-400 font-mono"
        />
      </div>
      <div id="connect-status-msg" class="text-sm mb-4 hidden"></div>
      <div class="flex gap-3">
        <button id="connect-test-btn" onclick="testAndConnect()" class="flex-1 px-4 py-2.5 rounded-xl bg-brand-600 text-white font-medium hover:bg-brand-700 transition-colors text-sm">
          Test & Connect
        </button>
        ${currentUrl ? `
          <button onclick="disconnectBackend()" title="Disconnect UI from backend" class="px-3 py-2.5 rounded-xl border border-red-200 dark:border-red-900/50 text-red-600 dark:text-red-400 font-medium hover:bg-red-50 dark:hover:bg-red-950/30 transition-colors text-sm">
            Disconnect
          </button>
          <button id="stop-server-btn" onclick="stopBackendServer()" title="Shut down the backend process" class="px-3 py-2.5 rounded-xl border border-red-500 bg-red-600 text-white font-medium hover:bg-red-700 transition-colors text-sm">
            Stop Server
          </button>
        ` : ''}
      </div>
    </div>
  `;

  document.body.appendChild(backdrop);
  backdrop.addEventListener('click', (e) => {
    if (e.target === backdrop) closeConnectModal();
  });
  lucide.createIcons();
  document.getElementById('connect-url-input').value = currentUrl;
  document.getElementById('connect-url-input').focus();
}

function closeConnectModal() {
  const backdrop = document.getElementById('connect-modal-backdrop');
  if (backdrop) backdrop.remove();
}

async function testAndConnect() {
  const input = document.getElementById('connect-url-input');
  const msg = document.getElementById('connect-status-msg');
  const btn = document.getElementById('connect-test-btn');
  const url = input.value.trim();

  if (!url) {
    msg.className = 'text-sm mb-4 text-red-500';
    msg.textContent = '⚠️ Please enter a URL';
    msg.classList.remove('hidden');
    return;
  }

  btn.disabled = true;
  btn.innerHTML = '<span class="spinner spinner-dark"></span> Testing...';
  msg.className = 'text-sm mb-4 text-slate-500 dark:text-slate-400';
  msg.textContent = '⏳ Connecting...';
  msg.classList.remove('hidden');

  try {
    const { health, legacy } = await StudioAPI.connect(url);
    msg.className = 'text-sm mb-4 text-emerald-600 dark:text-emerald-400';
    msg.textContent = `✅ Connected! GPU: ${health.gpu_available ? '✅' : '❌'} | FFmpeg: ${health.ffmpeg_available ? '✅' : '❌'}${legacy ? ' | Older backend: renderer readiness unverified.' : ' | Renderer ready'}`;
    updateConnectIndicator();

    setTimeout(() => closeConnectModal(), 1200);
  } catch (err) {
    msg.className = 'text-sm mb-4 text-red-500';
    msg.textContent = `❌ Connection failed: ${err.message}`;
    updateConnectIndicator();
  } finally {
    btn.disabled = false;
    btn.innerHTML = 'Test & Connect';
  }
}

function disconnectBackend() {
  StudioAPI.clearBackendUrl();
  updateConnectIndicator();
  closeConnectModal();
}

async function stopBackendServer() {
  const btn = document.getElementById('stop-server-btn');
  const msg = document.getElementById('connect-status-msg');

  if (!confirm('Are you sure you want to STOP the Kaggle backend server? You will have to restart the notebook cell manually.')) {
    return;
  }

  if (btn) {
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner spinner-light"></span> Stopping...';
  }

  if (msg) {
    msg.className = 'text-sm mb-4 text-slate-500 dark:text-slate-400';
    msg.textContent = '⏳ Sending shutdown signal...';
    msg.classList.remove('hidden');
  }

  try {
    await StudioAPI.shutdownServer();
    if (msg) {
      msg.className = 'text-sm mb-4 text-emerald-600 dark:text-emerald-400';
      msg.textContent = '✅ Server stopped successfully.';
    }
  } catch (err) {
    if (msg) {
      msg.className = 'text-sm mb-4 text-red-500';
      msg.textContent = `Shutdown could not be confirmed: ${err.message}`;
    } else {
      alert(`Shutdown could not be confirmed: ${err.message}`);
    }
    if (btn) { btn.disabled = false; btn.textContent = 'Stop Server'; }
    return;
  }
    setTimeout(() => {
      StudioAPI.clearBackendUrl();
      updateConnectIndicator();
      closeConnectModal();
    }, 1500);
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
  updateConnectIndicator();
  StudioAPI.checkConnection().catch(() => {});
  setInterval(() => StudioAPI.checkConnection().catch(() => {}), 15000);
});
window.addEventListener('studio:connection', updateConnectIndicator);
