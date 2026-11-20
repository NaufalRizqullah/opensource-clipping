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
    let url;
    try { url = new URL(value.trim()); }
    catch (_) { throw StudioI18n.error('api.invalidUrl'); }
    if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash) {
      throw StudioI18n.error('api.invalidUrl');
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
    if (!base) throw StudioI18n.error('api.noBackend');
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
        const error = typeof detail === 'string' ? new Error(detail) : StudioI18n.error('api.requestFailed', { status: res.status });
        error.status = res.status;
        error.body = body;
        throw error;
      }
      if (body === null) throw StudioI18n.error('api.unexpectedPage');
      return body;
    } catch (error) {
      if (!error.status && base === getBackendUrl()) setConnection('offline');
      if (error.name === 'AbortError') throw StudioI18n.error('api.timeout');
      if (error.name === 'TypeError') throw StudioI18n.error('api.network');
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
    if (health.status !== 'ok') throw StudioI18n.error('api.healthFailed');
    let readiness;
    try {
      readiness = await requestAt(base, '/api/ready', { timeoutMs: 150000 });
    } catch (error) {
      // Older backends have liveness only. Keep compatibility, visibly unverified.
      if (error.status === 404) return { health, ready: null, legacy: true };
      if (error.status === 503) {
        const failure = StudioI18n.error('api.notReady');
        failure.notReady = true;
        throw failure;
      }
      throw error;
    }
    if (readiness.ready !== true) {
      const error = StudioI18n.error('api.notReady');
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

  function uploadYoutubeCookies(file) {
    return request('/api/settings/youtube-cookies', {
      method: 'PUT', headers: { 'Content-Type': 'application/octet-stream' },
      body: file, redirect: 'error',
    });
  }

  function removeYoutubeCookies() {
    return request('/api/settings/youtube-cookies', { method: 'DELETE', redirect: 'error' });
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
          throw StudioI18n.error('api.liveUnavailable');
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
    uploadYoutubeCookies,
    removeYoutubeCookies,
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
  return StudioI18n.formatDate(dateStr);
}

/** Escape backend-provided text before inserting it into an HTML template. */
function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, char => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[char]));
}

/** Known API status codes stay unchanged; only their display labels are localized. */
function statusKey(status, cancelRequested = false) {
  if (cancelRequested && !['completed', 'failed', 'cancelled'].includes(status)) return 'status.cancelling';
  return ['queued', 'downloading', 'transcribing', 'analyzing', 'rendering', 'completed', 'failed', 'cancelled'].includes(status)
    ? `status.${status}` : 'common.message';
}

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
      StudioI18n.text(label, 'connection.host', { host: new URL(url).hostname, readiness: connection.legacy ? { key: 'connection.unverified' } : '' });
    } catch {
      StudioI18n.text(label, 'connection.connected');
    }
    if (btn) StudioI18n.text(btn, 'common.change');
    if (stopBtn) stopBtn.classList.remove('hidden');
  } else {
    dot.className = 'connect-dot offline';
    StudioI18n.text(label, connection.status === 'not_ready' ? 'connection.notReady'
      : StudioAPI.isConfigured() ? (connection.status === 'unknown' ? 'connection.checking' : 'connection.unreachable')
      : 'connection.offline');
    if (btn) StudioI18n.text(btn, StudioAPI.isConfigured() ? 'common.reconnect' : 'common.connect');
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
        <h3 class="font-display font-bold text-lg text-slate-900 dark:text-white">${StudioI18n.html('connection.title')}</h3>
        <button onclick="closeConnectModal()" aria-label="${tr('connection.close')}" data-i18n-aria-label="connection.close" class="p-1 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors">
          <i data-lucide="x" class="w-5 h-5 text-slate-400"></i>
        </button>
      </div>
      <p class="text-sm text-slate-500 dark:text-slate-400 mb-4">
        ${StudioI18n.html('connection.help')}
      </p>
      <div class="mb-4">
        <input id="connect-url-input" type="url" aria-label="${tr('connection.url')}" data-i18n-aria-label="connection.url"
          placeholder="https://xxxx-xx-xx.ngrok-free.app"
          class="w-full px-4 py-3 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500/50 focus:border-brand-500 text-slate-900 dark:text-white placeholder-slate-400 font-mono"
        />
      </div>
      <div id="connect-status-msg" class="text-sm mb-4 hidden"></div>
      <div class="flex gap-3">
        <button id="connect-test-btn" onclick="testAndConnect()" class="flex-1 px-4 py-2.5 rounded-xl bg-brand-600 text-white font-medium hover:bg-brand-700 transition-colors text-sm">
          ${StudioI18n.html('connection.test')}
        </button>
        ${currentUrl ? `
          <button onclick="disconnectBackend()" title="${tr('connection.disconnectHint')}" data-i18n-title="connection.disconnectHint" class="px-3 py-2.5 rounded-xl border border-red-200 dark:border-red-900/50 text-red-600 dark:text-red-400 font-medium hover:bg-red-50 dark:hover:bg-red-950/30 transition-colors text-sm">
            ${StudioI18n.html('connection.disconnect')}
          </button>
          <button id="stop-server-btn" onclick="stopBackendServer()" title="${tr('connection.stopHint')}" data-i18n-title="connection.stopHint" class="px-3 py-2.5 rounded-xl border border-red-500 bg-red-600 text-white font-medium hover:bg-red-700 transition-colors text-sm">
            ${StudioI18n.html('connection.stop')}
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
    StudioI18n.text(msg, 'connection.enterUrl');
    msg.classList.remove('hidden');
    return;
  }

  btn.disabled = true;
  btn.innerHTML = `<span class="spinner spinner-dark"></span> ${StudioI18n.html('connection.testing')}`;
  msg.className = 'text-sm mb-4 text-slate-500 dark:text-slate-400';
  StudioI18n.text(msg, 'connection.connecting');
  msg.classList.remove('hidden');

  try {
    const { health, legacy } = await StudioAPI.connect(url);
    msg.className = 'text-sm mb-4 text-emerald-600 dark:text-emerald-400';
    StudioI18n.text(msg, 'connection.success', {
      gpu: health.gpu_available ? '✅' : '❌', ffmpeg: health.ffmpeg_available ? '✅' : '❌',
      readiness: { key: legacy ? 'connection.legacy' : 'connection.ready' },
    });
    updateConnectIndicator();

    setTimeout(() => closeConnectModal(), 1200);
  } catch (err) {
    msg.className = 'text-sm mb-4 text-red-500';
    StudioI18n.text(msg, 'connection.failed', { message: StudioI18n.errorMessage(err) });
    updateConnectIndicator();
  } finally {
    btn.disabled = false;
    btn.innerHTML = StudioI18n.html('connection.test');
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

  if (!confirm(tr('connection.stopConfirm'))) {
    return;
  }

  if (btn) {
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner spinner-light"></span> ${StudioI18n.html('connection.stopping')}`;
  }

  if (msg) {
    msg.className = 'text-sm mb-4 text-slate-500 dark:text-slate-400';
    StudioI18n.text(msg, 'connection.shutdownSignal');
    msg.classList.remove('hidden');
  }

  try {
    await StudioAPI.shutdownServer();
    if (msg) {
      msg.className = 'text-sm mb-4 text-emerald-600 dark:text-emerald-400';
      StudioI18n.text(msg, 'connection.stopped');
    }
  } catch (err) {
    if (msg) {
      msg.className = 'text-sm mb-4 text-red-500';
      StudioI18n.text(msg, 'connection.shutdownFailed', { message: StudioI18n.errorMessage(err) });
    } else {
      alert(tr('connection.shutdownFailed', { message: StudioI18n.errorMessage(err) }));
    }
    if (btn) { btn.disabled = false; btn.innerHTML = StudioI18n.html('connection.stop'); }
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
