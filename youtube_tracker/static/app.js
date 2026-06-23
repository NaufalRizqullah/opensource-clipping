/* ==========================================================================
   YouTube Tracker — Vanilla JS SPA
   All data comes from local API. No YouTube calls from frontend.
   ========================================================================== */

(function () {
  'use strict';

  const API = '';
  const $app = document.getElementById('app');
  const $sidebarNav = document.getElementById('sidebar-nav');
  const $toastContainer = document.getElementById('toast-container');
  const $modalOverlay = document.getElementById('modal-overlay');
  const $modalContent = document.getElementById('modal-content');

  // -----------------------------------------------------------------------
  // Helpers
  // -----------------------------------------------------------------------

  async function api(path, opts = {}) {
    const url = API + path;
    const config = { headers: {} };
    if (opts.method) config.method = opts.method;
    if (opts.body) {
      config.method = config.method || 'POST';
      config.headers['Content-Type'] = 'application/json';
      config.body = JSON.stringify(opts.body);
    }
    const res = await fetch(url, config);
    if (path.endsWith('/csv') && res.ok) {
      return res.text();
    }
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    return data;
  }

  function fmtDuration(s) {
    if (!s && s !== 0) return '--:--';
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = s % 60;
    if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
    return `${m}:${String(sec).padStart(2, '0')}`;
  }

  function fmtDate(d) {
    if (!d) return '';
    return d.substring(0, 10);
  }

  function escHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  function statusPill(status) {
    const s = status || 'unused';
    return `<span class="status-pill pill-${s}" data-status="${s}">${s}</span>`;
  }

  function progressBar(used, candidate, skipped, unused, total) {
    if (!total) return '';
    const pUsed = (used / total * 100).toFixed(1);
    const pCand = (candidate / total * 100).toFixed(1);
    const pSkip = (skipped / total * 100).toFixed(1);
    const pUnu = (unused / total * 100).toFixed(1);
    return `
      <div class="progress-wrap">
        <div class="progress-bar-track">
          <div class="progress-segment seg-used" style="width:${pUsed}%"></div>
          <div class="progress-segment seg-candidate" style="width:${pCand}%"></div>
          <div class="progress-segment seg-skipped" style="width:${pSkip}%"></div>
          <div class="progress-segment seg-unused" style="width:${pUnu}%"></div>
        </div>
        <div class="progress-legend">
          <span><span class="dot dot-used"></span>${used} used</span>
          <span><span class="dot dot-candidate"></span>${candidate} cand</span>
          <span><span class="dot dot-skipped"></span>${skipped} skip</span>
          <span><span class="dot dot-unused"></span>${unused} unused</span>
        </div>
      </div>`;
  }

  function thumbImg(url, alt, cls = 'video-thumb') {
    if (url) return `<img src="${escHtml(url)}" alt="${escHtml(alt)}" class="${cls}" loading="lazy" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'"><div class="${cls}-placeholder" style="display:none">🎬</div>`;
    return `<div class="${cls}-placeholder">🎬</div>`;
  }

  function toast(msg, type = 'info') {
    const el = document.createElement('div');
    el.className = `toast toast-${type}`;
    el.textContent = msg;
    $toastContainer.appendChild(el);
    setTimeout(() => { el.style.opacity = '0'; setTimeout(() => el.remove(), 300); }, 3500);
  }

  function showModal(html) {
    $modalContent.innerHTML = html;
    $modalOverlay.classList.add('active');
  }

  function hideModal() {
    $modalOverlay.classList.remove('active');
  }

  $modalOverlay.addEventListener('click', (e) => {
    if (e.target === $modalOverlay) hideModal();
  });

  function loading(text = 'Loading...') {
    return `<div class="loading-spinner"><div class="spinner"></div><div class="loading-text">${escHtml(text)}</div></div>`;
  }

  function emptyState(icon, title, text) {
    return `<div class="empty-state"><div class="empty-icon">${icon}</div><div class="empty-title">${escHtml(title)}</div><div class="empty-text">${escHtml(text)}</div></div>`;
  }

  // --- Bulk Actions State ---
  window._selectedVideos = new Set();

  window._toggleAll = (checked, idPrefix = 'chk-') => {
    document.querySelectorAll(`.${idPrefix}`).forEach(cb => {
      cb.checked = checked;
      if (checked) window._selectedVideos.add(cb.value);
      else window._selectedVideos.delete(cb.value);
    });
    window._updateBulkBar();
  };

  window._toggleRow = (val, checked) => {
    if (checked) window._selectedVideos.add(val);
    else window._selectedVideos.delete(val);
    window._updateBulkBar();
  };

  window._updateBulkBar = () => {
    const bar = document.getElementById('bulk-bar');
    if (!bar) return;
    const count = window._selectedVideos.size;
    if (count > 0) {
      bar.style.display = 'flex';
      document.getElementById('bulk-count').textContent = count;
    } else {
      bar.style.display = 'none';
    }
  };

  window._bulkAction = async (status) => {
    const ids = Array.from(window._selectedVideos);
    if (!ids.length) return;
    try {
      await api('/api/videos/bulk_status', { method: 'PATCH', body: { video_ids: ids, status } });
      toast(`${ids.length} videos marked as ${status}`, 'success');
      window._selectedVideos.clear();
      handleRoute();
    } catch (err) {
      toast(err.message, 'error');
    }
  };

  window._markUsed = async (ytVideoId) => {
    try {
      await api(`/api/videos/${ytVideoId}/status`, { method: 'PATCH', body: { status: 'used' } });
      toast('Marked as Used', 'success');
      handleRoute();
    } catch (err) {
      toast(err.message, 'error');
    }
  };

  function bulkBarHtml() {
    return `
      <div class="bulk-action-bar" id="bulk-bar" style="display:none">
        <span style="font-weight:600"><span id="bulk-count">0</span> selected</span>
        <div style="width:1px;height:16px;background:var(--border-medium);margin:0 8px"></div>
        <span style="font-size:0.8rem">Mark as:</span>
        <button class="btn btn-ghost btn-sm" onclick="window._bulkAction('unused')">Unused</button>
        <button class="btn btn-ghost btn-sm" onclick="window._bulkAction('candidate')">Candidate</button>
        <button class="btn btn-ghost btn-sm" onclick="window._bulkAction('used')">Used</button>
        <button class="btn btn-ghost btn-sm" onclick="window._bulkAction('skipped')">Skipped</button>
      </div>`;
  }


  // -----------------------------------------------------------------------
  // Router
  // -----------------------------------------------------------------------

  const routes = {};

  function registerRoute(pattern, handler) {
    routes[pattern] = handler;
  }

  function navigate(hash) {
    window.location.hash = hash;
  }

  function matchRoute(hash) {
    const path = hash.replace(/^#/, '') || '/';

    // Exact match first
    if (routes[path]) return { handler: routes[path], params: {} };

    // Pattern match
    for (const pattern of Object.keys(routes)) {
      const regex = new RegExp('^' + pattern.replace(/:(\w+)/g, '(?<$1>[^/]+)') + '$');
      const m = path.match(regex);
      if (m) return { handler: routes[pattern], params: m.groups || {} };
    }
    return null;
  }

  async function handleRoute() {
    const hash = window.location.hash || '#/';
    const match = matchRoute(hash);

    // Update sidebar active
    $sidebarNav.querySelectorAll('a').forEach(a => {
      const route = a.dataset.route;
      if (!route) return;
      const isActive = hash === '#' + route || hash.startsWith('#' + route + '/');
      a.classList.toggle('active', isActive);
    });

    if (match) {
      try {
        await match.handler(match.params);
      } catch (err) {
        $app.innerHTML = `<div class="error-state">Error: ${escHtml(err.message)}</div>`;
        console.error(err);
      }
    } else {
      $app.innerHTML = emptyState('🤔', 'Page not found', 'The page you are looking for does not exist.');
    }
  }

  window.addEventListener('hashchange', handleRoute);

  // -----------------------------------------------------------------------
  // Dashboard
  // -----------------------------------------------------------------------

  registerRoute('/', async () => {
    $app.innerHTML = loading('Loading dashboard...');

    const data = await api('/api/sources');
    const sources = data.sources || [];
    const stats = data.stats || {};

    let html = `
      <div class="page-header">
        <h2>Dashboard</h2>
        <p class="page-subtitle">Local YouTube playlist snapshot tracker</p>
      </div>

      <div class="stats-grid">
        <div class="stat-card stat-accent"><div class="stat-value">${stats.total_sources || 0}</div><div class="stat-label">Sources</div></div>
        <div class="stat-card stat-accent"><div class="stat-value">${stats.total_videos || 0}</div><div class="stat-label">Videos</div></div>
        <div class="stat-card stat-accent"><div class="stat-value">${stats.total_channels || 0}</div><div class="stat-label">Channels</div></div>
        <div class="stat-card stat-used"><div class="stat-value">${stats.total_used || 0}</div><div class="stat-label">Used</div></div>
        <div class="stat-card stat-unused"><div class="stat-value">${stats.total_unused || 0}</div><div class="stat-label">Unused</div></div>
        <div class="stat-card stat-candidate"><div class="stat-value">${stats.total_candidate || 0}</div><div class="stat-label">Candidate</div></div>
        <div class="stat-card stat-skipped"><div class="stat-value">${stats.total_skipped || 0}</div><div class="stat-label">Skipped</div></div>
      </div>

      <!-- Add Playlist / Video -->
      <div class="add-forms-grid" id="add-form">
        <div class="add-form-card">
          <h3><span style="font-size:1.2rem">📑</span> Add Playlist</h3>
          <div class="form-row">
            <div class="form-group">
              <input type="text" class="form-input" id="add-playlist-url" placeholder="https://www.youtube.com/playlist?list=..." />
            </div>
            <button class="btn btn-primary" id="btn-add-playlist" onclick="window._submitAdd('playlist')">Add</button>
          </div>
        </div>
        <div class="add-form-card">
          <h3><span style="font-size:1.2rem">🎬</span> Add Manual Video</h3>
          <div class="form-row">
            <div class="form-group">
              <input type="text" class="form-input" id="add-video-url" placeholder="https://www.youtube.com/watch?v=..." />
            </div>
            <button class="btn btn-primary" id="btn-add-video" onclick="window._submitAdd('video')">Add</button>
          </div>
        </div>
      </div>

      <!-- Source cards -->
      <div class="source-grid" id="source-grid">`;

    if (sources.length === 0) {
      html += emptyState('📋', 'No sources yet', 'Add a YouTube playlist or manual video to get started.');
    } else {
      for (const s of sources) {
        const total = s.total_videos || 0;
        const typeBadge = `badge-${s.source_type}`;
        html += `
          <div class="source-card" onclick="window.location.hash='#/source/${s.id}'">
            <div class="source-card-header">
              ${s.thumbnail_url
            ? `<img src="${escHtml(s.thumbnail_url)}" class="source-thumb" alt="" onerror="this.outerHTML='<div class=source-thumb-placeholder>📋</div>'">`
            : '<div class="source-thumb-placeholder">📋</div>'}
              <div>
                <div class="source-title">${escHtml(s.title)}</div>
                <div class="source-meta">${s.owner_channel_name ? escHtml(s.owner_channel_name) + ' · ' : ''}${total} videos</div>
                <span class="source-type-badge ${typeBadge}">${s.source_type}</span>
              </div>
            </div>
            ${progressBar(s.used_count || 0, s.candidate_count || 0, s.skipped_count || 0, s.unused_count || 0, total)}
            <div class="source-meta text-xs mt-2">
              ${s.last_pulled_at ? 'Last pull: ' + fmtDate(s.last_pulled_at) : 'Never pulled'}
            </div>
            <div class="source-card-actions" onclick="event.stopPropagation()">
              <button class="btn btn-secondary btn-sm" onclick="window._refreshSource(${s.id})">🔄 Pull Again</button>
              ${s.source_type !== 'manual' ? `<button class="btn btn-danger btn-sm" onclick="window._deleteSource(${s.id})">🗑</button>` : ''}
            </div>
          </div>`;
      }
    }

    html += `</div>

    <div style="margin-top: 24px; display:flex; gap:10px;">
      <button class="btn btn-secondary" onclick="window._exportJson()">📥 Export JSON</button>
      <button class="btn btn-secondary" onclick="window._exportCsv()">📥 Export CSV</button>
    </div>`;

    $app.innerHTML = html;

    window._submitAdd = async (type) => {
      const inputId = type === 'playlist' ? 'add-playlist-url' : 'add-video-url';
      const btnId = type === 'playlist' ? 'btn-add-playlist' : 'btn-add-video';
      const url = document.getElementById(inputId).value.trim();

      if (!url) { toast('Please enter a URL', 'error'); return; }

      const $btn = document.getElementById(btnId);
      $btn.disabled = true;
      $btn.textContent = 'Fetching...';

      try {
        if (type === 'playlist') {
          const res = await api('/api/sources/playlist', { body: { url } });
          const st = res.stats;
          const errCount = st.fetch_errors || 0;
          const msg = `Playlist added! ${st.total_entries} entries in playlist, ${st.videos_found} fetched, ${st.videos_added} new.${errCount ? ` (${errCount} errors)` : ''}`;
          toast(msg, 'success');
        } else {
          const res = await api('/api/videos/manual', { body: { url } });
          toast(`Video "${res.title}" added!`, 'success');
        }
        document.getElementById(inputId).value = '';
        handleRoute();
      } catch (err) {
        toast(err.message, 'error');
      } finally {
        $btn.disabled = false;
        $btn.textContent = 'Add';
      }
    };
  });

  // -----------------------------------------------------------------------
  // Playlists (All Playlists view)
  // -----------------------------------------------------------------------

  registerRoute('/playlists', async () => {
    $app.innerHTML = loading('Loading playlists...');

    const data = await api('/api/sources');
    const sources = (data.sources || []).filter(s => s.source_type === 'playlist');

    let html = `
      <div class="page-header">
        <h2>Playlists</h2>
        <p class="page-subtitle">Tracked YouTube playlists</p>
      </div>

      <div class="source-grid" id="source-grid">`;

    if (sources.length === 0) {
      html += emptyState('📑', 'No playlists yet', 'Add a YouTube playlist from the Dashboard to get started.');
    } else {
      for (const s of sources) {
        const total = s.total_videos || 0;
        html += `
          <div class="source-card" onclick="window.location.hash='#/source/${s.id}'">
            <div class="source-card-header">
              ${s.thumbnail_url
            ? `<img src="${escHtml(s.thumbnail_url)}" class="source-thumb" alt="" onerror="this.outerHTML='<div class=source-thumb-placeholder>📋</div>'">`
            : '<div class="source-thumb-placeholder">📋</div>'}
              <div>
                <div class="source-title">${escHtml(s.title)}</div>
                <div class="source-meta">${s.owner_channel_name ? escHtml(s.owner_channel_name) + ' · ' : ''}${total} videos</div>
                <span class="badge badge-playlist">PLAYLIST</span>
              </div>
            </div>
            ${progressBar(s.total_used || 0, s.total_candidate || 0, s.total_skipped || 0, s.total_unused || 0, total)}
            
            <div class="source-footer">
              <span class="last-pull">Last pull: ${fmtDate(s.last_pulled_at) || 'Never pulled'}</span>
              <div class="source-actions">
                <button class="btn btn-secondary btn-sm" onclick="event.stopPropagation(); window._refreshSource(${s.id})">🔄 Pull Again</button>
                <button class="btn btn-danger btn-sm" onclick="event.stopPropagation(); window._deleteSource(${s.id})" title="Delete Source">🗑️</button>
              </div>
            </div>
          </div>`;
      }
    }
    html += `</div>`;
    $app.innerHTML = html;
  });

  // -----------------------------------------------------------------------
  // Manual Videos View
  // -----------------------------------------------------------------------

  registerRoute('/manual', async () => {
    $app.innerHTML = loading('Loading manual videos...');
    try {
      const data = await api('/api/sources');
      const manualSource = (data.sources || []).find(s => s.source_type === 'manual');
      if (manualSource) {
        navigate('#/source/' + manualSource.id);
      } else {
        $app.innerHTML = emptyState('🎬', 'No manual videos yet', 'Add manual videos from the Dashboard to get started.');
      }
    } catch (e) {
      $app.innerHTML = `<div class="error-state">Error: ${escHtml(e.message)}</div>`;
    }
  });

  // -----------------------------------------------------------------------
  // Source Detail
  // -----------------------------------------------------------------------

  registerRoute('/source/:id', async (params) => {
    $app.innerHTML = loading('Loading source...');
    const sourceId = params.id;

    const source = await api(`/api/sources/${sourceId}`);
    if (!source) { $app.innerHTML = emptyState('🤔', 'Source not found', ''); return; }

    let currentFilters = { status: 'all', present: 'all', q: '', sort: 'position' };

    async function renderVideos() {
      const qs = new URLSearchParams(currentFilters).toString();
      const data = await api(`/api/sources/${sourceId}/videos?${qs}`);
      const videos = data.videos || [];
      const $list = document.getElementById('video-list');

      if (videos.length === 0) {
        $list.innerHTML = emptyState('🎬', 'No videos match', 'Try changing filters.');
        return;
      }

      window._selectedVideos.clear();

      $list.innerHTML = bulkBarHtml() + `
        <div style="display:flex;align-items:center;padding:0 16px 8px;font-size:0.8rem;color:var(--text-secondary)">
          <input type="checkbox" onchange="window._toggleAll(this.checked, 'chk-src')" style="margin-right:8px;accent-color:var(--accent)"> Select All
        </div>
      ` + '<div class="video-list">' + videos.map(v => {
        const isChecked = window._selectedVideos.has(v.youtube_video_id) ? 'checked' : '';
        return `
        <div class="video-row">
          <div class="checkbox-wrap">
            <input type="checkbox" class="row-checkbox chk-src" value="${escHtml(v.youtube_video_id)}" ${isChecked} onchange="window._toggleRow(this.value, this.checked)">
          </div>
          ${thumbImg(v.thumbnail_url, v.title)}
          <div class="video-info">
            <div class="video-title">${escHtml(v.title)}</div>
            <div class="video-channel">
              ${v.channel_db_id ? `<a href="#/channel/${v.channel_db_id}">${escHtml(v.channel_name || 'Unknown')}</a>` : escHtml(v.channel_name || '')}
              ${v.is_present_latest_pull === 0 ? '<span class="video-missing-badge">Missing from latest pull</span>' : ''}
              ${(v.source_count || 0) > 1 ? `<span class="video-dup-badge" onclick="event.stopPropagation();window.location.hash='#/duplicates'">In ${v.source_count} sources</span>` : ''}
            </div>
          </div>
          <div class="video-channel text-sm">${v.notes ? escHtml(v.notes.substring(0, 40)) : ''}</div>
          <div class="video-duration">${fmtDuration(v.duration_seconds)}</div>
          <div>${statusPill(v.status)}</div>
          <div class="video-actions">
            ${v.status !== 'used' ? `<button class="btn btn-primary btn-sm" onclick="event.stopPropagation();window._markUsed('${escHtml(v.youtube_video_id)}')">✓ Used</button>` : ''}
            <button class="btn btn-ghost btn-sm" title="Edit Status" onclick="event.stopPropagation();window._editStatus('${escHtml(v.youtube_video_id)}','${escHtml(v.status)}')">✏️</button>
            <button class="btn btn-ghost btn-sm" title="Copy Command" onclick="event.stopPropagation();window._copyCmd('${escHtml(v.youtube_video_id)}')">📋</button>
            <a class="btn btn-ghost btn-sm" href="${escHtml(v.url)}" target="_blank" title="Open on YouTube" onclick="event.stopPropagation()">▶️</a>
          </div>
        </div>`;
      }).join('') + '</div>';
      window._updateBulkBar();
    }

    const total = source.total_videos || 0;
    const pr = source.latest_pull_run;

    let html = `
      <div class="source-detail-header">
        ${source.thumbnail_url ? `<img src="${escHtml(source.thumbnail_url)}" class="source-detail-thumb" alt="" onerror="this.style.display='none'">` : ''}
        <div class="source-detail-info">
          <h2>${escHtml(source.title)}</h2>
          ${source.url ? `<a href="${escHtml(source.url)}" target="_blank" class="source-url">${escHtml(source.url)}</a>` : ''}
          <div class="source-meta mt-2">
            <span class="source-type-badge badge-${source.source_type}">${source.source_type}</span>
            ${source.owner_channel_name ? ' · ' + escHtml(source.owner_channel_name) : ''}
            · ${total} videos
            ${source.last_pulled_at ? ' · Last pull: ' + fmtDate(source.last_pulled_at) : ''}
          </div>
          ${progressBar(source.used_count || 0, source.candidate_count || 0, source.skipped_count || 0, source.unused_count || 0, total)}
          <div class="source-detail-actions">
            ${source.source_type !== 'manual' ? `<button class="btn btn-primary btn-sm" id="pull-again-btn" onclick="window._refreshSource(${source.id})">🔄 Pull Again</button>` : ''}
            <button class="btn btn-secondary btn-sm" onclick="window.location.hash='#/'">← Back</button>
          </div>
        </div>
      </div>

      <div class="snapshot-notice">
        <span class="notice-icon">📸</span>
        This playlist is a local snapshot. It will not update unless you pull again.
      </div>

      ${pr ? `
      <div class="snapshot-notice" style="background:rgba(0,196,140,0.06);border-color:rgba(0,196,140,0.15);">
        <span class="notice-icon">📊</span>
        Latest pull: ${pr.videos_found} fetched of ${source.total_videos} in tracker, ${pr.videos_added} added, ${pr.videos_already_exists} existing, ${pr.videos_missing_from_latest_pull} missing · ${pr.status}
      </div>` : ''}

      <!-- Filters -->
      <div class="filter-bar">
        <input type="text" class="form-input search-input" placeholder="Search title or channel..." id="source-search" />
        <div class="filter-chips" id="status-filters">
          <button class="filter-chip active" data-val="all">All</button>
          <button class="filter-chip" data-val="unused">Unused</button>
          <button class="filter-chip" data-val="candidate">Candidate</button>
          <button class="filter-chip" data-val="used">Used</button>
          <button class="filter-chip" data-val="skipped">Skipped</button>
          <button class="filter-chip" data-val="not_used_yet">Not Used Yet</button>
        </div>
        <select class="form-input" style="width:auto;min-width:120px" id="source-sort">
          <option value="position">Playlist Order</option>
          <option value="title">Title</option>
          <option value="channel">Channel</option>
          <option value="duration">Duration</option>
          <option value="status">Status</option>
          <option value="date">Upload Date</option>
        </select>
        <div class="filter-chips" id="present-filters">
          <button class="filter-chip active" data-val="all">All</button>
          <button class="filter-chip" data-val="present">Present</button>
          <button class="filter-chip" data-val="missing">Missing</button>
        </div>
      </div>

      <div id="video-list">${loading('Loading videos...')}</div>`;

    $app.innerHTML = html;

    // Bind filters
    document.querySelectorAll('#status-filters .filter-chip').forEach(chip => {
      chip.addEventListener('click', () => {
        document.querySelectorAll('#status-filters .filter-chip').forEach(c => c.classList.remove('active'));
        chip.classList.add('active');
        currentFilters.status = chip.dataset.val;
        renderVideos();
      });
    });

    document.querySelectorAll('#present-filters .filter-chip').forEach(chip => {
      chip.addEventListener('click', () => {
        document.querySelectorAll('#present-filters .filter-chip').forEach(c => c.classList.remove('active'));
        chip.classList.add('active');
        currentFilters.present = chip.dataset.val;
        renderVideos();
      });
    });

    document.getElementById('source-sort').addEventListener('change', (e) => {
      currentFilters.sort = e.target.value;
      renderVideos();
    });

    let searchTimeout;
    document.getElementById('source-search').addEventListener('input', (e) => {
      clearTimeout(searchTimeout);
      searchTimeout = setTimeout(() => {
        currentFilters.q = e.target.value;
        renderVideos();
      }, 300);
    });

    renderVideos();
  });

  // -----------------------------------------------------------------------
  // Channels
  // -----------------------------------------------------------------------

  registerRoute('/channels', async () => {
    $app.innerHTML = loading('Loading channels...');
    const data = await api('/api/channels');
    const channels = data.channels || [];

    let html = `
      <div class="page-header">
        <h2>Channels</h2>
        <p class="page-subtitle">Videos in your tracker grouped by channel</p>
      </div>`;

    if (channels.length === 0) {
      html += emptyState('📺', 'No channels yet', 'Channels appear when you add playlists or videos.');
    } else {
      html += '<div class="channel-grid">';
      for (const ch of channels) {
        const initials = (ch.name || '?').substring(0, 2).toUpperCase();
        const avatarHtml = ch.thumbnail_url 
          ? `<img src="${escHtml(ch.thumbnail_url)}" class="channel-avatar" alt="" onerror="this.outerHTML='<div class=\\'channel-avatar\\'>${initials}</div>'">`
          : `<div class="channel-avatar">${initials}</div>`;
        html += `
          <div class="channel-card" onclick="window.location.hash='#/channel/${ch.id}'">
            <div class="channel-card-header">
              ${avatarHtml}
              <div>
                <div class="channel-name">${escHtml(ch.name)}</div>
                <div class="channel-handle">${ch.handle ? escHtml(ch.handle) : ''}</div>
              </div>
            </div>
            <div class="channel-stats-row">
              <span class="mini-stat"><strong>${ch.total_videos}</strong> videos</span>
              <span class="mini-stat" style="color:var(--color-used)"><strong>${ch.used_count}</strong> used</span>
              <span class="mini-stat" style="color:var(--color-unused)"><strong>${ch.unused_count}</strong> unused</span>
              <span class="mini-stat" style="color:var(--color-candidate)"><strong>${ch.candidate_count}</strong> cand</span>
              <span class="mini-stat" style="color:var(--color-skipped)"><strong>${ch.skipped_count}</strong> skip</span>
            </div>
            ${progressBar(ch.used_count, ch.candidate_count, ch.skipped_count, ch.unused_count, ch.total_videos)}
          </div>`;
      }
      html += '</div>';
    }

    $app.innerHTML = html;
  });

  // -----------------------------------------------------------------------
  // Channel Detail
  // -----------------------------------------------------------------------

  registerRoute('/channel/:id', async (params) => {
    $app.innerHTML = loading('Loading channel...');
    const chId = params.id;
    let currentFilters = { status: 'all', q: '' };

    async function render() {
      const qs = new URLSearchParams(currentFilters).toString();
      const data = await api(`/api/channels/${chId}/videos?${qs}`);
      const ch = data.channel;
      const stats = data.stats;
      const videos = data.videos || [];

      const initials = (ch.name || '?').substring(0, 2).toUpperCase();
      const avatarHtml = ch.thumbnail_url 
        ? `<img src="${escHtml(ch.thumbnail_url)}" class="channel-avatar" style="width:64px;height:64px;font-size:1.6rem;" alt="" onerror="this.outerHTML='<div class=\\'channel-avatar\\' style=\\'width:64px;height:64px;font-size:1.6rem;\\'>${initials}</div>'">`
        : `<div class="channel-avatar" style="width:64px;height:64px;font-size:1.6rem;">${initials}</div>`;

      let html = `
        <div class="page-header" style="display:flex;align-items:center;gap:16px;">
          ${avatarHtml}
          <div>
            <h2 style="margin-bottom:4px;">${escHtml(ch.name || 'Unknown Channel')}</h2>
            ${ch.url ? `<p class="page-subtitle" style="margin:0;"><a href="${escHtml(ch.url)}" target="_blank" style="color:var(--text-link)">${escHtml(ch.url)}</a></p>` : ''}
          </div>
        </div>

        <div class="snapshot-notice">
          <span class="notice-icon">📸</span>
          Known videos from this channel in your tracker. This does not fetch all videos from YouTube.
        </div>

        <div class="stats-grid">
          <div class="stat-card"><div class="stat-value">${stats.total}</div><div class="stat-label">Known Videos</div></div>
          <div class="stat-card stat-used"><div class="stat-value">${stats.used}</div><div class="stat-label">Used</div></div>
          <div class="stat-card stat-unused"><div class="stat-value">${stats.unused}</div><div class="stat-label">Unused</div></div>
          <div class="stat-card stat-candidate"><div class="stat-value">${stats.candidate}</div><div class="stat-label">Candidate</div></div>
          <div class="stat-card stat-skipped"><div class="stat-value">${stats.skipped}</div><div class="stat-label">Skipped</div></div>
          <div class="stat-card" style="border-color:var(--accent)"><div class="stat-value" style="color:var(--accent)">${stats.not_used_yet}</div><div class="stat-label">Not Used Yet</div></div>
        </div>

        <div class="filter-bar">
          <input type="text" class="form-input search-input" placeholder="Search videos..." id="ch-search" value="${escHtml(currentFilters.q)}" />
          <div class="filter-chips" id="ch-status-filters">
            <button class="filter-chip ${currentFilters.status === 'all' ? 'active' : ''}" data-val="all">All</button>
            <button class="filter-chip ${currentFilters.status === 'unused' ? 'active' : ''}" data-val="unused">Unused</button>
            <button class="filter-chip ${currentFilters.status === 'candidate' ? 'active' : ''}" data-val="candidate">Candidate</button>
            <button class="filter-chip ${currentFilters.status === 'used' ? 'active' : ''}" data-val="used">Used</button>
            <button class="filter-chip ${currentFilters.status === 'skipped' ? 'active' : ''}" data-val="skipped">Skipped</button>
            <button class="filter-chip ${currentFilters.status === 'not_used_yet' ? 'active' : ''}" data-val="not_used_yet">Not Used Yet</button>
          </div>
          <button class="btn btn-secondary btn-sm" onclick="window.location.hash='#/channels'">← Back</button>
        </div>`;

      if (videos.length === 0) {
        html += emptyState('🎬', 'No videos match', 'Try changing filters.');
      } else {
        window._selectedVideos.clear();
        html += bulkBarHtml();
        html += `
        <div style="display:flex;align-items:center;padding:0 16px 8px;font-size:0.8rem;color:var(--text-secondary)">
          <input type="checkbox" onchange="window._toggleAll(this.checked, 'chk-ch')" style="margin-right:8px;accent-color:var(--accent)"> Select All
        </div>
        <div class="video-list" id="ch-video-list">`;

        for (const v of videos) {
          const isChecked = window._selectedVideos.has(v.youtube_video_id) ? 'checked' : '';
          const srcTags = (v.sources || []).map(s => `<a class="dup-source-tag" href="#/source/${s.id}">${escHtml(s.title)}</a>`).join('');
          html += `
            <div class="video-row">
              <div class="checkbox-wrap">
                <input type="checkbox" class="row-checkbox chk-ch" value="${escHtml(v.youtube_video_id)}" ${isChecked} onchange="window._toggleRow(this.value, this.checked)">
              </div>
              ${thumbImg(v.thumbnail_url, v.title)}
              <div class="video-info">
                <div class="video-title">${escHtml(v.title)}</div>
                <div class="video-channel">${escHtml(v.channel_name || '')}</div>
                <div class="dup-sources">${srcTags}</div>
              </div>
              <div class="video-channel text-sm">${v.notes ? escHtml(v.notes.substring(0, 40)) : ''}</div>
              <div class="video-duration">${fmtDuration(v.duration_seconds)}</div>
              <div>${statusPill(v.status)}</div>
              <div class="video-actions">
                ${v.status !== 'used' ? `<button class="btn btn-primary btn-sm" onclick="event.stopPropagation();window._markUsed('${escHtml(v.youtube_video_id)}')">✓ Used</button>` : ''}
                <button class="btn btn-ghost btn-sm" onclick="event.stopPropagation();window._editStatus('${escHtml(v.youtube_video_id)}','${escHtml(v.status)}')">✏️</button>
                <button class="btn btn-ghost btn-sm" onclick="event.stopPropagation();window._copyCmd('${escHtml(v.youtube_video_id)}')">📋</button>
                <a class="btn btn-ghost btn-sm" href="${escHtml(v.url)}" target="_blank" onclick="event.stopPropagation()">▶️</a>
              </div>
            </div>`;
        }
        html += '</div>';
      }

      $app.innerHTML = html;
      window._updateBulkBar();

      // Bind filters
      document.querySelectorAll('#ch-status-filters .filter-chip').forEach(chip => {
        chip.addEventListener('click', () => {
          currentFilters.status = chip.dataset.val;
          render();
        });
      });

      let sTimeout;
      const $search = document.getElementById('ch-search');
      if ($search) {
        $search.addEventListener('input', (e) => {
          clearTimeout(sTimeout);
          sTimeout = setTimeout(() => { currentFilters.q = e.target.value; render(); }, 300);
        });
      }
    }

    await render();
  });

  // -----------------------------------------------------------------------
  // Duplicates
  // -----------------------------------------------------------------------

  registerRoute('/duplicates', async () => {
    $app.innerHTML = loading('Checking duplicates...');
    let currentStatus = 'all';

    async function render() {
      const data = await api(`/api/duplicates?status=${currentStatus}`);
      const stats = data.stats;
      const videos = data.videos || [];

      let html = `
        <div class="page-header">
          <h2>Duplicates in Tracker</h2>
          <p class="page-subtitle">Videos appearing in more than one source</p>
        </div>

        <div class="stats-grid">
          <div class="stat-card stat-candidate"><div class="stat-value">${stats.duplicate_videos}</div><div class="stat-label">Duplicate Videos</div></div>
          <div class="stat-card stat-accent"><div class="stat-value">${stats.total_duplicate_source_links}</div><div class="stat-label">Total Source Links</div></div>
        </div>

        <div class="filter-bar">
          <div class="filter-chips" id="dup-filters">
            <button class="filter-chip ${currentStatus === 'all' ? 'active' : ''}" data-val="all">All</button>
            <button class="filter-chip ${currentStatus === 'unused' ? 'active' : ''}" data-val="unused">Unused</button>
            <button class="filter-chip ${currentStatus === 'candidate' ? 'active' : ''}" data-val="candidate">Candidate</button>
            <button class="filter-chip ${currentStatus === 'used' ? 'active' : ''}" data-val="used">Used</button>
            <button class="filter-chip ${currentStatus === 'skipped' ? 'active' : ''}" data-val="skipped">Skipped</button>
            <button class="filter-chip ${currentStatus === 'not_used_yet' ? 'active' : ''}" data-val="not_used_yet">Not Used Yet</button>
          </div>
        </div>

        <div class="video-list" id="dup-list">`;

      if (videos.length === 0) {
        html += emptyState('✨', 'No duplicates', 'No videos appear in multiple sources.');
      } else {
        window._selectedVideos.clear();
        html += bulkBarHtml();
        html += `
        <div style="display:flex;align-items:center;padding:0 16px 8px;font-size:0.8rem;color:var(--text-secondary)">
          <input type="checkbox" onchange="window._toggleAll(this.checked, 'chk-dup')" style="margin-right:8px;accent-color:var(--accent)"> Select All
        </div>
        <div class="video-list">`;

        for (const v of videos) {
          const isChecked = window._selectedVideos.has(v.youtube_video_id) ? 'checked' : '';
          const srcTags = (v.sources || []).map(s =>
            `<a class="dup-source-tag" href="#/source/${s.id}">${escHtml(s.title)} <span style="opacity:0.5">(${s.source_type})</span></a>`
          ).join('');
          html += `
            <div class="video-row">
              <div class="checkbox-wrap">
                <input type="checkbox" class="row-checkbox chk-dup" value="${escHtml(v.youtube_video_id)}" ${isChecked} onchange="window._toggleRow(this.value, this.checked)">
              </div>
              ${thumbImg(v.thumbnail_url, v.title)}
              <div class="video-info">
                <div class="video-title">${escHtml(v.title)}</div>
                <div class="video-channel">${escHtml(v.channel_name || '')} · <span style="color:var(--color-candidate)">In ${v.source_count} sources</span></div>
                <div class="dup-sources">${srcTags}</div>
              </div>
              <div class="video-channel text-sm"></div>
              <div class="video-duration">${fmtDuration(v.duration_seconds)}</div>
              <div>${statusPill(v.status)}</div>
              <div class="video-actions">
                ${v.status !== 'used' ? `<button class="btn btn-primary btn-sm" onclick="event.stopPropagation();window._markUsed('${escHtml(v.youtube_video_id)}')">✓ Used</button>` : ''}
                <button class="btn btn-ghost btn-sm" onclick="event.stopPropagation();window._editStatus('${escHtml(v.youtube_video_id)}','${escHtml(v.status)}')">✏️</button>
                <button class="btn btn-ghost btn-sm" onclick="event.stopPropagation();window._copyCmd('${escHtml(v.youtube_video_id)}')">📋</button>
                <a class="btn btn-ghost btn-sm" href="${escHtml(v.url)}" target="_blank" onclick="event.stopPropagation()">▶️</a>
              </div>
            </div>`;
        }
        html += '</div>';
      }

      $app.innerHTML = html;
      window._updateBulkBar();

      document.querySelectorAll('#dup-filters .filter-chip').forEach(chip => {
        chip.addEventListener('click', () => {
          currentStatus = chip.dataset.val;
          render();
        });
      });
    }

    await render();
  });

  // -----------------------------------------------------------------------
  // Search
  // -----------------------------------------------------------------------

  registerRoute('/search', async () => {
    let html = `
      <div class="page-header">
        <h2>Global Search</h2>
        <p class="page-subtitle">Search videos, channels, notes across all sources</p>
      </div>

      <div class="filter-bar">
        <input type="text" class="form-input search-input" placeholder="Search by title, channel, notes, clip title..." id="global-search" autofocus />
        <div class="filter-chips" id="search-status-filters">
          <button class="filter-chip active" data-val="">All</button>
          <button class="filter-chip" data-val="unused">Unused</button>
          <button class="filter-chip" data-val="candidate">Candidate</button>
          <button class="filter-chip" data-val="used">Used</button>
          <button class="filter-chip" data-val="skipped">Skipped</button>
          <button class="filter-chip" data-val="not_used_yet">Not Used Yet</button>
        </div>
      </div>

      <div id="search-results">${emptyState('🔍', 'Search something', 'Type to search across all your tracked videos.')}</div>`;

    $app.innerHTML = html;

    let currentStatus = '';

    async function doSearch() {
      const q = document.getElementById('global-search').value.trim();
      const $results = document.getElementById('search-results');
      if (!q) {
        $results.innerHTML = emptyState('🔍', 'Search something', 'Type to search across all your tracked videos.');
        return;
      }
      $results.innerHTML = loading('Searching...');
      try {
        const data = await api(`/api/search?q=${encodeURIComponent(q)}&status=${currentStatus}`);
        const videos = data.videos || [];
        if (videos.length === 0) {
          $results.innerHTML = emptyState('🤷', 'No results', `No videos found for "${q}".`);
          return;
        }

        window._selectedVideos.clear();
        $results.innerHTML = `<div class="text-secondary text-sm mb-4">${videos.length} result(s)</div>` +
          bulkBarHtml() + `
          <div style="display:flex;align-items:center;padding:0 16px 8px;font-size:0.8rem;color:var(--text-secondary)">
            <input type="checkbox" onchange="window._toggleAll(this.checked, 'chk-search')" style="margin-right:8px;accent-color:var(--accent)"> Select All
          </div>
          ` +
          '<div class="video-list">' + videos.map(v => {
            const isChecked = window._selectedVideos.has(v.youtube_video_id) ? 'checked' : '';
            const srcTags = (v.sources || []).map(s => `<a class="dup-source-tag" href="#/source/${s.id}">${escHtml(s.title)}</a>`).join('');
            return `
              <div class="video-row">
                <div class="checkbox-wrap">
                  <input type="checkbox" class="row-checkbox chk-search" value="${escHtml(v.youtube_video_id)}" ${isChecked} onchange="window._toggleRow(this.value, this.checked)">
                </div>
                ${thumbImg(v.thumbnail_url, v.title)}
                <div class="video-info">
                  <div class="video-title">${escHtml(v.title)}</div>
                  <div class="video-channel">${v.channel_db_id ? `<a href="#/channel/${v.channel_db_id}">${escHtml(v.channel_name || '')}</a>` : escHtml(v.channel_name || '')}</div>
                  <div class="dup-sources">${srcTags}</div>
                </div>
                <div class="video-channel text-sm">${v.notes ? escHtml(v.notes.substring(0, 40)) : ''}</div>
                <div class="video-duration">${fmtDuration(v.duration_seconds)}</div>
                <div>${statusPill(v.status)}</div>
                <div class="video-actions">
                  ${v.status !== 'used' ? `<button class="btn btn-primary btn-sm" onclick="event.stopPropagation();window._markUsed('${escHtml(v.youtube_video_id)}')">✓ Used</button>` : ''}
                  <button class="btn btn-ghost btn-sm" onclick="event.stopPropagation();window._editStatus('${escHtml(v.youtube_video_id)}','${escHtml(v.status)}')">✏️</button>
                  <button class="btn btn-ghost btn-sm" onclick="event.stopPropagation();window._copyCmd('${escHtml(v.youtube_video_id)}')">📋</button>
                  <a class="btn btn-ghost btn-sm" href="${escHtml(v.url)}" target="_blank" onclick="event.stopPropagation()">▶️</a>
                </div>
              </div>`;
          }).join('') + '</div>';
        window._updateBulkBar();
      } catch (err) {
        $results.innerHTML = `<div class="error-state">${escHtml(err.message)}</div>`;
      }
    }

    let searchTimeout;
    document.getElementById('global-search').addEventListener('input', () => {
      clearTimeout(searchTimeout);
      searchTimeout = setTimeout(doSearch, 400);
    });

    document.querySelectorAll('#search-status-filters .filter-chip').forEach(chip => {
      chip.addEventListener('click', () => {
        document.querySelectorAll('#search-status-filters .filter-chip').forEach(c => c.classList.remove('active'));
        chip.classList.add('active');
        currentStatus = chip.dataset.val;
        doSearch();
      });
    });
  });

  // -----------------------------------------------------------------------
  // Settings
  // -----------------------------------------------------------------------

  registerRoute('/settings', async () => {
    $app.innerHTML = loading('Loading settings...');
    const settings = await api('/api/settings');

    const fields = [
      { key: 'clips', label: 'Default Clips Count', placeholder: 'e.g. 3', type: 'text' },
      { key: 'ratio', label: 'Aspect Ratio', placeholder: 'e.g. 9:16', type: 'text' },
      { key: 'font_style', label: 'Font Style', placeholder: 'e.g. bold', type: 'text' },
      { key: 'no_bgm', label: 'No BGM', placeholder: '', type: 'checkbox' },
      { key: 'split_screen', label: 'Split Screen', placeholder: '', type: 'checkbox' },
      { key: 'output_dir', label: 'Output Directory', placeholder: 'e.g. outputs/', type: 'text' },
    ];

    let html = `
      <div class="page-header">
        <h2>Settings</h2>
        <p class="page-subtitle">Default clipping command options</p>
      </div>

      <div class="settings-grid">`;

    for (const f of fields) {
      const val = settings[f.key] || '';
      if (f.type === 'checkbox') {
        const checked = val && val.toLowerCase() !== 'false' && val !== '0' ? 'checked' : '';
        html += `
          <div class="setting-card">
            <label>${escHtml(f.label)}</label>
            <label style="display:flex;align-items:center;gap:8px;cursor:pointer;font-weight:400;color:var(--text-primary)">
              <input type="checkbox" id="setting-${f.key}" ${checked} style="width:18px;height:18px;accent-color:var(--accent)" />
              Enabled
            </label>
          </div>`;
      } else {
        html += `
          <div class="setting-card">
            <label>${escHtml(f.label)}</label>
            <input type="text" class="form-input" id="setting-${f.key}" value="${escHtml(val)}" placeholder="${escHtml(f.placeholder)}" />
          </div>`;
      }
    }

    html += `
      </div>
      <div class="mt-4">
        <button class="btn btn-primary" id="save-settings">💾 Save Settings</button>
      </div>

      <div style="margin-top:32px; padding-top:24px; border-top:1px solid var(--border-subtle)">
        <h3 style="font-size:1.1rem;margin-bottom:12px">Export Data</h3>
        <div style="display:flex;gap:10px">
          <button class="btn btn-secondary" onclick="window._exportJson()">📥 Export JSON</button>
          <button class="btn btn-secondary" onclick="window._exportCsv()">📥 Export CSV</button>
        </div>
      </div>`;

    $app.innerHTML = html;

    document.getElementById('save-settings').addEventListener('click', async () => {
      const payload = {};
      for (const f of fields) {
        const el = document.getElementById(`setting-${f.key}`);
        if (f.type === 'checkbox') {
          payload[f.key] = el.checked ? 'true' : 'false';
        } else {
          payload[f.key] = el.value.trim();
        }
      }
      try {
        await api('/api/settings', { method: 'PATCH', body: payload });
        toast('Settings saved!', 'success');
      } catch (err) {
        toast(err.message, 'error');
      }
    });
  });

  // -----------------------------------------------------------------------
  // Global Actions (attached to window)
  // -----------------------------------------------------------------------

  window._editStatus = (ytVideoId, currentStatus) => {
    showModal(`
      <div class="modal-header">
        <h3>Edit Video Status</h3>
        <button class="modal-close" onclick="window._hideModal()">✕</button>
      </div>
      <div class="form-group">
        <label>Status</label>
        <select class="form-input" id="modal-status">
          <option value="unused" ${currentStatus === 'unused' ? 'selected' : ''}>Unused</option>
          <option value="candidate" ${currentStatus === 'candidate' ? 'selected' : ''}>Candidate</option>
          <option value="used" ${currentStatus === 'used' ? 'selected' : ''}>Used</option>
          <option value="skipped" ${currentStatus === 'skipped' ? 'selected' : ''}>Skipped</option>
        </select>
      </div>
      <div class="form-group">
        <label>Clip Title</label>
        <input type="text" class="form-input" id="modal-clip-title" placeholder="Title of the clip output" />
      </div>
      <div class="form-group">
        <label>Used At</label>
        <input type="date" class="form-input" id="modal-used-at" />
      </div>
      <div class="form-group">
        <label>Local Output Path</label>
        <input type="text" class="form-input" id="modal-output-path" placeholder="e.g. outputs/clip1.mp4" />
      </div>
      <div class="form-group">
        <label>Published URL</label>
        <input type="text" class="form-input" id="modal-published-url" placeholder="URL where clip was published" />
      </div>
      <div class="form-group">
        <label>Notes</label>
        <textarea class="form-input" id="modal-notes" placeholder="Any notes..."></textarea>
      </div>
      <div class="modal-footer">
        <button class="btn btn-secondary" onclick="window._hideModal()">Cancel</button>
        <button class="btn btn-primary" id="modal-save">Save</button>
      </div>
    `);

    // Pre-fill from API
    api(`/api/videos/${ytVideoId}`).then(v => {
      if (v.clip_title) document.getElementById('modal-clip-title').value = v.clip_title;
      if (v.used_at) document.getElementById('modal-used-at').value = v.used_at.substring(0, 10);
      if (v.status_local_output_path) document.getElementById('modal-output-path').value = v.status_local_output_path;
      if (v.status_published_url) document.getElementById('modal-published-url').value = v.status_published_url;
      if (v.notes) document.getElementById('modal-notes').value = v.notes;
    }).catch(() => { });

    document.getElementById('modal-save').addEventListener('click', async () => {
      const payload = {
        status: document.getElementById('modal-status').value,
        clip_title: document.getElementById('modal-clip-title').value,
        used_at: document.getElementById('modal-used-at').value || null,
        local_output_path: document.getElementById('modal-output-path').value,
        published_url: document.getElementById('modal-published-url').value,
        notes: document.getElementById('modal-notes').value,
      };
      try {
        await api(`/api/videos/${ytVideoId}/status`, { method: 'PATCH', body: payload });
        toast('Status updated!', 'success');
        hideModal();
        handleRoute(); // Refresh current page
      } catch (err) {
        toast(err.message, 'error');
      }
    });
  };

  window._hideModal = hideModal;

  window._copyCmd = async (ytVideoId) => {
    try {
      const data = await api('/api/command', { body: { youtube_video_id: ytVideoId } });
      await navigator.clipboard.writeText(data.command);
      toast('Command copied!', 'success');
    } catch (err) {
      toast(err.message, 'error');
    }
  };

  window._refreshSource = async (sourceId) => {
    const btn = event.target;
    btn.disabled = true;
    btn.textContent = '⏳ Pulling...';
    try {
      const result = await api(`/api/sources/${sourceId}/refresh`, { method: 'POST' });
      const s = result.stats;
      const errCount = s.fetch_errors || 0;
      const msg = `Pull complete: ${s.total_entries} entries, ${s.videos_found} fetched, ${s.videos_added} new, ${s.videos_missing_from_latest_pull} missing.${errCount ? ` (${errCount} errors)` : ''}`;
      toast(msg, 'success');
      handleRoute();
    } catch (err) {
      toast(`Pull failed: ${err.message}`, 'error');
      btn.disabled = false;
      btn.textContent = '🔄 Pull Again';
    }
  };

  window._deleteSource = async (sourceId) => {
    if (!confirm('Delete this source? Videos will remain in the database.')) return;
    try {
      await api(`/api/sources/${sourceId}`, { method: 'DELETE' });
      toast('Source deleted.', 'success');
      handleRoute();
    } catch (err) {
      toast(err.message, 'error');
    }
  };

  window._exportJson = async () => {
    try {
      const data = await api('/api/export/json');
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'youtube_tracker_export.json';
      a.click();
      URL.revokeObjectURL(url);
      toast('JSON exported!', 'success');
    } catch (err) {
      toast(err.message, 'error');
    }
  };

  window._exportCsv = async () => {
    try {
      const csv = await api('/api/export/csv');
      const blob = new Blob([csv], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'youtube_tracker_export.csv';
      a.click();
      URL.revokeObjectURL(url);
      toast('CSV exported!', 'success');
    } catch (err) {
      toast(err.message, 'error');
    }
  };

  // -----------------------------------------------------------------------
  // Init
  // -----------------------------------------------------------------------

  handleRoute();

})();
