const assert = require('node:assert/strict');
const { test } = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const studioPath = path.join(__dirname, '..', 'docs', 'studio');
const i18nSource = fs.readFileSync(path.join(studioPath, 'i18n.js'), 'utf8');
const apiSource = fs.readFileSync(path.join(studioPath, 'api.js'), 'utf8');
const flush = () => new Promise(resolve => setImmediate(resolve));
const json = (body, status = 200) => new Response(JSON.stringify(body), {
  status, headers: { 'Content-Type': 'application/json' },
});

function createHarness(fetchImpl = async () => json({ status: 'ok' }), options = {}) {
  const elements = new Map();
  const storage = new Map(Object.entries(options.storage || {}));
  const timers = new Map();
  const intervals = new Map();
  const window = new EventTarget();
  window.location = { search: '?id=abc123', href: '' };
  function element(id) {
    if (!elements.has(id)) {
      const classes = new Set(['hidden']);
      const attributes = new Map();
      elements.set(id, {
        id, textContent: '', innerHTML: '', value: '', disabled: false, style: {},
        setAttribute: (key, value) => attributes.set(key, String(value)),
        getAttribute: key => attributes.get(key) ?? null,
        removeAttribute: key => attributes.delete(key),
        hasAttribute: key => attributes.has(key),
        classList: {
          add: name => classes.add(name), remove: name => classes.delete(name),
          contains: name => classes.has(name),
          toggle(name, force) {
            const enabled = force === undefined ? !classes.has(name) : force;
            if (enabled) classes.add(name); else classes.delete(name);
            return enabled;
          },
        },
        focus() {}, remove() {}, addEventListener() {},
      });
    }
    return elements.get(id);
  }
  class DetailEvent extends Event {
    constructor(type, options = {}) { super(type); this.detail = options.detail; }
  }
  const context = vm.createContext({
    window, URL, URLSearchParams, AbortController, TextDecoder,
    CustomEvent: DetailEvent, console, fetch: fetchImpl,
    navigator: { language: options.browserLanguage || 'id-ID' },
    confirm: () => true, alert: () => {}, lucide: { createIcons() {} },
    localStorage: {
      getItem: key => { if (options.blockStorage) throw new Error('Storage blocked'); return storage.get(key) || null; },
      setItem: (key, value) => { if (options.blockStorage) throw new Error('Storage blocked'); storage.set(key, value); },
      removeItem: key => storage.delete(key),
    },
    document: {
      getElementById: element, querySelector: element, createElement: element,
      querySelectorAll: selector => [...elements.values()].filter(el => el.hasAttribute(selector.slice(1, -1))),
      documentElement: element('html'), body: { appendChild() {} }, addEventListener() {},
    },
    setTimeout(fn, delay) { const timer = {}; timers.set(timer, { fn, delay }); return timer; },
    clearTimeout: timer => timers.delete(timer),
    setInterval(fn, delay) { const timer = {}; intervals.set(timer, { fn, delay }); return timer; },
    clearInterval: timer => intervals.delete(timer),
  });
  vm.runInContext(i18nSource, context, { filename: 'i18n.js' });
  vm.runInContext(apiSource, context, { filename: 'api.js' });
  function runPage(filename) {
    const html = fs.readFileSync(path.join(studioPath, filename), 'utf8');
    const scripts = [...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)];
    vm.runInContext(scripts.at(-1)[1], context, { filename });
  }
  return {
    context, storage, timers, intervals, window, element, runPage,
    api: vm.runInContext('StudioAPI', context),
    i18n: vm.runInContext('StudioI18n', context),
    event: type => window.dispatchEvent(new DetailEvent(type)),
  };
}

test('failed candidate connection preserves the working backend and readiness state', async () => {
  const calls = [];
  const h = createHarness(async (url, options) => {
    calls.push([url, options]);
    if (url.startsWith('https://broken.example')) throw new TypeError('Failed to fetch');
    return json(url.endsWith('/ready') ? { ready: true } : { status: 'ok', gpu_available: true });
  });
  let connects = 0;
  h.window.addEventListener('studio:connected', () => connects++);
  await h.api.connect('https://working.example/');
  await assert.rejects(h.api.connect('https://broken.example'), /Failed to fetch/);
  assert.equal(h.api.getBackendUrl(), 'https://working.example');
  assert.equal(h.api.getConnectionState().status, 'online');
  assert.equal(connects, 1);
  assert(calls.every(([, options]) => options.headers['ngrok-skip-browser-warning'] === 'true'));
});

test('reachable but unready backend cannot replace a saved connection', async () => {
  const h = createHarness(async url => json(url.endsWith('/ready') ? { ready: false } : { status: 'ok' }, url.endsWith('/ready') ? 503 : 200));
  h.api.setBackendUrl('https://saved.example');
  await assert.rejects(h.api.connect('https://new.example'), /renderer is not ready/);
  assert.equal(h.api.getBackendUrl(), 'https://saved.example');
  await assert.rejects(h.api.checkConnection(), /renderer is not ready/);
  assert.equal(h.api.getConnectionState().status, 'not_ready');
});

test('legacy readiness 404 is explicit and malformed tunnel pages are rejected', async () => {
  const h = createHarness(async url => url.endsWith('/ready') ? json({ detail: 'Not found' }, 404) : json({ status: 'ok' }));
  const result = await h.api.connect('https://legacy.example');
  assert.equal(result.legacy, true);
  assert.equal(h.api.getConnectionState().ready, null);
  h.context.fetch = async () => new Response('<html>ngrok warning</html>');
  await assert.rejects(h.api.connect('https://warning.example'), /unexpected page/);
  assert.equal(h.api.getBackendUrl(), 'https://legacy.example');
  await assert.rejects(h.api.connect('javascript:alert(1)'), /HTTP/);
});

test('fetch SSE sends the tunnel header and parses split CRLF frames before stopping', async () => {
  let request;
  const source = ': heartbeat\r\n\r\ndata: {"type":"progress",\r\ndata: "status":"rendering","message":"klip 🎬"}\r\n\r\ndata: {"type":"completed","status":"completed","clips":[{"rank":1}]}\r\n\r\n';
  const bytes = new TextEncoder().encode(source);
  const h = createHarness(async (url, options) => {
    request = { url, options };
    return new Response(new ReadableStream({
      start(controller) {
        for (let offset = 0; offset < bytes.length; offset += 7) controller.enqueue(bytes.slice(offset, offset + 7));
        controller.close();
      },
    }), { headers: { 'Content-Type': 'text/event-stream; charset=utf-8' } });
  });
  h.api.setBackendUrl('https://tunnel.example');
  const events = [];
  h.api.createSSE('abc123', event => events.push(event));
  await flush();
  assert.equal(request.options.headers['ngrok-skip-browser-warning'], 'true');
  assert.equal(request.options.headers.Accept, 'text/event-stream');
  assert.equal(events.length, 2);
  assert.equal(events[0].message, 'klip 🎬');
  assert.equal(events[1].clips[0].rank, 1);
  assert.equal(h.timers.size, 0, 'terminal events must not schedule reconnection');
});

test('SSE failure retries and explicit close cancels a scheduled retry', async () => {
  let calls = 0;
  const h = createHarness(async () => { calls++; throw new TypeError('offline'); });
  h.api.setBackendUrl('https://tunnel.example');
  let errors = 0;
  const stream = h.api.createSSE('abc123', () => {}, () => errors++);
  await flush();
  assert.equal(errors, 1);
  const timer = [...h.timers.values()][0];
  assert.equal(timer.delay, 3000);
  h.timers.clear();
  timer.fn();
  await flush();
  assert.equal(calls, 2);
  stream.close();
  assert.equal(h.timers.size, 0);
});

test('terminal progress keeps polling until canonical clips arrive, using the backend output route', async () => {
  const h = createHarness();
  h.api.setBackendUrl('https://tunnel.example');
  let onEvent;
  let closes = 0;
  let resolveFinal;
  let fetches = 0;
  h.api.fetchJob = () => ++fetches === 1
    ? Promise.resolve({ id: 'abc123', status: 'rendering', clips: [], progress: { percent: 95 } })
    : new Promise(resolve => { resolveFinal = resolve; });
  h.api.createSSE = (_id, callback) => { onEvent = callback; return { close() { closes++; } }; };
  h.runPage('job.html');
  await flush();
  assert.equal(h.intervals.size, 1);
  onEvent({ type: 'progress', status: 'completed', progress: { percent: 100 } });
  assert.equal(closes, 0);
  assert.equal(h.intervals.size, 1);
  resolveFinal({
    id: 'abc123', status: 'completed', progress: { percent: 100 },
    clips: [{ rank: 1, title: '<script>untrusted</script>', download_url: '/api/outputs/abc123/clip_1.mp4' }],
  });
  await flush();
  assert.equal(closes, 1);
  assert.equal(h.intervals.size, 0);
  const clips = h.element('clips-grid').innerHTML;
  assert.match(clips, /https:\/\/tunnel\.example\/api\/outputs\/abc123\/clip_1\.mp4/);
  assert.match(clips, /&lt;script&gt;untrusted&lt;\/script&gt;/);
  assert.equal(h.element('clips-section').classList.contains('hidden'), false);
});

test('job and settings pages reload when connected after their initial empty state', async () => {
  for (const page of ['job.html', 'settings.html']) {
    const h = createHarness();
    let fetches = 0;
    h.api.fetchJob = async () => { fetches++; return { id: 'abc123', status: 'completed', clips: [] }; };
    h.api.fetchSettings = async () => { fetches++; return { gpu_available: true }; };
    h.runPage(page);
    await flush();
    assert.equal(fetches, 0);
    h.api.setBackendUrl('https://tunnel.example');
    h.event('studio:connected');
    await flush();
    assert.equal(fetches, 1, `${page} must refresh immediately after connecting`);
    assert.equal(h.element(page === 'job.html' ? 'page-not-found' : 'no-connection-state').classList.contains('hidden'), true);
  }
});

test('Settings uploads only on explicit action and can restore backend-provided cookies', async () => {
  const calls = [];
  const h = createHarness(async (url, options) => {
    calls.push([url, options]);
    return json({ youtube_cookies_set: true, youtube_cookies_uploaded: options.method === 'PUT' });
  });
  h.api.setBackendUrl('https://backend.example');
  h.runPage('settings.html');
  await flush();
  const input = h.element('cookies-file');
  const file = new Blob(['# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tTRUE\t0\tsynthetic\tfake\n']);
  input.files = [file];
  vm.runInContext('selectCookies()', h.context);
  assert.equal(calls.length, 1);
  assert.equal(h.element('cookies-status').getAttribute('data-i18n'), 'cookies.backend');
  await vm.runInContext('changeCookies(false)', h.context);
  const [url, options] = calls[1];
  assert.equal(url, 'https://backend.example/api/settings/youtube-cookies');
  assert.equal(options.method, 'PUT');
  assert.equal(options.body, file);
  assert.equal(options.headers['Content-Type'], 'application/octet-stream');
  assert.equal(options.headers['ngrok-skip-browser-warning'], 'true');
  assert.equal(options.redirect, 'error');
  assert.equal(h.element('cookies-status').getAttribute('data-i18n'), 'cookies.uploaded');
  assert.equal(h.element('cookies-remove').hidden, false);
  await vm.runInContext('changeCookies(true)', h.context);
  assert.equal(calls[2][1].method, 'DELETE');
  assert.equal(h.element('cookies-status').getAttribute('data-i18n'), 'cookies.backend');
  assert.equal(h.element('cookies-remove').hidden, true);
  assert.deepEqual([...h.storage.keys()], ['osc_backend_url']);
});

test('Settings blocks oversize cookies and selections made for another backend', async () => {
  let calls = 0;
  const h = createHarness(async () => {
    calls++;
    return json({ youtube_cookies_set: false, youtube_cookies_uploaded: false });
  });
  h.api.setBackendUrl('https://backend.example');
  h.runPage('settings.html');
  await flush();
  h.element('cookies-file').files = [{ size: 1048577 }];
  vm.runInContext('selectCookies()', h.context);
  await vm.runInContext('changeCookies(false)', h.context);
  assert.equal(h.element('cookies-upload').disabled, true);
  assert.equal(h.element('cookies-message').getAttribute('data-i18n'), 'cookies.invalid');
  h.element('cookies-file').files = [{ size: 60 }];
  vm.runInContext('selectCookies()', h.context);
  h.api.setBackendUrl('https://another.example');
  await vm.runInContext('changeCookies(false)', h.context);
  assert.equal(calls, 1);
});

test('cancellation remains active until the backend reports a terminal state', async () => {
  const h = createHarness();
  h.api.setBackendUrl('https://tunnel.example');
  let job = { id: 'abc123', status: 'rendering', clips: [], cancel_requested: false };
  h.api.fetchJob = async () => job;
  h.api.createSSE = () => ({ close() {} });
  h.api.deleteJob = async () => { job = { ...job, cancel_requested: true }; return { cancel_requested: true }; };
  h.runPage('job.html');
  await flush();
  await vm.runInContext('cancelJob()', h.context);
  assert.equal(h.element('job-badge').textContent, 'Cancellation requested');
  assert.equal(h.element('cancel-job-btn').disabled, true);
  assert.equal(h.intervals.size, 1);
  assert.match(h.element('progress-message').textContent, /may finish before stopping/);
});

test('all Studio page scripts parse', () => {
  for (const page of ['index.html', 'job.html', 'settings.html', 'new-job.html']) {
    const html = fs.readFileSync(path.join(studioPath, page), 'utf8');
    for (const match of html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)) {
      new vm.Script(match[1], { filename: page });
    }
  }
});

test('first readiness probe gets enough time and periodic checks share one request', async () => {
  let finish;
  let requests = 0;
  const h = createHarness(async url => {
    requests++;
    return url.endsWith('/ready') ? new Promise(resolve => { finish = resolve; }) : json({ status: 'ok' });
  });
  h.api.setBackendUrl('https://tunnel.example');
  const first = h.api.checkConnection();
  await flush();
  const second = h.api.checkConnection();
  assert.equal(requests, 2); // One health request, one shared readiness request.
  assert([...h.timers.values()].some(timer => timer.delay === 150000));
  finish(json({ ready: true }));
  await Promise.all([first, second]);
  assert.equal(h.api.getConnectionState().status, 'online');
});

test('late polling response cannot overwrite completed clips', async () => {
  const h = createHarness();
  h.api.setBackendUrl('https://tunnel.example');
  const pending = [];
  let calls = 0;
  let onEvent;
  h.api.fetchJob = () => ++calls === 1
    ? Promise.resolve({ id: 'abc123', status: 'rendering', clips: [] })
    : new Promise(resolve => pending.push(resolve));
  h.api.createSSE = (_id, callback) => { onEvent = callback; return { close() {} }; };
  h.runPage('job.html');
  await flush();
  [...h.intervals.values()][0].fn();
  onEvent({ type: 'completed', status: 'completed' });
  pending[1]({ id: 'abc123', status: 'completed', clips: [{ rank: 1, download_url: '/api/outputs/abc123/clip.mp4' }] });
  await flush();
  pending[0]({ id: 'abc123', status: 'rendering', clips: [] });
  await flush();
  assert.equal(h.element('job-badge').textContent, 'Completed');
  assert.equal(h.element('clips-section').classList.contains('hidden'), false);
});

test('Studio defaults to English regardless of browser language and persists an explicit choice', () => {
  const h = createHarness(undefined, { browserLanguage: 'id-ID' });
  assert.equal(h.i18n.getLanguage(), 'en');
  assert.equal(h.element('html').lang, 'en');
  assert.equal(h.i18n.t('new.heading'), 'New Clipping Job');
  h.i18n.setLanguage('id');
  assert.equal(h.storage.get('osc_studio_language'), 'id');
  assert.equal(h.element('html').lang, 'id');
  const nextPage = createHarness(undefined, { storage: Object.fromEntries(h.storage) });
  assert.equal(nextPage.i18n.getLanguage(), 'id');
  assert.equal(nextPage.i18n.t('new.heading'), 'Job Clipping Baru');
  nextPage.i18n.setLanguage('en');
  assert.equal(nextPage.storage.get('osc_studio_language'), 'en');
});

test('unsupported preferences and unavailable storage still allow a working English default', () => {
  for (const options of [{ storage: { osc_studio_language: 'fr' } }, { blockStorage: true }]) {
    const h = createHarness(undefined, options);
    assert.equal(h.i18n.getLanguage(), 'en');
    h.i18n.setLanguage('id');
    assert.equal(h.i18n.t('settings.save'), 'Simpan Pengaturan');
    h.i18n.setLanguage('unsupported');
    assert.equal(h.i18n.getLanguage(), 'en');
    assert.equal(h.i18n.t('missing.key'), 'missing.key');
  }
});

test('switching language updates text, attributes, dates and selectors without altering form values', () => {
  const h = createHarness();
  const input = h.element('input-url');
  input.value = 'https://youtube.com/watch?v=keep-me';
  input.setAttribute('data-i18n-placeholder', 'new.reuseExample');
  const keyInput = h.element('input-google-key');
  keyInput.value = 'unsaved-test-key';
  const toggle = h.element('toggle-use_broll');
  toggle.checked = false;
  const provider = h.element('input-ai');
  provider.value = 'vllm';
  const selector = h.element('language');
  selector.setAttribute('data-language-select', '');
  const title = h.element('title');
  h.i18n.text(title, 'dashboard.title');
  const help = h.element('help');
  help.setAttribute('data-i18n-title', 'settings.visibility');
  help.setAttribute('data-i18n-aria-label', 'nav.menu');
  const date = h.element('created');
  const timestamp = '2026-11-19T05:30:00Z';
  date.setAttribute('data-i18n-date', timestamp);
  h.i18n.setLanguage('id');
  assert.equal(title.textContent, 'Dasbor — OSC Studio');
  assert.equal(input.getAttribute('placeholder'), 'Contoh: d20b47341e08');
  assert.equal(help.getAttribute('title'), 'Tampilkan/sembunyikan');
  assert.equal(help.getAttribute('aria-label'), 'Buka navigasi');
  assert.equal(date.textContent, new Date(timestamp).toLocaleString('id-ID', {
    day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit',
  }));
  assert.equal(selector.value, 'id');
  h.i18n.setLanguage('en');
  assert.equal(title.textContent, 'Dashboard — OSC Studio');
  assert.equal(date.textContent, new Date(timestamp).toLocaleString('en-US', {
    day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit',
  }));
  assert.equal(input.value, 'https://youtube.com/watch?v=keep-me');
  assert.equal(keyInput.value, 'unsaved-test-key');
  assert.equal(toggle.checked, false);
  assert.equal(provider.value, 'vllm');
});

test('validation and in-flight client errors translate again when the language changes', async () => {
  const h = createHarness();
  h.runPage('new-job.html');
  await vm.runInContext('handleSubmit({ preventDefault() {} })', h.context);
  assert.match(h.element('form-error-text').textContent, /Backend not connected/);
  h.i18n.setLanguage('id');
  assert.match(h.element('form-error-text').textContent, /Backend belum terhubung/);
  h.api.setBackendUrl('https://tunnel.example');
  await vm.runInContext('handleSubmit({ preventDefault() {} })', h.context);
  assert.equal(h.element('form-error-text').textContent, 'URL tidak boleh kosong');
  h.i18n.setLanguage('en');
  assert.equal(h.element('form-error-text').textContent, 'URL cannot be empty');
  let failure;
  try { await h.api.connect('not a URL'); } catch (error) { failure = error; }
  h.i18n.text(h.element('connect-status-msg'), 'connection.failed', { message: h.i18n.errorMessage(failure) });
  h.i18n.setLanguage('id');
  assert.match(h.element('connect-status-msg').textContent, /^❌ Koneksi gagal: Masukkan URL backend/);
});

test('localized template parameters are escaped and server content is preserved', () => {
  const h = createHarness();
  const unsafe = '<img src=x onerror=alert(1)>"&';
  const rendered = h.i18n.html('common.message', { message: unsafe });
  assert(!rendered.includes('<img'));
  assert.match(rendered, /&lt;img/);
  h.i18n.text(h.element('server-message'), 'common.message', { message: unsafe });
  h.i18n.setLanguage('id');
  assert.equal(h.element('server-message').textContent, unsafe);
  h.i18n.text(h.element('model'), 'settings.serverConfig');
  h.i18n.literal(h.element('model'), 'custom/model-alias');
  h.i18n.setLanguage('en');
  assert.equal(h.element('model').textContent, 'custom/model-alias');
});

test('changing language preserves job polling and clip markup while updating cancellation status', async () => {
  const h = createHarness();
  h.api.setBackendUrl('https://tunnel.example');
  let fetches = 0;
  h.api.fetchJob = async () => {
    fetches++;
    return { id: 'abc123', status: 'rendering', cancel_requested: true, clips: [] };
  };
  h.api.createSSE = () => ({ close() {} });
  h.runPage('job.html');
  await flush();
  const timer = [...h.intervals.keys()][0];
  h.i18n.setLanguage('id');
  assert.equal(h.element('job-badge').textContent, 'Pembatalan diminta');
  assert.equal(h.element('cancel-job-btn').disabled, true);
  assert.match(h.element('progress-message').textContent, /Pembatalan diminta/);
  assert.equal(h.intervals.has(timer), true);
  assert.equal(fetches, 1);
  vm.runInContext("renderJob({ id: 'abc123', status: 'completed', clips: [{ rank: 1, download_url: '/api/outputs/abc123/clip.mp4' }] })", h.context);
  const clips = h.element('clips-grid').innerHTML;
  assert.match(clips, /Buka video/);
  h.i18n.setLanguage('en');
  assert.equal(h.element('job-badge').textContent, 'Completed');
  assert.equal(h.element('clips-grid').innerHTML, clips, 'language changes must not replace video elements');
});

test('both catalogs cover every key and placeholder, and all pages load localization before their API client', () => {
  const catalog = JSON.parse(i18nSource.match(/const messages = (\{[\s\S]*?\n\s*\});/)[1]);
  assert.deepEqual(Object.keys(catalog.en).sort(), Object.keys(catalog.id).sort());
  for (const key of Object.keys(catalog.en)) {
    assert(catalog.en[key].trim() && catalog.id[key].trim(), key);
    const placeholders = value => [...value.matchAll(/\{(\w+)\}/g)].map(match => match[1]).sort();
    assert.deepEqual(placeholders(catalog.en[key]), placeholders(catalog.id[key]), key);
  }
  for (const page of ['index.html', 'new-job.html', 'job.html', 'settings.html']) {
    const source = fs.readFileSync(path.join(studioPath, page), 'utf8');
    assert(source.indexOf('src="i18n.js"') < source.indexOf('src="api.js"'), page);
    assert.match(source, /<html lang="en">/);
    assert.match(source, /data-language-select/);
    assert.match(source, /value="en" lang="en">EN · English/);
    assert.match(source, /value="id" lang="id">ID · Bahasa Indonesia/);
    for (const match of source.matchAll(/data-i18n(?:-placeholder|-title|-aria-label|-content)?="([\w.]+)"/g)) {
      assert(Object.hasOwn(catalog.en, match[1]), `${page}: ${match[1]}`);
    }
  }
});
