const assert = require('node:assert/strict');
const { test } = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const studioPath = path.join(__dirname, '..', 'docs', 'studio');
const apiSource = fs.readFileSync(path.join(studioPath, 'api.js'), 'utf8');
const flush = () => new Promise(resolve => setImmediate(resolve));
const json = (body, status = 200) => new Response(JSON.stringify(body), {
  status, headers: { 'Content-Type': 'application/json' },
});

function createHarness(fetchImpl = async () => json({ status: 'ok' })) {
  const elements = new Map();
  const storage = new Map();
  const timers = new Map();
  const intervals = new Map();
  const window = new EventTarget();
  window.location = { search: '?id=abc123', href: '' };
  function element(id) {
    if (!elements.has(id)) {
      const classes = new Set(['hidden']);
      elements.set(id, {
        id, textContent: '', innerHTML: '', value: '', disabled: false, style: {},
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
    confirm: () => true, alert: () => {}, lucide: { createIcons() {} },
    localStorage: {
      getItem: key => storage.get(key) || null,
      setItem: (key, value) => storage.set(key, value),
      removeItem: key => storage.delete(key),
    },
    document: {
      getElementById: element, querySelector: element, createElement: element,
      documentElement: element('html'), body: { appendChild() {} }, addEventListener() {},
    },
    setTimeout(fn, delay) { const timer = {}; timers.set(timer, { fn, delay }); return timer; },
    clearTimeout: timer => timers.delete(timer),
    setInterval(fn, delay) { const timer = {}; intervals.set(timer, { fn, delay }); return timer; },
    clearInterval: timer => intervals.delete(timer),
  });
  vm.runInContext(apiSource, context, { filename: 'api.js' });
  function runPage(filename) {
    const html = fs.readFileSync(path.join(studioPath, filename), 'utf8');
    const scripts = [...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)];
    vm.runInContext(scripts.at(-1)[1], context, { filename });
  }
  return {
    context, storage, timers, intervals, window, element, runPage,
    api: vm.runInContext('StudioAPI', context),
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
