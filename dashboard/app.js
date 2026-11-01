(() => {
'use strict';

const $ = (s) => document.querySelector(s);
const QUOTA = 30;
const HOUR = 3600e3;
const CFG_VERSION = 3;
const DEFAULTS = {
  v: CFG_VERSION,
  url: 'wss://io.adafruit.com:443/mqtt/',
  feedMag: 'accel-mag',
  feedMotion: 'motion-status',
  feedAlarm: 'buzzer-command',
  feedFall: 'fall-detected',
  feedPressure: 'pressure',
  feedAltitude: 'altitude',
  intervalS: 2.5,
  alarmS: 2,
};
const BARO_FEEDS = { pressure: [300, 1100], altitude: [-10000, 10000] };
const FALLBACK_PATHS = ['/mqtt/', '/mqtt', '/'];
const BACKOFF_S = [3, 6, 12, 24, 30];
const PREVIEW_MSG = 'Pratinjau ini tidak diizinkan membuka koneksi MQTT. Buka file index.html langsung di browser (atau dari GitHub Pages) untuk menyambung ke gelang. Mode demo tetap bisa dicoba di sini.';

const store = {
  get(k) { try { return localStorage.getItem('mitifall.' + k); } catch (e) { return null; } },
  set(k, v) { try { localStorage.setItem('mitifall.' + k, v); } catch (e) {  } },
  del(k) { try { localStorage.removeItem('mitifall.' + k); } catch (e) {  } },
};
const inPreview = (() => { try { return window.top !== window.self; } catch (e) { return true; } })();

function clampNum(v, lo, hi, dflt) {
  if (v === null || v === undefined || (typeof v === 'string' && v.trim() === '')) return dflt;
  const n = Number(typeof v === 'string' ? v.replace(',', '.') : v);
  return Number.isFinite(n) ? Math.min(hi, Math.max(lo, n)) : dflt;
}
function sanitizeCfg(raw) {
  const cfg = Object.assign({}, DEFAULTS);
  for (const k of ['url', 'feedMag', 'feedMotion', 'feedAlarm', 'feedFall', 'feedPressure', 'feedAltitude']) {
    const v = raw && typeof raw[k] === 'string' ? raw[k].trim() : '';
    if (v) cfg[k] = v;
  }
  cfg.intervalS = clampNum(raw && raw.intervalS, 0.5, 60, DEFAULTS.intervalS);
  cfg.alarmS = clampNum(raw && raw.alarmS, 0.2, 120, DEFAULTS.alarmS);
  return cfg;
}
function loadCfg() {
  let raw = {};
  try { raw = JSON.parse(store.get('cfg') || '{}') || {}; } catch (e) { raw = {}; }

  if (raw.v !== CFG_VERSION) raw = { url: raw.url };
  return sanitizeCfg(raw);
}

const pad = (n) => String(n).padStart(2, '0');
function fmtTime(t) { const d = new Date(t); return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`; }
function fmtClock(t) { const d = new Date(t); return `${pad(d.getHours())}:${pad(d.getMinutes())}`; }
function fmtNum(v, d) { return Number(v).toFixed(d).replace('.', ','); }

function fmtSigned(v, d) {
  const s = fmtNum(Math.abs(v), d);
  if (Number(Math.abs(v).toFixed(d)) === 0) return s;
  return (v < 0 ? '\u2212' : '+') + s;
}

function parseNum(text) {
  let s = String(text).trim();
  if (s.charAt(0) === '{') {
    try { s = String(JSON.parse(s).value).trim(); } catch (e) { return NaN; }
  }
  return s === '' ? NaN : Number(s.replace(',', '.'));
}
function fmtAgo(ms) {
  const s = Math.max(0, Math.round(ms / 1000));
  if (s < 5) return 'baru saja';
  if (s < 60) return `${s} dtk lalu`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m} mnt lalu`;
  return `${Math.floor(m / 60)} jam lalu`;
}
function isOn(v) { return /^(1|on|true)$/i.test(String(v).trim()); }
function decode(payload) {
  if (typeof payload === 'string') return payload;
  try { return new TextDecoder().decode(payload); } catch (e) { return String(payload); }
}

const S = {
  mode: 'idle',
  broker: 'off',
  brokerText: '',
  cfg: loadCfg(),
  range: [5, 15, 60].map((m) => m * 60e3).includes(Number(store.get('range'))) ? Number(store.get('range')) : 5 * 60e3,
  user: '', key: '', topics: null, url: '',
  points: [],
  motion: [],
  motionNow: null,
  alarmMarks: [],
  alarm: { active: false, until: 0, last: null },
  quota: [],
  pendingOwn: [],
  syncUntil: {},
  baro: { pressure: null, altitude: null },
  baroSeen: false,
  lastMagAt: 0, lastMsgAt: 0, received: 0,
  deviceState: 'off',
  connectedAt: 0, historyLoaded: false, cooldownUntil: 0,
  log: [], logVersion: 0,
};

function resetData() {
  Object.assign(S, {
    points: [], motion: [], motionNow: null, alarmMarks: [],
    alarm: { active: false, until: 0, last: null },
    quota: [], pendingOwn: [], syncUntil: {},
    baro: { pressure: null, altitude: null }, baroSeen: false,
    lastMagAt: 0, lastMsgAt: 0, received: 0, deviceState: 'wait',
    connectedAt: 0, historyLoaded: false, cooldownUntil: 0, url: '',
  });
}
function topicsFor(user, cfg) {
  return {
    mag: `${user}/feeds/${cfg.feedMag}`,
    motion: `${user}/feeds/${cfg.feedMotion}`,
    alarm: `${user}/feeds/${cfg.feedAlarm}`,
    fall: `${user}/feeds/${cfg.feedFall}`,
    pressure: `${user}/feeds/${cfg.feedPressure}`,
    altitude: `${user}/feeds/${cfg.feedAltitude}`,
    throttle: `${user}/throttle`,
    errors: `${user}/errors`,
  };
}

function addLog(text, level = 'info', t = Date.now()) {

  const entry = { t, text, level };
  const i = S.log.findIndex((e) => e.t < t);
  if (i < 0) S.log.push(entry); else S.log.splice(i, 0, entry);
  if (S.log.length > 300) S.log.length = 300;
  S.logVersion++;
  UI.dirty();
}

function quotaUsed(now = Date.now()) {
  const cut = now - 60e3;
  while (S.quota.length && S.quota[0] <= cut) S.quota.shift();
  return S.quota.length;
}
function quotaCount(t = Date.now()) { S.quota.push(t); }
function quotaFreeIn(now = Date.now()) {

  quotaUsed(now);
  return S.quota.length ? Math.max(1, Math.ceil((S.quota[0] + 60e3 - now) / 1000)) : 0;
}

function applyMag(v, t, opts = {}) {
  if (!Number.isFinite(v)) return;
  const pts = S.points;
  if (!pts.length || t >= pts[pts.length - 1].t) pts.push({ t, v });
  else {
    let i = pts.length - 1;
    while (i >= 0 && pts[i].t > t) i--;
    if (i >= 0 && pts[i].t === t) return;
    pts.splice(i + 1, 0, { t, v });
  }
  const cut = Date.now() - HOUR - 5 * 60e3;
  while (pts.length && pts[0].t < cut) pts.shift();
  if (!opts.history) { S.lastMagAt = t; S.received++; }
}

function applyBaro(field, v, t, opts = {}) {
  const [lo, hi] = BARO_FEEDS[field];
  if (!Number.isFinite(v) || v < lo || v > hi) return false;
  const cur = S.baro[field];
  if (opts.history && cur && (cur.live || cur.t >= t)) return true;
  S.baro[field] = { v, t, live: !opts.history };
  if (!opts.history) S.baroSeen = true;
  return true;
}

function magIntervalS() { return S.cfg.intervalS * (S.baroSeen ? 2 : 1); }

function applyMotion(on, t, opts = {}) {
  if (S.motionNow === on) return;
  S.motionNow = on;
  S.motion.push({ t, on });
  if (!opts.silent) addLog(on ? 'Gelang mulai bergerak' : 'Gelang kembali diam', on ? 'move' : 'info', t);
}
function motionAt(t) {
  let on = null;
  for (const e of S.motion) { if (e.t <= t) on = e.on; else break; }
  return on;
}

function srcText(source) {
  switch (source) {
    case 'dashboard': return 'dari dashboard ini';
    case 'device': return 'lewat tombol di gelang';
    case 'device-or-other': return 'dari luar dashboard, biasanya tombol di gelang';
    case 'other': return 'dari luar dashboard ini';
    case 'fall': return 'oleh gelang: jatuh terdeteksi';
    default: return '';
  }
}
function applyAlarm(on, t, source) {
  if (on) {
    S.alarm = { active: true, until: t + S.cfg.alarmS * 1000, last: { t, kind: 'on', source } };
    S.alarmMarks.push({ t, kind: 'on', source });
    addLog(`Alarm dinyalakan ${srcText(source)}`, 'crit', t);
  } else {
    const wasActive = S.alarm.active && t < S.alarm.until;
    S.alarm = { active: false, until: 0, last: { t, kind: 'off', source } };
    S.alarmMarks.push({ t, kind: 'off', source });
    addLog(wasActive ? `Alarm dimatikan ${srcText(source)}`
      : `Perintah matikan ${srcText(source)} (alarm sedang tidak berbunyi)`, 'ok', t);
  }
  UI.dirty();
}

function matchOwn(topic, value, now) {
  const i = S.pendingOwn.findIndex((p) => p.topic === topic && p.value === value && now - p.t < 8000);
  if (i < 0) return false;
  S.pendingOwn.splice(i, 1);
  return true;
}

let client = null;
let gen = 0;
let retryTimer = null;
let retryIdx = 0;
let everConnected = false;
let workingUrl = null;
let cspBlocked = false;

document.addEventListener('securitypolicyviolation', (e) => {
  const d = e.effectiveDirective || e.violatedDirective || '';
  if (/connect-src|default-src/.test(d)) cspBlocked = true;
});

function setBroker(state, text) { S.broker = state; S.brokerText = text || ''; UI.dirty(); }

function candidateUrls(base) {
  let u;
  try { u = new URL(base); } catch (e) { return [base]; }
  const origin = `${u.protocol}//${u.host}`;
  const list = [u.pathname || '/'];
  for (const p of FALLBACK_PATHS) if (!list.includes(p)) list.push(p);
  return list.map((p) => origin + p);
}

function stopClient() {
  gen++;
  clearTimeout(retryTimer);
  retryTimer = null;
  if (client) { try { client.end(true); } catch (e) {  } }
  client = null;
}

function failConnect(msg, pillText) {
  stopClient();
  S.mode = 'idle';
  setBroker('error', pillText || 'MQTT gagal tersambung');
  UI.showConnect(msg);
}

function startLive(user, key) {
  Demo.stop();
  stopClient();
  if (typeof window.mqtt === 'undefined' || !window.mqtt.connect) {
    failConnect('Library MQTT gagal dimuat. Pastikan komputer terhubung ke internet, lalu muat ulang halaman.');
    return;
  }
  if (S.user !== user || S.mode === 'demo') resetData();
  UI.clearBanner('demo');
  S.user = user; S.key = key; S.mode = 'live'; S.deviceState = 'wait';
  S.topics = topicsFor(user, S.cfg);
  retryIdx = 0; everConnected = false; workingUrl = null; cspBlocked = false;
  addLog(`Menghubungkan ke ${S.cfg.url}`, 'info');
  openClient(candidateUrls(S.cfg.url), 0);
}

function openClient(urls, i) {
  const myGen = ++gen;
  const url = urls[i];
  setBroker('connecting', i === 0 ? 'Menghubungkan…' : 'Mencoba alamat lain…');
  let c;
  try {
    c = window.mqtt.connect(url, {
      clientId: 'mitifall-web-' + Math.random().toString(16).slice(2, 10),
      username: S.user,
      password: S.key,
      clean: true,
      keepalive: 30,
      reconnectPeriod: 0,
      connectTimeout: 10000,
      protocolVersion: 4,
      resubscribe: false,
    });
  } catch (err) {
    failConnect('Alamat broker tidak valid: ' + ((err && err.message) || err));
    return;
  }
  client = c;
  let gotConnack = false;
  let authErr = false;

  c.on('connect', () => {
    if (myGen !== gen) { c.end(true); return; }
    gotConnack = true;
    everConnected = true;
    workingUrl = url;
    retryIdx = 0;
    onConnected(c, url);
  });
  c.on('message', (topic, payload) => { if (myGen === gen) onMessage(topic, payload); });
  c.on('error', (err) => {
    if (myGen !== gen) return;
    const msg = String((err && err.message) || err || '');
    if ((err && (err.code === 4 || err.code === 5 || err.code === 134 || err.code === 135))
        || /bad user ?name or password|not authori[sz]ed/i.test(msg)) authErr = true;
  });
  c.on('close', () => {
    if (myGen !== gen || S.mode !== 'live') return;
    if (authErr) {
      addLog('Login ke Adafruit IO ditolak', 'crit');
      failConnect('Username atau AIO Key ditolak Adafruit IO. Cek lagi keduanya — key diawali “aio_” dan username peka huruf besar/kecil.', 'Login ditolak');
      return;
    }
    if (gotConnack) {
      addLog('Koneksi MQTT terputus', 'warn');
      scheduleRetry();
      return;
    }
    if (!everConnected) {
      if (cspBlocked) { failConnect(PREVIEW_MSG, 'Diblokir di pratinjau'); return; }
      if (i + 1 < urls.length) { try { c.end(true); } catch (e) {  } openClient(urls, i + 1); return; }
      addLog('Broker tidak bisa dihubungi', 'crit');
      failConnect('Tidak bisa tersambung ke broker. Cek koneksi internet lalu coba lagi. Beberapa jaringan kampus memblokir WebSocket — coba lewat hotspot HP.');
      return;
    }
    scheduleRetry();
  });
}

function scheduleRetry() {
  clearTimeout(retryTimer);
  const s = BACKOFF_S[Math.min(retryIdx, BACKOFF_S.length - 1)];
  retryIdx++;
  setBroker('retry', `Terputus · coba lagi ${s} dtk`);
  retryTimer = setTimeout(() => {
    if (S.mode === 'live') openClient([workingUrl || S.cfg.url], 0);
  }, s * 1000);
}

function onConnected(c, url) {
  S.url = url;
  S.connectedAt = Date.now();
  if (url !== S.cfg.url) {
    S.cfg.url = url;
    store.set('cfg', JSON.stringify(S.cfg));
  }
  setBroker('on', 'MQTT tersambung');
  addLog(`Tersambung ke Adafruit IO sebagai ${S.user}`, 'ok');
  const T = S.topics;
  c.subscribe([T.mag, T.motion, T.alarm, T.fall, T.pressure, T.altitude, T.throttle, T.errors], { qos: 0 }, (err, granted) => {
    if (err) { addLog('Gagal berlangganan feed: ' + (err.message || err), 'crit'); return; }
    const bad = (granted || []).filter((g) => g.qos === 128).map((g) => g.topic);
    if (bad.length) {
      addLog('Broker menolak langganan: ' + bad.join(', '), 'warn');
      UI.banner('subs', 'warn', `Feed ditolak broker: ${bad.join(', ')}. Cek nama feed dan username di Pengaturan lanjutan.`, 120e3);
    }
  });
  UI.showDash();
  if (!S.historyLoaded) loadHistory();
}

function onMessage(topic, payload) {
  const now = Date.now();
  const text = decode(payload).trim();
  const T = S.topics;
  S.lastMsgAt = now;

  const isBaro = topic === T.pressure || topic === T.altitude;
  const isData = topic === T.mag || topic === T.motion || topic === T.alarm || topic === T.fall || isBaro;
  const isSync = (S.syncUntil[topic] || 0) > now;
  if (isSync) delete S.syncUntil[topic];

  if (isData && !isSync) {
    if (matchOwn(topic, text, now)) return;
    quotaCount(now);
  }

  if (topic === T.mag) {
    applyMag(parseNum(text), now);
  } else if (isBaro) {
    const field = topic === T.pressure ? 'pressure' : 'altitude';
    if (!applyBaro(field, parseNum(text), now)) addLog(`Nilai ${field} dari gelang tidak terbaca: ${text.slice(0, 40)}`, 'warn', now);
  } else if (topic === T.motion) {
    applyMotion(isOn(text), now, { silent: isSync });
  } else if (topic === T.fall) {

    if (isOn(text)) {
      addLog('JATUH TERDETEKSI oleh gelang', 'crit', now);
      UI.banner('fall', 'crit', `Jatuh terdeteksi oleh gelang pukul ${fmtTime(now)}.`, 5 * 60e3);
      applyAlarm(true, now, 'fall');
    }
  } else if (topic === T.alarm) {
    if (!isSync) applyAlarm(isOn(text), now, isOn(text) ? 'other' : 'device-or-other');
  } else if (topic === T.throttle) {
    addLog('Adafruit IO membatasi kiriman: ' + text, 'warn');
    UI.banner('throttle', 'warn', 'Adafruit IO membatasi kiriman akun ini: ' + text, 60e3);
  } else if (topic === T.errors) {
    addLog('Adafruit IO melaporkan galat: ' + text, 'crit');
    UI.banner('ioerror', 'crit', 'Adafruit IO melaporkan galat: ' + text, 60e3);
  }
  UI.dirty();
}

async function getFeed(feed, limit, sinceIso) {
  const base = `https://io.adafruit.com/api/v2/${encodeURIComponent(S.user)}/feeds/${encodeURIComponent(feed)}/data`;
  const url = `${base}?limit=${limit}&start_time=${encodeURIComponent(sinceIso)}&include=value,created_at`;
  const r = await fetch(url, { headers: { 'X-AIO-Key': S.key }, cache: 'no-store' });
  if (!r.ok) throw new Error('HTTP ' + r.status);
  const j = await r.json();
  if (!Array.isArray(j)) throw new Error('format tidak dikenal');
  return j
    .map((d) => ({ t: Date.parse(d.created_at), value: d.value }))
    .filter((d) => Number.isFinite(d.t))
    .sort((a, b) => a.t - b.t);
}

async function loadHistory() {
  S.historyLoaded = true;
  let host = '';
  try { host = new URL(S.url).hostname; } catch (e) {  }
  if (!/(^|\.)adafruit\.com$/i.test(host)) { requestMotionSync(); return; }

  const since = new Date(Date.now() - HOUR).toISOString();
  const [mag, mot, alm, prs, alt] = await Promise.allSettled([
    getFeed(S.cfg.feedMag, 1000, since),
    getFeed(S.cfg.feedMotion, 200, since),
    getFeed(S.cfg.feedAlarm, 100, since),
    getFeed(S.cfg.feedPressure, 5, since),
    getFeed(S.cfg.feedAltitude, 5, since),
  ]);
  if (S.mode !== 'live') return;

  for (const [res, field] of [[prs, 'pressure'], [alt, 'altitude']]) {
    const newest = res.status === 'fulfilled' ? res.value[res.value.length - 1] : null;
    if (newest) applyBaro(field, parseNum(newest.value), newest.t, { history: true });
  }

  if (mag.status === 'fulfilled') {
    const firstLive = S.points.length ? S.points[0].t : Infinity;
    let n = 0;
    for (const d of mag.value) {
      const v = parseFloat(String(d.value).replace(',', '.'));
      if (d.t < firstLive - 500 && Number.isFinite(v)) { applyMag(v, d.t, { history: true }); n++; }
    }
    addLog(n ? `Riwayat 60 menit dimuat: ${n} titik |a|` : 'Belum ada riwayat |a| dalam 60 menit terakhir', 'info');
  } else {
    addLog('Riwayat |a| tidak bisa dimuat dari browser; grafik terisi dari data live', 'info');
  }

  if (mot.status === 'fulfilled' && mot.value.length) {
    const firstLive = S.motion.length ? S.motion[0].t : Infinity;
    const hist = [];
    let prev = null;
    for (const d of mot.value) {
      const on = isOn(d.value);
      if (d.t < firstLive && on !== prev) { hist.push({ t: d.t, on }); prev = on; }
    }
    S.motion = hist.concat(S.motion);
    if (S.motionNow === null && hist.length) S.motionNow = hist[hist.length - 1].on;
  } else if (S.motionNow === null) {
    requestMotionSync();
  }

  if (alm.status === 'fulfilled') {
    for (const d of alm.value) S.alarmMarks.push({ t: d.t, kind: isOn(d.value) ? 'on' : 'off', source: 'history' });
    S.alarmMarks.sort((a, b) => a.t - b.t);
  }
  UI.dirty();
}

function requestMotionSync() {
  if (!client || !client.connected || S.motionNow !== null) return;
  const topic = S.topics.motion;
  S.syncUntil[topic] = Date.now() + 5000;
  client.publish(topic + '/get', '\0', { qos: 0 });
}

function sendAlarm(on) {
  const now = Date.now();
  if (now < S.cooldownUntil) return;
  if (S.mode === 'demo') {
    S.cooldownUntil = now + 1200;
    Demo.command(on);
    return;
  }
  if (S.mode !== 'live' || !client || !client.connected) return;
  if (QUOTA - quotaUsed(now) < 1) {
    addLog('Perintah ditahan: kuota Adafruit IO menit ini penuh', 'warn');
    UI.dirty();
    return;
  }
  const value = on ? '1' : '0';
  S.cooldownUntil = now + 3000;
  client.publish(S.topics.alarm, value, { qos: 0 }, (err) => {
    if (err) {
      S.cooldownUntil = 0;
      addLog('Perintah alarm gagal dikirim: ' + (err.message || err), 'crit');
      return;
    }
    const t = Date.now();
    S.pendingOwn.push({ topic: S.topics.alarm, value, t });
    quotaCount(t);
    applyAlarm(on, t, 'dashboard');
  });
  UI.dirty();
}

function disconnect() {
  const wasDemo = S.mode === 'demo';
  stopClient();
  Demo.stop();
  S.mode = 'idle';
  S.deviceState = 'off';
  setBroker('off', '');
  addLog(wasDemo ? 'Demo mode dihentikan' : 'Koneksi diputus dari dashboard', 'info');
  UI.clearBanner('demo');
  UI.showConnect();
}

function coreTick(now) {
  if (S.mode !== 'idle') {
    const lim = magIntervalS() * 1000 * 3;
    const st = !S.lastMagAt ? 'wait' : (now - S.lastMagAt <= lim ? 'ok' : 'stale');
    if (st !== S.deviceState) {
      if (st === 'stale') addLog('Data dari gelang berhenti masuk', 'warn', now);
      else if (st === 'ok' && S.deviceState === 'stale') addLog('Data dari gelang masuk lagi', 'ok', now);
      else if (st === 'ok' && S.deviceState === 'wait' && S.mode === 'live') addLog('Data pertama dari gelang diterima', 'ok', now);
      S.deviceState = st;
    }
  }
  if (S.alarm.active && now >= S.alarm.until) {
    const t = S.alarm.until;
    S.alarm = { active: false, until: 0, last: { t, kind: 'end', source: S.alarm.last && S.alarm.last.source } };
    addLog(`Alarm selesai sendiri (durasi firmware ${fmtNum(S.cfg.alarmS, S.cfg.alarmS % 1 ? 1 : 0)} dtk)`, 'info', t);
  }
  S.pendingOwn = S.pendingOwn.filter((p) => now - p.t < 8000);
}

const Demo = (() => {
  let timer = null;
  let phase = { moving: false, until: 0 };
  let reported = null;
  let lastPub = -Infinity;
  let slot = 0;
  let baro = null;
  const FLOOR_M = 3.2;
  const CLIMB_M_S = 0.25;

  function sample(moving) {
    if (!moving) return 1 + (Math.random() - 0.5) * 0.014;
    const base = 1.03 + Math.random() * 0.2;
    return base + (Math.random() < 0.08 ? 0.2 + Math.random() * 0.35 : 0);
  }
  function nextPhase(t) {
    phase.moving = !phase.moving;
    phase.until = t + (phase.moving ? 25e3 + Math.random() * 70e3 : 30e3 + Math.random() * 100e3);

    if (phase.moving && Math.random() < 0.35) baro.target = baro.target ? 0 : FLOOR_M;
  }

  function baroInit(t) {
    baro = { t0: t, at: t, pBoot: 1009.2 + Math.random() * 1.5, h: 0, target: 0, arm: 0 };
  }
  function baroStep(t) {
    const dt = Math.max(0, (t - baro.at) / 1000);
    baro.at = t;
    const d = baro.target - baro.h;
    baro.h += Math.sign(d) * Math.min(Math.abs(d), CLIMB_M_S * dt);
    baro.arm = phase.moving ? (Math.random() - 0.5) * 0.5 : 0;
  }
  function baroPressure(t) {
    const weather = baro.pBoot + 0.1 * Math.sin((2 * Math.PI * (t - baro.t0)) / (4 * HOUR));
    const h = baro.h + baro.arm;
    return weather * Math.pow(1 - h / 44330, 1 / 0.1902949) + (Math.random() - 0.5) * 0.024;
  }
  const baroAltitude = (p) => 44330 * (1 - Math.pow(p / baro.pBoot, 0.1902949));
  const round2 = (v) => Math.round(v * 100) / 100;

  function runSlot(t, history) {
    baroStep(t);
    const k = slot++;
    if (k % 2 === 0) {
      applyMag(sample(phase.moving), t, { history });
      return true;
    }
    const feed = (k >> 1) % 3;
    const p = baroPressure(t);
    if (feed === 0) applyBaro('pressure', round2(p), t);
    else if (feed === 1) applyBaro('altitude', round2(baroAltitude(p)), t);
    return feed !== 2;
  }

  function maybeReportMotion(t, silent) {
    if (phase.moving !== reported && (reported === null || t - lastPub >= 15e3)) {
      applyMotion(phase.moving, t, { silent });
      reported = phase.moving;
      lastPub = t;
      return true;
    }
    return false;
  }
  function seed(now) {
    const step = S.cfg.intervalS * 1000;
    let t = now - 22 * 60e3;
    phase = { moving: false, until: t + 80e3 };
    reported = null;
    lastPub = -Infinity;
    slot = 0;
    baroInit(t);
    for (; t < now - step; t += step) {
      if (t >= phase.until) nextPhase(t);

      if (runSlot(t, true) && t > now - 60e3) quotaCount(t);
      maybeReportMotion(t, true);
    }
    const ended = `Alarm selesai sendiri (durasi firmware ${fmtNum(S.cfg.alarmS, S.cfg.alarmS % 1 ? 1 : 0)} dtk)`;
    const a1 = now - 13 * 60e3;
    S.alarmMarks.push({ t: a1, kind: 'on', source: 'dashboard' });
    addLog('Alarm dinyalakan dari dashboard ini', 'crit', a1);
    if (S.cfg.alarmS > 2.6) {
      S.alarmMarks.push({ t: a1 + 2600, kind: 'off', source: 'device' });
      addLog('Alarm dimatikan lewat tombol di gelang', 'ok', a1 + 2600);
    } else {
      addLog(ended, 'info', a1 + S.cfg.alarmS * 1000);
    }
    const a2 = now - 3 * 60e3 - 40e3;
    S.alarmMarks.push({ t: a2, kind: 'on', source: 'dashboard' });
    addLog('Alarm dinyalakan dari dashboard ini', 'crit', a2);
    addLog(ended, 'info', a2 + S.cfg.alarmS * 1000);
  }
  function tick() {
    const now = Date.now();
    if (now >= phase.until) nextPhase(now);
    if (runSlot(now, false)) quotaCount(now);
    S.lastMsgAt = now;
    if (maybeReportMotion(now, false)) quotaCount(now);
    UI.dirty();
  }
  function start() {
    stopClient();
    stop();
    resetData();
    S.mode = 'demo';
    S.user = 'demo';
    S.topics = topicsFor('demo', S.cfg);
    S.connectedAt = Date.now();
    setBroker('demo', 'Demo mode');
    addLog('Demo mode dimulai — data contoh, bukan dari gelang', 'info');
    seed(Date.now());
    tick();
    timer = setInterval(tick, S.cfg.intervalS * 1000);
    UI.banner('demo', 'demo', 'Demo mode: semua angka di halaman ini data contoh, bukan dari gelang.', 0, { label: 'Exit demo', run: disconnect });
    UI.showDash();
  }
  function stop() { clearInterval(timer); timer = null; }
  function command(on) {
    const t = Date.now();
    quotaCount(t);
    applyAlarm(on, t, 'dashboard');
  }
  function pressButton() {
    const t = Date.now();
    if (S.alarm.active && t < S.alarm.until) {
      quotaCount(t);
      applyAlarm(false, t, 'device');
    } else {
      addLog('Tombol gelang ditekan saat alarm tidak berbunyi', 'info', t);
    }
  }
  return { start, stop, command, pressButton };
})();

const UI = (() => {
  const el = {
    pillBroker: $('#pillBroker'), pillDevice: $('#pillDevice'), btnDisconnect: $('#btnDisconnect'),
    banners: $('#banners'), connectView: $('#connectView'), dashView: $('#dashView'),
    form: $('#connectForm'), fUser: $('#fUser'), fKey: $('#fKey'), fRemember: $('#fRemember'),
    btnShowKey: $('#btnShowKey'), fUrl: $('#fUrl'), fFeedMag: $('#fFeedMag'), fFeedMotion: $('#fFeedMotion'),
    fFeedAlarm: $('#fFeedAlarm'), fFeedPressure: $('#fFeedPressure'), fFeedAltitude: $('#fFeedAltitude'),
    fInterval: $('#fInterval'), fAlarmS: $('#fAlarmS'), advBox: $('#advBox'),
    formError: $('#formError'), btnConnect: $('#btnConnect'), btnDemo: $('#btnDemo'),
    kMag: $('#kMag'), kMagSub: $('#kMagSub'), kMotion: $('#kMotion'), kMotionSub: $('#kMotionSub'),
    kAlarmBox: $('#kAlarmBox'), kAlarm: $('#kAlarm'), kAlarmSub: $('#kAlarmSub'), kAlarmNote: $('#kAlarmNote'),
    btnAlarmOn: $('#btnAlarmOn'), btnAlarmOff: $('#btnAlarmOff'), btnDemoPress: $('#btnDemoPress'),
    kQuota: $('#kQuota'), kQuotaMeter: $('#kQuotaMeter'), kQuotaSub: $('#kQuotaSub'),
    kAlt: $('#kAlt'), kAltSub: $('#kAltSub'), kAltNote: $('#kAltNote'),
    rangeSeg: $('#rangeSeg'), btnTable: $('#btnTable'), chartTitle: $('#chartTitle'), chartMeta: $('#chartMeta'),
    chart: $('#chart'), svg: $('#chartSvg'), chartEmpty: $('#chartEmpty'), tip: $('#tip'),
    tableWrap: $('#tableWrap'), tableBody: $('#tableBody'),
    log: $('#log'), btnClearLog: $('#btnClearLog'), infoList: $('#infoList'), btnCsv: $('#btnCsv'),
  };

  let raf = 0;
  let renderedLog = -1;
  let geom = null;
  let hoverT = null;
  const banners = new Map();

  function dirty() {
    if (!raf) raf = requestAnimationFrame(() => { raf = 0; render(); });
  }

  const BANNER_ICON = { demo: '#i-demo', warn: '#i-warn', crit: '#i-crit', info: '#i-info' };
  function banner(id, kind, text, ttlMs, action) {
    banners.set(id, { kind, text, until: ttlMs ? Date.now() + ttlMs : 0, action });
    drawBanners();
  }
  function clearBanner(id) { if (banners.delete(id)) drawBanners(); }
  function drawBanners() {
    el.banners.replaceChildren();
    for (const [id, b] of banners) {
      const div = document.createElement('div');
      div.className = 'banner';
      div.dataset.kind = b.kind;
      div.setAttribute('role', b.kind === 'crit' ? 'alert' : 'status');
      div.innerHTML = `<svg aria-hidden="true"><use href="${BANNER_ICON[b.kind] || '#i-info'}"/></svg>`;
      const span = document.createElement('span');
      span.className = 'grow';
      span.textContent = b.text;
      div.appendChild(span);
      if (b.action) {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'btn small';
        btn.textContent = b.action.label;
        btn.addEventListener('click', b.action.run);
        div.appendChild(btn);
      } else if (b.kind !== 'demo') {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'btn small';
        btn.textContent = 'Tutup';
        btn.addEventListener('click', () => clearBanner(id));
        div.appendChild(btn);
      }
      el.banners.appendChild(div);
    }
  }

  function showConnect(errorMsg) {
    el.connectView.hidden = false;
    el.dashView.hidden = true;
    el.btnDisconnect.hidden = true;
    el.btnConnect.disabled = false;
    el.btnConnect.lastChild.textContent = 'Connect';
    if (errorMsg) {
      el.formError.hidden = false;
      el.formError.querySelector('span').textContent = errorMsg;
    } else {
      el.formError.hidden = true;
    }
    dirty();
  }
  function showDash() {
    el.connectView.hidden = true;
    el.dashView.hidden = false;
    el.btnDisconnect.hidden = S.mode === 'demo';
    el.btnDisconnect.textContent = 'Disconnect';
    el.formError.hidden = true;
    renderedLog = -1;
    dirty();
  }

  function render() {
    const now = Date.now();
    renderPills(now);
    if (el.dashView.hidden) return;
    renderKpis(now);
    renderChart(now);
    if (!el.tableWrap.hidden) renderTable(now);
    if (renderedLog !== S.logVersion) renderLog();
    if (el.infoList) renderInfo(now);
  }

  function setPill(node, state, text) {
    node.dataset.state = state;
    node.lastChild.textContent = text;
  }
  function renderPills(now) {
    const brokerText = {
      off: 'MQTT terputus', connecting: S.brokerText || 'Menghubungkan…', on: 'MQTT tersambung',
      retry: S.brokerText || 'Terputus', error: S.brokerText || 'MQTT gagal', demo: 'Mode demo',
    }[S.broker] || 'MQTT';
    if (el.pillBroker) setPill(el.pillBroker, S.broker === 'retry' ? 'stale' : S.broker, brokerText);
    let dState = 'off';
    let dText = 'Gelang: belum ada data';
    if (S.mode !== 'idle') {
      if (S.deviceState === 'ok') { dState = 'ok'; dText = S.mode === 'demo' ? 'Gelang contoh aktif' : 'Gelang aktif'; }
      else if (S.deviceState === 'stale') { dState = 'stale'; dText = `Data terhenti · ${fmtAgo(now - S.lastMagAt)}`; }
      else { dState = 'connecting'; dText = 'Menunggu gelang'; }
    }
    if (el.pillDevice) setPill(el.pillDevice, dState, dText);
  }

  function useIcon(node, href) {
    const use = node.querySelector('use');
    if (use.getAttribute('href') !== href) use.setAttribute('href', href);
  }

  function renderKpis(now) {
    const iv = magIntervalS();
    const ivTxt = fmtNum(iv, iv % 1 ? 1 : 0);

    const last = S.points[S.points.length - 1];
    if (!last) {
      el.kMag.textContent = '—';
      const waitedLong = S.mode === 'live' && S.broker === 'on' && now - S.connectedAt > 12000;
      el.kMagSub.textContent = waitedLong
        ? 'Belum ada data. Pastikan Serial Monitor gelang menampilkan “# mqtt connected”.'
        : 'Menunggu data dari gelang…';
    } else {
      el.kMag.innerHTML = `${fmtNum(last.v, 3)}<small>g</small>`;
      let peak = null;
      for (let i = S.points.length - 1; i >= 0 && S.points[i].t >= now - S.range; i--) {
        if (peak === null || S.points[i].v > peak) peak = S.points[i].v;
      }
      const rangeMin = Math.round(S.range / 60e3);
      void rangeMin; void ivTxt;
      el.kMagSub.textContent = '';
    }

    const ms = S.motionNow === null ? 'none' : S.motionNow ? 'moving' : 'still';
    if (el.kMotion) {
    el.kMotion.dataset.s = ms;
    useIcon(el.kMotion, ms === 'moving' ? '#i-move' : '#i-still');
    el.kMotion.lastChild.textContent = ms === 'none' ? 'Belum ada data' : ms === 'moving' ? 'Bergerak' : 'Diam';
    const lastEv = S.motion[S.motion.length - 1];
    el.kMotionSub.textContent = lastEv
      ? `Sejak ${fmtTime(lastEv.t)} · ${fmtAgo(now - lastEv.t)}`
      : '';
    }

    const a = S.alarm;
    const active = a.active && now < a.until;
    el.kAlarmBox.dataset.active = String(active);
    el.kAlarm.dataset.s = active ? 'alarm' : 'idle';
    useIcon(el.kAlarm, active ? '#i-crit' : '#i-bell-off');
    el.kAlarm.lastChild.textContent = active ? 'Berbunyi' : 'Siaga';
    let sub = 'Belum ada perintah';
    if (active) sub = `Dinyalakan ${fmtTime(a.last.t)} ${srcText(a.last.source)} · selesai ±${Math.max(1, Math.ceil((a.until - now) / 1000))} dtk lagi`;
    else if (a.last && a.last.kind === 'off') sub = `Dimatikan ${fmtTime(a.last.t)} ${srcText(a.last.source)}`;
    else if (a.last && a.last.kind === 'end') sub = `Terakhir berbunyi sampai ${fmtTime(a.last.t)}`;
    el.kAlarmSub.textContent = sub;

    const canSend = S.mode === 'demo' || (S.mode === 'live' && S.broker === 'on');
    const used = quotaUsed(now);
    const full = S.mode === 'live' && used >= QUOTA;
    const cooling = now < S.cooldownUntil;
    el.btnAlarmOn.disabled = !canSend || full || cooling;
    el.btnAlarmOff.disabled = !canSend || full || cooling;
    let note = '';
    if (!canSend) note = S.mode === 'live' ? 'Menunggu koneksi MQTT untuk mengirim perintah.' : 'Hubungkan dashboard untuk mengirim perintah.';
    else if (full) note = `Kuota menit ini penuh — bisa kirim lagi ±${quotaFreeIn(now)} dtk.`;
    else if (cooling) note = 'Perintah terkirim.';
    el.kAlarmNote.textContent = note;
    el.btnDemoPress.hidden = S.mode !== 'demo';

    const level = used >= QUOTA ? 'crit' : used >= QUOTA - 3 ? 'warn' : 'ok';
    if (el.kQuota) {
    el.kQuota.innerHTML = `${used} <small>/ ${QUOTA} data per menit</small>`;
    el.kQuotaMeter.dataset.level = level;
    el.kQuotaMeter.setAttribute('aria-valuenow', String(used));
    el.kQuotaMeter.firstElementChild.style.width = `${Math.min(100, (used / QUOTA) * 100)}%`;
    if (level === 'ok') {
      el.kQuotaSub.textContent = '';
    } else {
      el.kQuotaSub.innerHTML = '';
      const flag = document.createElement('span');
      flag.className = 'quota-flag';
      flag.dataset.level = level;
      flag.innerHTML = `<svg aria-hidden="true"><use href="${level === 'crit' ? '#i-crit' : '#i-warn'}"/></svg>`;
      flag.appendChild(document.createTextNode(level === 'crit'
        ? 'Penuh: kiriman berikutnya bisa ditolak Adafruit IO.'
        : 'Hampir penuh: perintah dari dashboard bisa ditolak.'));
      el.kQuotaSub.appendChild(flag);
    }
    }

    const { pressure: bp, altitude: ba } = S.baro;
    el.kAlt.innerHTML = ba ? `${fmtSigned(ba.v, 2)}<small>m</small>` : '—';
    el.kAltSub.textContent = `Tekanan: ${bp ? `${fmtNum(bp.v, 1)} hPa` : '—'}`;
    const baroAt = Math.max(ba ? ba.t : 0, bp ? bp.t : 0);
    let baroNote = 'Menunggu data BMP280 (dikirim tiap 15 dtk)…';
    if (baroAt) baroNote = '';
    else if (S.mode === 'live' && S.lastMagAt && now - S.connectedAt > 45e3) {
      baroNote = 'Belum ada data BMP280. Tanpa sensor ini gelang hanya mengirim |a|.';
    }
    el.kAltNote.textContent = baroNote;
  }

  function niceStep(raw) {
    const p = Math.pow(10, Math.floor(Math.log10(raw)));
    const n = raw / p;
    return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 2.5 ? 2.5 : n <= 5 ? 5 : 10) * p;
  }
  function decimalsFor(step) { return step >= 1 ? 0 : step >= 0.1 ? 1 : step >= 0.01 ? 2 : 3; }
  function lowerBound(arr, t) {
    let lo = 0, hi = arr.length;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (arr[mid].t < t) lo = mid + 1; else hi = mid; }
    return lo;
  }
  function motionSegments(x0, x1) {
    const segs = [];
    let start = null;
    for (const e of S.motion) {
      if (e.on && start === null) start = e.t;
      else if (!e.on && start !== null) { segs.push([start, e.t]); start = null; }
    }
    if (start !== null) segs.push([start, x1]);
    return segs.filter(([a, b]) => b > x0 && a < x1).map(([a, b]) => [Math.max(a, x0), Math.min(b, x1)]);
  }
  const r1 = (n) => Math.round(n * 10) / 10;

  function renderChart(now) {
    const W = Math.max(280, Math.round(el.chart.clientWidth));
    const narrow = W < 560;
    const H = narrow ? 236 : 286;
    const m = { l: 42, r: narrow ? 58 : 70, t: 24, b: 50 };
    const pw = W - m.l - m.r;
    const ph = H - m.t - m.b;
    const x1 = now, x0 = now - S.range;
    const step = magIntervalS() * 1000;
    const pts = S.points;
    const vis = [];
    for (let i = lowerBound(pts, x0); i < pts.length && pts[i].t <= x1; i++) vis.push(pts[i]);

    let ymin = 0.9, ymax = 1.1;
    for (const p of vis) { if (p.v < ymin) ymin = p.v; if (p.v > ymax) ymax = p.v; }
    const stepY = niceStep((ymax - ymin) / 4);
    ymin = Math.floor(ymin / stepY + 1e-9) * stepY;
    ymax = Math.ceil(ymax / stepY - 1e-9) * stepY;
    if (ymax - ymin < 2 * stepY) ymax = ymin + 2 * stepY;
    const X = (t) => m.l + ((t - x0) / (x1 - x0)) * pw;
    const Y = (v) => m.t + (1 - (v - ymin) / (ymax - ymin)) * ph;
    const dec = decimalsFor(stepY);

    let s = '';
    for (let v = ymin; v <= ymax + stepY / 2; v += stepY) {
      const y = r1(Y(v));
      s += `<line class="c-grid" x1="${m.l}" x2="${m.l + pw}" y1="${y}" y2="${y}"/>`;
      s += `<text class="axis-text" x="${m.l - 8}" y="${y}" text-anchor="end" dominant-baseline="middle">${fmtNum(v, dec)}</text>`;
    }
    if (1 > ymin && 1 < ymax) {
      const y = r1(Y(1));
      s += `<line class="c-ref" x1="${m.l}" x2="${m.l + pw}" y1="${y}" y2="${y}"/>`;
      s += `<text class="ref-label" x="${m.l + 6}" y="${y - 6}">1 g · diam</text>`;
    }
    const base = m.t + ph;
    s += `<line class="c-axis" x1="${m.l}" x2="${m.l + pw}" y1="${base}" y2="${base}"/>`;

    const tick = S.range <= 5 * 60e3 ? 60e3 : S.range <= 15 * 60e3 ? 3 * 60e3 : 10 * 60e3;
    for (let tt = Math.ceil(x0 / tick) * tick; tt <= x1; tt += tick) {
      const x = r1(X(tt));
      s += `<line class="c-axis" x1="${x}" x2="${x}" y1="${base}" y2="${base + 4}"/>`;
      if (x > m.l + 16 && x < m.l + pw - 12) s += `<text class="axis-text" x="${x}" y="${H - 6}" text-anchor="middle">${fmtClock(tt)}</text>`;
    }

    void motionSegments;

    let lastLabelX = -Infinity;
    for (const a of S.alarmMarks) {
      if (a.kind !== 'on' || a.t < x0 || a.t > x1) continue;
      const x = r1(X(a.t));
      s += `<line class="c-alarm" x1="${x}" x2="${x}" y1="${m.t}" y2="${base}"/>`;
      s += `<circle class="c-alarm-cap" cx="${x}" cy="${m.t}" r="3"/>`;
      if (x - lastLabelX > 46) {
        const flip = x > m.l + pw - 44;
        s += `<text class="alarm-label" x="${flip ? x - 6 : x + 6}" y="${m.t - 8}" text-anchor="${flip ? 'end' : 'start'}">alarm</text>`;
        lastLabelX = x;
      }
    }

    const gap = step * 3;
    const segs = [];
    let cur = [];
    for (const p of vis) {
      if (cur.length && p.t - cur[cur.length - 1].t > gap) { segs.push(cur); cur = []; }
      cur.push(p);
    }
    if (cur.length) segs.push(cur);
    const yBottom = r1(Y(ymin));
    for (const seg of segs) {
      if (seg.length === 1) {
        s += `<circle class="c-pt" cx="${r1(X(seg[0].t))}" cy="${r1(Y(seg[0].v))}" r="2.5"/>`;
        continue;
      }
      let d = '';
      seg.forEach((p, i) => { d += `${i ? 'L' : 'M'}${r1(X(p.t))},${r1(Y(p.v))}`; });
      const area = `${d}L${r1(X(seg[seg.length - 1].t))},${yBottom}L${r1(X(seg[0].t))},${yBottom}Z`;
      s += `<path class="c-area" d="${area}"/><path class="c-line" d="${d}"/>`;
    }

    const last = vis[vis.length - 1];
    if (last && now - last.t <= gap) {
      const cx = r1(X(last.t)), cy = r1(Y(last.v));
      s += `<circle class="c-dot" cx="${cx}" cy="${cy}" r="4"/>`;
      s += `<text class="end-label" x="${cx + 9}" y="${cy}" dominant-baseline="middle">${fmtNum(last.v, 2)} g</text>`;
    }

    s += `<g id="xh" visibility="hidden"><line id="xhLine" class="c-xh" y1="${m.t}" y2="${base}"/><circle id="xhDot" class="c-dot" r="4.5"/></g>`;

    el.svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    el.svg.setAttribute('height', String(H));
    el.svg.innerHTML = s;
    el.chartEmpty.hidden = vis.length > 0;
    const iv = magIntervalS();
    el.chartTitle.textContent = `|a| per jendela ${fmtNum(iv, iv % 1 ? 1 : 0)} detik`;

    geom = { X, Y, vis, m, pw, W, step, x0, x1 };
    if (hoverT !== null) {
      if (hoverT < x0) hideHover(); else showHover(nearestIndex(hoverT), false);
    }
  }

  function nearestIndex(t) {
    if (!geom || !geom.vis.length) return -1;
    const v = geom.vis;
    let i = lowerBound(v, t);
    if (i >= v.length) i = v.length - 1;
    if (i > 0 && Math.abs(v[i - 1].t - t) < Math.abs(v[i].t - t)) i--;
    return i;
  }

  function showHover(i, setT = true) {
    if (!geom || i < 0) { hideHover(); return; }
    const p = geom.vis[i];
    if (setT) hoverT = p.t;
    const x = geom.X(p.t), y = geom.Y(p.v);
    const g = el.svg.querySelector('#xh');
    if (!g) return;
    const line = g.querySelector('#xhLine');
    line.setAttribute('x1', r1(x)); line.setAttribute('x2', r1(x));
    const dot = g.querySelector('#xhDot');
    dot.setAttribute('cx', r1(x)); dot.setAttribute('cy', r1(y));
    g.setAttribute('visibility', 'visible');

    const mv = motionAt(p.t);
    const near = S.alarmMarks.find((a) => a.kind === 'on' && Math.abs(a.t - p.t) <= geom.step);
    el.tip.querySelector('.v').textContent = `${fmtNum(p.v, 3)} g`;
    el.tip.querySelector('.t').textContent = fmtTime(p.t);
    void mv;
    el.tip.querySelector('.x').textContent = near ? 'Alarm dinyalakan' : '';
    el.tip.hidden = false;
    const scale = el.svg.getBoundingClientRect().width / geom.W || 1;
    const tw = el.tip.offsetWidth, th = el.tip.offsetHeight;
    let left = x * scale + 14;
    if (left + tw > el.chart.clientWidth) left = x * scale - tw - 14;
    let top = y * scale - th - 12;
    if (top < 0) top = y * scale + 14;
    el.tip.style.transform = `translate(${Math.max(0, Math.round(left))}px, ${Math.round(top)}px)`;
  }
  function hideHover() {
    hoverT = null;
    el.tip.hidden = true;
    const g = el.svg.querySelector('#xh');
    if (g) g.setAttribute('visibility', 'hidden');
  }
  function pointerToIndex(e) {
    if (!geom) return -1;
    const r = el.svg.getBoundingClientRect();
    const x = (e.clientX - r.left) * (geom.W / r.width);
    const t = geom.x0 + ((x - geom.m.l) / geom.pw) * (geom.x1 - geom.x0);
    return nearestIndex(t);
  }

  function renderTable(now) {
    const rows = [];
    for (let i = S.points.length - 1; i >= 0 && rows.length < 240; i--) {
      const p = S.points[i];
      if (p.t < now - S.range) break;
      rows.push(p);
    }
    el.tableBody.replaceChildren(...rows.map((p) => {
      const tr = document.createElement('tr');
      const mv = motionAt(p.t);
      void mv;
      for (const [txt, cls] of [[fmtTime(p.t), ''], [fmtNum(p.v, 3), 'num']]) {
        const td = document.createElement('td');
        td.textContent = txt;
        if (cls) td.className = cls;
        tr.appendChild(td);
      }
      return tr;
    }));
    if (!rows.length) {
      const tr = document.createElement('tr');
      const td = document.createElement('td');
      td.colSpan = 2;
      td.textContent = 'Belum ada data di rentang ini.';
      tr.appendChild(td);
      el.tableBody.replaceChildren(tr);
    }
  }

  const LOG_ICON = { info: '#i-info', ok: '#i-ok', move: '#i-move', warn: '#i-warn', crit: '#i-crit' };
  function renderLog() {
    renderedLog = S.logVersion;
    if (!S.log.length) {
      const li = document.createElement('li');
      li.className = 'empty-log';
      li.textContent = 'Belum ada kejadian.';
      el.log.replaceChildren(li);
      return;
    }
    el.log.replaceChildren(...S.log.slice(0, 150).map((e) => {
      const li = document.createElement('li');
      li.dataset.level = e.level;
      li.innerHTML = `<svg aria-hidden="true"><use href="${LOG_ICON[e.level] || '#i-info'}"/></svg>`;
      const time = document.createElement('time');
      time.dateTime = new Date(e.t).toISOString();
      time.textContent = fmtTime(e.t);
      const span = document.createElement('span');
      span.textContent = e.text;
      li.append(time, span);
      return li;
    }));
  }

  function renderInfo(now) {
    const T = S.topics || topicsFor(S.user || '{username}', S.cfg);
    const iv = magIntervalS();
    const ivTxt = fmtNum(iv, iv % 1 ? 1 : 0);
    const rows = [
      ['Broker', S.mode === 'demo' ? 'Mode demo (tanpa broker)' : (S.url || S.cfg.url)],
      ['Akun', S.mode === 'demo' ? '—' : S.user],
      ['Feed |a|', T.mag, true],
      ['Feed gerak', T.motion, true],
      ['Feed alarm', T.alarm, true],
      ['Feed jatuh', T.fall, true],
      ['Feed tekanan', T.pressure, true],
      ['Feed ketinggian', T.altitude, true],
      ['Interval kirim', `${ivTxt} dtk per titik |a|` + (S.baroSeen ? ' (bergantian dengan BMP280)' : '')],
      ['Titik diterima', `${S.received} (sesi ini)`],
      ['Pesan terakhir', S.lastMsgAt ? `${fmtTime(S.lastMsgAt)} · ${fmtAgo(now - S.lastMsgAt)}` : '—'],
    ];
    const key = rows.map((r) => r.join('|')).join('/');
    if (el.infoList.dataset.key === key) return;
    el.infoList.dataset.key = key;
    el.infoList.replaceChildren(...rows.flatMap(([k, v, mono]) => {
      const dt = document.createElement('dt');
      dt.textContent = k;
      const dd = document.createElement('dd');
      dd.textContent = v;
      if (mono) dd.className = 'mono';
      return [dt, dd];
    }));
  }

  function downloadCsv() {
    const lines = ['waktu_iso,waktu_lokal,accel_mag_g,gerak'];
    for (const p of S.points) {
      const mv = motionAt(p.t);
      const d = new Date(p.t);
      const local = `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${fmtTime(p.t)}`;
      lines.push(`${d.toISOString()},${local},${p.v.toFixed(4)},${mv === null ? '' : mv ? 1 : 0}`);
    }
    const blob = new Blob([lines.join('\n') + '\n'], { type: 'text/csv' });
    const a = document.createElement('a');
    const d = new Date();
    a.href = URL.createObjectURL(blob);
    a.download = `mitifall-${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  }

  function fillForm() {
    el.fUser.value = store.get('user') || '';
    const savedKey = store.get('key');
    el.fKey.value = savedKey || '';
    el.fRemember.checked = !!savedKey;
  }

  function readCfgFromForm() {
    store.set('cfg', JSON.stringify(S.cfg));
  }

  function init() {
    fillForm();
    const rangeMin = String(Math.round(S.range / 60e3));
    for (const b of el.rangeSeg.querySelectorAll('button')) b.setAttribute('aria-checked', String(b.dataset.min === rangeMin));

    el.form.addEventListener('submit', (e) => {
      e.preventDefault();
      const user = el.fUser.value.trim();
      const key = el.fKey.value.trim();
      if (!user || !key) {
        showConnect(!user ? 'Isi username Adafruit IO dulu.' : 'Isi AIO Key dulu.');
        (user ? el.fKey : el.fUser).focus();
        return;
      }
      readCfgFromForm();
      store.set('user', user);
      if (el.fRemember.checked) store.set('key', key); else store.del('key');
      el.formError.hidden = true;
      el.btnConnect.disabled = true;
      el.btnConnect.lastChild.textContent = 'Connecting…';
      startLive(user, key);
    });
    el.btnShowKey.addEventListener('click', () => {
      const show = el.fKey.type === 'password';
      el.fKey.type = show ? 'text' : 'password';
      el.btnShowKey.textContent = show ? 'Hide' : 'Show';
      el.btnShowKey.setAttribute('aria-pressed', String(show));
    });
    el.btnDemo.addEventListener('click', () => { readCfgFromForm(); Demo.start(); });
    el.btnDisconnect.addEventListener('click', disconnect);
    el.btnAlarmOn.addEventListener('click', () => sendAlarm(true));
    el.btnAlarmOff.addEventListener('click', () => sendAlarm(false));
    el.btnDemoPress.addEventListener('click', () => Demo.pressButton());
    el.btnClearLog.addEventListener('click', () => { S.log = []; S.logVersion++; dirty(); });
    if (el.btnCsv) {
      el.btnCsv.addEventListener('click', downloadCsv);
      if (inPreview) el.btnCsv.hidden = true;
    }

    el.rangeSeg.addEventListener('click', (e) => {
      const b = e.target.closest('button[data-min]');
      if (!b) return;
      S.range = Number(b.dataset.min) * 60e3;
      store.set('range', String(S.range));
      for (const x of el.rangeSeg.querySelectorAll('button')) x.setAttribute('aria-checked', String(x === b));
      hideHover();
      dirty();
    });
    el.btnTable.addEventListener('click', () => {
      const show = el.tableWrap.hidden;
      el.tableWrap.hidden = !show;
      el.btnTable.setAttribute('aria-pressed', String(show));
      el.btnTable.textContent = show ? 'Hide table' : 'Table';
      dirty();
    });

    el.chart.addEventListener('pointermove', (e) => showHover(pointerToIndex(e)));
    el.chart.addEventListener('pointerdown', (e) => showHover(pointerToIndex(e)));
    el.chart.addEventListener('pointerleave', (e) => { if (e.pointerType === 'mouse') hideHover(); });
    el.chart.addEventListener('blur', hideHover);
    el.chart.addEventListener('keydown', (e) => {
      if (!geom || !geom.vis.length) return;
      let i = hoverT === null ? geom.vis.length - 1 : nearestIndex(hoverT);
      if (e.key === 'ArrowLeft') i = Math.max(0, i - 1);
      else if (e.key === 'ArrowRight') i = Math.min(geom.vis.length - 1, i + 1);
      else if (e.key === 'Home') i = 0;
      else if (e.key === 'End') i = geom.vis.length - 1;
      else if (e.key === 'Escape') { hideHover(); return; }
      else return;
      e.preventDefault();
      showHover(i);
    });
    el.chart.addEventListener('focus', () => { if (geom && geom.vis.length && hoverT === null) showHover(geom.vis.length - 1); });

    if (window.ResizeObserver) new ResizeObserver(() => dirty()).observe(el.chart);
    else window.addEventListener('resize', dirty);

    setInterval(() => {
      const now = Date.now();
      coreTick(now);
      for (const [id, b] of banners) if (b.until && now > b.until) banners.delete(id) && drawBanners();
      dirty();
    }, 1000);

    if (inPreview) {
      banner('preview', 'info', 'Halaman ini terbuka di dalam frame pratinjau, jadi mode demo dimulai otomatis. Untuk menyambung ke gelang, buka index.html langsung di browser atau dari GitHub Pages.', 0);
      Demo.start();
    } else if (el.fUser.value && el.fKey.value && store.get('key')) {
      showConnect();
      el.btnConnect.disabled = true;
      el.btnConnect.lastChild.textContent = 'Connecting…';
      startLive(el.fUser.value.trim(), el.fKey.value.trim());
    } else {
      showConnect();
    }
    render();
  }

  return { dirty, banner, clearBanner, showConnect, showDash, init };
})();

UI.init();
})();
