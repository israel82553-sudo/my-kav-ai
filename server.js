const express = require('express');
const crypto = require('crypto');
const dns = require('dns').promises;
const net = require('net');
const fs = require('fs');
const os = require('os');
const path = require('path');
const execFileP = require('util').promisify(require('child_process').execFile);
// ffmpeg comes from the ffmpeg-static package (no system install needed); without it only mp3/wav files can be played
const FFMPEG = process.env.FFMPEG_PATH || (() => { try { return require('ffmpeg-static'); } catch (e) { return null; } })();
const app = express();
app.use(express.urlencoded({ extended: true }));

// two separate key pools: GEMINI_API_KEYS for transcription (audio understanding), GEMINI_TTS_KEYS for read-aloud
const keyList = v => String(v || '').split(',').map(s => s.trim()).filter(Boolean);
const KEYS = keyList(process.env.GEMINI_API_KEYS);
const TTS_KEYS = keyList(process.env.GEMINI_TTS_KEYS);
// web search through Groq (browser_search tool of the gpt-oss models); several keys can be given separated by commas
const GROQ_KEYS = keyList(process.env.GROQ_API_KEYS);
const GROQ_BASE = process.env.GROQ_BASE || 'https://api.groq.com/openai/v1/chat/completions';
const GROQ_MODEL = process.env.GROQ_SEARCH_MODEL || 'openai/gpt-oss-120b';
const HAS_SEARCH = GROQ_KEYS.length > 0;
// Jamendo: free music API (full songs, free license); register at developer.jamendo.com for a client id
const JAMENDO_ID = String(process.env.JAMENDO_CLIENT_ID || '').trim();
const JAMENDO_BASE = process.env.JAMENDO_BASE || 'https://api.jamendo.com/v3.0';
// songs and podcasts can be searched via Apple Podcasts, Jamendo, or Groq web search
const HAS_FOUND = true;
const TEXT_MODEL = String(process.env.TEXT_MODEL || '').trim() || 'gemini-3.6-flash';
// backup text model, used when the main one is overloaded (503/429/5xx); set TEXT_MODEL_FALLBACK to change it or to "none" to disable
const TEXT_FALLBACK_RAW = String(process.env.TEXT_MODEL_FALLBACK || '').trim() || 'gemini-3.6-flash';
const TEXT_FALLBACK = TEXT_FALLBACK_RAW === 'none' || TEXT_FALLBACK_RAW === TEXT_MODEL ? '' : TEXT_FALLBACK_RAW;
// TTS_MODEL can hold several models separated by commas: when one runs out of quota the next one is used
const TTS_LIST = String(process.env.TTS_MODEL || 'gemini-3.8-flash-tts').split(',').map(x => x.trim()).filter(Boolean);
const TTS_MODEL = TTS_LIST[0];
const VOICE = process.env.VOICE || 'Kore';
const EXT = (process.env.EXT_PATH || '1').replace(/^\/+|\/+$/g, '');
const TOKEN = process.env.YEMOT_TOKEN ||
  (process.env.YEMOT_API_USERNAME && process.env.YEMOT_API_PASSWORD ? process.env.YEMOT_API_USERNAME + ':' + process.env.YEMOT_API_PASSWORD : '');
const API_SECRET = String(process.env.API_SECRET || '').trim();
const MAX_TURNS_PER_HOUR = Math.max(0, parseInt(process.env.MAX_TURNS_PER_HOUR, 10) || 40);

const safeEq = (a, b) => {
  const x = Buffer.from(String(a)), y = Buffer.from(String(b));
  return x.length === y.length && crypto.timingSafeEqual(x, y);
};

// 1. Secret check for /yemot, /history, /selftest
function checkApiSecret(req) {
  if (!API_SECRET) return true;
  const k = String((req.query && req.query.k) || (req.body && req.body.k) || req.get('x-api-key') || '');
  return safeEq(k, API_SECRET);
}

// 3. Rate limiting per caller phone number (turns per hour)
const rateMap = new Map();
setInterval(() => {
  const cutoff = Date.now() - 3600000;
  for (const [ph, arr] of rateMap.entries()) {
    const fresh = arr.filter(t => t > cutoff);
    if (fresh.length) rateMap.set(ph, fresh);
    else rateMap.delete(ph);
  }
}, 600000).unref();

function checkRateLimit(ph) {
  if (!MAX_TURNS_PER_HOUR || !ph) return true;
  const now = Date.now();
  const cutoff = now - 3600000;
  const arr = (rateMap.get(ph) || []).filter(t => t > cutoff);
  if (arr.length >= MAX_TURNS_PER_HOUR) return false;
  arr.push(now);
  rateMap.set(ph, arr);
  return true;
}

const YM = process.env.YEMOT_BASE || 'https://www.call2all.co.il/ym/api/';
const GBASE = process.env.GEMINI_BASE || 'https://generativelanguage.googleapis.com/v1beta/models/';
const BUDGET = Number(process.env.TURN_BUDGET_MS) || 80000;
const REC_MAX = String(parseInt(process.env.REC_MAX, 10) || 30);

// Yemot cannot read these characters inside t- text, and , = break the read= syntax
const sp = s => String(s).replace(/[.\-"'&|,=]/g, ' ').replace(/\s+/g, ' ').trim();

// the user can change the voice or pace by speaking; the model signals it with a tag that the server strips before read-aloud
const VOICE_CTL = ' אם המשתמש מבקש לשנות את הקול, את קצב הדיבור או את אורך התשובות, התחל את התשובה בתג מתאים ואחריו אשר בקצרה במשפט אחד. התגים: [[VOICE:male]] לקול גברי, [[VOICE:female]] לקול נשי, [[VOICE:default]] לקול הרגיל, [[PACE:slow]] לדיבור לאט, [[PACE:fast]] לדיבור מהר, [[PACE:normal]] לקצב רגיל, [[LEN:short]] כשהמשתמש מבקש תשובות קצרות יותר, למשל: קצר את התשובה, תקצר, תענה קצר, בקצרה, בקיצור, [[LEN:long]] כשהוא מבקש לחזור לתשובות מפורטות או להרחיב. בקשה כזאת מתייחסת תמיד לאורך התשובות שלך ולא לחפץ או לנושא כלשהו, גם כשהיא מופיעה באותו משפט עם שאלה, ואז ענה על השאלה בקצרה. בקשת המשתמש לגבי אורך התשובה גוברת על כל הנחיית אורך אחרת, ולכן כבר בתשובה הזאת עצמה ענה לפי הבקשה החדשה, ואם יש בבקשה גם שאלה ענה עליה. אפשר לשלב כמה תגים. זה היוצא מן הכלל היחיד לכלל הסימנים המיוחדים. השתמש בתגים רק כשהמשתמש ביקש במפורש שינוי כזה, ולעולם אל תזכיר אותם בטקסט אחר.';
const SEARCH_HINT = ' יש לך כלי חיפוש באינטרנט בשם web_search. הפעל אותו בכל פעם שהמשתמש מבקש לחפש באינטרנט או באתר מסוים, או כשהוא שואל על חדשות, אירועים עדכניים, מזג אוויר, תוצאות ספורט, מחירים או כל מידע שדורש נתונים מהרשת. אם הוא ביקש אתר מסוים, העבר את שמו בשדה site. אחרי החיפוש ענה בצורה ברורה ומפורטת בלי להקריא כתובות אתרים.';
const FOUND_HINT = ' יש לך כלי search_media לחיפוש פודקאסטים, פרקים, שירים וקבצי שמע שאפשר להאזין להם, וכלי play_found להשמעת אחת התוצאות. אם המשתמש ביקש לחפש פודקאסט או שיר, הפעל search_media עם שאילתה מתאימה (לפודקאסטים העבר type="podcast"). אחרי שקיבלת את התוצאות הקרא אותן בקצרה, כל אחת עם מספר במילים ואחריו השם, ושאל איזו לנגן. כשהמשתמש מבקש לנגן אחת מהן, לפי מספר או לפי שם, הפעל play_found עם המספר, ואז אמור משפט קצר אחד שאתה מנגן עכשיו. אל תקריא כתובות.';
// audio or video files from the internet that the caller can ask to play ("נגן את הקובץ"); the owner lists them in MEDIA_FILES: [{"n":"name","u":"https://.../file.mp3"}]
let MEDIA = [];
try { MEDIA = JSON.parse(process.env.MEDIA_FILES || '[]'); } catch (e) { MEDIA = []; }
MEDIA = (Array.isArray(MEDIA) ? MEDIA : []).map(m => ({ n: String((m && m.n) || '').replace(/\s+/g, ' ').trim().slice(0, 80), u: String((m && m.u) || '').trim() })).filter(m => /^https?:\/\//i.test(m.u)).slice(0, 30);
MEDIA.forEach((m, i) => { if (!m.n) m.n = 'קובץ ' + (i + 1); });
const HAS_MEDIA = MEDIA.length > 0;
// 6. Aligned defaults with README (20 min, 100 MB) to stay safe within Render memory and MEDIA_PREP_MS
const MEDIA_MAX_SEC = (parseInt(process.env.MEDIA_MAX_MIN, 10) || 20) * 60;
const MEDIA_MAX_BYTES = (parseInt(process.env.MEDIA_MAX_MB, 10) || 100) * 1048576;
const MEDIA_PREP_MS = Number(process.env.MEDIA_PREP_MS) || 100000;
const MEDIA_HINT = ' יש לך קבצי שמע שאפשר להשמיע למתקשר. הפעל את הכלי play_media אך ורק אם המשתמש ביקש במפורש לנגן או להשמיע קובץ, והעבר את מספר הקובץ מהרשימה. אם יש רק קובץ אחד והמשתמש ביקש לנגן את הקובץ, הפעל אותו. הקבצים: ' + MEDIA.map((m, i) => (i + 1) + ' ' + m.n).join(', ') + '. אחרי שהפעלת את הכלי ענה במשפט קצר אחד בלבד שאתה מנגן עכשיו את הקובץ, בלי להקריא כתובות.';
// history is kept as text only, so a recorded question needs a transcript: the model adds it as a tag that the server strips
const TRANSCRIBE_HINT = ' אם השאלה הגיעה כהקלטה קולית, התחל את התשובה בתג [[Q:תמלול מדויק של השאלה במילים]] ואחריו התשובה עצמה. לשאלה שהוקלדה אל תוסיף את התג. לעולם אל תזכיר את התג בטקסט אחר.';
// detailed answers like Gemini: whatever MAX_WORDS says, the limit is never lower than 500 words
const MAXW = Math.min(1000, Math.max(500, parseInt(process.env.MAX_WORDS, 10) || 700));
const EXTRA = String(process.env.EXTRA_INSTRUCTION || '').replace(/\s+/g, ' ').trim().slice(0, 2000);
// 4. Open assistant instructions with minimal safety & accuracy floor
const RULES = `
אתה עוזר קולי חכם, פתוח, ידידותי ומקצועי בטלפון.
אתה עונה באופן חופשי, מקיף וענייני על כל שאלה ובכל נושא שהמשתמש שואל — כולל חדשות, אקטואליה, תרבות, מוזיקה, ספורט, מדע, היסטוריה, טכנולוגיה, יהדות, הלכה, בריאות וכל נושא אחר, ללא חסימות תוכן.
במקרה של מצוקה נפשית קשה, סכנה מיידית או מצב חירום רפואי דחוף, הפנה מיד לקבלת עזרה מקצועית או לחיוג למוקדי החירום 101 למגן דוד אדום או 100 למשטרה.
לעולם אל תמציא עובדות, נתונים או מקורות: אם אינך יודע או אינך בטוח בתשובה, אמור זאת בפירוש ובכנות ("אני לא בטוח בכך") והצע לחפש באינטרנט אם תתבקש.
אתה מדבר בשפה ברורה, נעימה, מכובדת וזורמת המתאימה להאזנה בטלפון.
`.replace(/\s+/g, ' ').trim();
const buildSys = (mode, extra) => 'אתה עוזר קולי בטלפון. ' + RULES +
  ' ענה בעברית כמו ג׳מיני: תשובה ארוכה, עשירה ומפורטת, זורמת ורכה לאוזן, בשפה טבעית של דיבור ובפסקאות קצרות. הרחב, הסבר את הרקע, תן דוגמאות, פרטים והקשר, והשלם את התמונה בלי למהר ובלי לקצר. אורך התשובה הוא עד ' + MAXW + ' מילים. רק לברכה או למשפט שיחתי קצר ענה בקצרה. בלי רשימות, בלי כוכביות ובלי סימנים מיוחדים, כך שיהיה נעים להקשיב לה בטלפון. המשתמש יכול לעצור את התשובה בכל רגע בהקשה על מקש ולהמשיך את השיחה, לכן שאלות המשך מתייחסות לתשובה הקודמת שלך. ' +
  (EXTRA ? 'הנחיה כללית מהבעלים: ' + EXTRA + ' ' : '') +
  (extra ? 'הנחיה לשלוחה הזאת מהבעלים: ' + extra + ' ' : '') + VOICE_CTL + TRANSCRIBE_HINT + (HAS_SEARCH ? SEARCH_HINT : '') + (HAS_FOUND ? FOUND_HINT : '') + (HAS_MEDIA ? MEDIA_HINT : '');

// one entry per Yemot extension: its own character mode, extra instruction and opening message
let CFG = [];
try { CFG = JSON.parse(process.env.EXTENSIONS || '[]'); } catch (e) { CFG = []; }
if (!Array.isArray(CFG) || !CFG.length) CFG = [{ p: EXT, m: process.env.MODE || '2', x: '', g: process.env.GREETING || '' }];
// one opening for both ways to ask: typing with the phone keys, or recording
const DEFAULT_GREET = 'שלום וברכה';
const EXTS = {};
for (const c of CFG) {
  const p = String(c.p || '').replace(/^\/+|\/+$/g, '');
  if (!/^[0-9]+(\/[0-9]+)*$/.test(p)) continue;
  EXTS[p.replace(/\//g, '_')] = {
    p, d: '/' + p,
    sys: buildSys(c.m, String(c.x || '').replace(/\s+/g, ' ').trim().slice(0, 1000)),
    g: sp(String(c.g || '').slice(0, 300)) || sp(DEFAULT_GREET)
  };
}
const EXT_FIRST = EXTS[Object.keys(EXTS)[0]] || { p: EXT, d: '/' + EXT, sys: buildSys('2', ''), g: sp(DEFAULT_GREET) };
const AGAIN = sp('לא שמעתי שאל שוב');
const ERR = sp('הייתה תקלה נסו שוב');

const EXTP = '/' + EXT;
const yp = (name, d) => 'ivr2:' + (d === undefined ? EXTP : d) + '/' + name + (/\.[a-z0-9]+$/.test(name) ? '' : '.wav');
const sleep = ms => new Promise(r => setTimeout(r, ms));
const isBudget = e => !!(e && e.budget);
const rot = { text: 0, tts: 0 };
const RETRY_STATUS = [429, 500, 502, 503, 504];
const calls = {};

// drop state of calls that never sent hangup
setInterval(() => {
  const now = Date.now();
  for (const id of Object.keys(calls)) if (now - calls[id].t > 3600000) delete calls[id];
}, 600000).unref();

// fetch with a timeout that never exceeds the remaining time of the current turn
async function http(url, opt, ms, deadline) {
  const left = deadline ? deadline - Date.now() : ms;
  if (left < 800) { const e = new Error('turn budget exhausted'); e.budget = true; throw e; }
  return fetch(url, { ...opt, signal: AbortSignal.timeout(Math.min(ms, left)) });
}

// 2. SSRF Protection: block private/internal IPs and validate redirects manually
function isPrivateIp(ip) {
  if (!ip) return true;
  if (ip.startsWith('::ffff:')) ip = ip.slice(7);
  if (net.isIPv4(ip)) {
    const p = ip.split('.').map(Number);
    if (p[0] === 0 || p[0] === 10 || p[0] === 127) return true;
    if (p[0] === 169 && p[1] === 254) return true; // link-local / cloud metadata
    if (p[0] === 172 && p[1] >= 16 && p[1] <= 31) return true;
    if (p[0] === 192 && p[1] === 168) return true;
    if (p[0] === 100 && p[1] >= 64 && p[1] <= 127) return true; // CGNAT
    if (p[0] >= 224) return true; // multicast / reserved
    return false;
  }
  if (net.isIPv6(ip)) {
    const l = ip.toLowerCase();
    if (l === '::' || l === '::1') return true;
    if (l.startsWith('fc') || l.startsWith('fd') || l.startsWith('fe80')) return true;
    return false;
  }
  return true;
}

async function assertSafeUrl(rawUrl) {
  let u;
  try { u = new URL(rawUrl); } catch (e) { throw new Error('invalid url'); }
  if (u.protocol !== 'http:' && u.protocol !== 'https:') throw new Error('blocked protocol');
  const host = u.hostname.replace(/^\[|\]$/g, '').toLowerCase();
  if (!host || host === 'localhost' || host.endsWith('.local') || host.endsWith('.internal')) throw new Error('blocked host');
  if (net.isIP(host)) {
    if (isPrivateIp(host)) throw new Error('blocked private ip');
    return u.toString();
  }
  const addrs = await dns.lookup(host, { all: true });
  if (!addrs || !addrs.length) throw new Error('dns lookup empty');
  for (const a of addrs) {
    if (isPrivateIp(a.address)) throw new Error('blocked resolved private ip');
  }
  return u.toString();
}

async function safeFetch(rawUrl, opt, ms, deadline, maxRedirects = 4, streamMode = false) {
  let cur = await assertSafeUrl(rawUrl);
  for (let i = 0; i <= maxRedirects; i++) {
    let r;
    if (streamMode) {
      const ctrl = new AbortController();
      const hdrTimer = setTimeout(() => ctrl.abort(new Error('header timeout')), ms || 30000);
      if (hdrTimer.unref) hdrTimer.unref();
      try {
        r = await fetch(cur, { ...opt, redirect: 'manual', signal: ctrl.signal });
      } finally {
        clearTimeout(hdrTimer);
      }
    } else {
      r = await http(cur, { ...opt, redirect: 'manual' }, ms, deadline);
    }
    if (r.status >= 300 && r.status < 400) {
      const loc = r.headers.get('location');
      if (!loc) throw new Error('redirect without location');
      cur = await assertSafeUrl(new URL(loc, cur).toString());
      continue;
    }
    return r;
  }
  throw new Error('too many redirects');
}

// opt.retries: extra tries beyond one pass over the keys (default 1); opt.ms: timeout of a single try
async function gem(model, body, deadline, pool, opt) {
  pool = pool || 'text';
  opt = opt || {};
  const list = pool === 'tts' ? TTS_KEYS : KEYS;
  if (!list.length) throw new Error(pool === 'tts' ? 'GEMINI_TTS_KEYS is empty' : 'GEMINI_API_KEYS is empty');
  const tries = list.length + (opt.retries === undefined ? 1 : opt.retries);
  let err = model + ' failed';
  let status = 0;
  for (let i = 0; i < tries; i++) {
    const key = list[rot[pool]++ % list.length];
    try {
      const r = await http(GBASE + model + ':generateContent', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'x-goog-api-key': key },
        body: JSON.stringify(body)
      }, opt.ms || 25000, deadline);
      if (r.ok) return await r.json();
      status = r.status;
      err = model + ' ' + r.status + ' ' + (await r.text()).slice(0, 200);
      if (!RETRY_STATUS.includes(r.status)) break;
    } catch (e) {
      if (isBudget(e)) throw e;
      status = 0;
      err = model + ' ' + (e.name === 'TimeoutError' ? 'timeout' : e.message);
    }
    if (i < tries - 1) await sleep(400 * (i + 1));
  }
  const e = new Error(err);
  e.status = status;
  throw e;
}

// fastest thinking level each model accepts; remembered per model after the first call
const THINK = ['minimal', 'low', null];
const thinkIdx = {};
// a model that just failed is skipped for a while, so every turn does not first wait for it to time out again
const downUntil = {};
const DOWN_MS = 120000;
async function textGen(body, deadline) {
  const models = TEXT_FALLBACK ? [TEXT_MODEL, TEXT_FALLBACK] : [TEXT_MODEL];
  let lastErr;
  for (let m = 0; m < models.length; m++) {
    const model = models[m];
    const hasNext = m < models.length - 1;
    if (hasNext && (downUntil[model] || 0) > Date.now()) continue;
    for (;;) {
      const lvl = THINK[thinkIdx[model] || 0];
      const gc = { maxOutputTokens: 8192 };
      if (lvl) gc.thinkingConfig = { thinkingLevel: lvl };
      try {
        // 503 means the model itself is overloaded, so more keys will not help: with a backup model, give up on this one quickly
        return await gem(model, { ...body, generationConfig: gc }, deadline, 'text', hasNext ? { retries: 0, ms: 30000 } : { ms: 40000 });
      } catch (e) {
        lastErr = e;
        if (e.status === 400 && /think/i.test(e.message) && (thinkIdx[model] || 0) < THINK.length - 1) { thinkIdx[model] = (thinkIdx[model] || 0) + 1; continue; }
        break;
      }
    }
    // only overload, quota and network errors are worth a different model
    if (isBudget(lastErr) || !(RETRY_STATUS.includes(lastErr.status) || lastErr.status === 0)) throw lastErr;
    if (hasNext) { downUntil[model] = Date.now() + DOWN_MS; console.error('text model', model, 'unavailable (' + (lastErr.message || '').slice(0, 60) + '), skipping it for 2 minutes and using', models[m + 1]); }
  }
  throw lastErr;
}

// Groq: the gpt-oss models search the web on the server side and answer with what they found
let grot = 0;
async function groqSearch(query, site, deadline) {
  let last = 'groq search failed';
  for (let i = 0; i < GROQ_KEYS.length; i++) {
    const key = GROQ_KEYS[grot++ % GROQ_KEYS.length];
    try {
      const r = await http(GROQ_BASE, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key },
        body: JSON.stringify({
          model: GROQ_MODEL,
          messages: [
            { role: 'system', content: 'Search the web and answer in up to 150 words, in plain text, with accurate facts. No links and no citations.' },
            { role: 'user', content: site ? 'Look at the website "' + site + '" (open it directly if you can). Request: ' + (query || 'the current main headlines') : query }
          ],
          tools: [{ type: 'browser_search' }],
          tool_choice: 'required',
          reasoning_effort: 'low',
          max_completion_tokens: 900,
          stream: false
        })
      }, 15000, deadline);
      if (r.ok) {
        const j = await r.json();
        const m = ((j.choices || [])[0] || {}).message || {};
        // drop citation markers such as 【2†L6-L10】
        const text = String(m.content || '').replace(/【[^】]*】/g, '').replace(/\s+/g, ' ').trim();
        if (text) return { answer: text.slice(-1500), results: [] };
        last = 'groq empty answer';
        continue;
      }
      last = 'groq ' + r.status + ' ' + (await r.text()).slice(0, 150);
      if (r.status === 400 || r.status === 401) break;
    } catch (e) {
      if (isBudget(e)) throw e;
      last = 'groq ' + (e.name === 'TimeoutError' ? 'timeout' : e.message);
    }
  }
  console.error(last);
  return { error: 'search unavailable' };
}

async function webSearch(query, site, deadline) {
  query = String(query || '').trim().slice(0, 300);
  site = String(site || '').trim().slice(0, 100);
  if (!query && !site) return { error: 'empty query' };
  if (!GROQ_KEYS.length) return { error: 'search unavailable' };
  return groqSearch(query, site, deadline);
}

const SEARCH_DECL = {
  name: 'web_search',
  description: 'Search the web for current news, information, facts, or a specific website. Returns a summary answer.',
  parameters: { type: 'OBJECT', properties: { query: { type: 'STRING', description: 'A short search query' }, site: { type: 'STRING', description: 'Optional: the website the user asked to look at, as spoken, such as N12 or ynet' } }, required: ['query'] }
};

const MEDIA_DECL = {
  name: 'play_media',
  description: 'Play an audio or video file from the list to the caller. Call ONLY when the user explicitly asked to play a file.',
  parameters: { type: 'OBJECT', properties: { file: { type: 'INTEGER', description: 'The number of the file in the list' } }, required: ['file'] }
};

// downloads the file safely (with SSRF checks), converts and slices into phone wav segments (8 kHz mono) so playback starts immediately
const MEDIA_SEG_SEC = Math.min(300, Math.max(30, parseInt(process.env.MEDIA_SEG_SEC, 10) || 120)); // 2-minute chunks by default
const MEDIA_MAX_SEGS = Math.min(90, Math.max(1, parseInt(process.env.MEDIA_MAX_SEGS, 10) || 45)); // up to 90 minutes total
const MEDIA_CACHE_TTL_MS = 2 * 3600 * 1000; // keep cached job metadata for 2 hours max
const mediaJobs = {}; // key -> { parts: [], done: false, err: null, t: number, firstReady: Promise }

// Periodically prune expired mediaJobs from memory
setInterval(() => {
  const cutoff = Date.now() - MEDIA_CACHE_TTL_MS;
  for (const k of Object.keys(mediaJobs)) {
    if (mediaJobs[k].done && (mediaJobs[k].t || 0) < cutoff) delete mediaJobs[k];
  }
}, 900000).unref();

async function prepareMedia(m, d, dl0) {
  const base = 'mf' + crypto.createHash('sha1').update(m.u).digest('hex').slice(0, 12);
  const key = d + '/' + base;
  const cached = mediaJobs[key];
  if (cached && !cached.err && (Date.now() - (cached.t || 0) < MEDIA_CACHE_TTL_MS)) {
    await cached.firstReady;
    return cached;
  }

  let resolveFirst, rejectFirst;
  const firstReady = new Promise((res, rej) => { resolveFirst = res; rejectFirst = rej; });
  const job = { key, d, base, parts: [], done: false, err: null, t: Date.now(), firstReady };
  mediaJobs[key] = job;

  (async () => {
    // Allow up to 15 minutes for full background streaming/slicing of long podcasts
    const streamDl = Math.max(dl0 || 0, Date.now() + 900000);
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), base + '-'));
    const inFile = path.join(dir, 'in.bin');
    let ff = null;
    try {
      // 30s timeout applies ONLY to initial HTTP headers, NOT the full body stream
      const r = await safeFetch(m.u, {}, 30000, streamDl, 4, true);
      if (!r.ok) throw new Error('media download ' + r.status);
      if (Number(r.headers.get('content-length') || 0) > MEDIA_MAX_BYTES) throw new Error('media file too large');

      // Fallback when ffmpeg is unavailable: single-file upload
      if (!FFMPEG) {
        const out = fs.createWriteStream(inFile);
        const reader = r.body.getReader();
        let got = 0;
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          got += value.length;
          if (got > MEDIA_MAX_BYTES) { reader.cancel().catch(() => {}); throw new Error('media file too large'); }
          if (!out.write(value)) await new Promise(res => out.once('drain', res));
        }
        await new Promise((res, rej) => out.end(e => (e ? rej(e) : res())));
        const buf = fs.readFileSync(inFile);
        await upload(base, buf, streamDl, d, 60000);
        job.parts.push(base);
        job.done = true;
        job.t = Date.now();
        resolveFirst(job);
        return;
      }

      // Stream download directly into ffmpeg while slicing into MEDIA_SEG_SEC wav segments in real time
      const { spawn } = require('child_process');
      const maxDur = MEDIA_SEG_SEC * MEDIA_MAX_SEGS;
      ff = spawn(FFMPEG, [
        '-y', '-v', 'error',
        '-protocol_whitelist', 'file,pipe',
        '-i', 'pipe:0',
        '-vn', '-t', String(maxDur),
        '-ar', '8000', '-ac', '1', '-c:a', 'pcm_s16le',
        '-f', 'segment', '-segment_time', String(MEDIA_SEG_SEC), '-reset_timestamps', '1',
        path.join(dir, 'p%03d.wav')
      ], { stdio: ['pipe', 'ignore', 'pipe'] });

      let ffErr = '';
      ff.stderr.on('data', c => { if (ffErr.length < 300) ffErr += c.toString(); });
      ff.stdin.on('error', () => {});
      const ffExit = new Promise((res, rej) => {
        ff.on('error', rej);
        ff.on('close', code => (code === 0 || job.parts.length > 0 ? res() : rej(new Error('ffmpeg exit ' + code + ' ' + ffErr))));
      });

      // Pump HTTP stream into ffmpeg stdin with explicit error capture
      let pumpErr = null;
      const pump = (async () => {
        const reader = r.body.getReader();
        let got = 0;
        try {
          for (;;) {
            if (Date.now() > streamDl) { reader.cancel().catch(() => {}); throw new Error('stream deadline exceeded'); }
            const { done, value } = await reader.read();
            if (done) break;
            got += value.length;
            if (got > MEDIA_MAX_BYTES) {
              reader.cancel().catch(() => {});
              throw new Error('media stream exceeded max bytes');
            }
            if (!ff.stdin.destroyed && !ff.stdin.write(value)) {
              await new Promise(res => ff.stdin.once('drain', res));
            }
          }
        } catch (e) {
          pumpErr = e;
        } finally {
          if (!ff.stdin.destroyed) ff.stdin.end();
        }
      })();

      // Upload each completed segment as soon as the next segment starts (or ffmpeg finishes)
      let nextIdx = 0;
      let ffFinished = false;
      ffExit.then(() => { ffFinished = true; }, () => { ffFinished = true; });

      while (nextIdx < MEDIA_MAX_SEGS) {
        const curName = 'p' + String(nextIdx).padStart(3, '0') + '.wav';
        const nextName = 'p' + String(nextIdx + 1).padStart(3, '0') + '.wav';
        const curPath = path.join(dir, curName);
        const nextPath = path.join(dir, nextName);

        const readyToUpload = fs.existsSync(curPath) && (fs.existsSync(nextPath) || ffFinished);
        if (!readyToUpload) {
          if (ffFinished) break;
          if (Date.now() > streamDl) throw new Error('media prep deadline exceeded');
          await sleep(300);
          continue;
        }

        const segName = base + 'x' + String(nextIdx).padStart(3, '0');
        const buf = fs.readFileSync(curPath);
        fs.rmSync(curPath, { force: true }); // free disk and RAM immediately
        await upload(segName, buf, streamDl, d, 30000);
        job.parts.push(segName);
        if (nextIdx === 0) resolveFirst(job);
        nextIdx++;
      }

      await Promise.allSettled([pump, ffExit]);
      if (pumpErr && !job.parts.length) throw pumpErr;
      if (!job.parts.length) throw new Error('no audio segments produced ' + ffErr);
      job.done = true;
      job.t = Date.now();
      resolveFirst(job);
    } catch (e) {
      job.err = e;
      job.done = true;
      job.t = Date.now();
      rejectFirst(e);
    } finally {
      try { if (ff && !ff.killed) ff.kill('SIGKILL'); } catch (e) {}
      fs.rmSync(dir, { recursive: true, force: true });
    }
  })();

  await firstReady;
  return job;
}

// ---------- podcasts (Apple Podcasts / iTunes API + RSS) and songs found on the web ----------
const FOUND_SEARCH_DECL = {
  name: 'search_media',
  description: 'Search for podcasts, podcast episodes, songs, or audio files the user can listen to. Returns a numbered list of titles.',
  parameters: {
    type: 'OBJECT',
    properties: {
      query: { type: 'STRING', description: 'Search query (podcast name, topic, or song name)' },
      type: { type: 'STRING', description: 'podcast for podcasts/episodes, music for songs (default: podcast if user mentions podcast/episode/show, otherwise auto)' },
      sort: { type: 'STRING', description: 'new for newest, popular for most popular, or relevance (default)' }
    },
    required: ['query']
  }
};
const FOUND_PLAY_DECL = {
  name: 'play_found',
  description: 'Play one of the results of the last search_media to the caller. Call ONLY when the user asked to play one of them.',
  parameters: { type: 'OBJECT', properties: { number: { type: 'INTEGER', description: 'The number of the result in the list' } }, required: ['number'] }
};

// 8. Decode named and numeric (decimal & hex) XML/HTML entities
function decodeXml(str) {
  return String(str || '')
    .replace(/<!\[CDATA\[([\s\S]*?)\]\]>/g, '$1')
    .replace(/&#x([0-9a-fA-F]+);/g, (_, hex) => {
      const cp = parseInt(hex, 16);
      return cp > 0 && cp <= 0x10ffff ? String.fromCodePoint(cp) : '';
    })
    .replace(/&#(\d+);/g, (_, dec) => {
      const cp = parseInt(dec, 10);
      return cp > 0 && cp <= 0x10ffff ? String.fromCodePoint(cp) : '';
    })
    .replace(/&amp;/g, '&')
    .replace(/&quot;/g, '"')
    .replace(/&#39;|&apos;/g, "'")
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&nbsp;/g, ' ')
    .replace(/<[^>]+>/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function parseRssEpisodes(xml, showTitle, maxItems) {
  const items = [];
  const itemRe = /<item\b[\s\S]*?<\/item>/gi;
  let m;
  while ((m = itemRe.exec(xml)) && items.length < (maxItems || 6)) {
    const block = m[0];
    const encMatch = /<enclosure\b[^>]*\burl=["']([^"']+)["']/i.exec(block);
    if (!encMatch) continue;
    const epUrl = encMatch[1].replace(/&amp;/g, '&').trim();
    if (!/^https?:\/\//i.test(epUrl)) continue;
    const titleMatch = /<title\b[^>]*>([\s\S]*?)<\/title>/i.exec(block);
    const epTitle = titleMatch ? decodeXml(titleMatch[1]) : 'פרק ' + (items.length + 1);
    const fullTitle = (showTitle ? showTitle + ' - ' + epTitle : epTitle).slice(0, 120);
    items.push({ title: fullTitle, url: epUrl, src: 'podcast' });
  }
  return items;
}

// 7. Read only up to maxBytes (1.5 MB) of an RSS stream since latest episodes are at the top
async function readTextLimited(response, maxBytes = 1572864) {
  if (!response.body) return '';
  const reader = response.body.getReader();
  const dec = new TextDecoder('utf-8');
  let got = 0;
  let out = '';
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    got += value.length;
    out += dec.decode(value, { stream: true });
    if (got >= maxBytes) {
      reader.cancel().catch(() => {});
      break;
    }
  }
  out += dec.decode();
  return out;
}

async function podcastSearch(query, deadline) {
  query = String(query || '').trim();
  if (!query) return [];
  const countries = ['IL', 'US'];
  let feeds = [];
  for (const country of countries) {
    const u = new URL('https://itunes.apple.com/search');
    u.searchParams.set('media', 'podcast');
    u.searchParams.set('entity', 'podcast');
    u.searchParams.set('country', country);
    u.searchParams.set('limit', '4');
    u.searchParams.set('term', query);
    const r = await http(u.toString(), {}, 10000, deadline);
    if (!r.ok) continue;
    const j = await r.json().catch(() => ({}));
    feeds = (j.results || []).filter(x => x && x.feedUrl);
    if (feeds.length) break;
  }
  if (!feeds.length) return [];

  const episodes = [];
  await Promise.all(feeds.slice(0, 3).map(async f => {
    try {
      const r = await safeFetch(f.feedUrl, { headers: { 'User-Agent': 'Mozilla/5.0 PodcastPhoneBot/1.0' } }, 10000, deadline, 3);
      if (!r.ok) return;
      const xml = await readTextLimited(r, 1572864);
      const showName = decodeXml(String(f.collectionName || f.trackName || '')).trim();
      const parsed = parseRssEpisodes(xml, showName, feeds.length === 1 ? 6 : 2);
      episodes.push(...parsed);
    } catch (e) {
      console.error('rss fetch failed:', String(e.message).slice(0, 80));
    }
  }));
  return episodes.slice(0, 6);
}

async function groqText(system, user, deadline, ms) {
  let last = 'groq failed';
  for (let i = 0; i < GROQ_KEYS.length; i++) {
    const key = GROQ_KEYS[grot++ % GROQ_KEYS.length];
    try {
      const r = await http(GROQ_BASE, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key },
        body: JSON.stringify({
          model: GROQ_MODEL,
          messages: [{ role: 'system', content: system }, { role: 'user', content: user }],
          tools: [{ type: 'browser_search' }],
          tool_choice: 'required',
          reasoning_effort: 'low',
          max_completion_tokens: 1500,
          stream: false
        })
      }, ms || 30000, deadline);
      if (r.ok) {
        const j = await r.json();
        const m = ((j.choices || [])[0] || {}).message || {};
        const text = String(m.content || '').replace(/【[^】]*】/g, '').trim();
        if (text) return text;
        last = 'groq empty answer';
        continue;
      }
      last = 'groq ' + r.status + ' ' + (await r.text()).slice(0, 150);
      if (r.status === 400 || r.status === 401) break;
    } catch (e) {
      if (isBudget(e)) throw e;
      last = 'groq ' + (e.name === 'TimeoutError' ? 'timeout' : e.message);
    }
  }
  throw new Error(last);
}
const parseList = t => {
  const m = /\[[\s\S]*\]/.exec(String(t || ''));
  if (!m) return [];
  try { const a = JSON.parse(m[0]); return Array.isArray(a) ? a : []; } catch (e) { return []; }
};
const MEDIA_JSON_SYS = 'Search the web. Reply with ONLY a JSON array and no other text. Each item is an object {"title": string, "url": string}. ';
async function jamendoSearch(s, query, sort, deadline) {
  const u = new URL(JAMENDO_BASE + '/tracks/');
  u.searchParams.set('client_id', JAMENDO_ID);
  u.searchParams.set('format', 'json');
  u.searchParams.set('limit', '6');
  u.searchParams.set('audioformat', 'mp32');
  if (query) u.searchParams.set('search', query);
  const so = String(sort || '').toLowerCase();
  if (so === 'new') u.searchParams.set('order', 'releasedate_desc');
  else if (so === 'popular') u.searchParams.set('order', 'popularity_week');
  const r = await http(u.toString(), {}, 15000, deadline);
  const j = await r.json().catch(() => ({}));
  if (!r.ok || !j.headers || j.headers.status !== 'success') throw new Error('jamendo ' + r.status + ' ' + JSON.stringify(j.headers || {}).slice(0, 120));
  return (j.results || []).filter(x => x && x.audio).map(x => ({ title: (String(x.name || '') + (x.artist_name ? ' - ' + x.artist_name : '')).replace(/\s+/g, ' ').trim().slice(0, 100), url: String(x.audio), src: 'jamendo' })).filter(x => x.title);
}
async function searchMedia(s, query, deadline, sort, mediaType) {
  query = String(query || '').trim().slice(0, 200);
  const mType = String(mediaType || '').toLowerCase();
  const isPodcastReq = mType === 'podcast' || /פודקאסט|פרק|תוכנית|הסכת|podcast/i.test(query);

  if (isPodcastReq || (!JAMENDO_ID && /[א-ת]/.test(query))) {
    try {
      const cleanQ = query.replace(/פודקאסט|הסכת|podcast/gi, '').trim() || query;
      const pods = await podcastSearch(cleanQ, deadline);
      if (pods.length) {
        s.found = pods;
        return { results: s.found.map((x, i) => ({ number: i + 1, title: x.title })) };
      }
    } catch (e) {
      if (isBudget(e)) throw e;
      console.error('podcast search failed:', String(e.message).slice(0, 150));
    }
  }

  if (JAMENDO_ID && !isPodcastReq) {
    try {
      s.found = await jamendoSearch(s, query, sort, deadline);
      if (s.found.length) return { results: s.found.map((x, i) => ({ number: i + 1, title: x.title })) };
      if (!GROQ_KEYS.length) return { error: 'nothing found' };
    } catch (e) {
      if (isBudget(e)) throw e;
      console.error('jamendo search failed:', String(e.message).slice(0, 150));
      if (!GROQ_KEYS.length) return { error: 'search unavailable' };
    }
  }

  if (!isPodcastReq && query) {
    try {
      const pods = await podcastSearch(query, deadline);
      if (pods.length) {
        s.found = pods;
        return { results: s.found.map((x, i) => ({ number: i + 1, title: x.title })) };
      }
    } catch (e) {
      if (isBudget(e)) throw e;
    }
  }

  if (!query || !GROQ_KEYS.length) return { error: 'nothing found' };
  try {
    const t = await groqText(MEDIA_JSON_SYS + 'Return up to 6 results for the request. title = the song, podcast episode or file name (with the artist/show when known), in its original language. url = a direct link to an audio or video file (mp3, m4a, mp4, wav) when you found one, otherwise an empty string.', query, deadline, 35000);
    const items = parseList(t).map(x => ({ title: String((x && x.title) || '').replace(/\s+/g, ' ').trim().slice(0, 100), url: /^https?:\/\//i.test(String((x && x.url) || '')) ? String(x.url).trim() : '' })).filter(x => x.title);
    const seen = new Set();
    s.found = items.filter(x => !seen.has(x.title) && seen.add(x.title)).slice(0, 6);
    if (!s.found.length) return { error: 'nothing found' };
    return { results: s.found.map((x, i) => ({ number: i + 1, title: x.title })) };
  } catch (e) {
    if (isBudget(e)) throw e;
    console.error('media search failed:', String(e.message).slice(0, 150));
    return { error: 'search unavailable' };
  }
}
async function findMediaUrls(title, deadline) {
  const t = await groqText(MEDIA_JSON_SYS + 'Find direct links to an audio or video file (mp3, m4a, mp4, wav) of the requested item. Return up to 4 items, title = the file name, url = the direct link to the file itself and not to a web page.', title, deadline, 35000);
  return parseList(t).map(x => String((x && x.url) || '').trim()).filter(u => /^https?:\/\//i.test(u)).slice(0, 4);
}
async function prepareFound(item, d) {
  const dl = Date.now() + MEDIA_PREP_MS + 40000;
  const tried = new Set();
  const attempt = async url => { tried.add(url); return prepareMedia({ n: item.title, u: url }, d, dl); };
  if (item.url) { try { return await attempt(item.url); } catch (e) { if (isBudget(e)) throw e; console.error('found file failed:', String(e.message).slice(0, 120)); } }
  if (item.src === 'jamendo' || item.src === 'podcast' || !GROQ_KEYS.length) throw new Error('no playable file found');
  for (const u of await findMediaUrls(item.title, dl)) {
    if (tried.has(u)) continue;
    try { return await attempt(u); } catch (e) { if (isBudget(e)) throw e; console.error('found file failed:', String(e.message).slice(0, 120)); }
  }
  throw new Error('no playable file found');
}
function startFound(s, n) {
  const item = ((s && s.found) || [])[Number(n) - 1];
  if (!item) return { error: 'no such result' };
  s.mediaJob = null;
  s.mediaIdx = 0;
  s.mediaP = prepareFound(item, s.d).catch(e => { console.error('media failed:', String(e.message).slice(0, 150)); return { error: e.message }; });
  return { status: 'the file will be played right after your reply', title: item.title };
}

function startMedia(s, idx) {
  const m = MEDIA[Number(idx) - 1];
  if (!m || !s) return { error: 'no such file' };
  s.mediaJob = null;
  s.mediaIdx = 0;
  s.mediaP = prepareMedia(m, s.d).catch(e => { console.error('media failed:', String(e.message).slice(0, 150)); return { error: e.message }; });
  return { status: 'the file will be played right after your reply', file: m.n };
}
async function takeMedia(s, deadline) {
  if (!s.mediaP) return '';
  const left = deadline ? Math.max(3000, deadline - Date.now()) : 45000;
  const r = await Promise.race([
    s.mediaP,
    sleep(left).then(() => ({ error: 'media first segment timeout' }))
  ]);
  s.mediaP = null;
  if (r && r.parts && r.parts.length) {
    s.mediaJob = r;
    s.mediaIdx = 1;
    return 'f-' + s.d + '/' + r.parts[0];
  }
  return 't-' + sp('לא הצלחתי להכין את הקובץ');
}
async function nextMediaSegment(s) {
  const job = s.mediaJob;
  if (!job) return '';
  while (s.mediaIdx >= job.parts.length && !job.done) {
    await sleep(350);
  }
  if (s.mediaIdx < job.parts.length) {
    const seg = job.parts[s.mediaIdx++];
    return 'f-' + s.d + '/' + seg;
  }
  s.mediaJob = null;
  s.mediaIdx = 0;
  return '';
}

// 5. Privacy-safe logging (no user query/search text printed to logs)
async function askModel(sys, contents, deadline, s) {
  const base = { systemInstruction: { parts: [{ text: sys }] } };
  const decls = [];
  if (HAS_SEARCH) decls.push(SEARCH_DECL);
  if (HAS_FOUND) decls.push(FOUND_SEARCH_DECL, FOUND_PLAY_DECL);
  if (HAS_MEDIA) decls.push(MEDIA_DECL);
  if (!decls.length) return textGen({ ...base, contents }, deadline);
  const tools = [{ functionDeclarations: decls }];
  for (let i = 0; i < 3; i++) {
    const a = await textGen({ ...base, contents, tools }, deadline);
    const cand = (a.candidates || [])[0] || {};
    const fcalls = ((cand.content || {}).parts || []).filter(p => p.functionCall);
    if (!fcalls.length) return a;
    const resp = [];
    for (const c of fcalls) {
      const args = c.functionCall.args || {};
      let out;
      if (c.functionCall.name === 'play_media') {
        console.log('tool: play_media', Number(args.file) || 0);
        out = startMedia(s, args.file);
      } else if (c.functionCall.name === 'search_media') {
        console.log('tool: search_media', args.type || 'auto', 'len=' + String(args.query || '').length);
        out = await searchMedia(s, args.query, deadline, args.sort, args.type);
      } else if (c.functionCall.name === 'play_found') {
        console.log('tool: play_found', Number(args.number) || 0);
        out = startFound(s, args.number);
      } else {
        console.log('tool: web_search len=' + String(args.query || '').length);
        if (s) { s.searching = true; if (s.onSearch) s.onSearch(); }
        out = await webSearch(args.query, args.site, deadline);
      }
      resp.push({ functionResponse: { name: c.functionCall.name, response: out } });
    }
    contents = [...contents, cand.content, { role: 'user', parts: resp }];
  }
  return textGen({ ...base, contents }, deadline);
}

function wav(pcm, rate) {
  const h = Buffer.alloc(44);
  h.write('RIFF'); h.writeUInt32LE(36 + pcm.length, 4); h.write('WAVEfmt ', 8);
  h.writeUInt32LE(16, 16); h.writeUInt16LE(1, 20); h.writeUInt16LE(1, 22);
  h.writeUInt32LE(rate, 24); h.writeUInt32LE(rate * 2, 28); h.writeUInt16LE(2, 32); h.writeUInt16LE(16, 34);
  h.write('data', 36); h.writeUInt32LE(pcm.length, 40);
  return Buffer.concat([h, pcm]);
}

const isNewTts = m => /^gemini-(3\.[8-9]|[4-9])/.test(m);
const voicesFor = (model, name) => isNewTts(model)
  ? [{ voice: name }, { prebuiltVoiceConfig: { voiceName: name } }]
  : [{ prebuiltVoiceConfig: { voiceName: name } }, { voice: name }];
const vIdx = {};
const ttsDown = {};

const VOICE_BY = { male: process.env.VOICE_MALE || 'Charon', female: process.env.VOICE_FEMALE || process.env.VOICE || 'Kore' };
const SHORT_NOTE = ' הנחיה מהמשתמש שגוברת על כל הנחיית אורך אחרת: ענה בקצרה, בשניים עד ארבעה משפטים ועד חמישים מילים, בלי הרחבות, עד שהמשתמש יבקש לחזור לתשובות מפורטות.';
const PACE_PREFIX = { slow: 'Say slowly and calmly: ', fast: 'Say quickly: ' };

const SHORT_RE = /^\s*(?:ת?קצר(?:י)? את התשוב(?:ה|ות)|תענה(?: יותר)? קצר|תענה בקצרה|בקצרה|בקיצור)\s*/;
const LONG_RE = /^\s*(?:תפרט|תרחיב|תאריך את התשוב(?:ה|ות)|תענה(?: יותר)? בהרחבה)\s*/;
function parseCtl(text) {
  const ctl = {};
  let q = '';
  const noQ = text.replace(/\[\[\s*Q\s*:([^\]]*)\]\]/gi, (m, v) => { if (!q) q = v.replace(/\s+/g, ' ').trim(); return ' '; });
  const clean = noQ.replace(/\[\[\s*(VOICE|PACE|LEN)\s*:\s*([a-z]+)\s*\]\]/gi, (m, k, v) => { ctl[k.toLowerCase()] = v.toLowerCase(); return ' '; });
  return { text: clean.replace(/\s+/g, ' ').trim(), ctl, q: q.slice(0, 500) };
}

const TTS_CHUNK = 900;
function splitText(text) {
  const out = [];
  let cur = '';
  for (let sn of text.split(/(?<=[.!?:;])\s+/)) {
    while (sn.length > TTS_CHUNK) {
      let k = sn.lastIndexOf(' ', TTS_CHUNK);
      if (k < 200) k = TTS_CHUNK;
      if (cur) { out.push(cur); cur = ''; }
      out.push(sn.slice(0, k).trim());
      sn = sn.slice(k).trim();
    }
    if (!sn) continue;
    if (cur && cur.length + sn.length + 1 > TTS_CHUNK) { out.push(cur); cur = sn; }
    else cur = cur ? cur + ' ' + sn : sn;
  }
  if (cur) out.push(cur);
  return out.filter(Boolean);
}
function pcmOf(buf) {
  let p = 12, rate = 24000;
  while (p + 8 <= buf.length) {
    const id = buf.toString('ascii', p, p + 4), sz = buf.readUInt32LE(p + 4);
    if (id === 'fmt ') rate = buf.readUInt32LE(p + 12);
    if (id === 'data') return { rate, pcm: buf.subarray(p + 8, Math.min(buf.length, p + 8 + sz)) };
    p += 8 + sz + (sz & 1);
  }
  return { rate, pcm: buf.subarray(44) };
}
function to8k(x) {
  const f = x.rate / 8000;
  if (f === 1 || !Number.isInteger(f)) return x;
  const n = Math.floor(x.pcm.length / 2 / f), out = Buffer.alloc(n * 2);
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let k = 0; k < f; k++) sum += x.pcm.readInt16LE((i * f + k) * 2);
    out.writeInt16LE(Math.round(sum / f), i * 2);
  }
  return { rate: 8000, pcm: out };
}
async function tts(text, deadline, o) {
  const parts = splitText(String(text).slice(0, 7000));
  if (parts.length <= 1) return ttsOne(text, deadline, o);
  const bufs = await Promise.all(parts.map(x => ttsOne(x, deadline, o)));
  const gap = Buffer.alloc(3200);
  const pcms = [];
  bufs.forEach((b, i) => { if (i) pcms.push(gap); pcms.push(to8k(pcmOf(b)).pcm); });
  return wav(Buffer.concat(pcms), 8000);
}

async function ttsOne(text, deadline, o) {
  o = o || {};
  const pre = PACE_PREFIX[o.pace] || '';
  let last = new Error('no audio from tts');
  if (TTS_LIST.every(m => (ttsDown[m] || 0) > Date.now())) { last = new Error('all tts models are out of quota'); last.status = 429; throw last; }
  for (let mi = 0; mi < TTS_LIST.length; mi++) {
    const model = TTS_LIST[mi];
    const hasNext = mi < TTS_LIST.length - 1;
    if ((ttsDown[model] || 0) > Date.now()) continue;
    const V = voicesFor(model, o.voice || VOICE);
    for (let i = 0; i < 3; i++) {
      try {
        const t = await gem(model, {
          contents: [{ role: 'user', parts: [{ text: pre + text.slice(0, 3000) }] }],
          generationConfig: { responseModalities: ['AUDIO'], speechConfig: { voiceConfig: V[vIdx[model] || 0] } }
        }, deadline, 'tts', hasNext ? { ms: 50000, retries: 0 } : { ms: 50000 });
        const parts = (((t.candidates || [])[0] || {}).content || {}).parts || [];
        const part = parts.find(p => p.inlineData && p.inlineData.data);
        if (part) {
          let pcm = Buffer.from(part.inlineData.data, 'base64');
          if (pcm.subarray(0, 4).toString() !== 'RIFF') {
            const m = /rate=(\d+)/.exec(part.inlineData.mimeType || '');
            pcm = wav(pcm, m ? Number(m[1]) : 24000);
          }
          return pcm;
        }
        last = new Error('no audio from tts');
      } catch (e) {
        last = e;
        if (isBudget(e)) throw e;
        if (e.status === 400 && /voice|speech/i.test(e.message) && (vIdx[model] || 0) < V.length - 1) { vIdx[model] = (vIdx[model] || 0) + 1; continue; }
        if (e.status === 429 || e.status === 404 || e.status >= 500) break;
      }
    }
    if (hasNext || last.status === 429) {
      ttsDown[model] = Date.now() + (last.status === 429 ? 600000 : 120000);
      console.error('tts model', model, 'unavailable (' + String(last.message || '').slice(0, 60) + ')' + (hasNext ? ', using ' + TTS_LIST[mi + 1] : ''));
    }
  }
  throw last;
}

async function download(name, deadline, d) {
  let last = '';
  for (let i = 0; i < 4; i++) {
    try {
      const r = await http(YM + 'DownloadFile?token=' + encodeURIComponent(TOKEN) + '&path=' + encodeURIComponent(yp(name, d)), {}, 8000, deadline);
      const b = Buffer.from(await r.arrayBuffer());
      if (r.ok && b.length > 100 && b.subarray(0, 4).toString() === 'RIFF') return b;
      last = r.status + ' ' + b.subarray(0, 80).toString('utf8').replace(/\s+/g, ' ');
    } catch (e) {
      if (isBudget(e)) throw e;
      last = e.message;
    }
    await sleep(500);
  }
  throw new Error('download failed ' + last);
}

async function upload(name, buf, deadline, d, ms) {
  const path = yp(name, d);
  const attempts = [
    { q: true, field: 'file' }, { q: true, field: 'upload' },
    { q: false, field: 'upload' }, { q: false, field: 'file' }
  ];
  let last = '';
  for (const a of attempts) {
    try {
      const f = new FormData();
      let url = YM + 'UploadFile';
      if (a.q) {
        url += '?token=' + encodeURIComponent(TOKEN) + '&path=' + encodeURIComponent(path) + '&convertAudio=1';
      } else {
        f.append('token', TOKEN); f.append('path', path); f.append('convertAudio', '1');
      }
      f.append(a.field, new Blob([buf], { type: 'audio/wav' }), name + '.wav');
      const r = await http(url, { method: 'POST', body: f }, ms || 15000, deadline);
      const j = await r.json().catch(() => ({}));
      if (r.ok && j.responseStatus === 'OK') return;
      last = JSON.stringify(j).slice(0, 200);
    } catch (e) {
      if (isBudget(e)) throw e;
      last = e.message;
      if (ms && e.name === 'TimeoutError') break;
    }
  }
  throw new Error('upload failed ' + last);
}

async function uploadText(name, text, deadline, d) {
  const what = yp(name + '.txt', d);
  let last = '';
  try {
    const r = await http(YM + 'UploadTextFile', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({ token: TOKEN, what, contents: text })
    }, 15000, deadline);
    const j = await r.json().catch(() => ({}));
    if (r.ok && j.responseStatus === 'OK') return;
    last = JSON.stringify(j).slice(0, 200);
  } catch (e) {
    if (isBudget(e)) throw e;
    last = e.message;
  }
  try {
    const f = new FormData();
    f.append('file', new Blob([text], { type: 'text/plain' }), name + '.txt');
    const r = await http(YM + 'UploadFile?token=' + encodeURIComponent(TOKEN) + '&path=' + encodeURIComponent(what) + '&convertAudio=0', { method: 'POST', body: f }, 15000, deadline);
    const j = await r.json().catch(() => ({}));
    if (r.ok && j.responseStatus === 'OK') return;
    last += ' | ' + JSON.stringify(j).slice(0, 150);
  } catch (e) {
    if (isBudget(e)) throw e;
    last += ' | ' + e.message;
  }
  throw new Error('text upload failed ' + last);
}
async function downloadText(name, deadline, d) {
  const r = await http(YM + 'DownloadFile?token=' + encodeURIComponent(TOKEN) + '&path=' + encodeURIComponent(yp(name + '.txt', d)), {}, 8000, deadline);
  const t = await r.text();
  if (!r.ok || /^\s*\{\s*"responseStatus"/.test(t)) throw new Error('text download failed ' + t.slice(0, 80));
  return t;
}

function del(name, d) {
  fetch(YM + 'FileAction?token=' + encodeURIComponent(TOKEN) + '&action=delete&what=' + encodeURIComponent(yp(name, d)), { signal: AbortSignal.timeout(8000) }).catch(() => {});
}

app.get('/', (q, r) => { r.set('Access-Control-Allow-Origin', '*'); r.send('ok'); });

// ---------- shared helpers of the call flow ----------
const idOf = q => String(q.ApiCallId || 'x').replace(/[^a-zA-Z0-9]/g, '').slice(-24) || 'x';
const newSession = (d, sys, g, extra) => Object.assign({ d, sys, g, n: 0, hn: 0, mode: '', job: null, vc: '', pace: '', len: '', h: [], p: '', name: '', r: '', t: 0, e: crypto.randomBytes(3).toString('hex') }, extra || {});

const askFor = (s, pre) => {
  s.n++;
  s.mode = 'rec';
  s.p = 'Q' + s.e + s.n;
  s.name = 'q' + s.e + 'n' + s.n;
  return 'read=' + pre + '=' + s.p + ',no,record,' + s.d + ',' + s.name + ',no,,,1,' + REC_MAX;
};
const tapFor = (s, pre) => {
  s.n++;
  s.p = 'M' + s.e + s.n;
  return 'read=' + pre + '=' + s.p + ',no,1,1,7,No';
};

// ---------- asking: the caller either types the question with the phone keys or records it ----------
const KB_KEYS = { '3': 'אבג', '2': 'דהו', '6': 'זחט', '5': 'יכךל', '4': 'מםנן', '9': 'סעפף', '8': 'צץק', '7': 'רשת' };
const KB_NAME = { 'א': 'אלף', 'ב': 'בית', 'ג': 'גימל', 'ד': 'דלת', 'ה': 'הא', 'ו': 'ואו', 'ז': 'זין', 'ח': 'חית', 'ט': 'טית', 'י': 'יוד', 'כ': 'כף', 'ל': 'למד', 'מ': 'מם', 'נ': 'נון', 'ס': 'סמך', 'ע': 'עין', 'פ': 'פא', 'צ': 'צדי', 'ק': 'קוף', 'ך': 'כף סופית', 'ם': 'מם סופית', 'ן': 'נון סופית', 'ף': 'פא סופית', 'ץ': 'צדי סופית', 'ר': 'ריש', 'ש': 'שין', 'ת': 'תו' };
const KB_WAIT = Math.min(60, Math.max(3, parseInt(process.env.KBD_WAIT, 10) || 20));
const KB_FINAL = { 'כ': 'ך', 'מ': 'ם', 'נ': 'ן', 'פ': 'ף', 'צ': 'ץ' };
const KB_MORE = sp('לשאלה נוספת');
const KB_START = sp('הקלד את השאלה. כל אות לפי מקומה על המקש. בין שתי אותיות על אותו מקש הקש אחת. לרווח הקש אפס. ובסיום הקש סולמית');
const KB_SHORT = sp('הקלד את השאלה ובסיום הקש סולמית');
const MENU = sp('להקלטה הקש 1 להקלדה הקש 2');
const MENU_OPEN = sp('להקלטת שאלה הקש 1 להקלדה הקש 2');
const WAIT_SEC = Math.min(15, Math.max(2, parseInt(process.env.INPUT_WAIT_SEC, 10) || 7));
const EMPTY = 'None';
const TAP_EMPTY = ',No,no,no,,,,Ok,' + EMPTY;
const digitsOf = v => { const x = String(v == null ? '' : v).trim(); return x === EMPTY ? '' : x; };
const kbFinal = t => t.replace(/(?<=[א-ת])[כמנפצ](?=\s|$)/g, c => KB_FINAL[c]);
const beepOk = {};
const beepP = {};
const beepFail = {};
function makeBeep() {
  const rate = 8000, n = Math.round(rate * 0.35), pcm = Buffer.alloc(n * 2);
  for (let i = 0; i < n; i++) {
    const env = Math.min(1, i / 80, (n - i) / 80);
    pcm.writeInt16LE(Math.round(Math.sin(2 * Math.PI * 1000 * i / rate) * 12000 * env), i * 2);
  }
  return wav(pcm, rate);
}
function ensureBeep(d) {
  if (!TOKEN || beepOk[d]) return Promise.resolve(!!beepOk[d]);
  if (Date.now() - (beepFail[d] || 0) < 60000) return Promise.resolve(false);
  if (!beepP[d]) {
    beepP[d] = (async () => {
      const dl = Date.now() + 25000;
      try {
        const r = await http(YM + 'GetIVR2Dir?token=' + encodeURIComponent(TOKEN) + '&path=' + encodeURIComponent('ivr2:' + d), {}, 6000, dl);
        const j = await r.json().catch(() => ({}));
        if (j.responseStatus !== 'OK') throw new Error('dir ' + JSON.stringify(j).slice(0, 100));
        if (!JSON.stringify(j).includes('beep.wav')) await upload('beep', makeBeep(), dl, d);
        beepOk[d] = true;
        return true;
      } catch (e) {
        console.error('beep ' + d + ' failed:', String(e.message).slice(0, 120));
        beepFail[d] = Date.now();
        return false;
      } finally { delete beepP[d]; }
    })();
  }
  return beepP[d];
}
const tapAny = (s, pre, menu) => {
  s.n++;
  s.mode = 'menu';
  s.p = 'K' + s.e + s.n;
  return 'read=' + pre + '.t-' + (menu || MENU) + '=' + s.p + ',no,1,1,' + WAIT_SEC + TAP_EMPTY;
};
const ansFor = (s, pre) => {
  s.n++;
  s.mode = 'ans';
  s.p = 'A' + s.e + s.n;
  return 'read=' + pre + '=' + s.p + ',no,1,1,1' + TAP_EMPTY;
};
const recPre = s => 't-' + sp('דבר אחרי הצליל') + (beepOk[s.d] ? '.f-' + s.d + '/beep' : '');
const kbAsk = (s, pre) => {
  s.n++;
  s.mode = 'type';
  s.p = 'K' + s.e + s.n;
  return 'read=' + pre + '=' + s.p + ',no,*,1,' + KB_WAIT + TAP_EMPTY;
};
function kbKey(s, raw) {
  const d = String(raw || '').replace(/#/g, '').trim();
  for (const r of d.match(/(\d)\1*/g) || []) {
    const k = r[0], n = r.length;
    if (k === '1') { s.buf += '1'.repeat(Math.floor(n / 2)); continue; }
    if (n === 5) { s.buf += k; continue; }
    if (k === '0') {
      if (n === 1) s.buf += ' ';
      else if (n === 2) s.buf = s.buf.slice(0, -1);
      continue;
    }
    const L = KB_KEYS[k];
    if (L && n <= L.length) s.buf += L[n - 1];
  }
}
async function kbStep(s, q, id, T0, first) {
  const d = digitsOf(first !== undefined ? first : q[s.p]);
  if (!d) return tapAny(s, 't-' + sp('לא הוקלד כלום'));
  s.buf = '';
  kbKey(s, d);
  const text = kbFinal(s.buf.replace(/\s+/g, ' ').trim().slice(0, 600));
  s.buf = '';
  if (!text) return kbAsk(s, 't-' + sp('לא הוקלד טקסט נסו שוב'));
  return runTurn(s, q, id, T0, text);
}
async function turnStep(s, q, id, T0) {
  if (s.job) return holdStep(s, id);
  if (s.mode === 'ans') {
    const d = digitsOf(q[s.p]);
    if (d) {
      s.mediaJob = null;
      s.mediaIdx = 0;
      const lm = s.h[s.h.length - 1];
      if (lm && lm.role === 'model' && lm.parts && lm.parts[0]) lm.parts[0].text += ' (המשתמש עצר את ההקראה לפני סופה)';
      if (d === '1' || d === '2') s.mode = 'menu';
    } else if (s.mediaJob) {
      const nextSeg = await nextMediaSegment(s);
      if (nextSeg) return ansFor(s, nextSeg);
    }
    if (s.mode === 'ans') return tapAny(s, 't-' + KB_MORE);
  }
  if (s.mode === 'menu') {
    const d = digitsOf(q[s.p]);
    if (d === '1') { s.miss = 0; return askFor(s, recPre(s)); }
    if (d === '2') { s.miss = 0; s.buf = ''; const help = s.kbSeen ? KB_SHORT : KB_START; s.kbSeen = true; return kbAsk(s, 't-' + help); }
    s.miss = (s.miss || 0) + 1;
    if (s.miss >= 3) return 'id_list_message=t-' + sp('להתראות');
    return tapAny(s, 't-' + sp(d ? 'המקש לא מוכר' : 'לא התקבלה בחירה'));
  }
  if (s.mode === 'type') return kbStep(s, q, id, T0);
  return runTurn(s, q, id, T0);
}

function endCall(id) {
  const old = calls[id];
  if (old && old.r) del(old.r, old.d);
  delete calls[id];
}

// spoken status messages while the answer is being prepared
const WAIT_MSG = sp('חכה שנייה');
const SEARCH_MSG = sp('חכה שנייה מחפש באינטרנט');
function statusMsg(s) {
  let msg = '';
  if (!s.waitSaid) { s.waitSaid = true; msg = WAIT_MSG; }
  else if (s.searching && !s.searchSaid) { s.searchSaid = true; msg = SEARCH_MSG; }
  if (!msg) return '';
  s.hn++;
  return 'read=t-' + msg + '=H' + s.e + s.hn + ',no,1,1,1' + TAP_EMPTY;
}

function runTurn(s, q, id, T0, typed) {
  if (!typed && !q[s.p]) return tapAny(s, 't-' + AGAIN);
  if (!checkRateLimit(s.ph)) {
    return 'id_list_message=t-' + sp('הגעת למגבלת השאלות לשעה נסה שוב מאוחר יותר');
  }
  s.jobT0 = Date.now();
  s.hold = null;
  s.waitSaid = false;
  s.searching = false;
  s.searchSaid = false;
  s.searchSig = new Promise(res => { s.onSearch = res; });
  s.holdP = songsFor(s.ph).catch(() => []);
  s.job = doTurn(s, q, id, s.jobT0, typed).catch(e => { console.error(id, 'turn crashed', e && e.message); return tapAny(s, 't-' + ERR); });
  return holdStep(s, id);
}

async function holdStep(s, id) {
  if (!s.waitSaid) return statusMsg(s);
  const files = await Promise.race([s.holdP, sleep(2500).then(() => [])]);
  if (!files || !files.length) {
    const r = await Promise.race([s.job, s.searchSig.then(() => null)]);
    if (r !== null) { s.job = null; return r; }
    const st = statusMsg(s);
    if (st) return st;
    const r2 = await s.job; s.job = null; return r2;
  }
  const r = await Promise.race([s.job, sleep(HOLD_LAG_MS).then(() => null)]);
  if (r !== null) { s.job = null; return r; }
  { const st = statusMsg(s); if (st) return st; }
  if (Date.now() - s.jobT0 > BUDGET + 15000) { s.job = null; return tapAny(s, 't-' + ERR); }
  if (!s.hold) s.hold = { i: Math.floor(Math.random() * files.length) };
  const msgs = [];
  for (let k = 0; k < HOLD_CHAIN; k++) msgs.push('f-' + SD + '/' + files[(s.hold.i + k) % files.length]);
  s.hold.i = (s.hold.i + HOLD_CHAIN) % files.length;
  s.hn++;
  return 'read=' + msgs.join('.') + '=H' + s.e + s.hn + ',no,1,1,1' + TAP_EMPTY;
}

async function doTurn(s, q, id, T0, typed) {
  const ask = pre => tapAny(s, pre);
  const more = pre => ansFor(s, pre);
  const deadline = T0 + BUDGET;
  const qn = s.name;
  try {
    let userPart;
    let qAudio = null;
    let tq = typed;
    if (typed) {
      let m;
      if ((m = SHORT_RE.exec(tq))) { s.len = 'short'; tq = tq.slice(m[0].length).trim() || 'אשר לי בקצרה שמעכשיו התשובות יהיו קצרות'; }
      else if ((m = LONG_RE.exec(tq))) { s.len = ''; tq = tq.slice(m[0].length).trim() || 'אשר לי בקצרה שמעכשיו התשובות יהיו מפורטות'; }
      userPart = { text: 'השאלה הוקלדה במקשי הטלפון ולכן ייתכנו בה שגיאות כתיב ובלי סימני פיסוק. אם בתחילתה יש בקשה לקצר או להאריך את התשובות, זו הנחיה לגבי אורך התשובה שלך ולא חלק מהשאלה. הבן את הכוונה וענה: ' + tq };
    } else {
      const audio = await download(qn, deadline, s.d);
      del(qn, s.d);
      qAudio = audio;
      userPart = { inlineData: { mimeType: 'audio/wav', data: audio.toString('base64') } };
    }
    if (s.r) { del(s.r, s.d); s.r = ''; }
    const t1 = Date.now();

    const a = await askModel(s.len === 'short' ? s.sys + SHORT_NOTE : s.sys, [...(s.base || []), ...s.h.slice(-8), { role: 'user', parts: [userPart] }], deadline, s);
    const parts = (((a.candidates || [])[0] || {}).content || {}).parts || [];
    let text = parts.filter(p => !p.thought).map(p => p.text || '').join('').trim();
    if (!text) throw new Error('empty answer ' + JSON.stringify((a.promptFeedback || {})).slice(0, 100));
    const ctl = parseCtl(text);
    text = ctl.text || 'בסדר';
    if (ctl.ctl.voice === 'male' || ctl.ctl.voice === 'female') s.vc = ctl.ctl.voice;
    else if (ctl.ctl.voice === 'default') s.vc = '';
    if (ctl.ctl.len === 'short') s.len = 'short';
    else if (ctl.ctl.len === 'long') s.len = '';
    if (ctl.ctl.pace === 'slow' || ctl.ctl.pace === 'fast') s.pace = ctl.ctl.pace;
    else if (ctl.ctl.pace === 'normal') s.pace = '';
    const qtext = tq || ctl.q;
    const histQ = { role: 'user', parts: [{ text: qtext || '[שאלה קולית]' }] };
    s.save = { qt: qtext || 'לא תומללה', at: text };
    const t2 = Date.now();

    let pcm = null;
    try {
      pcm = await tts(text, deadline, { voice: VOICE_BY[s.vc], pace: s.pace });
    } catch (te) {
      console.error(id, 'gemini voice unavailable, Yemot reads the answer:', String(te.message).slice(0, 80));
    }
    const t3 = Date.now();
    if (pcm && qAudio && s.save) { s.save.qa = qAudio; s.save.aa = pcm; }
    const media = await takeMedia(s, deadline);
    const tail = media ? '.' + media : '';
    s.h.push(histQ, { role: 'model', parts: [{ text }] });
    if (s.h.length > 12) s.h = s.h.slice(-12);

    if (!pcm) {
      console.log(id, 'turn', s.n, 'ok (yemot voice)', 'gemini', t2 - t1, 'ms total', Date.now() - T0, 'ms');
      return more('t-' + (sp(text) || ERR) + tail);
    }

    const rn = 'r' + s.e + 'n' + s.n;
    await upload(rn, pcm, deadline, s.d, 30000);
    s.r = rn;
    console.log(id, 'turn', s.n, typed ? 'ok (typed)' : 'ok', 'gemini', t2 - t1, 'ms tts', t3 - t2, 'ms total', Date.now() - T0, 'ms');
    return more('f-' + s.d + '/' + rn + tail);
  } catch (e) {
    console.error(id, 'turn', s.n, 'failed after', Date.now() - T0, 'ms:', e.message, e.stderr ? '| ' + String(e.stderr).replace(/\s+/g, ' ') : '');
    s.mediaP = null;
    s.save = null;
    if (!typed) del(qn, s.d);
    return ask('t-' + ERR);
  }
}

// ---------- history ----------
const HIST = String(process.env.HISTORY_EXT || '').replace(/^\/+|\/+$/g, '');
const HD = /^[0-9]+(\/[0-9]+)*$/.test(HIST) ? '/' + HIST : '';
const CONT_TURNS = 2;
const phoneOf = q => { const p = String(q.ApiPhone || '').replace(/\D/g, ''); return p.length >= 7 && p.length <= 15 ? p : ''; };
const hName = (ph, cid, suffix) => 'p' + ph + 'c' + cid + suffix;
const sortedTurns = c => Object.keys(c.turns).map(Number).filter(n => c.turns[n].q && c.turns[n].a).sort((x, y) => x - y);

async function saveTurn(s, sv) {
  if (!HD || !s.ph) return;
  if (!s.cid) { s.cid = Date.now().toString(36).padStart(8, '0'); s.tn = 0; }
  const cid = s.cid, t = ++s.tn;
  const dl = Date.now() + 120000;
  const vo = { voice: VOICE_BY[s.vc], pace: s.pace };
  let qa = null;
  if (sv.qt && sv.qt !== 'לא תומללה') {
    try { qa = await tts(sv.qt, Date.now() + 40000, vo); } catch (e) { console.error('history question voice failed:', String(e.message).slice(0, 80)); }
  }
  if (!qa) qa = sv.qa || null;
  let aa = sv.aa || null;
  if (!aa) {
    try { aa = await tts(sv.at, Date.now() + 40000, vo); } catch (e) { console.error('history answer voice failed:', String(e.message).slice(0, 80)); }
  }
  if (qa && aa) {
    await upload(hName(s.ph, cid, 't' + t + 'q'), qa, dl, HD, 20000);
    await upload(hName(s.ph, cid, 't' + t + 'a'), aa, dl, HD, 30000);
    console.log('history saved (wav)', cid, 't' + t);
    return;
  }
  await uploadText(hName(s.ph, cid, 't' + t + 'q'), sv.qt, dl, HD);
  await uploadText(hName(s.ph, cid, 't' + t + 'a'), sv.at, dl, HD);
  console.log('history saved (txt)', cid, 't' + t);
}
function flushSave(s) {
  const sv = s.save; s.save = null;
  if (sv && HD && s.ph) saveTurn(s, sv).catch(e => console.error('history save failed:', e.message));
}

async function histList(ph, deadline) {
  const r = await http(YM + 'GetIVR2Dir?token=' + encodeURIComponent(TOKEN) + '&path=' + encodeURIComponent('ivr2:' + HD), {}, 10000, deadline);
  const j = await r.json().catch(() => ({}));
  if (j.responseStatus !== 'OK') throw new Error('history list failed ' + JSON.stringify(j).slice(0, 120));
  const txt = JSON.stringify(j);
  console.log('history dir: ' + (txt.match(/\.txt/g) || []).length + ' txt, ' + (txt.match(/\.wav/g) || []).length + ' wav mentions');
  const re = new RegExp('p' + ph + 'c([0-9a-z]{8})(?:t(\\d+)([qa])|n)\\.(wav|txt)', 'g');
  const map = new Map();
  let m;
  while ((m = re.exec(txt))) {
    const c = map.get(m[1]) || { cid: m[1], turns: {}, name: false };
    if (m[2]) { const tt = c.turns[m[2]] = c.turns[m[2]] || {}; tt[m[3]] = true; if (m[4] === 'txt') tt.txt = true; } else c.name = true;
    map.set(m[1], c);
  }
  return [...map.values()].filter(c => sortedTurns(c).length).sort((x, y) => (x.cid < y.cid ? 1 : -1));
}

const dateOf = cid => {
  try {
    const parts = new Intl.DateTimeFormat('en-GB', { timeZone: 'Asia/Jerusalem', day: 'numeric', month: 'numeric' }).formatToParts(new Date(parseInt(cid, 36)));
    const g = t => (parts.find(x => x.type === t) || {}).value;
    return Number(g('day')) + ' ' + Number(g('month'));
  } catch (e) { return ''; }
};
const HMENU = sp('להשמעה הקש 1 להמשך השיחה הקש 2 למחיקה הקש 3 לשינוי שם הקש 4 לשיחה הבאה הקש 5 לשיחה הקודמת הקש 6 ליציאה הקש 0');

function showItem(s, intro) {
  const c = s.list[s.i];
  const msgs = [];
  if (intro) msgs.push('t-' + sp(intro));
  msgs.push('t-' + sp('שיחה מספר ' + (s.i + 1) + ' מתוך ' + s.list.length));
  msgs.push(c.name ? 'f-' + HD + '/' + hName(s.ph, c.cid, 'n') : 't-' + sp('מתאריך ' + dateOf(c.cid)));
  msgs.push('t-' + HMENU);
  s.hm = 'menu';
  return tapFor(s, msgs.join('.'));
}

async function playConv(s) {
  const c = s.list[s.i];
  const msgs = [];
  const dl = Date.now() + 20000;
  for (const n of sortedTurns(c)) {
    if (c.turns[n].txt) {
      let qt = '', at = '';
      try { [qt, at] = await Promise.all([downloadText(hName(s.ph, c.cid, 't' + n + 'q'), dl, HD), downloadText(hName(s.ph, c.cid, 't' + n + 'a'), dl, HD)]); } catch (e) { console.error('history text failed:', e.message); }
      msgs.push('t-' + sp('שאלה'), 't-' + (sp(qt) || sp('לא זמין')), 't-' + sp('תשובה'), 't-' + (sp(at) || sp('לא זמין')));
      continue;
    }
    msgs.push('t-' + sp('שאלה'), 'f-' + HD + '/' + hName(s.ph, c.cid, 't' + n + 'q'), 't-' + sp('תשובה'), 'f-' + HD + '/' + hName(s.ph, c.cid, 't' + n + 'a'));
  }
  msgs.push('t-' + HMENU);
  s.hm = 'menu';
  return tapFor(s, msgs.join('.'));
}

async function continueConv(s) {
  const c = s.list[s.i];
  const all = sortedTurns(c);
  const last = all.slice(-CONT_TURNS);
  const dl = Date.now() + 30000;
  const parts = [{ text: 'זה ההקשר משיחה קודמת עם המשתמש. אחריו תגיע שאלה חדשה שלו, והמשך את השיחה בהתאם.' }];
  for (const n of last) {
    if (c.turns[n].txt) {
      const [qt, at] = await Promise.all([downloadText(hName(s.ph, c.cid, 't' + n + 'q'), dl, HD), downloadText(hName(s.ph, c.cid, 't' + n + 'a'), dl, HD)]);
      parts.push({ text: 'שאלה קודמת של המשתמש: ' + qt }, { text: 'התשובה הקודמת שלך: ' + at });
      continue;
    }
    const [qa, aa] = await Promise.all([download(hName(s.ph, c.cid, 't' + n + 'q'), dl, HD), download(hName(s.ph, c.cid, 't' + n + 'a'), dl, HD)]);
    parts.push({ text: 'שאלה קודמת של המשתמש בהקלטה:' }, { inlineData: { mimeType: 'audio/wav', data: qa.toString('base64') } });
    parts.push({ text: 'התשובה הקודמת שלך בהקלטה:' }, { inlineData: { mimeType: 'audio/wav', data: aa.toString('base64') } });
  }
  s.base = [{ role: 'user', parts }, { role: 'model', parts: [{ text: 'הבנתי אמשיך את השיחה.' }] }];
  s.h = [];
  s.cid = c.cid;
  s.tn = Math.max(...Object.keys(c.turns).map(Number));
  s.hm = 'talk';
  return tapAny(s, 't-' + sp('אפשר להמשיך'));
}

async function historyStep(s, q) {
  const deadline = Date.now() + 25000;
  if (!s.hm) {
    s.list = await histList(s.ph, deadline);
    if (!s.list.length) return 'id_list_message=t-' + sp('אין עדיין שיחות שמורות');
    s.i = 0;
    return showItem(s, 'שלום יש לך ' + s.list.length + ' שיחות שמורות');
  }
  const d = String(q[s.p] || '').trim();
  const c = s.list[s.i];
  if (s.hm === 'del') {
    if (d !== '1') return showItem(s, 'המחיקה בוטלה');
    for (const n of Object.keys(c.turns)) { const x = c.turns[n].txt ? '.txt' : ''; del(hName(s.ph, c.cid, 't' + n + 'q') + x, HD); del(hName(s.ph, c.cid, 't' + n + 'a') + x, HD); }
    if (c.name) del(hName(s.ph, c.cid, 'n'), HD);
    s.list.splice(s.i, 1);
    if (!s.list.length) return 'id_list_message=t-' + sp('השיחה נמחקה ואין עוד שיחות שמורות');
    s.i = Math.min(s.i, s.list.length - 1);
    return showItem(s, 'השיחה נמחקה');
  }
  if (s.hm === 'rename') {
    c.name = !!q[s.p];
    return showItem(s, c.name ? 'השם נשמר' : 'לא הוקלט שם');
  }
  switch (d) {
    case '1': return await playConv(s);
    case '2':
      try { return await continueConv(s); } catch (e) { console.error('continue failed:', e.message); return showItem(s, 'הייתה תקלה'); }
    case '3':
      s.hm = 'del';
      return tapFor(s, 't-' + sp('למחיקת השיחה לצמיתות הקש 1 לביטול הקש 2'));
    case '4': {
      const f = hName(s.ph, c.cid, 'n');
      if (c.name) del(f, HD);
      s.hm = 'rename';
      s.n++;
      s.p = 'R' + s.e + s.n;
      return 'read=t-' + sp('אחרי הצליל אמור את השם החדש ובסיום הקש סולמית') + '=' + s.p + ',no,record,' + HD + ',' + f + ',no,,,1,8';
    }
    case '5': s.i = (s.i + 1) % s.list.length; return showItem(s);
    case '6': s.i = (s.i - 1 + s.list.length) % s.list.length; return showItem(s);
    case '0': return 'id_list_message=t-' + sp('להתראות');
    default: return showItem(s);
  }
}

// ---------- songs folder ----------
const SONGS = String(process.env.SONGS_EXT || '').replace(/^\/+|\/+$/g, '');
const SD = /^[0-9]+(\/[0-9]+)*$/.test(SONGS) ? '/' + SONGS : '';
const SONGS_PASS = String(process.env.SONGS_PASS || '');
const SEG_SEC = Math.min(30, Math.max(3, parseInt(process.env.SONG_SEG_SEC, 10) || 5));
const HOLD_CHAIN = Math.min(6, Math.max(1, parseInt(process.env.HOLD_CHAIN, 10) || 2));
const HOLD_LAG_MS = 1200;
const SONG_MAX_SEC = (parseInt(process.env.SONG_MAX_MIN, 10) || 6) * 60;
const SONG_RE = 'h(?:p(\\d{7,15}))?s([0-9a-f]{6})x(\\d{3})\\.wav';
const enc = encodeURIComponent;
let songCache = { t: 0, tracks: new Map() };

async function songList(deadline) {
  const r = await http(YM + 'GetIVR2Dir?token=' + enc(TOKEN) + '&path=' + enc('ivr2:' + SD), {}, 10000, deadline);
  const j = await r.json().catch(() => ({}));
  if (j.responseStatus !== 'OK') throw new Error('songs list failed ' + JSON.stringify(j).slice(0, 120));
  const txt = JSON.stringify(j);
  const tracks = new Map();
  for (const m of txt.matchAll(new RegExp('(?<![A-Za-z0-9])' + SONG_RE, 'g'))) {
    const key = (m[1] || '') + 's' + m[2];
    const t = tracks.get(key) || { owner: m[1] || '', id: m[2], parts: new Set() };
    t.parts.add(m[0].replace(/\.wav$/, ''));
    tracks.set(key, t);
  }
  const chunk = new RegExp('^' + SONG_RE + '$');
  const raw = [...new Set([...txt.matchAll(/(?<![A-Za-z0-9_\-])([A-Za-z0-9_\-]+\.(?:wav|mp3|m4a|mp4|ogg|aac))/gi)].map(m => m[1]))].filter(n => !chunk.test(n));
  return { tracks, raw };
}

async function songsFor(ph) {
  if (!SD) return [];
  const now = Date.now();
  if (now - songCache.t > 120000) {
    try { songCache = { t: now, tracks: (await songList(now + 8000)).tracks }; }
    catch (e) { console.error(String(e.message).slice(0, 150)); songCache.t = now - 90000; }
  }
  const all = [...songCache.tracks.values()];
  const mine = ph ? all.filter(t => t.owner === ph) : [];
  const use = (mine.length ? mine : all.filter(t => !t.owner)).sort((x, y) => (x.id < y.id ? -1 : 1));
  return use.flatMap(t => [...t.parts].sort());
}

async function ingestSong(buf, owner) {
  if (!SD) throw new Error('SONGS_EXT לא הוגדר');
  if (!FFMPEG) throw new Error('ffmpeg לא זמין בשרת');
  const id = crypto.randomBytes(3).toString('hex');
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'song-'));
  const done = [];
  try {
    fs.writeFileSync(path.join(dir, 'in.bin'), buf);
    await execFileP(FFMPEG, ['-y', '-v', 'error', '-protocol_whitelist', 'file', '-i', path.join(dir, 'in.bin'), '-vn', '-t', String(SONG_MAX_SEC), '-ar', '8000', '-ac', '1', '-c:a', 'pcm_s16le', '-f', 'segment', '-segment_time', String(SEG_SEC), '-reset_timestamps', '1', path.join(dir, 'p%03d.wav')], { timeout: 120000 });
    const files = fs.readdirSync(dir).filter(f => /^p\d{3}\.wav$/.test(f)).sort();
    if (!files.length) throw new Error('הקובץ לא זוהה כקובץ שמע');
    const dl = Date.now() + 240000;
    const base = 'h' + (owner ? 'p' + owner : '') + 's' + id + 'x';
    let next = 0, failed = null;
    const worker = async () => {
      while (!failed) {
        const k = next++;
        if (k >= files.length) return;
        const name = base + String(k).padStart(3, '0');
        try { await upload(name, fs.readFileSync(path.join(dir, files[k])), dl, SD, 30000); done.push(name); } catch (e) { failed = e; }
      }
    };
    await Promise.all([worker(), worker(), worker()]);
    if (failed) { for (const n of done) del(n, SD); throw failed; }
    songCache.t = 0;
    return { id, parts: files.length };
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
}

let scanning = false;
const rawFail = {};
async function scanRaw() {
  if (!SD || scanning || !TOKEN) return;
  scanning = true;
  try {
    const l = await songList(Date.now() + 10000);
    for (const name of l.raw) {
      if ((rawFail[name] || 0) > Date.now()) continue;
      try {
        const r = await http(YM + 'DownloadFile?token=' + enc(TOKEN) + '&path=' + enc(yp(name, SD)), {}, 60000, Date.now() + 90000);
        const b = Buffer.from(await r.arrayBuffer());
        if (!r.ok || b.length < 1000 || b[0] === 0x7b) throw new Error('download failed ' + r.status + ' ' + b.length + ' ' + b.subarray(0, 60).toString('utf8').replace(/\s+/g, ' '));
        const mo = /^p(\d{7,15})[_\-]/.exec(name);
        await ingestSong(b, mo ? mo[1] : '');
        del(name, SD);
        console.log('song ingested from', name);
      } catch (e) {
        rawFail[name] = Date.now() + 3600000;
        console.error('song ingest failed', name, String(e.message).slice(0, 120));
      }
    }
  } catch (e) {
    console.error('songs scan failed:', String(e.message).slice(0, 120));
  } finally {
    scanning = false;
  }
}
if (SD) { setTimeout(scanRaw, 20000).unref(); setInterval(scanRaw, 600000).unref(); }

// Clean up Yemot media/podcast files (mf*.wav) older than 7 days
const MEDIA_RETENTION_MS = Math.max(1, parseInt(process.env.MEDIA_RETENTION_DAYS, 10) || 7) * 86400000;
function parseYemotMtime(item) {
  if (!item || typeof item !== 'object') return 0;
  for (const k of ['mtime', 'modificationTime', 'date', 'created', 'time', 'timestamp']) {
    const v = item[k];
    if (!v) continue;
    if (typeof v === 'number') return v > 1e12 ? v : v * 1000;
    const s = String(v).trim();
    if (/^\d{10,13}$/.test(s)) { const n = Number(s); return n > 1e12 ? n : n * 1000; }
    // Handle DD/MM/YYYY HH:mm:ss or YYYY-MM-DD HH:mm:ss
    const dmy = /^(\d{1,2})[\/.-](\d{1,2})[\/.-](\d{4})(?:\s+(\d{1,2}):(\d{2})(?::(\d{2}))?)?/.exec(s);
    if (dmy) return new Date(Number(dmy[3]), Number(dmy[2]) - 1, Number(dmy[1]), Number(dmy[4] || 0), Number(dmy[5] || 0), Number(dmy[6] || 0)).getTime();
    const parsed = Date.parse(s);
    if (!Number.isNaN(parsed)) return parsed;
  }
  return 0;
}

async function cleanOldMedia() {
  if (!TOKEN) return;
  const dirs = [...new Set(Object.values(EXTS).map(e => e.d))];
  const cutoff = Date.now() - MEDIA_RETENTION_MS;
  for (const d of dirs) {
    try {
      const r = await http(YM + 'GetIVR2Dir?token=' + enc(TOKEN) + '&path=' + enc('ivr2:' + d), {}, 10000, Date.now() + 15000);
      const j = await r.json().catch(() => ({}));
      if (j.responseStatus !== 'OK') continue;
      const files = Array.isArray(j.files) ? j.files : [];
      let deleted = 0;
      for (const f of files) {
        const name = String((f && (f.name || f.fileName || f.what)) || '');
        if (!/^mf[0-9a-f]{12}(?:x\d{3})?\.wav$/i.test(name)) continue;
        const ts = parseYemotMtime(f);
        if (ts > 0 && ts < cutoff) {
          del(name, d);
          deleted++;
        }
      }
      if (deleted) {
        for (const k of Object.keys(mediaJobs)) {
          if (k.startsWith(d + '/')) delete mediaJobs[k];
        }
        console.log('cleaned', deleted, 'old media files in', d);
      }
    } catch (e) {
      console.error('media cleanup failed:', String(e.message).slice(0, 120));
    }
  }
}
setTimeout(cleanOldMedia, 60000).unref();
setInterval(cleanOldMedia, 24 * 3600000).unref();

const songsAuth = req => !!SONGS_PASS && safeEq(String(req.get('x-songs-pass') || ''), SONGS_PASS);
const SONGS_PAGE = `<!doctype html><html lang="he" dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>תקיית שירים</title>
<style>body{font-family:Arial,sans-serif;max-width:560px;margin:24px auto;padding:0 14px;line-height:1.5}label{display:block;margin-top:14px;font-weight:bold}input,button{font-size:16px;padding:8px;width:100%;box-sizing:border-box;margin-top:4px}button{cursor:pointer;margin-top:14px}#msg{margin-top:14px;font-weight:bold}li{margin:6px 0}.d{width:auto;padding:2px 10px;margin:0 8px}</style></head><body>
<h2>תקיית שירים</h2><p>השירים מושמעים למתקשר מרגע שליחת השאלה ועד שהתשובה מוכנה.</p>
<label>סיסמה<input id="pw" type="password"></label>
<label>מספר טלפון של המשתמש (ריק = שיר לכולם)<input id="ph" inputmode="numeric" placeholder="0501234567"></label>
<label>קובץ שמע או וידאו<input id="f" type="file" accept="audio/*,video/*"></label>
<button id="up">העלאה</button><button id="ls">הצגת השירים</button><div id="msg"></div><ul id="l"></ul>
<script>
var $=function(i){return document.getElementById(i)};
function say(t){$('msg').textContent=t}
function hd(){return{'x-songs-pass':$('pw').value}}
function ph(){return $('ph').value.replace(/\\D/g,'')}
$('up').onclick=async function(){var f=$('f').files[0];if(!f){say('בחרו קובץ');return}say('מעלה וחותך את הקובץ, זה יכול לקחת כחצי דקה...');
try{var r=await fetch('/songs/upload?phone='+ph(),{method:'POST',headers:hd(),body:await f.arrayBuffer()});var j=await r.json();say(j.ok?'השיר נוסף ('+j.parts+' חלקים)':(j.error||'שגיאה'))}catch(e){say('שגיאה: '+e.message)}};
$('ls').onclick=async function(){try{var r=await fetch('/songs/list?phone='+ph(),{headers:hd()});var j=await r.json();if(!r.ok){say(j.error||'שגיאה');return}say(j.songs.length?'':'אין שירים');
var l=$('l');l.innerHTML='';j.songs.forEach(function(s){var li=document.createElement('li');li.textContent='שיר '+s.id+' ('+s.seconds+' שניות, '+(s.owner||'לכולם')+')';var b=document.createElement('button');b.className='d';b.textContent='מחיקה';b.onclick=async function(){await fetch('/songs/delete?owner='+s.owner+'&id='+s.id,{method:'POST',headers:hd()});$('ls').click()};li.appendChild(b);l.appendChild(li)})}catch(e){say('שגיאה: '+e.message)}};
</script></body></html>`;
app.get('/songs', (req, res) => res.type('html').send(SONGS_PAGE));
app.post('/songs/upload', express.raw({ type: () => true, limit: '40mb' }), async (req, res) => {
  if (!SD) return res.status(400).json({ error: 'SONGS_EXT לא הוגדר בשירות' });
  if (!songsAuth(req)) return res.status(403).json({ error: 'סיסמה שגויה או שלא הוגדרה SONGS_PASS' });
  const ph = String(req.query.phone || '').replace(/\D/g, '');
  if (ph && (ph.length < 7 || ph.length > 15)) return res.status(400).json({ error: 'מספר טלפון לא תקין' });
  if (!Buffer.isBuffer(req.body) || req.body.length < 1000) return res.status(400).json({ error: 'לא התקבל קובץ' });
  try {
    const r = await ingestSong(req.body, ph);
    res.json({ ok: true, id: r.id, parts: r.parts });
  } catch (e) {
    console.error('song upload failed:', String(e.message).slice(0, 200));
    res.status(500).json({ error: String(e.message).slice(0, 200) });
  }
});
app.get('/songs/list', async (req, res) => {
  if (!SD) return res.status(400).json({ error: 'SONGS_EXT לא הוגדר בשירות' });
  if (!songsAuth(req)) return res.status(403).json({ error: 'סיסמה שגויה או שלא הוגדרה SONGS_PASS' });
  try {
    const ph = String(req.query.phone || '').replace(/\D/g, '');
    const l = await songList(Date.now() + 12000);
    const songs = [...l.tracks.values()].filter(t => !ph || t.owner === ph || !t.owner).map(t => ({ id: t.id, owner: t.owner, seconds: t.parts.size * SEG_SEC }));
    res.json({ songs });
  } catch (e) {
    res.status(500).json({ error: String(e.message).slice(0, 200) });
  }
});
app.post('/songs/delete', async (req, res) => {
  if (!SD) return res.status(400).json({ error: 'SONGS_EXT לא הוגדר בשירות' });
  if (!songsAuth(req)) return res.status(403).json({ error: 'סיסמה שגויה או שלא הוגדרה SONGS_PASS' });
  const id = String(req.query.id || ''), owner = String(req.query.owner || '').replace(/\D/g, '');
  if (!/^[0-9a-f]{6}$/.test(id)) return res.status(400).json({ error: 'מזהה לא תקין' });
  try {
    const l = await songList(Date.now() + 12000);
    const t = l.tracks.get(owner + 's' + id);
    if (!t) return res.status(404).json({ error: 'השיר לא נמצא' });
    for (const n of t.parts) del(n, SD);
    songCache.t = 0;
    res.json({ ok: true });
  } catch (e) {
    res.status(500).json({ error: String(e.message).slice(0, 200) });
  }
});

app.all(['/yemot', '/yemot/:e'], async (req, res) => {
  res.type('text/plain');
  if (!checkApiSecret(req)) {
    return res.status(403).send('id_list_message=t-' + sp('גישה לא מורשית'));
  }
  const q = { ...req.query, ...req.body };
  const id = idOf(q);
  const T0 = Date.now();
  const ext = EXTS[String(req.params.e || '')] || EXT_FIRST;
  try {
    if (q.hangup === 'yes') {
      endCall(id);
      return res.send('noop=bye');
    }

    const s = calls[id] || (calls[id] = newSession(ext.d, ext.sys, ext.g, { ph: phoneOf(q) }));
    s.t = Date.now();

    if (!s.mode) {
      const hasGreet = (await Promise.all([greetFile(s.d, Date.now() + 4000), Promise.race([ensureBeep(s.d), sleep(5000).then(() => false)])]))[0];
      const g = hasGreet ? 'f-' + s.d + '/greet' : 't-' + s.g;
      return res.send(tapAny(s, g, MENU_OPEN));
    }
    const inVal = q[s.p];
    console.log(id, 'req', s.mode || '-', 'inLen', inVal == null ? 0 : String(inVal).length);
    const out = await turnStep(s, q, id, T0);
    res.send(out);
    flushSave(s);
  } catch (e) {
    console.error(id, 'fatal', e.message);
    if (!res.headersSent) res.send('id_list_message=t-' + ERR);
  }
});

app.all('/history', async (req, res) => {
  res.type('text/plain');
  if (!checkApiSecret(req)) {
    return res.status(403).send('id_list_message=t-' + sp('גישה לא מורשית'));
  }
  const q = { ...req.query, ...req.body };
  const id = idOf(q);
  const T0 = Date.now();
  try {
    if (q.hangup === 'yes') {
      endCall(id);
      return res.send('noop=bye');
    }
    if (!HD) return res.send('id_list_message=t-' + sp('שלוחת ההיסטוריה עדיין לא הוגדרה'));
    const ph = phoneOf(q);
    if (!ph) return res.send('id_list_message=t-' + sp('לא זוהה מספר הטלפון ולכן אי אפשר להשמיע היסטוריה'));

    const s = calls[id] || (calls[id] = newSession(HD, EXT_FIRST.sys, '', { ph, hm: '', list: [], i: 0 }));
    s.t = Date.now();
    if (!s.hm) console.log(id, 'history start');

    if (s.hm === 'talk') {
      const inVal = q[s.p];
      console.log(id, 'req', s.mode || '-', 'inLen', inVal == null ? 0 : String(inVal).length);
      const out = await turnStep(s, q, id, T0);
      res.send(out);
      flushSave(s);
      return;
    }
    return res.send(await historyStep(s, q));
  } catch (e) {
    console.error(id, 'history fatal', e.message);
    if (!res.headersSent) res.send('id_list_message=t-' + ERR);
  }
});

const greetOk = {};
async function greetFile(d, deadline) {
  if (process.env.IGNORE_GREET_FILE === '1') return false;
  if (greetOk[d] !== undefined) return greetOk[d];
  try {
    const r = await http(YM + 'GetIVR2Dir?token=' + encodeURIComponent(TOKEN) + '&path=' + encodeURIComponent('ivr2:' + d), {}, 3500, deadline);
    const j = await r.json().catch(() => ({}));
    if (j.responseStatus !== 'OK') return false;
    greetOk[d] = JSON.stringify(j).includes('greet.wav');
    return greetOk[d];
  } catch (e) {
    return false;
  }
}

let lastRec = 0;
app.post('/record', async (req, res) => {
  res.set('Access-Control-Allow-Origin', '*').type('text/plain; charset=utf-8');
  try {
    if (!TOKEN || !safeEq((req.body || {}).token || '', TOKEN)) return res.send('ERR הטוקן לא תואם לטוקן ששמור בשירות');
    if (Date.now() - lastRec < 6000) return res.send('ERR חכו כמה שניות ונסו שוב');
    const dir = String(req.body.dir || '').replace(/^\/+|\/+$/g, '');
    const file = String(req.body.file || 'greet');
    if (dir && !/^[0-9]+(\/[0-9]+)*$/.test(dir)) return res.send('ERR מספר שלוחה לא תקין');
    if (!/^[A-Za-z0-9_]{1,20}$/.test(file)) return res.send('ERR שם קובץ לא תקין');
    const text = String(req.body.text || '').replace(/\s+/g, ' ').trim().slice(0, 600);
    if (!text) return res.send('ERR הטקסט ריק');
    lastRec = Date.now();
    const d = dir ? '/' + dir : '';
    const deadline = Date.now() + 80000;
    const pcm = await tts(text, deadline);
    await upload(file, pcm, deadline, d);
    if (file === 'greet') greetOk[d] = true;
    res.send('OK ההקלטה נוצרה על ידי ג׳מיני והועלתה אל ' + (dir ? 'שלוחה ' + dir : 'השלוחה הראשית') + ' בשם ' + file + '.wav (' + Math.round(pcm.length / 1024) + 'KB)');
  } catch (e) {
    res.send('ERR ' + String(e.message).slice(0, 200));
  }
});

let lastTest = 0;
app.get('/selftest', async (req, res) => {
  res.set('Access-Control-Allow-Origin', '*').type('text/plain; charset=utf-8');
  if (!checkApiSecret(req)) {
    return res.status(403).send('ERR גישה לא מורשית (חסר פרמטר k תואם ל-API_SECRET)');
  }
  if (Date.now() - lastTest < 20000) return res.send('ERR נבדק הרגע, חכו 20 שניות ונסו שוב');
  lastTest = Date.now();
  const out = [];
  const deadline = Date.now() + 90000;
  const step = async (name, fn) => {
    try { out.push('OK  ' + name + (await fn() || '')); }
    catch (e) { out.push('ERR ' + name + ': ' + String(e.message).slice(0, 220)); }
  };
  const silent = wav(Buffer.alloc(3200), 8000);

  await step('מפתחות ג׳מיני לתמלול', async () => { if (!KEYS.length) throw new Error('GEMINI_API_KEYS ריק'); return ' (' + KEYS.length + ')'; });
  await step('מפתחות ג׳מיני להקראה', async () => { if (!TTS_KEYS.length) throw new Error('GEMINI_TTS_KEYS ריק'); return ' (' + TTS_KEYS.length + ')'; });
  if (GROQ_KEYS.length) await step('חיפוש Groq (' + GROQ_KEYS.length + ' מפתחות)', async () => {
    const r = await groqSearch('weather today', '', deadline);
    if (r.error) throw new Error(r.error);
    return ' (' + String(r.answer || '').length + ' תווים)';
  });
  await step('טוקן ימות', async () => { if (!TOKEN || /undefined/.test(TOKEN)) throw new Error('YEMOT_TOKEN ריק'); });
  await step('ימות: קריאת השלוחה ' + EXT, async () => {
    const r = await http(YM + 'GetIVR2Dir?token=' + encodeURIComponent(TOKEN) + '&path=' + encodeURIComponent('ivr2:' + EXTP), {}, 10000, deadline);
    const j = await r.json().catch(() => ({}));
    if (j.responseStatus !== 'OK') throw new Error(JSON.stringify(j).slice(0, 180));
  });
  if (HD) await step('ימות: קריאת שלוחת ההיסטוריה ' + HD, async () => {
    const r = await http(YM + 'GetIVR2Dir?token=' + encodeURIComponent(TOKEN) + '&path=' + encodeURIComponent('ivr2:' + HD), {}, 10000, deadline);
    const j = await r.json().catch(() => ({}));
    if (j.responseStatus !== 'OK') throw new Error(JSON.stringify(j).slice(0, 180));
  });
  if (SD) await step('ימות: קריאת תקיית השירים ' + SD, async () => {
    const l = await songList(deadline);
    const n = [...l.tracks.values()].length;
    if (!FFMPEG) throw new Error('אין ffmpeg: אי אפשר לחתוך שירים');
    await execFileP(FFMPEG, ['-version'], { timeout: 8000 });
    return ' (' + n + ' שירים, ' + l.raw.length + ' קבצים שממתינים לחיתוך' + (SONGS_PASS ? '' : ', SONGS_PASS לא הוגדר: דף ההעלאה חסום') + ')';
  });
  await step('Apple Podcasts (חיפוש פודקאסטים)', async () => {
    const r = await podcastSearch('כאן הסכתים', deadline);
    if (!r.length) throw new Error('לא חזרו פרקים מ-Apple Podcasts');
    return ' (' + r.length + ' פרקים)';
  });
  if (JAMENDO_ID) await step('Jamendo (חיפוש שירים)', async () => {
    const r = await jamendoSearch({}, '', 'new', deadline);
    if (!r.length) throw new Error('לא חזרו שירים: בדקו את JAMENDO_CLIENT_ID');
    return ' (' + r.length + ' תוצאות)';
  });
  if (HAS_MEDIA) await step('קבצי שמע להשמעה (' + MEDIA.length + ')', async () => {
    if (!FFMPEG) return ' (ללא ffmpeg: יושמעו רק קבצי mp3 ו-wav)';
    await execFileP(FFMPEG, ['-version'], { timeout: 8000 });
  });
  await step('ימות: העלאה ומחיקה של קובץ', async () => {
    await upload('selftest', silent, deadline);
    const r = await http(YM + 'FileAction?token=' + encodeURIComponent(TOKEN) + '&action=delete&what=' + encodeURIComponent(yp('selftest')), {}, 8000, deadline);
    const j = await r.json().catch(() => ({}));
    if (j.responseStatus !== 'OK') throw new Error('מחיקה: ' + JSON.stringify(j).slice(0, 150));
  });
  if (HD) await step('ימות: שמירת טקסט להיסטוריה (שאלות מוקלדות)', async () => {
    await uploadText('selftesttxt', 'בדיקה', deadline, HD);
    const t = await downloadText('selftesttxt', deadline, HD);
    del('selftesttxt.txt', HD);
    if (t.trim() !== 'בדיקה') throw new Error('הטקסט שחזר שונה: ' + t.slice(0, 40));
  });
  await step('ג׳מיני ' + TEXT_MODEL + ' (קלט אודיו)', async () => {
    const a = await textGen({ contents: [{ role: 'user', parts: [{ inlineData: { mimeType: 'audio/wav', data: silent.toString('base64') } }, { text: 'ענה במילה אחת' }] }] }, deadline);
    if (!(((a.candidates || [])[0] || {}).content)) throw new Error('תשובה ריקה');
    return ' (thinking: ' + (THINK[thinkIdx[TEXT_MODEL] || 0] || 'ברירת מחדל') + ')';
  });
  await step('ג׳מיני ' + TTS_MODEL + ' (קול)', async () => {
    const p = await tts('בדיקה', deadline);
    return ' (' + Math.round(p.length / 1024) + 'KB)';
  });
  res.send(out.join('\n'));
});

// 9. Self-tests for T9 phone keypad logic (kbKey & kbFinal)
(function testKeyboard() {
  const assert = require('assert');
  const type = seq => { const s = { buf: '' }; kbKey(s, seq); return kbFinal(s.buf); };
  assert.strictEqual(type('70770777'), 'ר ש ת');
  assert.strictEqual(type('313'), 'אא');
  assert.strictEqual(type('11'), '1');
  assert.strictEqual(type('44444'), '4');
  assert.strictEqual(type('7755552224'), 'שלום');
  assert.strictEqual(type('33300'), '');
  assert.strictEqual(decodeXml('&#8217; &#x5D0; &amp;'), "’ א &");
})();

app.use((err, req, res, next) => {
  console.error('express', err && err.message);
  if (!res.headersSent) res.type('text/plain').send('id_list_message=t-' + ERR);
});
process.on('unhandledRejection', e => console.error('unhandledRejection', e && e.message));
process.on('uncaughtException', e => console.error('uncaughtException', e && e.message));

app.listen(process.env.PORT || 3000, () => console.log('kav-ai up'));
