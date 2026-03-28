const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const app = express();

app.use(cors());

// ── Static outputs directory ───────────────────────────────────
app.use('/outputs', express.static(path.join(__dirname, 'outputs')));

// ── Pipeline data endpoints ────────────────────────────────────
// In-memory cache so each file is read from disk only once.
const fileCache = new Map();
function loadJSON(filename) {
  if (fileCache.has(filename)) return fileCache.get(filename);
  const fpath = path.join(__dirname, 'outputs', filename);
  if (!fs.existsSync(fpath)) return null;
  const data = JSON.parse(fs.readFileSync(fpath, 'utf8'));
  fileCache.set(filename, data);
  return data;
}

app.get('/api/lii',             (_, res) => { const d = loadJSON('lii.json');                    d ? res.json(d) : res.status(503).json({error:'not exported yet'}); });
app.get('/api/density',         (_, res) => { const d = loadJSON('changepoint_density.json');    d ? res.json(d) : res.status(503).json({error:'not exported yet'}); });
app.get('/api/factors',         (_, res) => { const d = loadJSON('factor_trajectories.json');    d ? res.json(d) : res.status(503).json({error:'not exported yet'}); });
app.get('/api/events',          (_, res) => { const d = loadJSON('events.json');                 d ? res.json(d) : res.status(503).json({error:'not exported yet'}); });
app.get('/api/event-alignment', (_, res) => { const d = loadJSON('event_alignment.json');        d ? res.json(d) : res.status(503).json({error:'not exported yet'}); });
app.get('/api/word-index',      (_, res) => { const d = loadJSON('word_index.json');             d ? res.json(d) : res.status(503).json({error:'not exported yet'}); });

app.get('/api/word/:word', (req, res) => {
  const word = req.params.word.toLowerCase().replace(/[^a-z]/g, '');
  const fpath = path.join(__dirname, 'outputs', 'words', `${word}.json`);
  if (!fs.existsSync(fpath)) return res.status(404).json({ error: 'no model data for this word' });
  res.json(JSON.parse(fs.readFileSync(fpath, 'utf8')));
});

// ── Ngrams proxy (existing) ────────────────────────────────────
const cache = new Map();

app.get('/ngrams', async (req, res) => {
  const { content, year_start = 1800, year_end = 2019 } = req.query;

  if (!content) return res.status(400).json({ error: 'content param required' });

  const key = `${content}__${year_start}__${year_end}`;

  if (cache.has(key)) {
    return res.json(cache.get(key));
  }

  const url = `https://books.google.com/ngrams/json?content=${encodeURIComponent(content)}&year_start=${year_start}&year_end=${year_end}&corpus=en-US&smoothing=0`;

  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Google returned ${response.status}`);
    const data = await response.json();
    cache.set(key, data);
    res.json(data);
  } catch (err) {
    console.error(err);
    res.status(502).json({ error: 'failed to fetch from Google Ngrams' });
  }
});

app.get('/health', (_, res) => res.json({ ok: true }));

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`proxy running on ${PORT}`));
