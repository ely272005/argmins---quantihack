const express = require('express');
const cors = require('cors');
const app = express();

app.use(cors());

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
