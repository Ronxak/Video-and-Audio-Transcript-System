// Client-side transcript search helpers (case-insensitive, unicode-safe).
//
// Loaded once on page load via Gradio's demo.load(js=...). Registers three
// functions on `window`; the search box/button call __tsSearch / __tsClear.
// Everything is pure DOM work, so search is instant and never blocks the server.
//
// __tsHighlight is a pure string function (no DOM) and is unit-tested directly
// in tests/test_search_js.js — keep it side-effect free.
() => {
  const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const escRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const fmt = (t) => {
    t = Math.max(0, Math.floor(Number(t) || 0));
    const m = Math.floor(t / 60), s = t % 60;
    return (m < 10 ? '0' : '') + m + ':' + (s < 10 ? '0' : '') + s;
  };

  // Wrap every case-insensitive match of `query` in `raw` with <mark>, and
  // insert `label` (the timestamp) right after each match. Returns {html, count}.
  window.__tsHighlight = (raw, query, label) => {
    if (!query) return { html: esc(raw), count: 0 };
    let re;
    try { re = new RegExp(escRe(query), 'gi'); }
    catch (e) { return { html: esc(raw), count: 0 }; }
    let out = '', last = 0, m, count = 0;
    while ((m = re.exec(raw)) !== null) {
      count++;
      out += esc(raw.slice(last, m.index)) + '<mark>' + esc(m[0]) + '</mark>' + (label || '');
      last = m.index + m[0].length;
      if (m.index === re.lastIndex) re.lastIndex++;  // guard zero-width matches
    }
    out += esc(raw.slice(last));
    return { html: out, count: count };
  };

  // Remove all highlights and timestamp labels, restoring the original text.
  window.__tsClear = () => {
    const root = document.getElementById('ts-transcript');
    if (root) {
      root.querySelectorAll('.ts-cue').forEach((cue) => {
        const span = cue.querySelector('.ts-text');
        if (span) span.textContent = cue.dataset.text || '';
      });
    }
    const msg = document.getElementById('ts-search-msg');
    if (msg) { msg.textContent = ''; msg.style.display = 'none'; }
  };

  // Highlight all matches of `query` across every cue and show a result count.
  window.__tsSearch = (query) => {
    window.__tsClear();
    const root = document.getElementById('ts-transcript');
    const msg = document.getElementById('ts-search-msg');
    if (!root) return;
    query = (query || '').trim();
    if (!query) return;

    let total = 0, cuesHit = 0;
    root.querySelectorAll('.ts-cue').forEach((cue) => {
      const raw = cue.dataset.text || '';
      const label = '<sup class="ts-match">[' + fmt(cue.dataset.start) +
                    '–' + fmt(cue.dataset.end) + ']</sup>';
      const res = window.__tsHighlight(raw, query, label);
      if (res.count > 0) {
        total += res.count;
        cuesHit++;
        const span = cue.querySelector('.ts-text');
        if (span) span.innerHTML = res.html;
      }
    });

    if (msg) {
      msg.textContent = total === 0
        ? 'No matches for “' + query + '”.'
        : total + ' match' + (total === 1 ? '' : 'es') + ' in ' +
          cuesHit + ' cue' + (cuesHit === 1 ? '' : 's') + '.';
      msg.style.display = 'block';
    }
  };
}
