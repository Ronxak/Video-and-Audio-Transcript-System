// Unit tests for the transcript search highlighter (src/static/transcript_search.js).
//
// Exercises the *shipped* __tsHighlight (the pure, DOM-free core) — including
// case-insensitive matching, unicode (Hindi), HTML escaping, and regex-special
// queries. Run from the project root:
//
//     node tests/test_search_js.js
const fs = require("fs");
const path = require("path");

const src = fs.readFileSync(
  path.join(__dirname, "..", "src", "static", "transcript_search.js"),
  "utf8"
);

// The file is an arrow-function expression that registers helpers on `window`.
global.window = {};
(0, eval)(src)(); // indirect eval in global scope, then invoke to register
const { __tsHighlight } = global.window;

let failures = 0;
function check(name, cond) {
  if (cond) {
    console.log("ok   - " + name);
  } else {
    console.error("FAIL - " + name);
    failures++;
  }
}

// English, case-insensitive
let r = __tsHighlight("The cat sat on the mat", "the", "[ts]");
check("english: 2 matches, case-insensitive", r.count === 2);
check("english: wraps capitalised 'The'", r.html.includes("<mark>The</mark>"));
check("english: wraps lowercase 'the'", r.html.includes("<mark>the</mark>"));
check("english: label inserted after each match", r.html.includes("</mark>[ts]"));

// No matches
r = __tsHighlight("hello world", "xyz", "[ts]");
check("no-match: count is 0", r.count === 0);
check("no-match: no <mark> emitted", !r.html.includes("<mark>"));

// Unicode (Hindi / Devanagari)
r = __tsHighlight("नमस्ते दुनिया नमस्ते सब", "नमस्ते", "[ts]");
check("hindi: 2 matches", r.count === 2);
check("hindi: wraps नमस्ते", r.html.includes("<mark>नमस्ते</mark>"));

// Unicode, case-insensitive within the query path (mixed scripts shouldn't break)
r = __tsHighlight("Café CAFÉ café", "café", "");
check("accented: case-insensitive matches", r.count >= 2);

// HTML is escaped so transcript text can't inject markup
r = __tsHighlight("a <b> & c", "b", "");
check("escapes '<'", r.html.includes("&lt;"));
check("escapes '&'", r.html.includes("&amp;"));

// Regex-special characters in the query are treated literally
r = __tsHighlight("price is $5 (today)", "$5", "");
check("regex-special query '$5' matched literally", r.count === 1);
r = __tsHighlight("a.b a*b axb", "a.b", "");
check("regex-special query 'a.b' not treated as wildcard", r.count === 1);

if (failures > 0) {
  console.error("\n" + failures + " search-JS test(s) failed.");
  process.exit(1);
}
console.log("\nAll search-JS tests passed.");
