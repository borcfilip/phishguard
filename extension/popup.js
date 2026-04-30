// =============================================================
// KONFIGURÁCIA – zmeň na svoju Render URL po deploy
// =============================================================
const API_URL = "https://phishguard-api.onrender.com/analyze";
// =============================================================

const urlInput   = document.getElementById("urlInput");
const analyzeBtn = document.getElementById("analyzeBtn");
const loader     = document.getElementById("loader");
const errorBox   = document.getElementById("errorBox");
const card       = document.getElementById("card");
const vIcon      = document.getElementById("vIcon");
const vLabel     = document.getElementById("vLabel");
const vUrl       = document.getElementById("vUrl");
const vConf      = document.getElementById("vConf");
const featList   = document.getElementById("featList");
const lastUrlEl  = document.getElementById("lastUrl");

// Enter key shortcut
urlInput.addEventListener("keydown", e => {
  if (e.key === "Enter") analyzeBtn.click();
});

// ── Main handler ──────────────────────────────────────────────
analyzeBtn.addEventListener("click", async () => {
  const raw = urlInput.value.trim();
  if (!raw) return;

  // Basic URL validation
  let url = raw;
  if (!/^https?:\/\//i.test(url)) url = "https://" + url;
  try { new URL(url); } catch {
    showError("Zadaj platnú URL adresu.");
    return;
  }

  setLoading(true);
  hideAll();

  try {
    const resp = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });

    if (!resp.ok) {
      const txt = await resp.text().catch(() => "");
      throw new Error(`Server error ${resp.status}${txt ? ": " + txt : ""}`);
    }

    const data = await resp.json();
    showResult(data, url);

  } catch (err) {
    showError("Chyba: " + err.message);
  } finally {
    setLoading(false);
  }
});

// ── UI helpers ────────────────────────────────────────────────
function setLoading(on) {
  analyzeBtn.disabled = on;
  loader.classList.toggle("on", on);
}

function hideAll() {
  errorBox.classList.remove("on");
  card.classList.remove("on", "ok", "bad");
}

function showError(msg) {
  errorBox.textContent = msg;
  errorBox.classList.add("on");
}

function showResult(data, url) {
  const isPhish = data.prediction === "phishing";

  // Verdict
  card.className = "card on " + (isPhish ? "bad" : "ok");
  vIcon.textContent  = isPhish ? "⚠️" : "✓";
  vLabel.textContent = isPhish ? "PHISHING" : "LEGITIMATE";
  vConf.textContent  = data.confidence + "%";

  // Truncate URL for display
  const short = url.replace(/^https?:\/\//, "").slice(0, 42);
  vUrl.textContent   = short + (short.length < url.replace(/^https?:\/\//, "").length ? "…" : "");
  lastUrlEl.textContent = short.slice(0, 30) + (short.length > 30 ? "…" : "");

  // LIME features
  featList.innerHTML = "";
  const expl = data.explanation || [];
  const maxAbs = Math.max(...expl.map(f => Math.abs(f.impact)), 0.001);

  expl.forEach(feat => {
    // positive impact = pushes toward phishing (red), negative = toward legit (green)
    const isPos = feat.impact > 0;
    const pct   = (Math.abs(feat.impact) / maxAbs * 100).toFixed(1);
    const cls   = isPos ? "pos" : "neg";
    const sign  = isPos ? "+" : "";

    const row = document.createElement("div");
    row.className = "feat";
    row.innerHTML = `
      <span class="feat-name" title="${feat.condition || feat.feature}">${feat.feature}</span>
      <span class="feat-score ${cls}">${sign}${feat.impact.toFixed(3)}</span>
      <div class="feat-bar-bg">
        <div class="feat-bar-fill ${cls}" style="width:${pct}%"></div>
      </div>
    `;
    featList.appendChild(row);
  });

  card.classList.add("on");
}
