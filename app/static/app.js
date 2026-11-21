"use strict";
// All DOM updates use textContent (never innerHTML) so server data can never inject markup.
const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };
let info = null;

const pct = (x) => (x * 100).toFixed(1) + "%";
const headers = () => { const h = { "Content-Type": "application/json" }; const k = $("apikey").value.trim(); if (k) h["X-API-Key"] = k; return h; };

async function api(path, body) {
  const r = await fetch(path, body ? { method: "POST", headers: headers(), body: JSON.stringify(body) } : { headers: headers() });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Request failed (" + r.status + ")");
  return data;
}

function stat(label, value) { const d = el("div", "stat", label); d.prepend(el("b", "", value)); return d; }

function buildForm() {
  const form = $("form");
  for (const f of info.features) {
    const lab = el("label", "", f.replaceAll("_", " "));
    let input;
    if (info.categories[f]) {
      input = el("select"); input.id = "f_" + f;
      for (const c of info.categories[f]) { const o = el("option", "", c); o.value = c; input.append(o); }
      input.value = info.defaults[f];
    } else {
      input = el("input"); input.type = "number"; input.id = "f_" + f; input.step = "any"; input.value = Math.round(info.defaults[f]);
    }
    lab.append(input); form.append(lab);
  }
}

function person() {
  const p = {};
  for (const f of info.features) {
    const v = $("f_" + f).value;
    p[f] = info.categories[f] ? v : Number(v);
  }
  return p;
}

function show(nodes) { const r = $("result"); r.replaceChildren(...nodes); r.classList.remove("hidden"); }

function predictionView(title, pred) {
  const box = el("div");
  box.append(el("h2", "", title));
  const b = el("span", "badge", pred.label); box.append(b, document.createTextNode("  P(>50K) = " + pct(pred.probability)));
  const bar = el("div", "bar"); const fill = el("div"); fill.style.width = pct(pred.probability); bar.append(fill); box.append(bar);
  return box;
}

async function run(btn, fn) {
  btn.disabled = true;
  try { await fn(); } catch (e) { show([el("p", "error", "Error: " + e.message)]); } finally { btn.disabled = false; }
}

$("btn-predict").onclick = (e) => run(e.target, async () => {
  const r = await api("/predict", { instances: [person()] });
  show([predictionView("Prediction", r.predictions[0]), el("p", "stat", "model v" + r.model_version + " · " + r.latency_ms + " ms")]);
});

$("btn-explain").onclick = (e) => run(e.target, async () => {
  const r = await api("/explain", person());
  const nodes = [predictionView("Prediction", r), el("p", "stat", "Typical person scores " + pct(r.base_probability) + ". Green pushes towards >50K, red towards <=50K. Method: " + r.method)];
  const max = Math.max(...r.contributions.map((c) => Math.abs(c.contribution)), 1e-6);
  for (const c of r.contributions) {
    const row = el("div", "contrib");
    row.append(el("span", "", c.feature.replaceAll("_", " ") + " = " + c.value));
    const track = el("div", "track"); const fill = el("div", "fill " + (c.contribution >= 0 ? "pos" : "neg"));
    const w = (Math.abs(c.contribution) / max) * 50; fill.style.width = w + "%"; fill.style.left = c.contribution >= 0 ? "50%" : (50 - w) + "%";
    track.append(fill); row.append(track, el("span", "", (c.contribution >= 0 ? "+" : "") + (c.contribution * 100).toFixed(1) + " pp"));
    nodes.push(row);
  }
  show(nodes);
});

$("btn-cf").onclick = (e) => run(e.target, async () => {
  const r = await api("/counterfactual", person());
  const wrap = el("div", "grid");
  wrap.append(predictionView("As entered (" + r.original_sex + ")", r.original), predictionView("Only sex flipped (" + r.flipped_sex + ")", r.flipped));
  const sign = r.probability_change >= 0 ? "+" : "";
  show([el("h2", "", "Counterfactual fairness check"), wrap,
        el("p", "", "Change in probability: " + sign + (r.probability_change * 100).toFixed(2) + " percentage points"), el("p", "stat", r.interpretation)]);
});

(async function init() {
  try {
    info = await api("/model-info");
    const body = $("model-body");
    body.replaceChildren(stat("variant", info.variant), stat("accuracy", pct(info.performance.accuracy)), stat("ROC-AUC", info.performance.roc_auc.toFixed(3)),
      stat("disparate impact", info.fairness.disparate_impact.toFixed(2) + " (ideal 1)"), stat("stat. parity diff.", info.fairness.statistical_parity_difference.toFixed(3) + " (ideal 0)"),
      stat("serving", info.serving_file));
    buildForm();
  } catch (e) { $("model-body").textContent = "Model not available: " + e.message; }
})();
