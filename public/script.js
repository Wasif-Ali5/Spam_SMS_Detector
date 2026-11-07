const $ = (id) => document.getElementById(id);

const form = $("form");
const input = $("message");
const submitBtn = $("submit");
const errorBox = $("error");
const resultBox = $("result");

input.addEventListener("input", () => {
  $("counter").textContent = `${input.value.length} / 1000`;
});

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    input.value = chip.dataset.text;
    input.dispatchEvent(new Event("input"));
  });
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  errorBox.classList.add("hidden");
  submitBtn.disabled = true;
  submitBtn.textContent = "Checking...";

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: input.value }),
    });
    if (!res.ok) throw new Error(`Server returned ${res.status}`);
    showResult(await res.json());
  } catch (err) {
    resultBox.classList.add("hidden");
    errorBox.textContent = "Something went wrong: " + err.message;
    errorBox.classList.remove("hidden");
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Check message";
  }
});

function showResult(data) {
  const isSpam = data.prediction === "spam";
  const pct = (data.spam_probability * 100).toFixed(1);

  const badge = $("badge");
  badge.textContent = isSpam ? "SPAM" : "HAM (not spam)";
  badge.className = "badge " + (isSpam ? "spam" : "ham");
  $("probText").textContent = `Spam probability: ${pct}%`;

  const fill = $("barFill");
  fill.style.width = pct + "%";
  fill.style.background = isSpam ? "var(--spam)" : "var(--ham)";

  const words = $("words");
  words.innerHTML = "";
  if (data.top_words.length === 0) {
    words.textContent = "No known words found in this message.";
  }
  data.top_words.forEach((w) => {
    const span = document.createElement("span");
    span.className = "word " + w.effect;
    span.textContent = `${w.word} (${w.effect})`;
    words.appendChild(span);
  });

  resultBox.classList.remove("hidden");
}

async function loadMetrics() {
  try {
    const m = await (await fetch("/api/metrics")).json();

    const items = [
      ["Accuracy", m.accuracy], ["Precision", m.precision],
      ["Recall", m.recall], ["F1-score", m.f1],
    ];
    $("metrics").innerHTML = items
      .map(([label, v]) => `<div class="metric"><div class="value">${(v * 100).toFixed(1)}%</div><div class="label">${label}</div></div>`)
      .join("");

    $("matrix").innerHTML = `
      <div></div><div class="head">Predicted ham</div><div class="head">Predicted spam</div>
      <div class="head">Actual ham</div>
      <div class="cell good"><b>${m.tn}</b>correct (TN)</div>
      <div class="cell bad"><b>${m.fp}</b>ham blocked (FP)</div>
      <div class="head">Actual spam</div>
      <div class="cell bad"><b>${m.fn}</b>spam missed (FN)</div>
      <div class="cell good"><b>${m.tp}</b>spam caught (TP)</div>`;

    $("note").textContent = `Evaluated on ${m.test_size} unseen messages (trained on ${m.train_size}).`;
  } catch {
    $("note").textContent = "Could not load model metrics.";
  }
}

loadMetrics();