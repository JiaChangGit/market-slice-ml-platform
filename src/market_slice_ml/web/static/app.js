"use strict";

const state = { status: null, selectedJob: null, polling: null };
const statusText = {
  success: "就緒",
  partial: "部分完成",
  no_data: "無資料",
  disabled: "未啟用",
  failed: "失敗",
  queued: "待處理",
  running: "執行中",
  completed: "完成",
  completed_with_warnings: "完成但有警告",
  interrupted: "已中斷",
  ready: "就緒",
  missing: "缺少",
  waiting: "等待",
};
const jobTypeText = {
  fetch: "Fetch",
  canonical: "Canonical build",
  features: "Feature build",
  labels: "Label build",
  slices: "Slice build",
  train: "Model training",
  report: "Report build",
};
const directionText = { bullish: "偏多", neutral: "中性", bearish: "偏空" };

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function apiToken() { return localStorage.getItem("market-slice-api-token") || ""; }

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  headers.set("Accept", "application/json");
  if (options.body) headers.set("Content-Type", "application/json");
  if (apiToken()) headers.set("X-API-Token", apiToken());
  const response = await fetch(path, { ...options, headers });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(payload.message || `API 回應 ${response.status}`);
    error.payload = payload;
    throw error;
  }
  return payload;
}

function showToast(message, isError = false) {
  const toast = document.querySelector("#toast");
  toast.textContent = message;
  toast.classList.toggle("is-error", isError);
  toast.classList.add("is-visible");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => toast.classList.remove("is-visible"), 4200);
}

function chip(status) {
  return `<span class="chip ${escapeHtml(status)}">${escapeHtml(statusText[status] || status)}</span>`;
}

function formatTime(value) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toISOString().replace(".000Z", "Z");
}

async function loadStatus({ quiet = false } = {}) {
  try {
    state.status = await api("/api/status");
    renderStatus(state.status);
    if (state.selectedJob) {
      const updated = state.status.jobs.find((item) => item.job_id === state.selectedJob.job_id);
      if (updated) selectJob(updated, false);
    }
    if (!quiet) showToast("狀態已更新。", false);
  } catch (error) {
    showApiError(error);
  }
}

function renderStatus(data) {
  renderPipeline(data.stages || []);
  renderEnvironment(data.environment || {});
  renderProviders(data.providers || []);
  renderJobs(data.jobs || []);
  renderSlices(data.slices || []);
  renderModels(data.runs || []);
  renderReports(data.reports || []);
}

function renderPipeline(stages) {
  document.querySelector("#pipeline-stages").innerHTML = stages.map((stage, index) => `
    <li class="pipeline-stage ${escapeHtml(stage.status)}">
      <span class="stage-number">${index + 1}</span>
      <h3>${escapeHtml(stage.label_zh_tw)}</h3>
      <p>${escapeHtml(statusText[stage.status] || stage.detail)}</p>
    </li>`).join("");
}

function renderEnvironment(environment) {
  const rows = [
    ["Python", environment.python || "未偵測"],
    ["CPU", environment.cpu_baseline ? "Baseline" : "未就緒"],
    ["Optional GPU", environment.gpu_available ? "可使用" : "未使用"],
    ["Network", environment.network_enabled ? "已啟用" : "已停用"],
  ];
  document.querySelector("#environment-grid").innerHTML = rows.map(([label, value]) => `
    <div class="environment-item"><div class="label">${label}</div><div class="value">${escapeHtml(value)}</div></div>`).join("");
}

function renderProviders(providers) {
  const body = document.querySelector("#provider-rows");
  body.innerHTML = providers.length ? providers.map((item) => `
    <tr><td>${escapeHtml(item.provider_id)}</td><td>${chip(item.status)}</td><td>${escapeHtml(item.credentials)}</td><td>${escapeHtml(item.message)}</td><td>${escapeHtml(item.suggested_action || "—")}</td></tr>`).join("") : `<tr><td colspan="5" class="empty-row">沒有 Provider 設定。</td></tr>`;
}

function renderJobs(jobs) {
  const body = document.querySelector("#job-rows");
  document.querySelector("#job-count").textContent = `顯示 ${jobs.length} 筆 Jobs`;
  body.innerHTML = jobs.length ? jobs.map((job) => `
    <tr role="button" tabindex="0" data-job-id="${escapeHtml(job.job_id)}" class="${state.selectedJob?.job_id === job.job_id ? "is-selected" : ""}">
      <td class="metric">${escapeHtml(job.job_id.slice(0, 12))}</td><td>${escapeHtml(jobTypeText[job.job_type] || job.job_type)}</td><td>${chip(job.status)}</td><td class="metric">${escapeHtml(formatTime(job.started_at_utc || job.created_at_utc))}</td><td>${escapeHtml(job.message || "—")}</td>
    </tr>`).join("") : `<tr><td colspan="5" class="empty-row">目前沒有 Job。可從 Data 頁面開始建立資料。</td></tr>`;
  body.querySelectorAll("tr[data-job-id]").forEach((row) => {
    const activate = () => {
      const job = jobs.find((item) => item.job_id === row.dataset.jobId);
      if (job) selectJob(job);
    };
    row.addEventListener("click", activate);
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") activate();
    });
  });
}

function selectJob(job, open = true) {
  state.selectedJob = job;
  if (open) setDetailOpen(true);
  const warnings = (job.warnings || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("");
  const diagnostics = JSON.stringify(job.result || {}, null, 2);
  document.querySelector("#job-detail-content").innerHTML = `
    <div class="detail-title">${escapeHtml(job.job_id)}</div>
    ${chip(job.status)}
    <dl class="detail-grid">
      <dt>類型</dt><dd>${escapeHtml(jobTypeText[job.job_type] || job.job_type)}</dd>
      <dt>建立時間</dt><dd class="metric">${escapeHtml(formatTime(job.created_at_utc))}</dd>
      <dt>開始時間</dt><dd class="metric">${escapeHtml(formatTime(job.started_at_utc))}</dd>
      <dt>結束時間</dt><dd class="metric">${escapeHtml(formatTime(job.finished_at_utc))}</dd>
      <dt>Job log</dt><dd class="metric">${escapeHtml(job.log_path || "—")}</dd>
    </dl>
    <section class="detail-block"><h3>${job.status === "failed" ? "錯誤" : "結果"}</h3><p>${escapeHtml(job.message || "尚無訊息")}</p>${warnings ? `<ul>${warnings}</ul>` : ""}</section>
    <section class="detail-block suggestion"><h3>建議處理</h3><p>${escapeHtml(job.suggested_action || "不需要額外處理。")}</p></section>
    <section class="detail-block"><h3>Diagnostics</h3><pre class="diagnostics">${escapeHtml(diagnostics)}</pre></section>`;
  if (state.status) renderJobs(state.status.jobs || []);
}

function renderSlices(slices) {
  const body = document.querySelector("#slice-rows");
  body.innerHTML = slices.length ? slices.map((item) => `
    <tr><td>${escapeHtml(item.pair?.pair_id)}</td><td class="metric">${escapeHtml((item.symbols || []).length)}</td><td class="metric">${escapeHtml(item.pair?.train_end_utc)}</td><td class="metric">${escapeHtml(item.pair?.val_start_utc)}</td><td class="metric">${escapeHtml(String(item.fingerprint || "").slice(0, 18))}</td></tr>`).join("") : `<tr><td colspan="5" class="empty-row">尚未建立 Slice manifests。</td></tr>`;
}

function metric(run, model, name) { return run.metrics?.[model]?.[name]; }
function formatMetric(value) { return typeof value === "number" ? value.toFixed(4) : "—"; }

function renderModels(runs) {
  const body = document.querySelector("#model-rows");
  body.innerHTML = runs.length ? runs.map((run) => `
    <tr><td class="metric">${escapeHtml(String(run.run_id).slice(0, 12))}</td><td>${escapeHtml(run.pair_id)}</td><td>${escapeHtml(run.horizon)}</td><td class="metric">${escapeHtml(formatTime(run.created_at_utc))}</td><td class="metric">${formatMetric(metric(run, "gbm", "direction_accuracy"))}</td><td class="metric">${formatMetric(metric(run, "lstm", "direction_accuracy"))}</td><td class="metric">${formatMetric(metric(run, "gnn", "direction_accuracy"))}</td></tr>`).join("") : `<tr><td colspan="7" class="empty-row">尚未完成 Model run。</td></tr>`;
}

function renderReports(reports) {
  const root = document.querySelector("#report-list");
  root.innerHTML = reports.length ? reports.map((report) => `
    <div class="report-row"><span>${escapeHtml(report.name)}</span><a href="${escapeHtml(report.url)}" target="_blank" rel="noopener">開啟 Report</a></div>`).join("") : `<p class="empty-state">尚未建立 Report。</p>`;
}

async function submitJob(endpoint, payload = {}) {
  try {
    const job = await api(endpoint, { method: "POST", body: JSON.stringify(payload) });
    showToast(`Job ${job.job_id.slice(0, 8)} 已送出。`);
    selectJob(job);
    await loadStatus({ quiet: true });
  } catch (error) { showApiError(error); }
}

function showApiError(error) {
  const payload = error.payload || {};
  const action = payload.suggested_action ? ` ${payload.suggested_action}` : "";
  showToast(`${payload.message || error.message}${action}`, true);
}

function switchPanel(panel) {
  document.querySelectorAll(".nav-item").forEach((item) => item.classList.toggle("is-active", item.dataset.panel === panel));
  document.querySelectorAll(".panel-view").forEach((item) => item.classList.toggle("is-active", item.id === `panel-${panel}`));
  document.body.classList.remove("mobile-nav-open");
}

function setDetailOpen(open) {
  document.body.classList.toggle("detail-open", open);
  document.querySelector("#job-detail").setAttribute("aria-hidden", String(!open));
}

function initializeForms() {
  const today = new Date();
  const monthAgo = new Date(today);
  monthAgo.setUTCDate(today.getUTCDate() - 30);
  const dateValue = (date) => date.toISOString().slice(0, 10);
  document.querySelector('#fetch-form [name="start"]').value = dateValue(monthAgo);
  document.querySelector('#fetch-form [name="end"]').value = dateValue(today);

  document.querySelector("#fetch-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const values = new FormData(event.currentTarget);
    const symbols = String(values.get("symbols") || "").split(",").map((item) => item.trim()).filter(Boolean);
    submitJob("/api/jobs/fetch", {
      provider: values.get("provider"), interval: values.get("interval"), symbols,
      start_utc: `${values.get("start")}T00:00:00Z`, end_utc: `${values.get("end")}T23:59:59Z`,
    });
  });
  document.querySelectorAll(".job-action").forEach((button) => button.addEventListener("click", () => submitJob(`/api/jobs/${button.dataset.job}`, { symbols: [] })));
  document.querySelector("#build-slices").addEventListener("click", () => submitJob("/api/jobs/slices"));
  document.querySelector("#build-report").addEventListener("click", () => submitJob("/api/jobs/report"));
  document.querySelector("#train-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const values = new FormData(event.currentTarget);
    submitJob("/api/jobs/train", { horizon: values.get("horizon"), force: values.get("force") === "on" });
  });
  document.querySelector("#prediction-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const values = new FormData(event.currentTarget);
    try {
      const prediction = await api("/api/predictions", { method: "POST", body: JSON.stringify({ symbol: values.get("symbol"), horizon: values.get("horizon") }) });
      renderPrediction(prediction);
    } catch (error) { showApiError(error); }
  });
}

function renderPrediction(value) {
  const fields = [
    ["Symbol", value.symbol], ["Horizon", value.horizon], ["Direction", directionText[value.direction] || value.direction],
    ["Expected return", `${(value.expected_return * 100).toFixed(3)}%`],
    ["Expected volatility", `${(value.expected_volatility * 100).toFixed(2)}%`],
    ["Confidence", `${value.confidence_score.toFixed(1)} / 100`],
  ];
  document.querySelector("#prediction-result").innerHTML = `<div class="prediction-grid">${fields.map(([label, content]) => `<div class="prediction-field"><div class="label">${label}</div><div class="value metric">${escapeHtml(content)}</div></div>`).join("")}</div>`;
}

function initializeNavigation() {
  document.querySelectorAll(".nav-item").forEach((button) => button.addEventListener("click", () => switchPanel(button.dataset.panel)));
  document.querySelector("#collapse-nav").addEventListener("click", () => document.body.classList.toggle("nav-collapsed"));
  document.querySelector("#mobile-menu").addEventListener("click", () => document.body.classList.toggle("mobile-nav-open"));
  document.querySelector("#close-detail").addEventListener("click", () => setDetailOpen(false));
  document.querySelector("#refresh-status").addEventListener("click", () => loadStatus());
  document.querySelector("#open-diagnostics").addEventListener("click", () => {
    const environment = state.status?.environment || {};
    document.querySelector("#job-detail-content").innerHTML = `<div class="detail-title">環境 diagnostics</div><pre class="diagnostics">${escapeHtml(JSON.stringify(environment, null, 2))}</pre>`;
    setDetailOpen(true);
  });
  document.querySelector("#settings-button").addEventListener("click", () => {
    const value = window.prompt("輸入 WEB_API_TOKEN；留空會移除目前 token。", apiToken());
    if (value === null) return;
    if (value.trim()) localStorage.setItem("market-slice-api-token", value.trim());
    else localStorage.removeItem("market-slice-api-token");
    loadStatus({ quiet: true });
  });
}

document.addEventListener("DOMContentLoaded", async () => {
  initializeNavigation();
  initializeForms();
  await loadStatus({ quiet: true });
  state.polling = window.setInterval(() => loadStatus({ quiet: true }), 5000);
});
