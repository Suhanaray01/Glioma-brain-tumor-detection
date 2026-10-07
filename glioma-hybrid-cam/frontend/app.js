const tabs = [...document.querySelectorAll('.tab')];
const panels = [...document.querySelectorAll('.panel')];
const systemState = document.getElementById('systemState');
const systemStateText = document.getElementById('systemStateText');
const fileInput = document.getElementById('fileInput');
const explanationMode = document.getElementById('explainMode');
const analyzeButton = document.getElementById('analyzeBtn');
const inputPreviewFrame = document.getElementById('inputPreviewFrame');
const inputPreview = document.getElementById('inputPreview');
const resultState = document.getElementById('resultState');
const emptyState = document.getElementById('emptyState');
const resultContent = document.getElementById('resultContent');
let previewUrl = null;

for (const tab of tabs) {
  tab.addEventListener('click', () => {
    const target = tab.dataset.tab;
    for (const item of tabs) {
      const active = item === tab;
      item.classList.toggle('active', active);
      item.setAttribute('aria-selected', String(active));
    }
    for (const panel of panels) panel.classList.toggle('active', panel.id === target);
    if (target === 'performance') loadMetrics();
  });
}

async function loadStatus() {
  try {
    const response = await fetch('/api/health');
    const health = await response.json();
    const ready = response.ok && health.status === 'ok' && health.models_loaded > 0 && health.training_summary === 'ok';
    systemState.dataset.ready = String(ready);
    systemState.dataset.readyState = String(ready);
    systemStateText.textContent = ready ? 'Checkpoint available' : 'No eligible checkpoint';
  } catch {
    systemState.dataset.ready = 'false';
    systemStateText.textContent = 'API unavailable';
  }
}

fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  document.getElementById('fileName').textContent = file ? file.name : 'No image selected';
  analyzeButton.disabled = !file;
  resultContent.hidden = true;
  emptyState.hidden = false;
  resultState.textContent = 'Awaiting analysis';
  emptyState.querySelector('p').textContent = file ? 'Ready to analyze the selected image.' : 'Select an image to begin.';
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  inputPreviewFrame.hidden = !file;
  if (file) {
    previewUrl = URL.createObjectURL(file);
    inputPreview.src = previewUrl;
  } else {
    inputPreview.removeAttribute('src');
  }
});

function showWarnings(warnings) {
  const list = document.getElementById('warningList');
  list.replaceChildren();
  for (const warning of warnings || []) {
    const item = document.createElement('li');
    item.textContent = warning;
    list.append(item);
  }
}

function renderPrediction(prediction) {
  const label = prediction.label === 'glioma' ? 'Glioma class' : 'Notumor-derived class';
  document.getElementById('resultLabel').textContent = label;
  const probability = Number(prediction.glioma_probability);
  document.getElementById('gliomaProbability').textContent = `${(probability * 100).toFixed(1)}%`;
  document.getElementById('probabilityFill').style.width = `${Math.max(0, Math.min(100, probability * 100))}%`;
  document.getElementById('modelName').textContent = prediction.models.map((model) => model.name).join(', ') || '—';
  document.getElementById('aggregation').textContent = prediction.aggregation;
  document.getElementById('inferenceTime').textContent = `${Number(prediction.timing_ms).toFixed(1)} ms`;

  const explanation = prediction.explain || {};
  const heatmapFrame = document.getElementById('heatmapFrame');
  const heatmapImage = document.getElementById('heatmapImage');
  const hybridMode = explanation.requested === 'hybrid_cam';
  const overlay = hybridMode ? explanation.hybrid_cam_overlay : explanation.gradcam_overlay;
  document.getElementById('explanationMethod').textContent = hybridMode ? 'Hybrid-CAM · Grad-CAM + LIME' : 'Grad-CAM';
  heatmapFrame.hidden = !overlay;
  if (overlay) heatmapImage.src = overlay;
  else heatmapImage.removeAttribute('src');
  document.getElementById('explanationState').textContent = explanation.status === 'generated'
    ? `Generated · slice ${(explanation.slice_index ?? 0) + 1}`
    : explanation.status === 'disabled' ? 'Not requested' : 'Unavailable';
  document.getElementById('explanationNote').textContent = explanation.status === 'generated'
    ? hybridMode ? 'Fused attribution from Grad-CAM and positive LIME regions; it is not a tumor boundary or diagnosis.' : 'Grad-CAM shows model attribution, not a tumor boundary or diagnosis.'
    : explanation.reason || 'Grad-CAM could not be generated for this model.';
  const overlapStat = document.getElementById('attributionOverlap');
  overlapStat.hidden = !hybridMode || explanation.attribution_overlap === null || explanation.attribution_overlap === undefined;
  if (!overlapStat.hidden) {
    overlapStat.textContent = `Attribution overlap: ${(Number(explanation.attribution_overlap) * 100).toFixed(1)}% · not classification precision`;
  }
  showWarnings(prediction.warnings);

  emptyState.hidden = true;
  resultContent.hidden = false;
  resultState.textContent = 'Analysis complete';
}

analyzeButton.addEventListener('click', async () => {
  const file = fileInput.files[0];
  if (!file) return;
  if (file.size > 25 * 1024 * 1024) {
    resultState.textContent = 'File exceeds 25 MB';
    return;
  }

  analyzeButton.disabled = true;
  resultState.textContent = 'Analyzing';
  emptyState.hidden = false;
  emptyState.querySelector('p').textContent = 'Running model inference…';
  resultContent.hidden = true;
  try {
    const formData = new FormData();
    formData.append('files', file);
    formData.append('sequence', 'UNSPECIFIED');
    formData.append('explain', explanationMode.value);
    const response = await fetch('/api/predict', { method: 'POST', body: formData });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || 'Analysis failed');
    renderPrediction(payload);
  } catch (error) {
    emptyState.querySelector('p').textContent = error.message;
    resultState.textContent = 'Analysis failed';
  } finally {
    analyzeButton.disabled = !fileInput.files.length;
  }
});

function formatPercent(value) {
  return `${(Number(value) * 100).toFixed(2)}%`;
}

async function loadMetrics() {
  const state = document.getElementById('metricsState');
  const content = document.getElementById('metricsContent');
  state.hidden = false;
  content.hidden = true;
  state.querySelector('p').textContent = 'Loading saved evaluation…';
  try {
    const response = await fetch('/api/metrics');
    const payload = await response.json();
    const evaluation = payload.evaluation || {};
    if (!response.ok || !evaluation.metrics || !Object.keys(evaluation.metrics).length) {
      throw new Error('No held-out evaluation report is available.');
    }
    const labels = [
      ['accuracy', 'Accuracy'],
      ['sensitivity', 'Sensitivity'],
      ['specificity', 'Specificity'],
      ['precision', 'Precision'],
      ['f1', 'F1 score'],
      ['auc', 'ROC AUC'],
    ];
    const grid = document.getElementById('metricGrid');
    grid.replaceChildren();
    for (const [key, label] of labels) {
      if (evaluation.metrics[key] === undefined) continue;
      const cell = document.createElement('div');
      cell.className = 'metric-cell';
      const name = document.createElement('span');
      name.textContent = label;
      const value = document.createElement('strong');
      value.textContent = formatPercent(evaluation.metrics[key]);
      cell.append(name, value);
      grid.append(cell);
    }
    document.getElementById('testCount').textContent = `${evaluation.test_samples} held-out images`;
    document.getElementById('metricDataset').textContent = evaluation.dataset || 'Unspecified dataset';
    document.getElementById('metricModel').textContent = evaluation.model || 'Unspecified model';
    document.getElementById('metricCaveat').textContent = [
      evaluation.modality_note,
      evaluation.negative_class_note,
    ].filter(Boolean).join(' ');
    state.hidden = true;
    content.hidden = false;
  } catch (error) {
    state.querySelector('p').textContent = error.message;
  }
}

loadStatus();
