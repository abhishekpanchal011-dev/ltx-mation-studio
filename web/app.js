const state = {
  mode: 'image', duration: 10, aspect: '16:9', image: null,
  callId: null, jobId: null, jobStartedAt: null, estimateSeconds: 360,
  polling: null, grossUsage: Number(localStorage.getItem('ltx_gross_usage') || 0),
  deferredInstall: null,
};

const $ = id => document.getElementById(id);
const qs = s => document.querySelector(s);
const qsa = s => [...document.querySelectorAll(s)];
const ACTIVE_JOB_KEY = 'ltx_active_job_v2';

function toast(message) {
  const el = $('toast'); el.textContent = message; el.classList.add('show');
  setTimeout(() => el.classList.remove('show'), 2800);
}

function estimateCost(duration = state.duration) {
  const segments = duration <= 10 ? 1 : Math.ceil(duration / 10);
  const minutes = segments * 6;
  const gpu = (minutes / 60) * 1.9512;
  const overhead = gpu * 0.05;
  return { segments, minutes, cost: gpu + overhead };
}

function updateEstimate() {
  const x = estimateCost();
  const continuity = x.segments > 1 ? ` · ${x.segments} linked clips` : '';
  $('costEstimate').textContent = `~$${x.cost.toFixed(2)} · ~${x.minutes} min${continuity}`;
  $('durationHint').textContent = state.duration <= 10
    ? `Native ${state.duration}-second generation`
    : `${x.segments} continuity-preserving segments, stitched automatically`;
  $('mobileEstimateLabel').textContent = `${state.duration}s · ~${x.minutes} min`;
  $('mobileCostEstimate').textContent = `~$${x.cost.toFixed(2)}`;
  $('budgetUsed').textContent = `$${state.grossUsage.toFixed(2)}`;
  $('budgetBar').style.width = `${Math.min(100, (state.grossUsage / 32) * 100)}%`;
}

function switchView(view) {
  qsa('.nav-item,.mobile-nav-item').forEach(b => b.classList.toggle('active', b.dataset.view === view));
  qsa('.view').forEach(x => x.classList.remove('active-view'));
  $(`${view}View`).classList.add('active-view');
  const dock = qs('.mobile-generate-dock');
  if (dock) dock.style.display = view === 'create' ? '' : 'none';
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function setMode(mode) {
  state.mode = mode;
  qsa('.mode-tab').forEach(b => b.classList.toggle('active', b.dataset.mode === mode));
  $('imagePanel').style.display = mode === 'image' ? '' : 'none';
}
function setDuration(value) {
  state.duration = Number(value);
  qsa('#durationControl button').forEach(b => b.classList.toggle('active', Number(b.dataset.value) === state.duration));
  updateEstimate();
}
function setAspect(value) {
  state.aspect = value;
  qsa('#aspectControl button').forEach(b => b.classList.toggle('active', b.dataset.value === value));
  $('outputBadge').textContent = value === '16:9' ? '1280×720 · 24 fps' : '720×1280 · 24 fps';
}

function previewImage(file) {
  if (!file || !file.type.startsWith('image/')) return toast('Choose an image file.');
  if (file.size > 20 * 1024 * 1024) return toast('Image must be 20 MB or smaller.');
  state.image = file;
  $('imagePreview').src = URL.createObjectURL(file);
  $('dropZone').classList.add('has-image');
}
function removeImage() {
  state.image = null; $('imageInput').value = ''; $('cameraInput').value = '';
  $('imagePreview').removeAttribute('src'); $('dropZone').classList.remove('has-image');
}

async function checkBackend() {
  const dot = qs('.status-pill .dot');
  try {
    const r = await fetch('/api/health', { cache: 'no-store' }); if (!r.ok) throw new Error();
    const d = await r.json(); $('backendStatus').textContent = `${d.gpu} ready`; dot?.classList.add('ok');
  } catch { $('backendStatus').textContent = 'Backend unavailable'; dot?.classList.add('bad'); }
}

function setBusy(busy) {
  $('generateBtn').disabled = busy;
  $('mobileGenerateBtn').disabled = busy;
  qs('.mobile-generate-dock')?.classList.toggle('busy', busy);
}
function showProgress() {
  switchView('create');
  $('emptyOutput').classList.add('hidden'); $('videoOutput').classList.add('hidden');
  $('progressOutput').classList.remove('hidden'); setBusy(true); $('progressBar').style.width = '3%';
  updateProgressClock();
  setTimeout(() => $('progressOutput').scrollIntoView({ behavior: 'smooth', block: 'center' }), 100);
}
function hideProgressWithError(message) {
  stopPolling(); clearActiveJob(); $('progressOutput').classList.add('hidden');
  $('emptyOutput').classList.remove('hidden'); setBusy(false); toast(message);
}
function updateProgressClock() {
  if (!state.jobStartedAt) return;
  const elapsed = Math.floor((Date.now() - state.jobStartedAt) / 1000);
  const remaining = Math.max(0, state.estimateSeconds - elapsed);
  const fmt = s => `${Math.floor(s/60)}:${String(s%60).padStart(2,'0')}`;
  $('elapsedText').textContent = `${fmt(elapsed)} elapsed`;
  $('remainingText').textContent = remaining ? `~${fmt(remaining)} remaining` : 'Finishing output';
  $('progressBar').style.width = `${Math.min(94, 3 + (elapsed / Math.max(state.estimateSeconds, 1)) * 90)}%`;
  const segments = estimateCost(state.duration).segments;
  if (segments > 1) {
    const segment = Math.min(segments, Math.floor(elapsed / 360) + 1);
    $('progressTitle').textContent = `Generating segment ${segment} of ${segments}`;
    $('progressText').textContent = 'Maintaining identity and motion continuity from the previous segment.';
  } else {
    $('progressTitle').textContent = 'Generating with LTX‑2.5';
    $('progressText').textContent = '720p distilled render on an L40S 48 GB worker.';
  }
}
function stopPolling() { if (state.polling) clearInterval(state.polling); state.polling = null; }
function persistActiveJob() {
  localStorage.setItem(ACTIVE_JOB_KEY, JSON.stringify({
    callId: state.callId, jobId: state.jobId, startedAt: state.jobStartedAt,
    estimateSeconds: state.estimateSeconds, duration: state.duration, aspect: state.aspect,
  }));
}
function clearActiveJob() { localStorage.removeItem(ACTIVE_JOB_KEY); state.callId = null; state.jobId = null; }

function saveHistory(meta) {
  const history = JSON.parse(localStorage.getItem('ltx_history') || '[]');
  history.unshift({ jobId: meta.job_id, duration: meta.duration, seed: meta.seed, elapsed: meta.elapsed_seconds,
    resolution: meta.resolution, createdAt: new Date().toISOString(), prompt: $('prompt').value.trim().slice(0,160) });
  localStorage.setItem('ltx_history', JSON.stringify(history.slice(0,50))); renderHistory();
}

async function generate() {
  const prompt = $('prompt').value.trim();
  if (!prompt) return toast('Add a prompt first.');
  if (state.mode === 'image' && !state.image) return toast('Add a first frame.');
  const estimate = estimateCost();
  if (state.grossUsage + estimate.cost > 32) return toast('This job would cross your local $32 usage guard.');

  const data = new FormData();
  data.append('mode', state.mode); data.append('prompt', prompt); data.append('duration', String(state.duration));
  data.append('aspect_ratio', state.aspect); data.append('seed', $('seed').value || '-1');
  data.append('camera_motion', $('cameraMotion').value); data.append('prompt_enhance', $('promptEnhance').checked ? 'true' : 'false');
  data.append('generate_audio', $('audioToggle').checked ? 'true' : 'false'); if (state.image) data.append('image', state.image);

  state.jobStartedAt = Date.now(); state.estimateSeconds = estimate.minutes * 60; showProgress();
  try {
    const r = await fetch('/api/generate', { method: 'POST', body: data }); const body = await r.json();
    if (!r.ok) throw new Error(body.detail || 'Unable to start generation');
    state.callId = body.call_id; state.jobId = body.job_id; state.estimateSeconds = body.estimated_seconds || state.estimateSeconds;
    persistActiveJob(); await pollJob(); if (!state.polling && state.callId) state.polling = setInterval(pollJob, 4000);
  } catch (err) { hideProgressWithError(err.message || 'Generation could not be started.'); }
}

async function pollJob() {
  if (!state.callId) return; updateProgressClock();
  try {
    const r = await fetch(`/api/jobs/${encodeURIComponent(state.callId)}`, { cache: 'no-store' });
    if (r.status === 202) return;
    const body = await r.json(); if (!r.ok) throw new Error(body.detail || 'Generation failed'); completeJob(body);
  } catch (err) { hideProgressWithError(err.message || 'Generation failed.'); }
}

function completeJob(meta) {
  stopPolling(); clearActiveJob(); $('progressBar').style.width = '100%';
  const estimate = estimateCost(meta.duration || state.duration); state.grossUsage += estimate.cost;
  localStorage.setItem('ltx_gross_usage', String(state.grossUsage)); updateEstimate(); saveHistory(meta);
  setTimeout(() => {
    $('progressOutput').classList.add('hidden'); $('videoOutput').classList.remove('hidden'); setBusy(false);
    const url = `/api/download/${meta.job_id}`; $('resultVideo').src = url; $('downloadBtn').href = url;
    $('metaDuration').textContent = `${meta.duration}s`; $('metaSeed').textContent = meta.seed;
    $('metaRender').textContent = `${Math.max(1,Math.round(meta.elapsed_seconds/60))} min · ${meta.segments} segment${meta.segments>1?'s':''}`;
    toast('Video completed.'); setTimeout(() => $('videoOutput').scrollIntoView({ behavior:'smooth', block:'start' }), 100);
  }, 400);
}

async function cancelJob() {
  if (!state.callId) return;
  try { await fetch(`/api/jobs/${encodeURIComponent(state.callId)}/cancel`, { method:'POST' }); } catch {}
  stopPolling(); clearActiveJob(); $('progressOutput').classList.add('hidden'); $('emptyOutput').classList.remove('hidden'); setBusy(false); toast('Generation cancelled.');
}

function renderHistory() {
  const history = JSON.parse(localStorage.getItem('ltx_history') || '[]'); const list = $('historyList'); list.innerHTML='';
  if (!history.length) { list.innerHTML='<div class="history-empty">No generations yet.</div>'; return; }
  history.forEach(item => {
    const row=document.createElement('div'); row.className='history-item';
    row.innerHTML=`<div><strong>${escapeHtml(item.prompt||'LTX generation')}</strong><span>${item.duration}s · ${item.resolution} · seed ${item.seed} · ${new Date(item.createdAt).toLocaleString()}</span></div><div class="history-actions"><a class="secondary-button" href="/api/download/${item.jobId}" target="_blank">Open</a><a class="secondary-button" href="/api/download/${item.jobId}" download>Download</a></div>`;
    list.appendChild(row);
  });
}
function escapeHtml(s){return String(s).replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));}

function restoreActiveJob() {
  try {
    const saved = JSON.parse(localStorage.getItem(ACTIVE_JOB_KEY) || 'null'); if (!saved?.callId) return false;
    state.callId=saved.callId; state.jobId=saved.jobId; state.jobStartedAt=saved.startedAt || Date.now();
    state.estimateSeconds=saved.estimateSeconds || 360; if (saved.duration) setDuration(saved.duration); if (saved.aspect) setAspect(saved.aspect);
    showProgress(); pollJob(); state.polling=setInterval(pollJob,4000); return true;
  } catch { localStorage.removeItem(ACTIVE_JOB_KEY); return false; }
}

qsa('.mode-tab').forEach(b=>b.addEventListener('click',()=>setMode(b.dataset.mode)));
qsa('#durationControl button').forEach(b=>b.addEventListener('click',()=>setDuration(b.dataset.value)));
qsa('#aspectControl button').forEach(b=>b.addEventListener('click',()=>setAspect(b.dataset.value)));
qsa('.nav-item,.mobile-nav-item').forEach(b=>b.addEventListener('click',()=>switchView(b.dataset.view)));

$('imageInput').addEventListener('change',e=>e.target.files[0]&&previewImage(e.target.files[0]));
$('cameraInput').addEventListener('change',e=>e.target.files[0]&&previewImage(e.target.files[0]));
$('galleryBtn').addEventListener('click',e=>{e.preventDefault();e.stopPropagation();$('imageInput').click();});
$('cameraBtn').addEventListener('click',e=>{e.preventDefault();e.stopPropagation();$('cameraInput').click();});
$('removeImage').addEventListener('click',e=>{e.preventDefault();e.stopPropagation();removeImage();});
['dragenter','dragover'].forEach(ev=>$('dropZone').addEventListener(ev,e=>{e.preventDefault();$('dropZone').classList.add('dragging');}));
['dragleave','drop'].forEach(ev=>$('dropZone').addEventListener(ev,e=>{e.preventDefault();$('dropZone').classList.remove('dragging');}));
$('dropZone').addEventListener('drop',e=>{const f=e.dataTransfer.files[0];if(f)previewImage(f);});
$('prompt').addEventListener('input',()=> $('charCount').textContent=`${$('prompt').value.length} / 6000`);
$('promptExample').addEventListener('click',()=>{ $('prompt').value='A cinematic medium shot of the subject looking toward the horizon as wind moves through the scene. The subject turns naturally and takes a few steps forward while the background responds with realistic depth and parallax. Golden-hour light, subtle atmosphere, realistic motion, detailed textures, coherent hands and face, no text or logos.'; $('prompt').dispatchEvent(new Event('input')); });
$('randomSeed').addEventListener('click',()=> $('seed').value=Math.floor(Math.random()*2147483646)+1);
$('generateBtn').addEventListener('click',generate); $('mobileGenerateBtn').addEventListener('click',generate); $('cancelBtn').addEventListener('click',cancelJob);
$('newVideoBtn').addEventListener('click',()=>{ $('videoOutput').classList.add('hidden'); $('emptyOutput').classList.remove('hidden'); $('resultVideo').removeAttribute('src'); $('resultVideo').load(); window.scrollTo({top:0,behavior:'smooth'}); });
$('clearHistory').addEventListener('click',()=>{localStorage.removeItem('ltx_history');renderHistory();});

document.addEventListener('visibilitychange',()=>{ if(!document.hidden&&state.callId) pollJob(); });
window.addEventListener('beforeinstallprompt',e=>{e.preventDefault();state.deferredInstall=e;$('installBtn').classList.remove('hidden');});
$('installBtn').addEventListener('click',async()=>{if(!state.deferredInstall)return;state.deferredInstall.prompt();await state.deferredInstall.userChoice;state.deferredInstall=null;$('installBtn').classList.add('hidden');});
window.addEventListener('appinstalled',()=>toast('LTX Motion installed.'));
if('serviceWorker' in navigator) window.addEventListener('load',()=>navigator.serviceWorker.register('/sw.js').catch(()=>{}));

setMode('image'); setDuration(10); setAspect('16:9'); updateEstimate(); renderHistory(); checkBackend(); switchView('create');
restoreActiveJob();
