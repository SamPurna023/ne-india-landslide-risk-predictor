// Dashboard Application Logic
const DATA = window.DASHBOARD_DATA || [];

function probColor(prob) {
  if (prob < 25) return '#4E7C5F';
  if (prob < 60) return '#D9A441';
  return '#C1502E';
}

function markerColor(d) {
  if (!d.Correct) return '#D9A441';
  return d.Predicted_Label === 'At-Risk' ? '#C1502E' : '#4E7C5F';
}

function sparkline(slope, elev, prob) {
  const W = 58, H = 18;
  let s = (Math.floor(slope * 97 + elev * 0.07) & 0x7fffffff) || 137;
  const rand = () => { s = (s * 1664525 + 1013904223) & 0x7fffffff; return s / 0x7fffffff; };

  const n = 10;
  const slopeNorm = Math.min(slope / 40, 1);
  const pts = [[0, H]];

  for (let i = 1; i < n; i++) {
    const x = (i / n) * W;
    const floor = H * (1 - 0.1 - slopeNorm * 0.65);
    const jag   = slopeNorm * H * 0.55 * rand();
    pts.push([x, Math.max(H * 0.08, floor - jag)]);
  }
  pts.push([W, H]);

  const path = pts.map((p, i) => (i ? 'L' : 'M') + p[0].toFixed(1) + ' ' + p[1].toFixed(1)).join(' ') + ' Z';
  const col  = probColor(prob);
  const gid  = 'g' + (Math.abs(s) % 9999);

  return `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" style="display:block;overflow:visible">
    <defs>
      <linearGradient id="${gid}" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="${col}" stop-opacity="0.75"/>
        <stop offset="100%" stop-color="${col}" stop-opacity="0.08"/>
      </linearGradient>
    </defs>
    <path d="${path}" fill="url(#${gid})" stroke="${col}" stroke-width="1.2" stroke-linejoin="round"/>
  </svg>`;
}

// Leaflet Map Initialization
const map = L.map('map', { center: [25.7, 93.1], zoom: 7, zoomControl: false });
L.control.zoom({ position: 'bottomright' }).addTo(map);

L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', {
  attribution: '&copy; <a href="https://opentopomap.org">OpenTopoMap</a> contributors',
  maxZoom: 17,
}).addTo(map);

function popupHtml(d) {
  const col     = markerColor(d);
  const predCls = d.Predicted_Label === 'At-Risk' ? 'b-risk' : 'b-safe';
  const actCls  = d.Actual_Label    === 'At-Risk' ? 'b-risk' : 'b-safe';
  const predTxt = d.Predicted_Label === 'At-Risk' ? 'AT-RISK' : 'LOWER-RISK';
  const actTxt  = d.Actual_Label    === 'At-Risk' ? 'AT-RISK' : 'LOWER-RISK';
  
  const mismatchRow = d.Correct ? '' : `
    <div class="pop-verdict" style="border-color:rgba(217,164,65,0.4);background:rgba(217,164,65,0.08)">
      <span class="pop-verdict-lbl">&#9651; Mismatch</span>
      <span style="font-family:var(--f-mono);font-size:10px;color:#D9A441">
        Model: <span class="badge ${predCls}">${predTxt}</span>
        &nbsp;/&nbsp;
        Actual: <span class="badge ${actCls}">${actTxt}</span>
      </span>
    </div>`;
  return `<div class="pop-body">
    <div class="pop-name">${d.District}</div>
    <div class="pop-state">${d.State}</div>
    <div class="pop-grid">
      <div class="pcell"><div class="pcell-lbl">At-Risk Prob</div><div class="pcell-val" style="color:${col}">${d.AtRisk_Probability}%</div></div>
      <div class="pcell"><div class="pcell-lbl">Nat. Rank</div><div class="pcell-val">#${d.Rank}</div></div>
      <div class="pcell"><div class="pcell-lbl">Rainfall</div><div class="pcell-val">${d.Avg_Annual_Rainfall_mm} mm</div></div>
      <div class="pcell"><div class="pcell-lbl">Slope</div><div class="pcell-val">${d.Avg_Slope_Degrees}&deg;</div></div>
      <div class="pcell"><div class="pcell-lbl">Elevation</div><div class="pcell-val">${d.Avg_Elevation_m} m</div></div>
      <div class="pcell"><div class="pcell-lbl">NE Rank</div><div class="pcell-val">#${d.Model_Risk_Rank}</div></div>
    </div>
    <div class="pop-verdict">
      <span class="pop-verdict-lbl">Model</span>
      <span class="badge ${predCls}">${predTxt}</span>
      <span class="pop-verdict-lbl" style="margin-left:10px">Actual</span>
      <span class="badge ${actCls}">${actTxt}</span>
    </div>
    ${mismatchRow}
  </div>`;
}

const markers = {};
DATA.forEach((d, i) => {
  const col = markerColor(d);
  const r   = 6 + (d.AtRisk_Probability / 100) * 14;
  const m   = L.circleMarker([d.Latitude, d.Longitude], {
    radius: r,
    fillColor: col,
    color: d.Correct ? col : '#D9A441',
    weight: d.Correct ? 1.5 : 2.5,
    dashArray: d.Correct ? null : '5 3',
    fillOpacity: 0.78,
    opacity: 1,
  }).addTo(map).bindPopup(popupHtml(d), { maxWidth: 280 });
  m.on('click', () => select(i));
  markers[i] = m;
});

// Render Header Stats
const flagged   = DATA.filter(d => d.Predicted_Label === 'At-Risk').length;
const caught    = DATA.filter(d => d.Predicted_Label === 'At-Risk' && d.Actual_Label === 'At-Risk').length;
const totalAR   = DATA.filter(d => d.Actual_Label === 'At-Risk').length;
const acc       = Math.round(DATA.filter(d => d.Correct).length / DATA.length * 100);
const maxProb   = Math.max(...DATA.map(d => d.AtRisk_Probability));

document.getElementById('hstats').innerHTML = `
  <div class="hstat"><div class="hstat-val c-high">${flagged}</div><div class="hstat-lbl">Flagged</div></div>
  <div class="hstat"><div class="hstat-val c-mid">${caught}/${totalAR}</div><div class="hstat-lbl">Caught</div></div>
  <div class="hstat"><div class="hstat-val c-low">${acc}%</div><div class="hstat-lbl">Accuracy</div></div>
  <div class="hstat"><div class="hstat-val c-mute">${maxProb}%</div><div class="hstat-lbl">Max Prob</div></div>
`;

let curFilter = 'all';
let selIdx    = null;

function buildList() {
  const el = document.getElementById('dlist');
  el.innerHTML = '';
  DATA.forEach((d, i) => {
    const isRisk = d.Predicted_Label === 'At-Risk';
    if (curFilter === 'risk' && !isRisk) return;
    if (curFilter === 'safe' &&  isRisk) return;
    const col    = probColor(d.AtRisk_Probability);
    const mmIcon = d.Correct ? '' : '<span style="font-size:9px;color:#D9A441;margin-left:3px" title="Model/actual mismatch">&#9651;</span>';
    const item   = document.createElement('div');
    item.className = 'ditem' + (selIdx === i ? ' sel' : '');
    item.id = 'di' + i;
    item.innerHTML = `
      <span class="d-rank" style="color:${col}">${d.Model_Risk_Rank}</span>
      <div class="d-info">
        <div class="d-name">${d.District}${mmIcon}</div>
        <div class="d-state">${d.State}</div>
      </div>
      <div class="d-right">
        <span class="d-prob" style="color:${col}">${d.AtRisk_Probability}%</span>
        ${sparkline(d.Avg_Slope_Degrees, d.Avg_Elevation_m, d.AtRisk_Probability)}
      </div>`;
    item.onclick = () => select(i);
    el.appendChild(item);
  });
}

function setFilter(f) {
  curFilter = f;
  ['all','risk','safe'].forEach(k => {
    const t = document.getElementById('tab-' + k);
    t.className = 'ftab' + (k === f ? ' fa-' + k : '');
  });
  buildList();
}

function select(i) {
  selIdx = i;
  buildList();
  const d = DATA[i];
  const el = document.getElementById('di' + i);
  if (el) el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  map.flyTo([d.Latitude, d.Longitude], 9, { animate: true, duration: 0.7 });
  markers[i].openPopup();

  const col     = probColor(d.AtRisk_Probability);
  const predCls = d.Predicted_Label === 'At-Risk' ? 'b-risk' : 'b-safe';
  const actCls  = d.Actual_Label    === 'At-Risk' ? 'b-risk' : 'b-safe';

  document.getElementById('det-name').textContent = d.District + (d.Correct ? '' : ' \u25b3');
  document.getElementById('det-sub').textContent  =
    d.State.toUpperCase() + ' · NE MODEL RANK #' + d.Model_Risk_Rank;
  document.getElementById('det-grid').innerHTML = `
    <div class="dgcell"><div class="dgcell-lbl">Prob.</div><div class="dgcell-val" style="color:${col}">${d.AtRisk_Probability}%</div></div>
    <div class="dgcell"><div class="dgcell-lbl">Rainfall</div><div class="dgcell-val">${d.Avg_Annual_Rainfall_mm}</div></div>
    <div class="dgcell"><div class="dgcell-lbl">Slope</div><div class="dgcell-val">${d.Avg_Slope_Degrees}&deg;</div></div>
    <div class="dgcell"><div class="dgcell-lbl">Elev.</div><div class="dgcell-val">${d.Avg_Elevation_m}m</div></div>
    <div class="dgcell"><div class="dgcell-lbl">Model says</div><div class="dgcell-val"><span class="badge ${predCls}">${d.Predicted_Label === 'At-Risk' ? 'RISK' : 'SAFE'}</span></div></div>
    <div class="dgcell"><div class="dgcell-lbl">Actual label</div><div class="dgcell-val"><span class="badge ${actCls}">${d.Actual_Label === 'At-Risk' ? 'RISK' : 'SAFE'}</span></div></div>
  `;
  document.getElementById('detail').className = 'detail open';
}

buildList();
