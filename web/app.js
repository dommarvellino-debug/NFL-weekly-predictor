/* GridironML — data-driven renderers for the four static screens.
 * All numbers come from data/predictions.json and data/model-metrics.json.
 */

const DATA_URL = 'data/predictions.json';
const METRICS_URL = 'data/model-metrics.json';

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const TEAM_COLORS = {
  ARI: '#97233F', ATL: '#A71930', BAL: '#241773', BUF: '#00338D',
  CAR: '#0085CA', CHI: '#0B162A', CIN: '#FB4F14', CLE: '#311D00',
  DAL: '#041E42', DEN: '#FB4F14', DET: '#0076B6', GB: '#203731',
  HOU: '#03202F', IND: '#002C5F', JAX: '#006778', KC: '#E31837',
  LA: '#003594', LAC: '#0080C6', LV: '#000000', MIA: '#008E97',
  MIN: '#4F2683', NE: '#002244', NO: '#D3BC8D', NYG: '#0B2265',
  NYJ: '#125740', PHI: '#004C54', PIT: '#FFB612', SF: '#AA0000',
  SEA: '#002244', TB: '#D50A0A', TEN: '#0C2340', WAS: '#5A1414',
};
const DARK_TEXT_TEAMS = new Set(['PIT', 'NO']);

const TIER = {
  high: { label: 'High Conf', cls: 'bg-primary-fixed text-on-primary-fixed', icon: 'check_circle', bar: 'bg-primary', dot: 'bg-primary' },
  moderate: { label: 'Moderate', cls: 'bg-secondary-fixed text-on-secondary-fixed', icon: 'bolt', bar: 'bg-secondary', dot: 'bg-secondary' },
  low: { label: 'Low Conf', cls: 'bg-tertiary-fixed text-on-tertiary-fixed', icon: 'balance', bar: 'bg-tertiary', dot: 'bg-tertiary' },
};

const pickSide = (g) => (g.home.abbr === g.predicted_winner ? 'home' : 'away');
const pickTeam = (g) => g[pickSide(g)];
const shortName = (name) => name.split(' ').pop();
const pct = (p) => `${Math.round(p * 100)}%`;
const signed = (n, digits = 1) => `${n > 0 ? '+' : ''}${Number(n).toFixed(digits)}`;
const winCount = (record) => parseInt(record, 10) || 0;

function kickoff(g) {
  const [h, m] = g.gametime.split(':').map(Number);
  const ap = h >= 12 ? 'PM' : 'AM';
  const hh = h % 12 || 12;
  const wd = { Thursday: 'Thu', Sunday: 'Sun', Monday: 'Mon' }[g.weekday] || g.weekday.slice(0, 3);
  return `${wd} ${hh}:${String(m).padStart(2, '0')} ${ap} ET`;
}

function logo(abbr, size = 'w-8 h-8', textSize = 'text-headline-sm') {
  const bg = TEAM_COLORS[abbr] || '#283044';
  const fg = DARK_TEXT_TEAMS.has(abbr) ? 'text-[#101820]' : 'text-white';
  return `<div class="${size} rounded-lg ${fg} ${textSize} flex items-center justify-center font-bold shadow-sm flex-shrink-0" style="background-color:${bg}">${abbr}</div>`;
}

function tierBadge(tier, extra = '') {
  const t = TIER[tier] || TIER.low;
  return `<span class="font-label-sm text-label-sm ${t.cls} ${extra} font-bold px-space-xs py-0.5 rounded-full flex items-center gap-1"><span class="material-symbols-outlined text-[12px]">${t.icon}</span>${t.label}</span>`;
}

function probabilityBar(g, thick = false) {
  const hw = Math.round(g.home.win_prob * 100);
  const aw = 100 - hw;
  const bar = (TIER[g.confidence] || TIER.low).bar;
  const h = thick ? 'h-3' : 'h-2';
  return `<div class="flex flex-col gap-1">
    <div class="flex justify-between font-label-sm text-label-sm">
      <span class="font-bold ${g.home.win_prob >= g.away.win_prob ? 'text-primary' : 'text-on-surface-variant'}">${g.home.abbr} ${hw}%</span>
      <span class="font-semibold ${g.away.win_prob > g.home.win_prob ? 'text-primary' : 'text-on-surface-variant'}">${g.away.abbr} ${aw}%</span>
    </div>
    <div class="w-full ${h} rounded-full bg-surface-container overflow-hidden flex">
      <div class="h-full ${bar} rounded-l-full transition-all duration-700" style="width:${hw}%"></div>
      <div class="h-full bg-surface-variant rounded-r-full transition-all duration-700" style="width:${aw}%"></div>
    </div>
  </div>`;
}

function teamRow(team, side, isPick) {
  const color = isPick ? 'text-primary' : 'text-on-surface-variant';
  return `<div class="flex items-center justify-between">
    <div class="flex items-center gap-space-xs">
      ${logo(team.abbr)}
      <div>
        <div class="flex items-center gap-1">
          <span class="font-label-lg text-label-lg font-bold text-on-surface">${shortName(team.name)}</span>
          ${isPick ? '<span class="material-symbols-outlined text-primary text-[16px]" title="Model pick">verified</span>' : ''}
        </div>
        <span class="font-body-sm text-body-sm text-outline">${team.record} • ${side}</span>
      </div>
    </div>
    <span class="font-data-display-mobile text-data-display-mobile font-bold ${color}">${pct(team.win_prob)}</span>
  </div>`;
}

function isPrimetime(g) {
  const [h] = g.gametime.split(':').map(Number);
  return h >= 20 || g.weekday === 'Thursday';
}

function isUpset(g) {
  const loser = g[pickSide(g) === 'home' ? 'away' : 'home'];
  return winCount(pickTeam(g).record) < winCount(loser.record);
}

function isTossUp(g) {
  return g.edge_pp < 10;
}

function setText(selector, value) {
  const el = $(selector);
  if (el && value !== undefined && value !== null) el.textContent = value;
}

/* ---------------- Weekly board ---------------- */

function renderBoard(data, metrics) {
  const games = data.games;
  const high = games.filter((g) => g.confidence === 'high').length;
  const avgEdge = games.reduce((s, g) => s + g.edge_pp, 0) / games.length;

  setText('#stat-games', games.length);
  setText('#stat-locks', high);
  setText('#stat-edge', `+${avgEdge.toFixed(1)}`);
  setText('#model-name', 'Logistic Reg.');
  setText('#model-sub', `${data.model.seasons.length} seasons • ${data.model.features} features`);
  setText('#count-all', `All Games (${games.length})`);
  setText('#count-high', `High Confidence (${high})`);
  setText('#count-primetime', `Primetime (${games.filter(isPrimetime).length})`);
  setText('#count-upset', `Upset Alerts (${games.filter(isUpset).length})`);
  setText('#count-toss-up', `Toss-ups (${games.filter(isTossUp).length})`);
  setText('#schedule-title', `${kickoff(games.find((g) => g.weekday === 'Sunday') || games[0]).split(' ')[0]} & Monday Schedule`);
  setText('#schedule-subtitle', `Model win probabilities from ${data.model.training_games.toLocaleString()} games; no betting line available for ${data.season}.`);

  renderFeatured(games[0]);
  const grid = $('#matchup-grid');
  if (grid) grid.innerHTML = games.map(boardCard).join('');

  const filters = {
    all: () => true,
    high: (g) => g.confidence === 'high',
    primetime: isPrimetime,
    upset: isUpset,
    'toss-up': isTossUp,
  };
  $$('#filter-container .filter-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      $$('#filter-container .filter-btn').forEach((b) => {
        b.classList.remove('bg-on-surface', 'text-surface', 'shadow-sm');
        b.classList.add('bg-surface-container', 'text-on-surface-variant');
      });
      btn.classList.remove('bg-surface-container', 'text-on-surface-variant');
      btn.classList.add('bg-on-surface', 'text-surface', 'shadow-sm');
      const fn = filters[btn.dataset.filter] || filters.all;
      $$('#matchup-grid article').forEach((card) => {
        card.classList.toggle('hidden', !fn(games.find((g) => g.game_id === card.dataset.gameId)));
      });
    });
  });
}

function boardCard(g) {
  const t = TIER[g.confidence] || TIER.low;
  const pick = pickTeam(g);
  return `<article data-game-id="${g.game_id}" class="bg-surface-container-lowest rounded-xl shadow-sm hover:shadow-md transition-shadow p-space-md flex flex-col justify-between gap-space-md">
    <div class="flex items-center justify-between gap-1">
      <span class="font-label-sm text-label-sm font-semibold text-outline truncate">${kickoff(g)} • ${g.stadium}</span>
      ${tierBadge(g.confidence)}
    </div>
    <div class="flex flex-col gap-2">
      ${teamRow(g.away, 'Away', g.away.abbr === g.predicted_winner)}
      ${teamRow(g.home, 'Home', g.home.abbr === g.predicted_winner)}
    </div>
    ${probabilityBar(g)}
    <div class="pt-space-xs flex items-center justify-between bg-surface-container-low p-space-xs rounded-lg">
      <div class="flex items-center gap-space-xs min-w-0">
        <span class="font-label-sm text-label-sm bg-surface-container-lowest px-1.5 py-0.5 rounded font-bold ${t.dot.replace('bg-', 'text-')} truncate">Model edge: ${pick.abbr} +${g.edge_pp} pp</span>
        <span class="font-body-sm text-body-sm text-outline hidden sm:inline">No market line</span>
      </div>
      <a href="game-insights.html?game=${encodeURIComponent(g.game_id)}" class="text-primary hover:text-primary-container font-label-md text-label-md font-bold flex items-center gap-0.5 flex-shrink-0">Inspect <span class="material-symbols-outlined text-[16px]">chevron_right</span></a>
    </div>
  </article>`;
}

function renderFeatured(g) {
  const host = $('#featured-game');
  if (!host) return;
  const awayPick = g.away.abbr === g.predicted_winner;
  const hf = g.home.features, af = g.away.features;
  const rows = [
    ['speed', 'Rolling OFF EPA', signed(hf.off_epa - af.off_epa)],
    ['shield', 'EPA allowed diff', signed(hf.def_epa - af.def_epa)],
    ['pace', 'Pace diff', signed(hf.pace - af.pace)],
    ['event', 'Rest-day diff', signed(hf.rest_days - af.rest_days, 0)],
  ];
  const hw = Math.round(g.home.win_prob * 100);
  host.innerHTML = `<section class="relative bg-surface-container-lowest rounded-2xl shadow-md p-space-lg overflow-hidden">
    <div class="absolute -right-20 -top-20 w-80 h-80 rounded-full bg-primary-fixed/20 blur-3xl pointer-events-none"></div>
    <div class="absolute -left-20 -bottom-20 w-80 h-80 rounded-full bg-secondary-fixed/20 blur-3xl pointer-events-none"></div>
    <div class="relative z-10 flex flex-col lg:flex-row items-stretch lg:items-center justify-between gap-space-lg">
      <div class="flex flex-col gap-space-md flex-1">
        <div class="flex items-center gap-space-sm flex-wrap">
          <span class="font-label-sm text-label-sm bg-primary text-on-primary px-space-sm py-1 rounded-full uppercase tracking-wider font-bold flex items-center gap-1"><span class="material-symbols-outlined text-[14px]">lock</span> Lock of the Week</span>
          <span class="font-label-md text-label-md text-on-surface-variant">${kickoff(g)} • ${g.stadium}</span>
          ${tierBadge(g.confidence, 'bg-surface-container text-on-surface-variant')}
        </div>
        <div class="flex items-center gap-space-md">
          <div class="flex items-center gap-space-sm flex-1">
            ${logo(g.away.abbr, 'w-14 h-14', 'text-headline-md')}
            <div>
              <div class="flex items-center gap-1"><span class="font-headline-md text-headline-md text-on-surface font-bold">${shortName(g.away.name)}</span>${awayPick ? '<span class="material-symbols-outlined text-primary text-[18px]">verified</span>' : ''}</div>
              <span class="font-label-sm text-label-sm text-outline uppercase font-semibold">${g.away.record} • Road</span>
            </div>
          </div>
          <div class="flex flex-col items-center justify-center px-space-sm">
            <span class="font-headline-sm text-headline-sm font-extrabold text-outline-variant">AT</span>
            <span class="font-label-sm text-label-sm text-on-surface-variant">${kickoff(g).split(' ').slice(1).join(' ')}</span>
          </div>
          <div class="flex items-center gap-space-sm flex-1 justify-end text-right">
            <div>
              <div class="flex items-center justify-end gap-1"><span class="font-headline-md text-headline-md text-on-surface font-bold">${shortName(g.home.name)}</span>${awayPick ? '' : '<span class="material-symbols-outlined text-primary text-[18px]">verified</span>'}</div>
              <span class="font-label-sm text-label-sm text-outline uppercase font-semibold">${g.home.record} • Home</span>
            </div>
            ${logo(g.home.abbr, 'w-14 h-14', 'text-headline-md')}
          </div>
        </div>
        <div class="flex flex-col gap-1.5">
          <div class="flex items-center justify-between font-label-md text-label-md">
            <span class="font-bold text-primary flex items-center gap-1"><span class="w-2.5 h-2.5 rounded-full bg-primary inline-block"></span>${g.away.abbr} ${pct(g.away.win_prob)}</span>
            <span class="font-semibold text-secondary flex items-center gap-1">${g.home.abbr} ${pct(g.home.win_prob)}<span class="w-2.5 h-2.5 rounded-full bg-secondary-container inline-block"></span></span>
          </div>
          <div class="w-full h-3 rounded-full bg-surface-container overflow-hidden flex p-0.5">
            <div class="h-full bg-primary-container rounded-l-full transition-all duration-700" style="width:${100 - hw}%"></div>
            <div class="h-full bg-secondary-container rounded-r-full transition-all duration-700" style="width:${hw}%"></div>
          </div>
        </div>
      </div>
      <div class="flex flex-col justify-between gap-space-md lg:w-80 bg-surface-container-low p-space-md rounded-xl">
        <div class="flex items-center justify-between">
          <span class="font-label-sm text-label-sm uppercase tracking-wider text-outline font-bold">Model Inputs</span>
          <span class="font-label-sm text-label-sm px-space-xs py-0.5 rounded bg-primary-fixed text-on-primary-fixed font-bold">${g.edge_pp.toFixed(1)} pp Edge</span>
        </div>
        <div class="flex flex-col gap-1">
          ${rows.map(([icon, label, value]) => `<div class="flex items-center justify-between text-on-surface bg-surface-container-lowest px-space-sm py-1 rounded font-body-sm text-body-sm"><span class="flex items-center gap-1 text-on-surface-variant"><span class="material-symbols-outlined text-[15px] text-primary">${icon}</span>${label}</span><span class="font-semibold text-primary">${value}</span></div>`).join('')}
        </div>
        <div class="bg-primary text-on-primary px-space-sm py-2 rounded-lg flex items-center justify-between">
          <div class="flex flex-col"><span class="font-label-sm text-label-sm tracking-wider uppercase opacity-80">Model Recommendation</span><span class="font-label-lg text-label-lg font-bold">${g.predicted_winner} to win (${pct(pickTeam(g).win_prob)})</span></div>
          <a href="game-insights.html?game=${encodeURIComponent(g.game_id)}" aria-label="View full breakdown" class="w-8 h-8 rounded-lg bg-primary-container hover:bg-surface-tint flex items-center justify-center text-on-primary transition-all active:scale-95"><span class="material-symbols-outlined text-[18px]">arrow_forward</span></a>
        </div>
      </div>
    </div>
  </section>`;
}

/* ---------------- Pick simulator ---------------- */

const simState = { data: null, picks: {} };

function renderSimulator(data) {
  simState.data = data;
  data.games.forEach((g) => { simState.picks[g.game_id] = g.predicted_winner; });
  setText('#sim-title', `My Week ${data.week} Slate Simulator`);
  const grid = $('#sim-grid');
  if (grid) grid.innerHTML = data.games.map(simCard).join('');
  recalcSim();
}

function simCard(g) {
  const t = TIER[g.confidence] || TIER.low;
  const option = (team) => `<button type="button" data-team="${team.abbr}" class="team-option flex-1 flex items-center gap-space-xs p-space-xs rounded-lg border border-transparent hover:bg-surface-container transition-all text-left">
      ${logo(team.abbr)}
      <div class="flex-1 min-w-0">
        <div class="font-label-lg text-label-lg font-bold text-on-surface truncate">${shortName(team.name)}</div>
        <div class="font-body-sm text-body-sm text-outline">${team.record} • ${pct(team.win_prob)}</div>
      </div>
      <span class="material-symbols-outlined pick-radio text-[18px] text-outline">radio_button_unchecked</span>
    </button>`;
  return `<div class="game-card bg-surface-container-lowest rounded-xl p-space-md shadow-sm transition-all hover:shadow-md" data-game-id="${g.game_id}" data-ml-pick="${g.predicted_winner}">
    <div class="flex items-center justify-between gap-1 mb-space-sm">
      <span class="font-label-sm text-label-sm font-semibold text-outline truncate">${kickoff(g)} • ${g.stadium}</span>
      ${tierBadge(g.confidence)}
    </div>
    <div class="flex items-center gap-space-xs">${option(g.away)}${option(g.home)}</div>
    <div class="mt-space-sm flex items-center justify-between font-label-sm text-label-sm text-on-surface-variant">
      <span>Model pick: <span class="font-bold text-primary">${g.predicted_winner}</span></span>
      <span>Edge ${g.edge_pp.toFixed(1)} pp</span>
    </div>
  </div>`;
}

function recalcSim() {
  const games = simState.data?.games || [];
  if (!games.length) return;
  const picks = simState.picks;
  const made = games.filter((g) => picks[g.game_id]).length;
  const agree = games.filter((g) => picks[g.game_id] === g.predicted_winner).length;
  setText('#stat-picks-count', `${made} / ${games.length}`);
  setText('#stat-agreement-pct', `${Math.round((agree / games.length) * 100)}%`);
  setText('#stat-proj-score', `${agree}-${games.length - agree}`);

  $$('.game-card').forEach((card) => {
    const g = games.find((x) => x.game_id === card.dataset.gameId);
    if (!g) return;
    const chosen = picks[g.game_id];
    $$('.team-option', card).forEach((btn) => {
      const selected = btn.dataset.team === chosen;
      btn.classList.toggle('bg-surface-container', selected);
      btn.classList.toggle('ring-1', selected);
      btn.classList.toggle('ring-primary/40', selected);
      const radio = $('.pick-radio', btn);
      if (radio) {
        radio.textContent = selected ? 'radio_button_checked' : 'radio_button_unchecked';
        radio.classList.toggle('text-primary', selected);
        radio.classList.toggle('text-outline', !selected);
      }
    });
  });
}

function initSimulatorInteractions() {
  $('#sim-grid')?.addEventListener('click', (e) => {
    const btn = e.target.closest('.team-option');
    if (!btn) return;
    const card = btn.closest('.game-card');
    if (card) simState.picks[card.dataset.gameId] = btn.dataset.team;
    recalcSim();
  });
  $('#auto-sync-btn')?.addEventListener('click', () => {
    (simState.data?.games || []).forEach((g) => { simState.picks[g.game_id] = g.predicted_winner; });
    recalcSim();
  });
  $('#save-slate-btn')?.addEventListener('click', (e) => {
    const btn = e.currentTarget;
    const original = btn.innerHTML;
    btn.innerHTML = '<span class="material-symbols-outlined text-[18px]">check_circle</span> Slate Saved Locally';
    setTimeout(() => { btn.innerHTML = original; }, 1800);
  });
  $('#export-btn')?.addEventListener('click', () => {
    alert(`Week ${simState.data?.week} slate exported (${Object.keys(simState.picks).length} picks).`);
  });
}

/* ---------------- Model metrics ---------------- */

function renderMetrics(metrics, data) {
  const wf = metrics.walk_forward;
  const last = metrics.recent_average;
  setText('#kpi-accuracy', pct(last.accuracy));
  setText('#kpi-auc', last.roc_auc.toFixed(3));
  setText('#kpi-brier', last.brier.toFixed(3));
  setText('#kpi-logloss', metrics.in_sample.log_loss.toFixed(3));
  setText('#acc-badge', `${pct(last.accuracy)} Acc`);
  setText('#week-badge', `Week ${data.week}`);
  setText('#metrics-note', `Walk-forward validation on ${wf.reduce((s, r) => s + r.games, 0)} held-out games (${wf[0].season}-${wf[wf.length - 1].season}).`);
  setText('#meta-summary', `${metrics.training_games.toLocaleString()} games • ${metrics.seasons[0]}-${metrics.seasons[metrics.seasons.length - 1]} • ${metrics.features} features • ${metrics.num_teams} team codes`);

  const bars = $('#season-bars');
  if (bars) bars.innerHTML = wf.map((r) => `<div class="flex items-center gap-space-sm">
      <span class="font-label-md text-label-md text-on-surface-variant w-10">${r.season}</span>
      <div class="flex-1 h-2 rounded-full bg-surface-container overflow-hidden"><div class="h-full bg-primary rounded-full" style="width:${r.accuracy * 100}%"></div></div>
      <span class="font-label-md text-label-md font-bold text-on-surface w-10 text-right">${pct(r.accuracy)}</span>
      <span class="font-label-sm text-label-sm text-outline w-12 text-right">AUC ${r.roc_auc.toFixed(2)}</span>
    </div>`).join('');

  renderCalibration(metrics.calibration);

  const base = $('#baseline-bars');
  if (base) {
    const rows = [
      ['Model (last 3 seasons)', last.accuracy, 'bg-primary'],
      ['Always home', metrics.baselines.always_home, 'bg-secondary'],
      ['Closing market favorite', metrics.baselines.market_favorite, 'bg-tertiary'],
    ];
    base.innerHTML = rows.map(([label, value, color]) => `<div class="flex items-center gap-space-sm">
        <span class="font-label-md text-label-md text-on-surface-variant w-48 flex-shrink-0">${label}</span>
        <div class="flex-1 h-2 rounded-full bg-surface-container overflow-hidden"><div class="h-full ${color} rounded-full" style="width:${value * 100}%"></div></div>
        <span class="font-label-md text-label-md font-bold text-on-surface w-10 text-right">${pct(value)}</span>
      </div>`).join('');
  }
}

function renderCalibration(cal) {
  const chart = $('#calibration-chart');
  if (chart) {
    const W = 560, H = 300, pad = 34;
    const x = (p) => pad + p * (W - pad * 2);
    const y = (p) => H - pad - p * (H - pad * 2);
    const pts = cal.map((b) => ({ x: b.mean_pred, y: b.actual, n: b.count })).filter((p) => p.n > 0);
    const path = pts.map((p, i) => `${i ? 'L' : 'M'}${x(p.x).toFixed(1)},${y(p.y).toFixed(1)}`).join(' ');
    chart.innerHTML = `<svg viewBox="0 0 ${W} ${H}" class="w-full h-auto" role="img" aria-label="Reliability curve">
      <line x1="${x(0)}" y1="${y(0)}" x2="${x(1)}" y2="${y(1)}" stroke="#bccac0" stroke-dasharray="5 5" />
      <line x1="${x(0)}" y1="${y(0)}" x2="${x(1)}" y2="${y(0)}" stroke="#bccac0" />
      <line x1="${x(0)}" y1="${y(0)}" x2="${x(0)}" y2="${y(1)}" stroke="#bccac0" />
      <path d="${path}" fill="none" stroke="#006948" stroke-width="2.5" />
      ${pts.map((p) => `<circle cx="${x(p.x).toFixed(1)}" cy="${y(p.y).toFixed(1)}" r="${Math.min(7, 3 + Math.sqrt(p.n) / 2).toFixed(1)}" fill="#fd761a" stroke="#ffffff" stroke-width="1.5"><title>${p.n} games · predicted ${pct(p.x)} · actual ${pct(p.y)}</title></circle>`).join('')}
      <text x="${W / 2}" y="${H - 6}" text-anchor="middle" font-size="11" fill="#6d7a72">Predicted win probability</text>
      <text x="12" y="${H / 2}" transform="rotate(-90 12 ${H / 2})" text-anchor="middle" font-size="11" fill="#6d7a72">Actual win rate</text>
      <text x="${x(1) - 4}" y="${y(0) + 16}" text-anchor="end" font-size="10" fill="#6d7a72">100%</text>
      <text x="${x(0) + 4}" y="${y(1) + 12}" font-size="10" fill="#6d7a72">100%</text>
    </svg>`;
  }
  const table = $('#calibration-table');
  if (table) table.innerHTML = `<table class="w-full text-left font-body-sm text-body-sm">
      <thead><tr class="text-outline font-label-sm text-label-sm uppercase"><th class="py-1">Bucket</th><th>Pred</th><th>Actual</th><th>Games</th></tr></thead>
      <tbody>${cal.map((b) => `<tr class="border-t border-surface-container"><td class="py-1 font-semibold">${b.bin}</td><td>${pct(b.mean_pred)}</td><td class="${b.actual >= b.mean_pred ? 'text-primary' : 'text-secondary'} font-semibold">${pct(b.actual)}</td><td>${b.count}</td></tr>`).join('')}</tbody>
    </table>`;
}

/* ---------------- Game insights ---------------- */

const insightState = { data: null, index: 0 };

function renderInsights(data) {
  insightState.data = data;
  const requested = new URLSearchParams(location.search).get('game');
  const found = data.games.findIndex((g) => g.game_id === requested);
  insightState.index = found >= 0 ? found : 0;

  const chips = $('#game-chips');
  if (chips) chips.innerHTML = data.games.map((g, i) => `<button type="button" data-index="${i}" class="game-chip font-label-sm text-label-sm px-2 py-1 rounded-full border border-surface-container text-on-surface-variant hover:bg-surface-container transition-all">${g.away.abbr} @ ${g.home.abbr}</button>`).join('');

  const slider = $('#game-slider');
  if (slider) { slider.max = String(data.games.length - 1); slider.value = String(insightState.index); }

  const slate = $('#slate-bars');
  if (slate) slate.innerHTML = data.games.map((g, i) => `<button type="button" data-index="${i}" class="slate-bar w-full flex items-center gap-space-xs py-1 text-left">
      <span class="font-label-sm text-label-sm text-on-surface-variant w-20 flex-shrink-0 truncate">${g.away.abbr}@${g.home.abbr}</span>
      <span class="flex-1 h-2 rounded-full bg-surface-container overflow-hidden"><span class="block h-full ${(TIER[g.confidence] || TIER.low).bar} rounded-full" style="width:${g.edge_pp * 2}%"></span></span>
      <span class="font-label-sm text-label-sm font-bold text-on-surface w-8 text-right">${Math.round(pickTeam(g).win_prob * 100)}%</span>
    </button>`).join('');

  renderSelectedGame();

  chips?.addEventListener('click', (e) => {
    const btn = e.target.closest('.game-chip');
    if (btn) selectGame(Number(btn.dataset.index));
  });
  slate?.addEventListener('click', (e) => {
    const btn = e.target.closest('.slate-bar');
    if (btn) selectGame(Number(btn.dataset.index));
  });
  slider?.addEventListener('input', (e) => selectGame(Number(e.target.value)));
}

function selectGame(index) {
  insightState.index = index;
  const slider = $('#game-slider');
  if (slider) slider.value = String(index);
  renderSelectedGame();
}

function renderSelectedGame() {
  const g = insightState.data?.games[insightState.index];
  if (!g) return;
  const awayPick = g.away.abbr === g.predicted_winner;
  const pick = pickTeam(g);
  const hf = g.home.features, af = g.away.features;

  setText('#slider-label', `Game ${insightState.index + 1} of ${insightState.data.games.length}: ${g.away.abbr} @ ${g.home.abbr} — ${pick.abbr} ${pct(pick.win_prob)}`);
  setText('#insight-summary', `${kickoff(g)} • ${g.stadium} • ${g.home.record} home / ${g.away.record} away`);

  const header = $('#insight-header');
  if (header) header.innerHTML = `<div class="flex flex-col sm:flex-row items-center justify-around gap-space-md">
      <div class="flex items-center gap-space-sm">
        ${logo(g.away.abbr, 'w-14 h-14', 'text-headline-md')}
        <div><div class="font-headline-sm text-headline-sm font-bold text-on-surface">${g.away.name}</div><div class="font-label-sm text-label-sm text-outline">${g.away.record} • Away • ${pct(g.away.win_prob)}</div></div>
      </div>
      <div class="flex flex-col items-center gap-1">
        <span class="font-label-sm text-label-sm uppercase tracking-wider text-outline font-bold">Model Pick</span>
        <span class="font-headline-xl text-headline-xl font-extrabold text-primary">${pick.abbr}</span>
        ${tierBadge(g.confidence)}
      </div>
      <div class="flex items-center gap-space-sm">
        <div class="text-right"><div class="font-headline-sm text-headline-sm font-bold text-on-surface">${g.home.name}</div><div class="font-label-sm text-label-sm text-outline">${g.home.record} • Home • ${pct(g.home.win_prob)}</div></div>
        ${logo(g.home.abbr, 'w-14 h-14', 'text-headline-md')}
      </div>
    </div>`;

  const prob = $('#insight-prob');
  if (prob) prob.innerHTML = probabilityBar(g, true);

  const features = $('#insight-features');
  if (features) {
    const rows = [
      ['Rolling OFF EPA', hf.off_epa, af.off_epa, 2, 'Higher is better'],
      ['EPA allowed (DEF)', hf.def_epa, af.def_epa, 2, 'Lower is better'],
      ['Pace (plays/game)', hf.pace, af.pace, 1, 'Offensive tempo'],
      ['Rest days', hf.rest_days, af.rest_days, 0, 'Days since last game'],
      ['Bye proxy', hf.bye ? 1 : 0, af.bye ? 1 : 0, 0, 'Rest ≥ 12 days'],
    ];
    features.innerHTML = rows.map(([label, h, a, digits, hint]) => {
      const diff = h - a;
      const better = label.includes('allowed') ? diff < 0 : diff > 0;
      const color = diff === 0 ? 'text-outline' : better ? 'text-primary' : 'text-secondary';
      const fmt = (v) => digits === 0 ? String(v) : Number(v).toFixed(digits);
      return `<div class="bg-surface-container-lowest rounded-lg p-space-sm">
        <div class="flex items-center justify-between font-label-md text-label-md">
          <span class="font-bold text-on-surface">${label}</span>
          <span class="${color} font-bold">${signed(diff, digits)}</span>
        </div>
        <div class="flex items-center justify-between font-body-sm text-body-sm text-outline mt-1">
          <span>${g.home.abbr} ${fmt(h)}</span><span>${hint}</span><span>${fmt(a)} ${g.away.abbr}</span>
        </div>
      </div>`;
    }).join('');
  }

  $$('.game-chip').forEach((btn) => btn.classList.toggle('bg-on-surface', Number(btn.dataset.index) === insightState.index));
  $$('.slate-bar').forEach((btn) => btn.classList.toggle('bg-surface-container', Number(btn.dataset.index) === insightState.index));
}

/* ---------------- Bootstrap ---------------- */

function commonBindings(data, metrics) {
  setText('#week-badge', `Week ${data.week}`);
  setText('#acc-badge', `${pct(metrics.recent_average.accuracy)} Acc`);
  setText('#footer-sync', `${data.season} Week ${data.week} • Model generated • ${data.games.length} games`);
  setText('#game-week-label', `Week ${data.week} Slate`);
}

async function boot() {
  const page = document.body.dataset.page;
  try {
    const [predRes, metricsRes] = await Promise.all([fetch(DATA_URL), fetch(METRICS_URL)]);
    if (!predRes.ok || !metricsRes.ok) throw new Error('Prediction data unavailable');
    const [data, metrics] = await Promise.all([predRes.json(), metricsRes.json()]);
    commonBindings(data, metrics);
    if (page === 'board') renderBoard(data, metrics);
    if (page === 'simulator') { renderSimulator(data); initSimulatorInteractions(); }
    if (page === 'metrics') renderMetrics(metrics, data);
    if (page === 'insights') renderInsights(data);
  } catch (err) {
    console.error('GridironML data load failed:', err);
    const targets = $$('[data-page-content]');
    targets.forEach((el) => {
      el.innerHTML = `<div class="bg-surface-container-low rounded-xl p-space-lg text-center">
        <span class="material-symbols-outlined text-outline text-[32px]">cloud_off</span>
        <p class="font-label-md text-label-md text-on-surface-variant mt-space-sm">Could not load prediction data. Serve the <code>web/</code> folder over HTTP (e.g. <code>python -m http.server 8000</code>) and run <code>python -m src.predict</code>.</p>
      </div>`;
    });
  }
}

document.addEventListener('DOMContentLoaded', boot);
