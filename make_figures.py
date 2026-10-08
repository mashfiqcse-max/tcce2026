"""All figures for TCCE 2026 Paper 89 (camera ready), drawn from the same pipeline as the tables.

Colour is used for the electronic version; every figure stays legible in black and white through
hatching, direct labels or a light to dark sequential ramp, as the Springer instructions require.
Usage: python make_figures.py <path to hai-21.03> <output dir> <r11_tau_sigma_grid.csv>
"""
import sys, os, glob, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Patch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support
import lightgbm as lgb, shap

D, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
LAB = ['attack', 'attack_P1', 'attack_P2', 'attack_P3']
W, S, VAR, BUDGET, TAU, SIG = 60, 10, 0.98, 0.5, 300, 0.8
G = ['P1', 'P2', 'P3', 'P4']
GNAME = {'P1': 'P1 boiler', 'P2': 'P2 turbine', 'P3': 'P3 water', 'P4': 'P4 HIL'}

# ------------------------------------------------------------------ style
BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
VIOLET, GREY, INK, INK2 = '#4a3aa7', '#9a9a96', '#0b0b0b', '#52514e'
SUBC = {'P1': BLUE, 'P2': ORANGE, 'P3': AQUA, 'P4': YELLOW}
SUBH = {'P1': '', 'P2': '////', 'P3': '....', 'P4': 'xxxx'}
SEQ = LinearSegmentedColormap.from_list('blue', ['#ffffff', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.5, 'axes.edgecolor': INK2,
    'axes.linewidth': 0.6, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.labelcolor': INK,
    'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False, 'hatch.linewidth': 0.5,
    'axes.grid': False, 'savefig.dpi': 800, 'figure.dpi': 100})
TW = 4.75   # LNCS text width in inches
def save(fig, name):
    fig.savefig(f'{OUT}/{name}', dpi=800, pil_kwargs={'quality': 95}, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)

# ------------------------------------------------------------------ pipeline (identical to revision.py)
def load(p):
    df = pd.read_csv(p); df.columns = [c.strip() for c in df.columns]
    df['time'] = pd.to_datetime(df['time']); return df.sort_values('time').reset_index(drop=True)
tr = [load(p) for p in sorted(glob.glob(D + '/train*.csv*'))]
te = [load(p) for p in sorted(glob.glob(D + '/test*.csv*'))]
SEN = [c for c in te[0].columns if c not in LAB + ['time']]
FEAT = [f'{s}|{st}' for st in ['mean', 'std', 'min', 'max', 'dmean'] for s in SEN]
GR = np.array([f.split('_')[0] for f in FEAT])
def wz(df, lab=True):
    V = df[SEN].to_numpy(np.float32); st = np.arange(0, len(V) - W + 1, S); F = []
    for i in range(0, len(st), 2000):
        s = st[i:i + 2000]; b = V[s[:, None] + np.arange(W)[None, :]]
        F.append(np.concatenate([b.mean(1), b.std(1), b.min(1), b.max(1), np.abs(np.diff(b, 1, 1)).mean(1)], 1).astype(np.float32))
    o = {'X': np.vstack(F), 't': df['time'].to_numpy()[st]}
    if lab:
        idx = st[:, None] + np.arange(W)[None, :]
        for L in LAB: o[L] = df[L].to_numpy(np.int8)[idx].max(1)
    return o
Xtr = np.vstack([wz(d, False)['X'] for d in tr]); Wte = [wz(d) for d in te]
cut = int(0.8 * len(Xtr)); Xfit, Xcal = Xtr[:cut], Xtr[cut:]; cal_h = len(Xcal) * S / 3600
def events(y):
    ev, st = [], None
    for i, v in enumerate(y):
        if v and st is None: st = i
        if not v and st is not None: ev.append((st, i - 1)); st = None
    if st is not None: ev.append((st, len(y) - 1))
    return ev
sc = StandardScaler().fit(Xfit); Zfit = sc.transform(Xfit); Zcal = sc.transform(Xcal)
K = int(np.argmax(np.cumsum(PCA(150, random_state=0).fit(Zfit).explained_variance_ratio_) >= VAR)) + 1
pca = PCA(K, random_state=0).fit(Zfit)
err = lambda Z: (Z - pca.inverse_transform(pca.transform(Z))) ** 2
bthr = lambda s, h, b: np.sort(s)[-max(1, int(round(b * h)))]
THR = bthr(err(Zcal).mean(1), cal_h, BUDGET)
iso = IsolationForest(n_estimators=100, random_state=0, n_jobs=-1).fit(Zfit[np.random.RandomState(0).choice(len(Zfit), 20000, replace=False)])
THR_IF = bthr(-iso.score_samples(Zcal), cal_h, BUDGET)
f1 = lambda y, p: precision_recall_fscore_support(y, p, average='binary', zero_division=0)[2]
for h, w in enumerate(Wte):
    w['ev'] = events(w['attack']); eo = -np.ones(len(w['attack']), int)
    for k, (a, b) in enumerate(w['ev']): eo[a:b + 1] = k
    w['eo'] = eo; w['sub'] = np.stack([w[f'attack_P{k}'] for k in (1, 2, 3)], 1)
    Z = sc.transform(w['X']); w['E'] = err(Z); w['s'] = w['E'].mean(1); w['pred'] = (w['s'] > THR).astype(int)
    w['gE'] = np.stack([w['E'][:, GR == g].sum(1) for g in G], 1)
    w['ifp'] = (-iso.score_samples(Z) > THR_IF).astype(int)
    w['hours'] = (w['t'].max() - w['t'].min()) / np.timedelta64(1, 'h')
    fit = [i for i in range(5) if i != h]
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, class_weight='balanced',
                             random_state=0, n_jobs=-1, verbose=-1).fit(np.vstack([Wte[i]['X'] for i in fit]),
                                                          np.concatenate([Wte[i]['attack'] for i in fit]))
    w['clf'] = clf; w['sp'] = (clf.predict_proba(w['X'])[:, 1] > 0.5).astype(int)
F1 = pd.DataFrame({'PCA reconstruction': [f1(w['attack'], w['pred']) for w in Wte],
                   'Isolation Forest': [f1(w['attack'], w['ifp']) for w in Wte],
                   'LightGBM (supervised)': [f1(w['attack'], w['sp']) for w in Wte]}, index=[f'File {i}' for i in range(1, 6)])

def cosv(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
def consolidate(t, sig, rule, tau=TAU, sigma=SIG):
    inc = np.zeros(len(t), int); k = 0
    for i in range(1, len(t)):
        same = (t[i] - t[i - 1]) / np.timedelta64(1, 's') <= tau
        if same and rule == 'cos': same = cosv(sig[i], sig[i - 1]) >= sigma
        if same and rule == 'dom': same = int(np.argmax(sig[i])) == int(np.argmax(sig[i - 1]))
        if not same: k += 1
        inc[i] = k
    return inc
tot = {}
for name, rule, key in [('time', 'time', 'gE'), ('cos', 'cos', 'gE'), ('cosF', 'cos', 'E')]:
    n = 0
    for w in Wte:
        idx = np.where(w['pred'] == 1)[0]; inc = consolidate(w['t'][idx], w[key][idx], rule)
        w[f'inc_{name}'] = (idx, inc); n += inc.max() + 1
    tot[name] = n
check = dict(K=K, meanF1=round(F1['PCA reconstruction'].mean(), 3), alerts=int(sum(w['pred'].sum() for w in Wte)), **tot)
print('consistency', check)
assert check == dict(K=77, meanF1=0.819, alerts=941, time=60, cos=64, cosF=200), 'pipeline does not reproduce the paper'

# ================================================================== Fig 1: pipeline overview
fig, ax = plt.subplots(figsize=(TW, 1.75)); ax.set_xlim(0, 100); ax.set_ylim(0, 34); ax.axis('off')
steps = [('HAI 21.03', '79 variables\nat 1 Hz\nP1 to P4', '#e8f1fc', BLUE),
         ('Windowing', '60 s windows\n10 s stride\n395 statistics', '#e8f1fc', BLUE),
         ('Detection', 'PCA residual\nk = 77\nbudget 0.5/h', '#fde9e0', ORANGE),
         ('Attribution', 'error or SHAP\nper feature,\nsummed per\nsubsystem', '#e2f5ee', AQUA),
         ('Merging', 'gap $\\leq$ 300 s\ncos $\\Phi$ $\\geq$ 0.8\nlabel = top $\\Phi$', '#fdf1d6', '#b77c00')]
bw, gap = 18.0, 2.5; xs = [i * (bw + gap) for i in range(5)]
for (title, body, fill, edge), x in zip(steps, xs):
    ax.add_patch(FancyBboxPatch((x + 0.4, 6.5), bw - 0.8, 25.5, boxstyle='round,pad=0.25,rounding_size=1.2', fc=fill, ec=edge, lw=0.9))
    ax.text(x + bw / 2, 28.2, title, ha='center', va='center', fontsize=7, weight='bold', color=INK)
    ax.text(x + bw / 2, 16.0, body, ha='center', va='center', fontsize=6.2, color=INK2, linespacing=1.3)
for x in xs[:-1]:
    ax.annotate('', xy=(x + bw + gap - 0.2, 19), xytext=(x + bw + 0.2, 19),
                arrowprops=dict(arrowstyle='-|>', color=INK2, lw=0.8, mutation_scale=7))
ax.text(50, 0.5, 'Fitted on normal operation only; nothing is tuned on test data.\nOutput: incidents, each carrying a subsystem label.',
        ha='center', va='bottom', fontsize=6.2, color=INK2, style='italic', linespacing=1.2)
save(fig, '89Fig1.jpg')

# ================================================================== Fig 2: detection per file
fig, ax = plt.subplots(figsize=(TW, 1.9))
cols = [(BLUE, ''), (ORANGE, '////'), (AQUA, '....')]
x = np.arange(6); wdt = 0.26
data = pd.concat([F1, F1.mean().to_frame('Mean').T])
for j, (c, (col, hat)) in enumerate(zip(data.columns, cols)):
    bars = ax.bar(x + (j - 1) * wdt, data[c], wdt * 0.92, color=col, hatch=hat, edgecolor='white', linewidth=0.4, label=c)
    for b, v in zip(bars, data[c]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f'{v:.2f}', ha='center', va='bottom', fontsize=6, color=INK2, rotation=90)
ax.axvline(4.5, color=GREY, lw=0.6, ls=':')
ax.set_xticks(x); ax.set_xticklabels(list(data.index)); ax.set_ylim(0, 1.18); ax.set_ylabel('F1 (attack class)')
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.legend(ncol=3, frameon=False, loc='upper left', bbox_to_anchor=(0, 1.13), handlelength=1.4, columnspacing=1.2)
save(fig, '89Fig2.jpg')

# ================================================================== Fig 3: anomaly score over time, file 4
w = Wte[3]; th = (w['t'] - w['t'][0]) / np.timedelta64(1, 'h')
fig, ax = plt.subplots(figsize=(TW, 1.75))
for a, b in w['ev']:
    lab = w['sub'][a:b + 1].max(0); subs = [G[k] for k in range(3) if lab[k]]
    c = SUBC[subs[0]] if len(subs) == 1 else VIOLET
    ax.axvspan(th[a], th[b] + W / 3600, color=c, alpha=0.35, lw=0)
    ax.text((th[a] + th[b]) / 2, 1.02, '+'.join(subs), ha='center', va='bottom', fontsize=6, color=INK2,
            transform=ax.get_xaxis_transform())
ax.plot(th, w['s'], color=INK, lw=0.45)
ax.axhline(THR, color=ORANGE, lw=0.9, ls='--')
ax.text(2.1, THR * 1.3, 'threshold (0.5 false alerts/h)', ha='left', va='bottom', fontsize=6, color=ORANGE)
ax.set_yscale('log'); ax.set_ylim(0.03, 900); ax.set_xlim(0, th[-1])
ax.set_xlabel('time since start of test file 4 (hours)'); ax.set_ylabel('reconstruction error')
handles = [Patch(color=SUBC['P1'], alpha=0.45, label='attack on one subsystem (P1)'), Patch(color=VIOLET, alpha=0.45, label='attack on two subsystems')]
ax.legend(handles=handles, ncol=2, frameon=False, loc='upper right', bbox_to_anchor=(1, 1.27), handlelength=1.2)
save(fig, '89Fig3.jpg')

# ================================================================== Fig 4: grouped error share, file 2 (colour version)
w = Wte[1]; a = np.where(w['pred'] == 1)[0]
share = w['gE'][a] / w['gE'][a].sum(1, keepdims=True)
fig, ax = plt.subplots(figsize=(TW, 1.45))
im = ax.imshow(share.T, aspect='auto', cmap=SEQ, vmin=0, vmax=1, interpolation='nearest')
ax.set_yticks(range(4)); ax.set_yticklabels([GNAME[g] for g in G])
ax.set_xlabel('alert window, time ordered (test file 2, 302 alerts)')
for s in ['top', 'right']: ax.spines[s].set_visible(False)
cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.015, aspect=12); cb.set_label('share of error'); cb.outline.set_linewidth(0.4)
save(fig, '89Fig4.jpg')

# ================================================================== Fig 5: SHAP beeswarm, supervised fold 2
w = Wte[1]; idx = np.where(w['sp'] == 1)[0]
sv = shap.TreeExplainer(w['clf']).shap_values(w['X'][idx]); sv = sv[1] if isinstance(sv, list) else sv
sv = sv[:, :, 1] if np.ndim(sv) == 3 else sv
order = np.argsort(-np.abs(sv).mean(0))[:10][::-1]
fig, axs = plt.subplots(1, 2, figsize=(TW, 2.25), gridspec_kw=dict(width_ratios=[2.3, 1], wspace=0.75))
ax = axs[0]; rng = np.random.RandomState(0)
for r, j in enumerate(order):
    v = sv[:, j]; fv = w['X'][idx, j]; q = (fv - np.percentile(fv, 2)) / (np.percentile(fv, 98) - np.percentile(fv, 2) + 1e-12)
    ax.scatter(v, r + rng.uniform(-0.28, 0.28, len(v)), c=np.clip(q, 0, 1), cmap=SEQ, vmin=-0.15, vmax=1, s=2.2, lw=0, rasterized=True)
ax.axvline(0, color=GREY, lw=0.5)
ax.set_yticks(range(len(order))); ax.set_yticklabels([FEAT[j].replace('|', ' ') for j in order], fontsize=6)
ax.set_xlabel('SHAP value (log odds of attack)')
sm = plt.cm.ScalarMappable(cmap=SEQ, norm=plt.Normalize(0, 1))
cb = fig.colorbar(sm, ax=ax, pad=0.02, aspect=18, shrink=0.85); cb.set_ticks([0, 1]); cb.set_ticklabels(['low', 'high'])
cb.ax.set_title('feature\nvalue', fontsize=6, color=INK2, pad=3); cb.outline.set_linewidth(0.4)
ax = axs[1]
gm = np.array([np.abs(sv[:, GR == g]).sum(1).mean() for g in G]); gm = gm / gm.sum()
bars = ax.barh(range(4)[::-1], gm, color=[SUBC[g] for g in G], hatch=None, edgecolor='white', lw=0.4)
for b, g in zip(bars, G): b.set_hatch(SUBH[g])
for k, v in enumerate(gm): ax.text(v + 0.02, 3 - k, f'{v:.2f}', va='center', fontsize=6, color=INK2)
ax.set_yticks(range(4)[::-1]); ax.set_yticklabels([GNAME[g] for g in G]); ax.set_xlim(0, 1.05)
ax.set_xlabel('share of |SHAP| mass')
save(fig, '89Fig5.jpg')

# ================================================================== Fig 6: consolidation around four real events
def incs_on(w, key, a, b):
    idx, inc = w[f'inc_{key}']; m = (idx >= a) & (idx <= b); return len(set(inc[m]))
cand = []
for fi, w in enumerate(Wte):
    for e, (a, b) in enumerate(w['ev']):
        if w['pred'][a:b + 1].sum() == 0: continue
        lab = w['sub'][a:b + 1].max(0)
        cand.append(dict(fi=fi, e=e, a=a, b=b, multi=int(lab.sum() >= 2), t=incs_on(w, 'time', a, b),
                         c=incs_on(w, 'cos', a, b), u=incs_on(w, 'cosF', a, b)))
C = pd.DataFrame(cand)
pick = []
pick.append(C[(C.multi == 0) & (C.c == 1) & (C.u == 1)].iloc[0] if ((C.multi == 0) & (C.c == 1) & (C.u == 1)).any()
            else C[(C.multi == 0) & (C.c == 1)].sort_values('u').iloc[0])
pick.append(C[(C.multi == 1) & (C.c == 1)].sort_values('u', ascending=False).iloc[0])
used = lambda: {(int(p.fi), int(p.e)) for p in pick}
free = lambda D: D[[(int(r.fi), int(r.e)) not in used() for r in D.itertuples()]]
pick.append(free(C).sort_values('u', ascending=False).iloc[0])
pick.append(free(C[C.c >= 2]).sort_values('c', ascending=False).iloc[0])
titles = ['single subsystem', 'two subsystems', 'worst ungrouped split', 'grouped rule splits']
rows = [('event', None), ('time only', 'time'), ('grouped cos.', 'cos'), ('ungrouped cos.', 'cosF')]
fig, axs = plt.subplots(1, 4, figsize=(TW, 1.75), sharey=True, gridspec_kw=dict(wspace=0.12))
for ax, pk, ttl in zip(axs, pick, titles):
    w = Wte[int(pk.fi)]; a, b = int(pk.a), int(pk.b)
    t0 = w['t'][a]; tm = lambda i: (w['t'][i] - t0) / np.timedelta64(1, 'm')
    lab = w['sub'][a:b + 1].max(0); subs = [G[k] for k in range(3) if lab[k]]
    xlo, xhi = -4, tm(b) + 6
    for r, (name, key) in enumerate(rows):
        y = len(rows) - 1 - r
        if key is None:
            c = SUBC[subs[0]] if len(subs) == 1 else VIOLET
            ax.add_patch(plt.Rectangle((0, y - 0.32), tm(b) + 1, 0.64, fc=c, ec='none')); continue
        idx, inc = w[f'inc_{key}']; shown = 0
        for k in sorted(set(inc)):
            m = idx[inc == k]; lo, hi = tm(m.min()), tm(m.max()) + 1
            if hi < xlo or lo > xhi: continue
            att = w['attack'][m] == 1
            top = G[int(w['gE'][m].sum(0)[:3].argmax())]
            c, hat = (SUBC[top], SUBH[top]) if att.any() else (GREY, '')
            ax.add_patch(plt.Rectangle((max(lo, xlo), y - 0.32), min(hi, xhi) - max(lo, xlo), 0.64, fc=c, ec='white', lw=0.6, hatch=hat))
            shown += 1
        ax.text(xhi - 0.2, y, str(shown), ha='right', va='center', fontsize=6.5, color=INK, weight='bold')
    ax.set_xlim(xlo, xhi); ax.set_ylim(-0.6, len(rows) - 0.25)
    ax.set_title(f'{ttl}\n(file {int(pk.fi) + 1}, {"+".join(subs)})', fontsize=6.2, color=INK, linespacing=1.15)
    ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
axs[0].set_yticks(range(len(rows))[::-1]); axs[0].set_yticklabels([r[0] for r in rows])
fig.text(0.56, -0.02, 'minutes from event start; bold numbers count the incidents shown in each row', ha='center', va='top', fontsize=6.5, color=INK2)
handles = [Patch(fc=SUBC['P1'], ec='white', label='labelled P1'), Patch(fc=SUBC['P2'], ec='white', hatch=SUBH['P2'], label='labelled P2'),
           Patch(fc=VIOLET, ec='white', label='event on two subsystems')]
fig.legend(handles=handles, ncol=3, frameon=False, loc='upper center', bbox_to_anchor=(0.56, -0.08), handlelength=1.1, columnspacing=1.2)
PICK6 = [dict(file=int(p.fi) + 1, event=int(p.e), time=int(p.t), grouped=int(p.c), ungrouped=int(p.u)) for p in pick]
save(fig, '89Fig8.jpg')   # events: printed as Fig. 8

# ================================================================== Fig 7: alert burden per file
fig, ax = plt.subplots(figsize=(TW, 1.9))
x = np.arange(5); wdt = 0.2
series = [('raw alerts', lambda w: w['pred'].sum(), GREY, ''),
          ('ungrouped cosine', lambda w: w['inc_cosF'][1].max() + 1, VIOLET, 'xxxx'),
          ('grouped cosine', lambda w: w['inc_cos'][1].max() + 1, BLUE, ''),
          ('time only', lambda w: w['inc_time'][1].max() + 1, ORANGE, '////')]
for j, (name, fn, col, hat) in enumerate(series):
    v = np.array([fn(w) / w['hours'] for w in Wte])
    bars = ax.bar(x + (j - 1.5) * wdt, v, wdt * 0.92, color=col, hatch=hat, edgecolor='white', lw=0.4, label=name)
    for b, val in zip(bars, v):
        ax.text(b.get_x() + b.get_width() / 2, val + 0.15, f'{val:.2f}', ha='center', va='bottom', fontsize=6, color=INK2, rotation=90)
ax.set_xticks(x); ax.set_xticklabels([f'File {i}' for i in range(1, 6)]); ax.set_ylabel('notifications per hour')
ax.set_ylim(0, 15.5)
ax.legend(ncol=4, frameon=False, loc='upper left', bbox_to_anchor=(0, 1.13), handlelength=1.4, columnspacing=1.2)
save(fig, '89Fig6.jpg')   # alert burden: printed as Fig. 6

# ================================================================== Fig 8: tau sigma sensitivity
g = pd.read_csv(sys.argv[3])   # r11_tau_sigma_grid.csv written by revision.py
c = g[g.rule == 'cos']; taus = sorted(c.tau.unique()); sigs = sorted(c.sigma.unique())
M = np.array([[c[(c.tau == t) & (c.sigma == s)].incidents.iloc[0] for s in sigs] for t in taus])
Sp = np.array([[c[(c.tau == t) & (c.sigma == s)].split.iloc[0] for s in sigs] for t in taus])
tm = g[g.rule == 'time'].set_index('tau').incidents; dm = g[g.rule == 'dom'].set_index('tau').incidents
fig, axs = plt.subplots(1, 2, figsize=(TW, 1.75), gridspec_kw=dict(width_ratios=[1.35, 1], wspace=0.45))
ax = axs[0]; im = ax.imshow(M, cmap=SEQ, vmin=50, vmax=90, aspect='auto')
for i in range(len(taus)):
    for j in range(len(sigs)):
        ax.text(j, i, f'{M[i, j]}\n{Sp[i, j]} split', ha='center', va='center', fontsize=6,
                color='white' if M[i, j] > 74 else INK, linespacing=1.0)
ax.set_xticks(range(len(sigs))); ax.set_xticklabels([str(s) for s in sigs]); ax.set_yticks(range(len(taus))); ax.set_yticklabels([str(t) for t in taus])
ax.set_xlabel('$\\sigma$ (cosine threshold)'); ax.set_ylabel('$\\tau$ (seconds)'); ax.set_title('grouped cosine: incidents', loc='left')
for s in ['top', 'right', 'left', 'bottom']: ax.spines[s].set_visible(False)
ax.add_patch(plt.Rectangle((sigs.index(0.8) - 0.5, taus.index(300) - 0.5), 1, 1, fill=False, ec=ORANGE, lw=1.2))
ax = axs[1]
xp = np.arange(len(taus))
ax.plot(xp, dm.loc[taus].values, color=AQUA, marker='^', ms=3.4, lw=1.0, label='dominant subsystem')
ax.plot(xp, M[:, sigs.index(0.8)], color=BLUE, marker='o', ms=3.2, lw=1.0, label='grouped cosine ($\\sigma$ = 0.8)')
ax.plot(xp, tm.loc[taus].values, color=ORANGE, marker='s', ms=3.2, lw=1.0, label='time only')
ax.set_xticks(xp); ax.set_xticklabels([str(t) for t in taus])
ax.set_ylim(40, 78); ax.set_xlabel('$\\tau$ (seconds)'); ax.set_ylabel('incidents'); ax.set_title('rules compared', loc='left')
ax.legend(frameon=False, loc='lower left', handlelength=1.6)
save(fig, '89Fig7.jpg')   # sensitivity: printed as Fig. 7

json.dump(dict(check, fig6_events=PICK6, F1=F1.round(3).to_dict(), beeswarm_features=[FEAT[j] for j in order[::-1]],
               shap_group_share=dict(zip(G, map(float, gm)))), open(f'{OUT}/figure_data.json', 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
print('figures written to', OUT)
