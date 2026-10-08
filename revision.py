"""TCCE 2026 Paper 89, revision experiments.

Reproduces the main pipeline (identical settings to the submitted paper) and adds
the analyses requested by the reviewers:
  R1.1  time only and dominant subsystem consolidation baselines, tau/sigma grid
  R1.2  incident purity: merged incidents, split events, false incidents,
        one to one event matching, incident span (chaining)
  R1.3  signed vs magnitude aggregation, P4 handling, alternative explainers
        (interventional TreeSHAP, a priori group occlusion), multi subsystem windows
  R1.4  breakdown of all alerts, unconditional localisation, missed events
  R1.5  block bootstrap confidence intervals, comparison against a stated margin
  R2    first alert (onset) localisation
Usage: python revision.py <path to hai-21.03> <output dir>
"""
import sys, os, glob, json, time
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import lightgbm as lgb, shap

t0 = time.time()
D = sys.argv[1] if len(sys.argv) > 1 else 'hai/hai-21.03'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'revision_out'
os.makedirs(OUT, exist_ok=True)
LAB = ['attack', 'attack_P1', 'attack_P2', 'attack_P3']
W, S, VAR, BUDGET = 60, 10, 0.98, 0.5
TAU, SIG = 300, 0.8
G = ['P1', 'P2', 'P3', 'P4']
R = {}

# ------------------------------------------------------------------ data
def load(p):
    df = pd.read_csv(p); df.columns = [c.strip() for c in df.columns]
    df['time'] = pd.to_datetime(df['time'])
    return df.sort_values('time').reset_index(drop=True)

tr = [load(p) for p in sorted(glob.glob(D + '/train*.csv*'))]
te = [load(p) for p in sorted(glob.glob(D + '/test*.csv*'))]
SEN = [c for c in te[0].columns if c not in LAB + ['time']]
FEAT = [f'{s}|{st}' for st in ['mean', 'std', 'min', 'max', 'dmean'] for s in SEN]
GR = np.array([f.split('_')[0] for f in FEAT])

def wz(df, lab=True):
    V = df[SEN].to_numpy(np.float32); st = np.arange(0, len(V) - W + 1, S); F = []
    for i in range(0, len(st), 2000):
        s = st[i:i + 2000]; b = V[s[:, None] + np.arange(W)[None, :]]
        F.append(np.concatenate([b.mean(1), b.std(1), b.min(1), b.max(1),
                                 np.abs(np.diff(b, 1, 1)).mean(1)], 1).astype(np.float32))
    o = {'X': np.vstack(F), 't': df['time'].to_numpy()[st]}
    if lab:
        idx = st[:, None] + np.arange(W)[None, :]
        for L in LAB: o[L] = df[L].to_numpy(np.int8)[idx].max(1)
        # first second of each window, for onset analysis
        for L in LAB[1:]: o[L + '_first'] = df[L].to_numpy(np.int8)[st]
    return o

Xtr = np.vstack([wz(d, False)['X'] for d in tr]); Wte = [wz(d) for d in te]
cut = int(0.8 * len(Xtr)); Xfit, Xcal = Xtr[:cut], Xtr[cut:]
cal_hours = len(Xcal) * S / 3600

def events(y):
    ev, st = [], None
    for i, v in enumerate(y):
        if v and st is None: st = i
        if not v and st is not None: ev.append((st, i - 1)); st = None
    if st is not None: ev.append((st, len(y) - 1))
    return ev

for w in Wte:
    w['ev'] = events(w['attack'])
    eo = -np.ones(len(w['attack']), int)
    for k, (a, b) in enumerate(w['ev']): eo[a:b + 1] = k
    w['eo'] = eo
    w['sub'] = np.stack([w[f'attack_P{k}'] for k in (1, 2, 3)], 1)

# ------------------------------------------------------------------ detector
sc = StandardScaler().fit(Xfit); Zfit = sc.transform(Xfit); Zcal = sc.transform(Xcal)
pfull = PCA(n_components=150, random_state=0).fit(Zfit)
K = int(np.argmax(np.cumsum(pfull.explained_variance_ratio_) >= VAR)) + 1
pca = PCA(n_components=K, random_state=0).fit(Zfit)
def errmat(Z): return (Z - pca.inverse_transform(pca.transform(Z))) ** 2
def bthr(s, h, b): return np.sort(s)[-max(1, int(round(b * h)))]
THR = bthr(errmat(Zcal).mean(1), cal_hours, BUDGET)
R['K'] = K; R['THR'] = float(THR)

for w in Wte:
    w['Z'] = sc.transform(w['X']); w['E'] = errmat(w['Z']); w['s'] = w['E'].mean(1)
    w['pred'] = (w['s'] > THR).astype(int)
    w['gE'] = np.stack([w['E'][:, GR == g].sum(1) for g in G], 1)   # grouped error, non negative
    w['hours'] = (w['t'].max() - w['t'].min()) / np.timedelta64(1, 'h')

# sanity: reproduce headline numbers
f1s = [precision_recall_fscore_support(w['attack'], w['pred'], average='binary', zero_division=0)[2] for w in Wte]
aucs = [roc_auc_score(w['attack'], w['s']) for w in Wte]
R['unsup_meanF1'] = float(np.mean(f1s)); R['unsup_meanAUC'] = float(np.mean(aucs))
print('K', K, 'mean F1 %.3f AUC %.3f' % (np.mean(f1s), np.mean(aucs)), 'alerts', sum(w['pred'].sum() for w in Wte))

# ------------------------------------------------------------------ supervised reference
sup = []
for h in range(5):
    fit = [i for i in range(5) if i != h]
    Xf = np.vstack([Wte[i]['X'] for i in fit]); yf = np.concatenate([Wte[i]['attack'] for i in fit])
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, class_weight='balanced',
                             random_state=0, n_jobs=-1, verbose=-1).fit(Xf, yf)
    pb = clf.predict_proba(Wte[h]['X'])[:, 1]
    Wte[h]['sp'] = (pb > 0.5).astype(int); Wte[h]['spb'] = pb; Wte[h]['clf'] = clf
    Wte[h]['bg'] = Xf[np.random.RandomState(0).choice(len(Xf), 200, replace=False)]
sf1 = [precision_recall_fscore_support(w['attack'], w['sp'], average='binary', zero_division=0)[2] for w in Wte]
R['sup_meanF1'] = float(np.mean(sf1))
print('sup mean F1 %.3f' % np.mean(sf1), 'elapsed', round(time.time() - t0))

# ------------------------------------------------------------------ consolidation
def cos(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def consolidate(t, sig, rule, tau=TAU, sigma=SIG):
    """t: alert times (sorted); sig: signature rows. Returns incident id per alert."""
    inc = np.zeros(len(t), int); k = 0
    for i in range(1, len(t)):
        gap = (t[i] - t[i - 1]) / np.timedelta64(1, 's')
        same = gap <= tau
        if same and rule == 'cos': same = cos(sig[i], sig[i - 1]) >= sigma
        if same and rule == 'dom': same = int(np.argmax(sig[i])) == int(np.argmax(sig[i - 1]))
        if not same: k += 1
        inc[i] = k
    return inc

def purity(w, idx, inc):
    """idx: alert window indices (time order), inc: incident ids."""
    eo = w['eo'][idx]; n_inc = inc.max() + 1 if len(inc) else 0
    inc_ev = [set(eo[inc == k]) - {-1} for k in range(n_inc)]
    false_inc = sum(1 for s in inc_ev if len(s) == 0)
    merged_inc = sum(1 for s in inc_ev if len(s) >= 2)
    ev_inc = [set(inc[eo == e]) for e in range(len(w['ev']))]
    detected = sum(1 for s in ev_inc if len(s) >= 1)
    split = sum(1 for s in ev_inc if len(s) >= 2)
    one2one = sum(1 for e, s in enumerate(ev_inc) if len(s) == 1 and inc_ev[next(iter(s))] == {e})
    spans = []
    for k in range(n_inc):
        tt = w['t'][idx[inc == k]]
        spans.append((tt.max() - tt.min()) / np.timedelta64(1, 's') + W)
    return dict(incidents=int(n_inc), false_inc=false_inc, merged_inc=merged_inc, events=len(w['ev']),
                detected=detected, split=split, one2one=one2one, spans=spans)

def run_variant(path, rule, tau=TAU, sigma=SIG):
    agg = dict(alerts=0, incidents=0, false_inc=0, merged_inc=0, events=0, detected=0, split=0, one2one=0, hours=0)
    spans = []
    for w in Wte:
        if path == 'unsup':
            idx = np.where(w['pred'] == 1)[0]; full = w['E'][idx]; grp = w['gE'][idx]
        else:
            idx = np.where(w['sp'] == 1)[0]; full = w['sv'] if 'sv' in w else None; grp = w['gS'] if 'gS' in w else None
        sig = {'time': grp, 'dom': grp, 'cos': grp, 'cosF': full}[rule]
        r = 'cos' if rule in ('cos', 'cosF') else rule
        inc = consolidate(w['t'][idx], sig, r, tau, sigma)
        p = purity(w, idx, inc)
        for k in agg:
            if k in p: agg[k] += p[k]
        agg['alerts'] += len(idx); agg['hours'] += w['hours']; spans += p['spans']
    agg['inc_per_h'] = agg['incidents'] / agg['hours']
    agg['median_span_s'] = float(np.median(spans)); agg['max_span_s'] = float(np.max(spans))
    agg['p90_span_s'] = float(np.percentile(spans, 90))
    return agg

# true event durations for context
evdur = [((w['t'][b] - w['t'][a]) / np.timedelta64(1, 's') + W) for w in Wte for a, b in w['ev']]
R['event_duration_s'] = dict(median=float(np.median(evdur)), max=float(np.max(evdur)), min=float(np.min(evdur)))

T_cons = []
for name, rule in [('Time only', 'time'), ('Dominant subsystem', 'dom'),
                   ('Grouped cosine (ours)', 'cos'), ('Ungrouped cosine', 'cosF')]:
    a = run_variant('unsup', rule); a['method'] = name; T_cons.append(a)
T_cons = pd.DataFrame(T_cons)
print(T_cons[['method', 'alerts', 'incidents', 'inc_per_h', 'false_inc', 'merged_inc', 'split', 'detected',
              'one2one', 'median_span_s', 'p90_span_s', 'max_span_s']].round(2).to_string(index=False))
T_cons.to_csv(f'{OUT}/r11_r12_consolidation_baselines.csv', index=False)

grid = []
for tau in [60, 120, 300, 600, 900]:
    for sigma in [0.6, 0.7, 0.8, 0.9, 0.95]:
        a = run_variant('unsup', 'cos', tau, sigma); a.update(tau=tau, sigma=sigma); grid.append(a)
    for rule in ['time', 'dom']:
        a = run_variant('unsup', rule, tau); a.update(tau=tau, sigma=np.nan, rule=rule); grid.append(a)
T_grid = pd.DataFrame(grid); T_grid['rule'] = T_grid['rule'].fillna('cos')
T_grid.to_csv(f'{OUT}/r11_tau_sigma_grid.csv', index=False)
print(T_grid[['rule', 'tau', 'sigma', 'incidents', 'merged_inc', 'split', 'one2one', 'false_inc']].to_string(index=False))
print('elapsed', round(time.time() - t0))


# ------------------------------------------------------------------ R1.1 stress test: concurrent events
# The five test files are overlaid on one timeline (each aligned to its own start), emulating one
# operator console that receives several concurrent attack campaigns. Ground truth event identity is
# (file, event). This is a constructed scenario, reported as such.
rel_t, sigs, fulls, keys, subs = [], [], [], [], []
for fi, w in enumerate(Wte):
    idx = np.where(w['pred'] == 1)[0]
    rel_t.append((w['t'][idx] - w['t'][0]).astype('timedelta64[s]').astype(np.int64))
    sigs.append(w['gE'][idx]); fulls.append(w['E'][idx])
    keys += [(fi, int(e)) if e >= 0 else None for e in w['eo'][idx]]
    subs += [tuple(r) for r in w['sub'][idx]]
rel_t = np.concatenate(rel_t); o = np.argsort(rel_t, kind='stable')
tt = (rel_t[o] * 1_000_000_000).astype('timedelta64[ns]') + np.datetime64('2000-01-01')
sg = np.vstack(sigs)[o]; fl = np.vstack(fulls)[o]; ky = [keys[i] for i in o]
allev = sorted({k for k in ky if k is not None})
def purity2(inc):
    n = inc.max() + 1; inc_ev = [set(ky[i] for i in np.where(inc == k)[0]) - {None} for k in range(n)]
    ev_inc = {e: set() for e in allev}
    for i, k in enumerate(ky):
        if k is not None: ev_inc[k].add(inc[i])
    merged = sum(1 for s in inc_ev if len(s) >= 2)
    split = sum(1 for s in ev_inc.values() if len(s) >= 2)
    one = sum(1 for e, s in ev_inc.items() if len(s) == 1 and inc_ev[next(iter(s))] == {e})
    return dict(incidents=int(n), merged_inc=merged, split=split, one2one=one, events=len(allev))
ST = []
for name, rule, sig in [('Time only', 'time', sg), ('Dominant subsystem', 'dom', sg),
                        ('Grouped cosine (ours)', 'cos', sg), ('Ungrouped cosine', 'cos', fl)]:
    r = purity2(consolidate(tt, sig, rule)); r['method'] = name; ST.append(r)
ST = pd.DataFrame(ST); print(ST.to_string(index=False)); ST.to_csv(f'{OUT}/r11_stress_concurrent.csv', index=False)

# ------------------------------------------------------------------ attribution variants (R1.3)
def occlusion(Z):
    """a priori group attribution: score drop when a whole subsystem is set to its training mean (0)."""
    base = errmat(Z).mean(1); out = []
    for g in G:
        Zo = Z.copy(); Zo[:, GR == g] = 0.0
        out.append(base - errmat(Zo).mean(1))
    return np.stack(out, 1)

for h, w in enumerate(Wte):
    idx = np.where(w['sp'] == 1)[0]
    ex = shap.TreeExplainer(w['clf'])
    sv = ex.shap_values(w['X'][idx]); sv = sv[1] if isinstance(sv, list) else sv
    sv = sv[:, :, 1] if np.ndim(sv) == 3 else sv
    exi = shap.TreeExplainer(w['clf'], data=w['bg'], feature_perturbation='interventional', model_output='raw')
    svi = exi.shap_values(w['X'][idx], check_additivity=False); svi = svi[1] if isinstance(svi, list) else svi
    svi = svi[:, :, 1] if np.ndim(svi) == 3 else svi
    w['sv'] = sv; w['gS'] = np.stack([sv[:, GR == g].sum(1) for g in G], 1)
    w['gSabs'] = np.stack([np.abs(sv[:, GR == g]).sum(1) for g in G], 1)
    w['gSi'] = np.stack([svi[:, GR == g].sum(1) for g in G], 1)
    w['occ'] = occlusion(w['Z'])           # every window, unsupervised path
print('shap done', round(time.time() - t0))

def loc_acc(sigrows, sub, use4=False):
    """sigrows: grouped scores (n x 4). sub: (n x 3) labels. single subsystem windows only."""
    m = sub.sum(1) == 1
    if m.sum() == 0: return np.nan, 0
    s = sigrows[m] if use4 else sigrows[m][:, :3]
    top = s.argmax(1); return float((top == sub[m].argmax(1)).mean()), int(m.sum())

rows = []
def add(name, path, key, use4=False):
    hit = n = 0; per_ev = []
    for w in Wte:
        if path == 'unsup':
            idx = np.where(w['pred'] == 1)[0]; sig = w[key][idx]
        else:
            idx = np.where(w['sp'] == 1)[0]; sig = w[key]
        att = w['attack'][idx] == 1
        a, k = loc_acc(sig[att], w['sub'][idx][att], use4)
        if k: hit += a * k; n += k
    rows.append(dict(variant=name, n=n, acc=hit / n))

add('Unsup, grouped error (P1-P3)', 'unsup', 'gE')
add('Unsup, grouped error incl. P4', 'unsup', 'gE', True)
add('Unsup, a priori group occlusion', 'unsup', 'occ')
add('Sup, signed TreeSHAP, path dependent', 'sup', 'gS')
add('Sup, magnitude TreeSHAP, path dependent', 'sup', 'gSabs')
add('Sup, signed TreeSHAP, interventional', 'sup', 'gSi')
add('Sup, signed TreeSHAP incl. P4', 'sup', 'gS', True)
T_attr = pd.DataFrame(rows); print(T_attr.round(3).to_string(index=False))
T_attr.to_csv(f'{OUT}/r13_attribution_variants.csv', index=False)

# cancellation: signed vs magnitude disagreement
dis = tot = 0
for w in Wte:
    dis += int((w['gS'][:, :3].argmax(1) != w['gSabs'][:, :3].argmax(1)).sum()); tot += len(w['gS'])
R['signed_vs_magnitude_disagree'] = dict(n=tot, disagree=dis)
# occlusion vs post hoc agreement on unsup alerts
agree = tot = 0
for w in Wte:
    idx = np.where(w['pred'] == 1)[0]
    agree += int((w['gE'][idx, :3].argmax(1) == w['occ'][idx, :3].argmax(1)).sum()); tot += len(idx)
R['occlusion_vs_posthoc_agree'] = dict(n=tot, agree=agree)

# ------------------------------------------------------------------ R1.4 populations
brk = dict(alerts=0, normal=0, attack=0, single=0, multi=0, nolabel=0)
mh = mn = 0; t2h = 0; det_all_h = det_all_n = 0
for w in Wte:
    idx = np.where(w['pred'] == 1)[0]; y = w['attack'][idx]; ns = w['sub'][idx].sum(1)
    brk['alerts'] += len(idx); brk['normal'] += int((y == 0).sum()); brk['attack'] += int((y == 1).sum())
    brk['single'] += int(((y == 1) & (ns == 1)).sum()); brk['multi'] += int(((y == 1) & (ns >= 2)).sum())
    brk['nolabel'] += int(((y == 1) & (ns == 0)).sum())
    a = (y == 1) & (ns >= 1)
    top = w['gE'][idx][a][:, :3].argmax(1); sub = w['sub'][idx][a]
    det_all_h += int(sub[np.arange(len(top)), top].sum()); det_all_n += len(top)
    m = (y == 1) & (ns >= 2)
    if m.any():
        g = w['gE'][idx][m][:, :3]; sb = w['sub'][idx][m]
        tp = g.argmax(1); mh += int(sb[np.arange(len(tp)), tp].sum()); mn += len(tp)
        top2 = np.argsort(-g, 1)[:, :2]
        t2h += int(sum(set(np.where(sb[i])[0]) <= set(top2[i]) for i in range(len(sb))))
R['alert_breakdown'] = brk
R['multi_subsystem'] = dict(n=mn, top1_in_labelled=mh, top2_cover_all=t2h)
R['unconditional_detected_attack_windows'] = dict(n=det_all_n, hit=det_all_h)

# all true attack windows, detected or not (attribution exists below threshold too)
ah = an = 0; mh2 = mn2 = 0; missed_events = []
for fidx, w in enumerate(Wte):
    a = (w['attack'] == 1) & (w['sub'].sum(1) >= 1)
    top = w['gE'][a][:, :3].argmax(1); sub = w['sub'][a]
    ah += int(sub[np.arange(len(top)), top].sum()); an += len(top)
    miss = (w['attack'] == 1) & (w['pred'] == 0) & (w['sub'].sum(1) >= 1)
    if miss.any():
        top = w['gE'][miss][:, :3].argmax(1); sub = w['sub'][miss]
        mh2 += int(sub[np.arange(len(top)), top].sum()); mn2 += len(top)
    for e, (a0, b0) in enumerate(w['ev']):
        if w['pred'][a0:b0 + 1].sum() == 0:
            seg = slice(a0, b0 + 1); s = w['sub'][seg]
            missed_events.append(dict(file=fidx + 1, windows=b0 - a0 + 1,
                                      dur_s=float((w['t'][b0] - w['t'][a0]) / np.timedelta64(1, 's') + W),
                                      subsystems=''.join(str(k + 1) for k in range(3) if s[:, k].any()),
                                      max_score_over_thr=float(w['s'][seg].max() / THR)))
R['all_attack_windows'] = dict(n=an, hit=ah)
R['undetected_attack_windows'] = dict(n=mn2, hit=mh2)
R['missed_events'] = missed_events


# ------------------------------------------------------------------ incident level localisation
# Each incident is labelled with the subsystem holding the largest grouped error summed over its
# windows. Scored against the subsystem labels present in the incident's attack windows.
IL = []
for name, rule in [('Time only', 'time'), ('Grouped cosine (ours)', 'cos')]:
    hit = n = 0
    for w in Wte:
        idx = np.where(w['pred'] == 1)[0]; inc = consolidate(w['t'][idx], w['gE'][idx], rule)
        for k in range(inc.max() + 1):
            m = inc == k; a = w['attack'][idx][m] == 1
            if not a.any(): continue
            lab = w['sub'][idx][m][a].max(0)
            if lab.sum() == 0: continue
            top = int(w['gE'][idx][m].sum(0)[:3].argmax()); hit += int(lab[top]); n += 1
    IL.append(dict(method=name, attack_incidents=n, correct_subsystem=hit))
R['incident_localisation'] = IL; print(IL)

# ------------------------------------------------------------------ R2 onset localisation and causal timing
onset = []
for fi, w in enumerate(Wte):
    for e, (a0, b0) in enumerate(w['ev']):
        al = np.where(w['pred'][a0:b0 + 1] == 1)[0]
        if not len(al): continue
        i = a0 + al[0]
        # subsystem whose label switches on first inside the event (per second of window start)
        first_on = []
        for k in range(3):
            on = np.where(w[f'attack_P{k + 1}'][a0:b0 + 1] == 1)[0]
            first_on.append(on[0] if len(on) else 10 ** 9)
        lead = int(np.argmin(first_on)); multi = sum(v < 10 ** 9 for v in first_on) >= 2
        tie = sum(v == min(first_on) for v in first_on) >= 2
        top = int(w['gE'][i, :3].argmax())
        active = w['sub'][i]   # subsystems labelled inside the first alerted window
        onset.append(dict(file=fi + 1, event=e, latency_s=float(al[0] * S), lead=lead + 1, top=top + 1,
                          hit=int(top == lead), multi=int(multi), tie=int(tie), hit_active=int(active[top] == 1),
                          n_active=int(active.sum())))
O = pd.DataFrame(onset); O.to_csv(f'{OUT}/r2_onset_localisation.csv', index=False)
R['onset'] = dict(n=len(O), hit=int(O.hit.sum()), multi_events=int(O.multi.sum()),
                  multi_hit=int(O[O.multi == 1].hit.sum()), single_events=int((O.multi == 0).sum()),
                  single_hit=int(O[O.multi == 0].hit.sum()), ties=int(O.tie.sum()),
                  hit_active=int(O.hit_active.sum()), multi_hit_active=int(O[O.multi == 1].hit_active.sum()),
                  median_latency_s=float(O.latency_s.median()))

# ------------------------------------------------------------------ R1.5 uncertainty
rng = np.random.RandomState(0); B = 1000; BL = 360   # one hour blocks of windows
blocks = []
for fi, w in enumerate(Wte):
    n = len(w['attack']); blocks.append([np.arange(i, min(i + BL, n)) for i in range(0, n, BL)])
def f1(y, p): return precision_recall_fscore_support(y, p, average='binary', zero_division=0)[2]
bu, bs, bd = [], [], []
for _ in range(B):
    fu, fs = [], []
    for fi, w in enumerate(Wte):
        pick = rng.randint(0, len(blocks[fi]), len(blocks[fi]))
        ix = np.concatenate([blocks[fi][j] for j in pick])
        fu.append(f1(w['attack'][ix], w['pred'][ix])); fs.append(f1(w['attack'][ix], w['sp'][ix]))
    bu.append(np.mean(fu)); bs.append(np.mean(fs)); bd.append(np.mean(fs) - np.mean(fu))
ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
R['boot'] = dict(unsup_F1=ci(bu), sup_F1=ci(bs), diff_sup_minus_unsup=ci(bd),
                 diff_point=float(np.mean(sf1) - np.mean(f1s)),
                 per_file_diff=[float(a - b) for a, b in zip(sf1, f1s)])

# event level bootstrap of localisation (unsup, single subsystem detected windows)
ev_hits = []
for w in Wte:
    idx = np.where(w['pred'] == 1)[0]
    for e in range(len(w['ev'])):
        m = (w['eo'][idx] == e) & (w['sub'][idx].sum(1) == 1)
        if m.any():
            top = w['gE'][idx][m][:, :3].argmax(1)
            ev_hits.append((int((top == w['sub'][idx][m].argmax(1)).sum()), int(m.sum())))
ev_hits = np.array(ev_hits); bl = []
for _ in range(B):
    s = ev_hits[rng.randint(0, len(ev_hits), len(ev_hits))]; bl.append(s[:, 0].sum() / s[:, 1].sum())
R['boot']['loc_unsup'] = ci(bl); R['boot']['loc_events'] = int(len(ev_hits))
# Wilson interval for event coverage
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); r = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [float((c - r) / d), float((c + r) / d)]
cov = T_cons.loc[T_cons.method == 'Grouped cosine (ours)'].iloc[0]
R['coverage_wilson'] = wilson(int(cov.detected), int(cov.events))

json.dump(R, open(f'{OUT}/revision_summary.json', 'w'), indent=1, default=float)
print(json.dumps(R, indent=1, default=float))
print('total elapsed', round(time.time() - t0))
