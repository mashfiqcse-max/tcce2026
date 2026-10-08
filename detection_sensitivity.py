"""Detection sensitivity to the two free choices (explained variance criterion, false alarm budget).
Pooled over all five HAI 21.03 test files. Usage: python detection_sensitivity.py <hai-21.03> <out.csv>"""
import sys, glob, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
D, OUT = sys.argv[1], sys.argv[2]
LAB = ['attack', 'attack_P1', 'attack_P2', 'attack_P3']; W, S = 60, 10
def load(p):
    df = pd.read_csv(p); df.columns = [c.strip() for c in df.columns]
    df['time'] = pd.to_datetime(df['time']); return df.sort_values('time').reset_index(drop=True)
tr = [load(p) for p in sorted(glob.glob(D + '/train*.csv*'))]; te = [load(p) for p in sorted(glob.glob(D + '/test*.csv*'))]
SEN = [c for c in te[0].columns if c not in LAB + ['time']]
def wz(df, lab=True):
    V = df[SEN].to_numpy(np.float32); st = np.arange(0, len(V) - W + 1, S); F = []
    for i in range(0, len(st), 2000):
        s = st[i:i + 2000]; b = V[s[:, None] + np.arange(W)[None, :]]
        F.append(np.concatenate([b.mean(1), b.std(1), b.min(1), b.max(1), np.abs(np.diff(b, 1, 1)).mean(1)], 1).astype(np.float32))
    X = np.vstack(F)
    return (X, df['attack'].to_numpy(np.int8)[st[:, None] + np.arange(W)[None, :]].max(1), df['time'].to_numpy()[st]) if lab else X
Xtr = np.vstack([wz(d, False) for d in tr]); T = [wz(d) for d in te]
cut = int(0.8 * len(Xtr)); sc = StandardScaler().fit(Xtr[:cut]); Zf = sc.transform(Xtr[:cut]); Zc = sc.transform(Xtr[cut:])
cal_h = len(Zc) * S / 3600
Za = sc.transform(np.vstack([t[0] for t in T])); y = np.concatenate([t[1] for t in T])
hours = sum((t[2].max() - t[2].min()) / np.timedelta64(1, 'h') for t in T)
cum = np.cumsum(PCA(150, random_state=0).fit(Zf).explained_variance_ratio_)
rows = []
for v in [0.90, 0.95, 0.98, 0.99]:
    k = int(np.argmax(cum >= v)) + 1; p = PCA(k, random_state=0).fit(Zf)
    e = lambda Z: ((Z - p.inverse_transform(p.transform(Z))) ** 2).mean(1)
    ec, ea = e(Zc), e(Za); auc = roc_auc_score(y, ea)
    for b in [0.25, 0.5, 1.0]:
        thr = np.sort(ec)[-max(1, int(round(b * cal_h)))]; pr = (ea > thr).astype(int)
        P, R, F, _ = precision_recall_fscore_support(y, pr, average='binary', zero_division=0)
        rows.append(dict(variance=v, k=k, budget=b, P=P, R=R, F1=F, AUC=auc, alerts_per_h=pr.sum() / hours))
df = pd.DataFrame(rows); df.to_csv(OUT, index=False); print(df.round(3).to_string(index=False))
