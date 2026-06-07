#!/usr/bin/env python3
###############################################################################
#  Deep recurrent autoencoders for first-year growth phenotyping and early
#  prediction of one-year malnutrition in preterm infants (KMC cohort)
#
#  Python "engine of record" — reproduces the manuscript tables and figures.
#  Mirrors the companion R script (reproduce_KMC_deep_growth.R) section for
#  section. This is the code that produced the numbers reported in the paper.
#
#    1. data preparation        2. LCTM (latent class growth analysis)
#    3. LSTM/GRU autoencoders    4. outcome-by-phenotype
#    5. early prediction         6. tables & figures
#
#  Requirements:
#    pip install numpy pandas scikit-learn matplotlib openpyxl torch
###############################################################################
import os, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             roc_curve, adjusted_rand_score, normalized_mutual_info_score)

# ----------------------------- configuration --------------------------------
DATA_XLSX = "KMC_results_Prof_Arashi_growth_only.xlsx"   # <- set path
OUTDIR, FIGDIR = "outputs", "figs"
os.makedirs(OUTDIR, exist_ok=True); os.makedirs(FIGDIR, exist_ok=True)

SEED        = 42
AE_SEEDS    = [13, 21, 34]      # autoencoder restarts for phenotyping
AE_EPOCHS   = 400
CLF_EPOCHS  = 160               # LSTM/GRU outcome classifiers
BOOT_N      = 2000              # bootstrap reps for AUROC CIs
np.random.seed(SEED); torch.manual_seed(SEED)

INDICES   = ["WAZ", "LAZ", "WLZ", "HCZ"]
KPAPER    = {"WAZ": 3, "LAZ": 3, "WLZ": 3, "HCZ": 2}
GRIDS     = {"WAZ": np.arange(32, 89, 2.0), "LAZ": np.arange(38, 89, 2.0),
             "WLZ": np.arange(40, 89, 2.0), "HCZ": np.arange(38, 89, 2.0)}
NAMEORDER = {"WAZ": ["WAZ catch-up", "gradual WAZ gain", "WAZ faltering"],
             "LAZ": ["LAZ catch-up", "gradual LAZ gain", "LAZ faltering"],
             "WLZ": ["WLZ catch-up", "WLZ maintenance", "WLZ faltering"],
             "HCZ": ["HCZ gain", "HCZ maintenance"]}
OUTCOMES  = ["underweight", "stunting", "wasting", "overweight"]
CUT       = 50.0                # PMA (weeks): Fenton below, WHO at/above

###############################################################################
# 1. DATA PREPARATION
###############################################################################
def load_long():
    df = pd.read_excel(DATA_XLSX)
    df["infant"] = df["Visit_ID"].astype(str).str.split("-").str[0]
    df["visit"]  = df["Visit_ID"].astype(str).str.split("-").str[1].astype(int)
    df = df.sort_values(["infant", "PMA_W"]).reset_index(drop=True)
    def comb(fen, who):                       # Fenton <50 wk PMA, WHO thereafter
        v = np.where((df["PMA_W"] < CUT) & df[fen].notna(), df[fen], df[who])
        return pd.to_numeric(pd.Series(v, index=df.index), errors="coerce")
    df["WAZ"]  = comb("F13_W_Z",  "CA_WAZ")
    df["LAZ"]  = comb("F13_L_Z",  "CA_HAZ")
    df["HCZ"]  = comb("F13_HC_Z", "CA_HCZ")
    df["WLZ"]  = pd.to_numeric(df["CA_WHZ"], errors="coerce")   # WHO only
    df["BMIZ"] = pd.to_numeric(df["CA_BAZ"], errors="coerce")
    return df

full = load_long()
# analysis window: PMA 200-650 days; infants with >=3 measurements in window
win = full[(full["PMA_W"] >= 200/7) & (full["PMA_W"] <= 650/7)].copy()
keep = win.groupby("infant").size(); keep = keep[keep >= 3].index
win = win[win["infant"].isin(keep)].copy()

# baseline covariates (birth = lowest PMA row)
birth = full.sort_values("PMA_W").groupby("infant").first()
base = pd.DataFrame(index=sorted(win["infant"].unique()))
base["sex_male"] = (birth["SEX"].reindex(base.index) == "Male").astype(int)
base["GA"]  = birth["GA"].reindex(base.index)
base["BWZ"] = birth["F13_W_Z"].reindex(base.index)
early = (full[(full["PMA_W"] < CUT) & full["F13_W_Z"].notna()]
         .sort_values("PMA_W").groupby("infant")["F13_W_Z"].last())
base["earlyWAZgain"] = early.reindex(base.index) - base["BWZ"]

# one-year outcomes at last visit in window (missing z -> 0, i.e. not malnourished)
last = win.sort_values("PMA_W").groupby("infant").last()
for c in ["WAZ", "LAZ", "WLZ", "HCZ", "BMIZ"]:
    base[f"{c}_last"] = last[c].reindex(base.index)
base["CA_M_last"]   = last["CA_M"].reindex(base.index)
base["underweight"] = (base["WAZ_last"] < -2).astype(int)
base["stunting"]    = (base["LAZ_last"] < -2).astype(int)
base["wasting"]     = (base["WLZ_last"] < -2).astype(int)
base["overweight"]  = (base["BMIZ_last"] > 2).astype(int)
base = base.dropna(subset=["BWZ", "GA", "earlyWAZgain"])
ids  = list(base.index)
print(f"Infants: {len(base)} | outcomes U/S/W/O = "
      f"{base.underweight.sum()}/{base.stunting.sum()}/{base.wasting.sum()}/{base.overweight.sum()}")

def grid_matrix(z):
    """Interpolate each infant's trajectory onto the regular grid (flat-hold ends)."""
    g = GRIDS[z]; M = np.full((len(ids), len(g)), np.nan)
    for i, inf in enumerate(ids):
        d = win[(win["infant"] == inf) & win[z].notna()].sort_values("PMA_W")
        if len(d) >= 2:
            M[i] = np.interp(g, d["PMA_W"].values, d[z].values)
    col = np.nanmean(M, 0)
    for j in range(M.shape[1]):
        M[np.isnan(M[:, j]), j] = col[j]
    return g, M

###############################################################################
# 2. LATENT CLASS TRAJECTORY MODELLING (reference)
#    Latent class growth analysis: EM mixture of class-specific quadratic
#    regressions on the raw (PMA, z) observations. Class count fixed to the
#    published solution; classes labelled by attained z-level.
###############################################################################
def seqs(z):
    out = []
    for inf in ids:
        d = win[(win["infant"] == inf) & win[z].notna()].sort_values("PMA_W")
        out.append((d["PMA_W"].values.astype(float), d[z].values.astype(float)))
    return out

def design(t, tm, ts):
    tt = (t - tm) / ts
    return np.column_stack([np.ones_like(tt), tt, tt**2])

def em_mixreg(S, k, tm, ts, restarts=12, iters=120):
    Xs = [design(t, tm, ts) for t, _ in S]; Ys = [y for _, y in S]; ni = len(S); best = None
    for r in range(restarts):
        R = np.random.RandomState(r).dirichlet(np.ones(k), ni)
        for _ in range(iters):
            B = np.zeros((k, 3)); s2 = np.zeros(k); pi = R.mean(0) + 1e-9
            for c in range(k):
                XtX = 1e-6 * np.eye(3); Xty = np.zeros(3); ss = 0.0; w = 0.0
                for i in range(ni):
                    if len(Ys[i]):
                        XtX += R[i, c] * Xs[i].T @ Xs[i]; Xty += R[i, c] * Xs[i].T @ Ys[i]
                B[c] = np.linalg.solve(XtX, Xty)
                for i in range(ni):
                    if len(Ys[i]):
                        res = Ys[i] - Xs[i] @ B[c]; ss += R[i, c] * np.sum(res**2); w += R[i, c] * len(Ys[i])
                s2[c] = max(ss / max(w, 1e-9), 1e-3)
            logR = np.zeros((ni, k))
            for i in range(ni):
                for c in range(k):
                    res = Ys[i] - Xs[i] @ B[c]
                    logR[i, c] = (np.log(pi[c]) - 0.5*len(res)*np.log(2*np.pi*s2[c])
                                  - 0.5*np.sum(res**2)/s2[c])
            m = logR.max(1, keepdims=True); P = np.exp(logR - m); P /= P.sum(1, keepdims=True)
            ll = float(np.sum(m + np.log(np.exp(logR - m).sum(1, keepdims=True)))); R = P
        if best is None or ll > best[0]:
            best = (ll, B, s2, pi, R)
    return best

lctm = {}
print("\nLCTM (latent class growth analysis):")
for z in INDICES:
    S = seqs(z); allt = np.concatenate([t for t, _ in S]); tm, ts = allt.mean(), allt.std()
    ll, B, s2, pi, R = em_mixreg(S, KPAPER[z], tm, ts)
    raw = R.argmax(1); g = GRIDS[z]; Xg = design(g, tm, ts)
    ends = {c: (Xg @ B[c])[-1] for c in range(KPAPER[z])}
    order = sorted(range(KPAPER[z]), key=lambda c: -ends[c])      # highest attained first
    remap = {old: NAMEORDER[z][i] for i, old in enumerate(order)}
    lab = np.array([remap[c] for c in raw])
    appa = {remap[c]: R[raw == c, c].mean() for c in range(KPAPER[z])}
    ent  = 1 - (-(R * np.log(R + 1e-12)).sum()) / (len(R) * np.log(KPAPER[z]))
    curve = {remap[c]: Xg @ B[c] for c in range(KPAPER[z])}
    lctm[z] = dict(lab=lab, curve=curve, grid=g, appa=appa, entropy=ent, order=NAMEORDER[z])
    sizes = ", ".join(f"{nm} {100*np.mean(lab==nm):.1f}%" for nm in NAMEORDER[z])
    print(f"  {z}: entropy={ent:.2f} | {sizes}")

###############################################################################
# 3. DEEP RECURRENT AUTOENCODERS (LSTM / GRU) FOR PHENOTYPING
###############################################################################
class AE(nn.Module):
    def __init__(self, T, cell="LSTM", H=16, L=4):
        super().__init__(); self.T = T
        rnn = nn.LSTM if cell == "LSTM" else nn.GRU
        self.enc = rnn(1, H, batch_first=True); self.toL = nn.Linear(H, L)
        self.fromL = nn.Linear(L, H); self.dec = rnn(H, H, batch_first=True); self.out = nn.Linear(H, 1)
    def forward(self, x):
        _, h = self.enc(x); h = h[0] if isinstance(h, tuple) else h
        zc = self.toL(h[-1]); d = self.fromL(zc).unsqueeze(1).repeat(1, self.T, 1)
        o, _ = self.dec(d); return self.out(o), zc

def run_ae(z, cell, seed):
    g, M = grid_matrix(z); T = len(g)
    Xn = (M - M.mean(0)) / (M.std(0) + 1e-6)
    X = torch.tensor(Xn, dtype=torch.float32).unsqueeze(-1)
    torch.manual_seed(seed); np.random.seed(seed); m = AE(T, cell)
    opt = torch.optim.Adam(m.parameters(), lr=0.01); lf = nn.MSELoss()
    for _ in range(AE_EPOCHS):
        opt.zero_grad(); rec, _ = m(X); loss = lf(rec.squeeze(-1), X.squeeze(-1)); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        _, emb = m(X); mse = float(loss.detach())
    gm = GaussianMixture(KPAPER[z], covariance_type="full", n_init=10, random_state=seed).fit(emb.numpy())
    cl = gm.predict(emb.numpy())
    ends = {c: M[cl == c].mean(0)[-1] for c in range(KPAPER[z])}
    order = sorted(range(KPAPER[z]), key=lambda c: -ends[c])
    remap = {old: NAMEORDER[z][i] for i, old in enumerate(order)}
    lab = np.array([remap[c] for c in cl]); curve = {remap[c]: M[cl == c].mean(0) for c in range(KPAPER[z])}
    return lab, curve, mse, g, M

deep = {"LSTM": {}, "GRU": {}}; agree = {}
print("\nDeep phenotyping (autoencoder embeddings clustered, agreement with LCTM):")
for cell in ("LSTM", "GRU"):
    for z in INDICES:
        aris = []; best = None
        for s in AE_SEEDS:
            lab, curve, mse, g, M = run_ae(z, cell, s)
            aris.append(adjusted_rand_score(lctm[z]["lab"], lab))
            if best is None or mse < best["mse"]:
                best = dict(lab=lab, curve=curve, mse=mse, grid=g, M=M)
        deep[cell][z] = best
        agree[(cell, z)] = (float(np.mean(aris)), float(np.std(aris)),
                            float(normalized_mutual_info_score(lctm[z]["lab"], best["lab"])))
        print(f"  {cell} {z}: ARI={np.mean(aris):.2f} ({np.std(aris):.2f})  "
              f"NMI={agree[(cell,z)][2]:.2f}  recon_mse={best['mse']:.3f}")

###############################################################################
# 4. ONE-YEAR OUTCOME PREVALENCE BY PHENOTYPE
###############################################################################
def prevalence_table(label_of):
    rows = []
    for z in INDICES:
        for nm in NAMEORDER[z]:
            m = label_of[z] == nm; sub = base[m]
            rows.append(dict(Index=z, Phenotype=nm, n=int(m.sum()),
                Underweight=round(100*sub.underweight.mean(), 1),
                Stunting=round(100*sub.stunting.mean(), 1),
                Wasting=round(100*sub.wasting.mean(), 1),
                Overweight=round(100*sub.overweight.mean(), 1)))
    return pd.DataFrame(rows)

T3 = prevalence_table({z: lctm[z]["lab"] for z in INDICES})
T3.to_csv(f"{OUTDIR}/T3_outcome_by_phenotype.csv", index=False)

###############################################################################
# 5. EARLY PREDICTION OF ONE-YEAR MALNUTRITION (<50 wk PMA)
###############################################################################
EG = np.arange(30, 51, 2.0)
def early_seq(inf):
    d = full[(full["infant"] == inf) & (full["PMA_W"] < CUT) & full["F13_W_Z"].notna()].sort_values("PMA_W")
    if len(d) < 2:
        return np.full(len(EG), base.loc[inf, "BWZ"])
    return np.interp(EG, d["PMA_W"].values, d["F13_W_Z"].values)

SEQ  = np.array([early_seq(i) for i in ids])                 # (n, T) early weight-z
Xtab = base[["BWZ", "earlyWAZgain", "GA", "sex_male"]].values.astype(float)
STAT = base[["GA", "sex_male"]].values.astype(float)
Y    = {o: base[o].values.astype(int) for o in OUTCOMES}

def boot_ci(y, p, B=BOOT_N, seed=1):
    rng = np.random.RandomState(seed); a = []
    for _ in range(B):
        idx = rng.randint(0, len(y), len(y))
        if len(np.unique(y[idx])) > 1:
            a.append(roc_auc_score(y[idx], p[idx]))
    return np.percentile(a, [2.5, 97.5])

# --- tabular baselines (5-fold out-of-fold) ---
MODELS = {"Logistic":     lambda: LogisticRegression(class_weight="balanced", max_iter=1000),
          "RandomForest": lambda: RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED),
          "GradBoost":    lambda: GradientBoostingClassifier(random_state=SEED)}
def oof_tab(model_fn, y, k=5):
    skf = StratifiedKFold(k, shuffle=True, random_state=SEED); p = np.zeros(len(y))
    for tr, te in skf.split(Xtab, y):
        sc = StandardScaler().fit(Xtab[tr]); m = model_fn(); m.fit(sc.transform(Xtab[tr]), y[tr])
        p[te] = m.predict_proba(sc.transform(Xtab[te]))[:, 1]
    return p

# --- LSTM/GRU classifiers (early weight sequence + static covariates) ---
class Clf(nn.Module):
    def __init__(self, cell, T, H=16, ns=2):
        super().__init__(); rnn = nn.LSTM if cell == "LSTM" else nn.GRU
        self.rnn = rnn(1, H, batch_first=True)
        self.head = nn.Sequential(nn.Linear(H+ns, 16), nn.ReLU(), nn.Dropout(0.2), nn.Linear(16, 1))
    def forward(self, x, s):
        _, h = self.rnn(x); h = h[0] if isinstance(h, tuple) else h
        return self.head(torch.cat([h[-1], s], 1)).squeeze(-1)

def train_clf(cell, Xtr, Str, ytr, Xte, Ste):
    torch.manual_seed(SEED)
    xt = torch.tensor(Xtr, dtype=torch.float32).unsqueeze(-1); st = torch.tensor(Str, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)
    pos = max(ytr.sum(), 1); neg = len(ytr) - ytr.sum()
    pw = torch.tensor([neg/pos], dtype=torch.float32)
    m = Clf(cell, Xtr.shape[1]); opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=1e-4)
    lf = nn.BCEWithLogitsLoss(pos_weight=pw)
    for _ in range(CLF_EPOCHS):
        opt.zero_grad(); lf(m(xt, st), yt).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        xe = torch.tensor(Xte, dtype=torch.float32).unsqueeze(-1); se = torch.tensor(Ste, dtype=torch.float32)
        return torch.sigmoid(m(xe, se)).numpy()

def oof_deep(cell, y, k=5):
    skf = StratifiedKFold(k, shuffle=True, random_state=SEED); p = np.zeros(len(y))
    for tr, te in skf.split(SEQ, y):
        mu, sd = SEQ[tr].mean(), SEQ[tr].std() + 1e-9
        sm, ss = STAT[tr].mean(0), STAT[tr].std(0) + 1e-6
        p[te] = train_clf(cell, (SEQ[tr]-mu)/sd, (STAT[tr]-sm)/ss, y[tr], (SEQ[te]-mu)/sd, (STAT[te]-sm)/ss)
    return p

pred = {}
print("\nEarly prediction (5-fold OOF AUROC):")
for o in OUTCOMES:
    y = Y[o]; pred[o] = {}
    for name, fn in MODELS.items():
        pred[o][name] = oof_tab(fn, y)
    for cell in ("LSTM", "GRU"):
        pred[o][cell] = oof_deep(cell, y)
    line = []
    for name in ["Logistic", "RandomForest", "GradBoost", "LSTM", "GRU"]:
        line.append(f"{name} {roc_auc_score(y, pred[o][name]):.3f}")
    print(f"  {o:12s}: " + " | ".join(line))

def paired_delta(y, pa, pb, B=3000, seed=7):
    rng = np.random.RandomState(seed); d = []
    for _ in range(B):
        idx = rng.randint(0, len(y), len(y))
        if len(np.unique(y[idx])) > 1:
            d.append(roc_auc_score(y[idx], pa[idx]) - roc_auc_score(y[idx], pb[idx]))
    d = np.array(d); return d.mean(), np.percentile(d, [2.5, 97.5]), float(np.mean(d <= 0))

###############################################################################
# 6. TABLES & FIGURES
###############################################################################
def ms(x): return f"{np.mean(x):.2f} ({np.std(x, ddof=1):.2f})"
# ---- Table 1: cohort characteristics ----
t1 = [("n", str(len(base))),
      ("Male, n (%)", f"{int(base.sex_male.sum())} ({100*base.sex_male.mean():.1f})"),
      ("Gestational age, wk", ms(base.GA)),
      ("Birth weight z (Fenton)", ms(base.BWZ.dropna())),
      ("Early WAZ change to <50 wk", ms(base.earlyWAZgain.dropna())),
      ("Age at 1-yr, mo", ms(base.CA_M_last.dropna()))]
for z, lab in [("WAZ_last","1-yr WAZ"),("LAZ_last","1-yr LAZ"),("WLZ_last","1-yr WLZ"),
               ("HCZ_last","1-yr HCZ"),("BMIZ_last","1-yr BMI z")]:
    t1.append((lab, ms(base[z].dropna())))
for o, lab in [("underweight","Underweight, n (%)"),("stunting","Stunting, n (%)"),
               ("wasting","Wasting, n (%)"),("overweight","Overweight, n (%)")]:
    t1.append((lab, f"{int(base[o].sum())} ({100*base[o].mean():.1f})"))
pd.DataFrame(t1, columns=["Characteristic", "Value"]).to_csv(f"{OUTDIR}/T1_characteristics.csv", index=False)

# ---- Table 2: phenotype agreement ----
t2 = []
for z in INDICES:
    lm, ls, ln = agree[("LSTM", z)]; gm, gs, gn = agree[("GRU", z)]
    t2.append(dict(Index=z, Classes=KPAPER[z], Entropy=round(lctm[z]["entropy"], 2),
                   ARI_LSTM=f"{lm:.2f} ({ls:.2f})", ARI_GRU=f"{gm:.2f} ({gs:.2f})",
                   NMI_LSTM=round(ln, 2), NMI_GRU=round(gn, 2)))
pd.DataFrame(t2).to_csv(f"{OUTDIR}/T2_phenotype_agreement.csv", index=False)

# ---- Table 4: prediction performance + paired delta vs logistic ----
t4 = []
for o in OUTCOMES:
    y = Y[o]
    for name in ["Logistic", "RandomForest", "GradBoost", "LSTM", "GRU"]:
        p = pred[o][name]; auc = roc_auc_score(y, p); lo, hi = boot_ci(y, p)
        row = dict(Outcome=o, Model=name, AUROC=f"{auc:.3f} ({lo:.3f}-{hi:.3f})",
                   AUPRC=f"{average_precision_score(y, p):.3f}",
                   F1=f"{f1_score(y, (p>0.5).astype(int)):.3f}",
                   dAUROC_vs_Logistic="—", p="—")
        if name in ("LSTM", "GRU"):
            d, ci, pv = paired_delta(y, p, pred[o]["Logistic"])
            row["dAUROC_vs_Logistic"] = f"{d:+.3f} ({ci[0]:+.3f},{ci[1]:+.3f})"; row["p"] = f"{pv:.3f}"
        t4.append(row)
pd.DataFrame(t4).to_csv(f"{OUTDIR}/T4_prediction.csv", index=False)

# ---- Figure 1: phenotype mean trajectories (LCTM vs LSTM vs GRU) ----
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150})
def color(nm):
    if "catch-up" in nm or (nm.endswith("gain") and "gradual" not in nm): return "#2ca02c"
    if "gradual" in nm:     return "#1f77b4"
    if "maintenance" in nm: return "#7f7f7f"
    if "faltering" in nm:   return "#d62728"
    return "#9467bd"
labs = {"LCTM": {z: lctm[z]["lab"] for z in INDICES},
        "LSTM": {z: deep["LSTM"][z]["lab"] for z in INDICES},
        "GRU":  {z: deep["GRU"][z]["lab"]  for z in INDICES}}
fig, ax = plt.subplots(len(INDICES), 3, figsize=(11, 12), sharex="row")
for r, z in enumerate(INDICES):
    g, M = grid_matrix(z)
    for c, meth in enumerate(["LCTM", "LSTM", "GRU"]):
        a = ax[r, c]; Ldict = labs[meth][z]
        for nm in NAMEORDER[z]:
            m = Ldict == nm
            if m.sum() == 0: continue
            a.plot(g/4.345, M[m].mean(0), color=color(nm), lw=2.2,
                   label=f"{nm.replace(z+' ','').replace(' '+z,'')} ({100*m.mean():.0f}%)")
        a.axhline(0, color="k", lw=0.6, ls=":"); a.axhline(-2, color="gray", lw=0.6, ls="--")
        a.set_ylim(-3.6, 2.2)
        if r == 0: a.set_title(meth, fontsize=12, fontweight="bold")
        if c == 0: a.set_ylabel(f"{z}\n(z-score)", fontweight="bold")
        if r == len(INDICES)-1: a.set_xlabel("Age (months, corrected)")
        a.legend(fontsize=7.5, loc="lower right", framealpha=0.9)
fig.suptitle("First-year growth phenotypes: LCTM vs LSTM/GRU autoencoders",
             fontsize=13, fontweight="bold", y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.985]); fig.savefig(f"{FIGDIR}/fig1_phenotypes.png", bbox_inches="tight"); plt.close(fig)

# ---- Figure 2: AUROC comparison ----
cols = {"Logistic":"#4c72b0","RandomForest":"#55a868","GradBoost":"#c44e52","LSTM":"#8172b3","GRU":"#ccb974"}
fig, axb = plt.subplots(figsize=(9, 4.8)); x = np.arange(len(OUTCOMES)); w = 0.16
for i, mdl in enumerate(["Logistic","RandomForest","GradBoost","LSTM","GRU"]):
    vals = [roc_auc_score(Y[o], pred[o][mdl]) for o in OUTCOMES]
    cis  = [boot_ci(Y[o], pred[o][mdl]) for o in OUTCOMES]
    lo = [vals[k]-cis[k][0] for k in range(len(OUTCOMES))]; hi = [cis[k][1]-vals[k] for k in range(len(OUTCOMES))]
    axb.bar(x+(i-2)*w, vals, w, yerr=[lo, hi], capsize=2, label=mdl, color=cols[mdl], ec="white")
axb.axhline(0.5, color="k", lw=0.7, ls=":"); axb.set_ylim(0.45, 1.0)
axb.set_xticks(x); axb.set_xticklabels([o.capitalize() for o in OUTCOMES]); axb.set_ylabel("AUROC (5-fold OOF, 95% CI)")
axb.set_title("Prediction of 1-year malnutrition from early (<50 wk PMA) growth", fontweight="bold")
axb.legend(ncol=5, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig2_auroc.png", bbox_inches="tight"); plt.close(fig)

# ---- Figure 3: ROC curves ----
fig, axr = plt.subplots(2, 2, figsize=(9, 8))
for k, o in enumerate(OUTCOMES):
    a = axr[k//2, k%2]
    for mdl in ["Logistic", "RandomForest", "LSTM", "GRU"]:
        fpr, tpr, _ = roc_curve(Y[o], pred[o][mdl])
        a.plot(fpr, tpr, color=cols[mdl], lw=1.8, label=f"{mdl} ({roc_auc_score(Y[o], pred[o][mdl]):.2f})")
    a.plot([0, 1], [0, 1], "k:", lw=0.7); a.set_title(f"{o.capitalize()} (n+={int(Y[o].sum())})", fontweight="bold")
    a.set_xlabel("1 - specificity"); a.set_ylabel("Sensitivity"); a.legend(fontsize=8, loc="lower right")
fig.suptitle("ROC curves by outcome and model", fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(f"{FIGDIR}/fig3_roc.png", bbox_inches="tight"); plt.close(fig)

print(f"\nDone. Tables -> {OUTDIR}/  Figures -> {FIGDIR}/")
###############################################################################
# End of script
###############################################################################
