# ================================================================
# 02_score_models: score models (folders and seed are set in 00_config)
#   global:       FedAvg (one shared model)
#   personalized: FedAvg + local fine-tuning (best epoch by local validation BCE)
#   local_lr:     logistic regression trained separately in each hospital
# All models are saved in MODEL_DIR. Federated training exchanges model parameters, not patient records.
# ================================================================
import os, json, pickle, copy, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
assert "BASE_DIR" in globals(), "Lancez d'abord la cellule 00_Config"
SEEDS         = [SEED]            # set in 00_config
ROUNDS        = 100
LOCAL_EPOCHS  = 1
LR            = 1e-3
BATCH         = 64
PARTICIPATION = 0.5
HIDDEN        = [32, 16]
FT_EPOCHS     = 10                # personalization: maximum local fine-tuning epochs
FT_LR         = 1e-3
LOCAL_LR_C    = 1.0               # regularization of the local logistic regression
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------------------------------------------------
# DATA
# ------------------------------------------------------------
with open(os.path.join(CLIENT_DIR, "metadata.json")) as f:
    META = json.load(f)
CL = {}
for cid in range(META["n_clients"]):
    with open(os.path.join(CLIENT_DIR, f"client_{cid:02d}.pkl"), "rb") as f:
        CL[cid] = pickle.load(f)
IDS, INPUT_DIM = list(CL.keys()), META["input_dim"]
print(f"✓ {len(IDS)} hôpitaux | {INPUT_DIM} variables | device={DEVICE}")

def T(a): return torch.tensor(np.asarray(a), dtype=torch.float32)
# tensors are built once (much faster than a DataLoader at every round)
TR = {c: (T(CL[c]["X_train"]).to(DEVICE), T(CL[c]["y_train"]).to(DEVICE)) for c in IDS}

# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        layers, cur = [], d
        for h in hidden:
            layers += [nn.Linear(cur, h), nn.ReLU()]; cur = h
        layers += [nn.Linear(cur, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):                  # returns logits
        return self.net(x).squeeze(-1)

def train_epochs(model, c, epochs, lr):
    """Train `model` on the training set of hospital c (shuffled mini-batches)."""
    X, y = TR[c]
    opt, crit = torch.optim.Adam(model.parameters(), lr=lr), nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(y), device=DEVICE)
        for i in range(0, len(y), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad(); crit(model(X[b]), y[b]).backward(); opt.step()

@torch.no_grad()
def predict(model, X):
    model.eval()
    return torch.sigmoid(model(T(X).to(DEVICE))).cpu().numpy()

def bce(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6); y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

def auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan

def fedavg(states, weights):
    w = np.asarray(weights, float); w = w / w.sum()
    return {k: sum(wi * s[k].float() for wi, s in zip(w, states)) for k in states[0]}

# ------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------
X_val_all = np.vstack([CL[c]["X_val"] for c in IDS])
y_val_all = np.concatenate([CL[c]["y_val"] for c in IDS])

for seed in SEEDS:
    print(f"\n{'=' * 70}\n  SEED {seed}\n{'=' * 70}")
    torch.manual_seed(seed); np.random.seed(seed); rng = np.random.default_rng(seed)

    # --- 1. FedAvg (global model); best round by pooled validation BCE ---
    g = MLP(INPUT_DIM, HIDDEN).to(DEVICE)
    best_bce, best_state, best_round = np.inf, copy.deepcopy(g.state_dict()), 0
    n_part = max(1, int(PARTICIPATION * len(IDS)))
    t0 = time.time()
    for r in range(1, ROUNDS + 1):
        part = rng.choice(IDS, size=n_part, replace=False)
        states, ws = [], []
        for c in part:
            m = copy.deepcopy(g); train_epochs(m, c, LOCAL_EPOCHS, LR)
            states.append(m.state_dict()); ws.append(len(CL[c]["y_train"]))
        g.load_state_dict(fedavg(states, ws))
        vb = bce(predict(g, X_val_all), y_val_all)
        if vb < best_bce:
            best_bce, best_state, best_round = vb, copy.deepcopy(g.state_dict()), r
        if r % 20 == 0 or r == 1:
            print(f"  FedAvg round {r:>3} | BCE val poolée = {vb:.4f} | meilleure = {best_bce:.4f} (round {best_round}) "
                  f"| {time.time() - t0:.0f}s")
    g.load_state_dict(best_state)

    # --- 2. Personalization: local fine-tuning, best epoch by local validation BCE ---
    personal, ft_epochs = {}, {}
    for c in IDS:
        m = copy.deepcopy(g)
        best_b, best_s, best_e = bce(predict(m, CL[c]["X_val"]), CL[c]["y_val"]), copy.deepcopy(m.state_dict()), 0
        for e in range(1, FT_EPOCHS + 1):
            train_epochs(m, c, 1, FT_LR)
            b = bce(predict(m, CL[c]["X_val"]), CL[c]["y_val"])
            if b < best_b:
                best_b, best_s, best_e = b, copy.deepcopy(m.state_dict()), e
        personal[c], ft_epochs[c] = best_s, best_e

    # --- 3. Local logistic regression ---
    local_lr = {c: LogisticRegression(C=LOCAL_LR_C, max_iter=2000).fit(CL[c]["X_train"], CL[c]["y_train"]) for c in IDS}

    # --- 4. Summary (holdout AUROC; descriptive only) ---
    rows = []
    pm = MLP(INPUT_DIM, HIDDEN).to(DEVICE)
    for c in IDS:
        Xh, yh = CL[c]["X_test"], CL[c]["y_test"]
        pm.load_state_dict(personal[c])
        rows.append({"cid": c, "global": auc(yh, predict(g, Xh)), "personalized": auc(yh, predict(pm, Xh)),
                     "local_lr": auc(yh, local_lr[c].predict_proba(Xh)[:, 1]), "ft_epochs": ft_epochs[c]})
    S = pd.DataFrame(rows)
    print(f"\n  AUROC par hôpital sur le holdout (médiane | moyenne) :")
    for k in ["global", "personalized", "local_lr"]:
        print(f"    {k:<13}: {S[k].median():.4f} | {S[k].mean():.4f}")
    print(f"  Époques de fine-tuning retenues : médiane={int(S['ft_epochs'].median())} "
          f"| 0 (= garder le global) pour {(S['ft_epochs'] == 0).mean():.0%} des hôpitaux")

    torch.save({"global": best_state, "personal": personal, "hidden": HIDDEN, "input_dim": INPUT_DIM,
                "best_round": best_round, "ft_epochs": ft_epochs},
               os.path.join(MODEL_DIR, f"models_seed{seed}.pt"))
    with open(os.path.join(MODEL_DIR, f"local_lr_seed{seed}.pkl"), "wb") as f:
        pickle.dump(local_lr, f, protocol=pickle.HIGHEST_PROTOCOL)
    S.to_csv(os.path.join(MODEL_DIR, f"auroc_holdout_seed{seed}.csv"), index=False)
    print(f"  ✓ Modèles sauvegardés dans {MODEL_DIR}")
print("\n✅ Modèles de score prêts.")
