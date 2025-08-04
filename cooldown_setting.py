import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from tqdm import trange, tqdm
import warnings

warnings.filterwarnings("ignore")
sns.set(style="whitegrid", context="talk")

# MPS if available (Apple), otherwise CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)
print("Using device:", device)


# ============================ Parameters ==============================
rnd_seed      = 42
base_arms     = 5     # number of demographic groups
K             = base_arms
epsilon       = 0.1   # grid granularity
T             = 200   # total rounds
runs          = 3     # simulation replications
NN_epoch      = 100   # epochs for NN

# ----------------  multiple resource‐types & budgets -------------------
# four resource types: 1=Scholarship,2=Loan,3=WorkStudy,4=Waiver
budgets       = {1: B1,    2: B2,
                 3: B3,    4: B4}
# per‐type cool‐downs (in rounds)
cooldowns     = {1: cd1,   2: cd2,
                 3: cd3,   4: cd4}

# ------------ For population‐replacement every 8 rounds ----------------
cohort_length = 8               # one “program” = 8 rounds
# n_cohorts     = T // cohort_length #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< n-fold cross-validation <<<<<<<
# # we will do stratified‐kfold with n_cohorts splits

# pick *any* number of cross-val folds I like:
n_splits      = 5                # 5-fold CV
# two different RNG seeds, for splitting vs. assignment
seed_split    = 123
seed_assign   = 456
# compute how many cohorts you'll actually simulate
n_cohorts     = int(np.ceil(T / cohort_length))

# ========================== Data Preparation ==========================
print("== (1) Loading & cleaning data ==")
df = pd.read_csv('/resource_alloc/RA-data.csv')

# rename race columns
df = df.rename(columns={
    'Race_Asian, Hawaii/Pac. Islander': 'Asian',
    'Race_Black or African American':   'Black',
    'Race_Hispanic':                    'Hispanic',
    'Race_More than one race':          'MR',
    'Race_White':                       'White'
})
racial_cols = ['Asian','Black','Hispanic','MR','White'][:base_arms]

# keep only rows with at least one race indicator
df = df[df[racial_cols].sum(axis=1) > 0]

# binary resource column
df['resource_binary'] = (df['resource'] > 0).astype(int)

# drop unwanted columns
df = df.drop(columns=[
    'Unnamed: 0',
    'gender_Male',
    'Race_Amer. Indian/Alaska Native',
    'F3_Highest level of education',
    'resource'
]).dropna()

print(f"Cleaned dataset shape: {df.shape}")

# subgroup sizes
group_sizes = df[racial_cols].sum(axis=0).astype(int).tolist()
group_sizes_tensor = torch.tensor(group_sizes, dtype=torch.float32, device=device)

# features and target
feature_cols = [c for c in df.columns if c not in ['F3_GPA(all)']+racial_cols]
X = df[feature_cols].copy()
y = df['F3_GPA(all)'].values
race = df[racial_cols].values

# standardize numeric features
num_cols = X.select_dtypes(include='number').columns
scaler   = StandardScaler().fit(X[num_cols])
X[num_cols] = scaler.transform(X[num_cols])

# # train/test split (stratify by race)<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< old train-test spilit <<<<<<<<
# X_tr, X_te, y_tr, y_te, r_tr, r_te = train_test_split(
#     X.values, y, race,
#     test_size=0.2, random_state=rnd_seed,
#     stratify=race.argmax(axis=1)
# )

#---------------------------------Cross Validation-----------------------------

# 0) build the stratification labels
strata = df[racial_cols].values.argmax(axis=1)

# 1) create exactly n_splits CV folds
skf   = StratifiedKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=seed_split
)
folds = list(skf.split(df, strata))   # each entry is (train_idx, test_idx)

# 2) now assign one test_idx to each of my n_cohorts cohorts:
test_idx_per_cohort = []
for c in range(n_cohorts):
    if c < n_splits:
        # use each fold once, in order
        _, test_idx = folds[c]
    else:
        # for the leftover cohorts, resample a fold at random
        rng = np.random.RandomState(seed_assign + c)
        pick = rng.randint(0, n_splits)
        _, test_idx = folds[pick]
    test_idx_per_cohort.append(test_idx)

#-------------------------------------------------------------------------------

# combine for NN training
X_all    = torch.tensor(np.vstack([X_tr, X_te]), dtype=torch.float32).to(device)
race_all = torch.tensor(np.vstack([r_tr, r_te]), dtype=torch.float32).to(device)
race_all_np = race_all.cpu().numpy()   # shape (N, K)
race_idx    = race_all_np.argmax(axis=1)  # each student’s group 0..K-1
y_all    = np.hstack([y_tr, y_te])
print(f"Combined samples: {X_all.shape[0]}")

# ========================== Neural-Net for μ_k,σ²_k ==========================
print("== (2) Constructing Neural Net ==")
class SharedBackbone(nn.Module):
    def __init__(self, in_dim, hid=128, drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hid, hid),    nn.ReLU(), nn.Dropout(drop)
        )
    def forward(self, x):
        return self.net(x)

class MeanPoolingGaussianLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, emb, race):
        grouped    = emb.unsqueeze(1) * race.unsqueeze(-1)  # [N,K,H]
        pooled_sum = grouped.sum(dim=0)                    # [K,H]
        counts     = race.sum(dim=0, keepdim=True).T       # [K,1]
        pooled     = pooled_sum / (counts + 1e-6)          # [K,H]
        out        = self.mlp(pooled)                      # [K,2]
        mu, lv     = out[:,0], out[:,1]
        return mu, torch.exp(lv)                           # each [K]

class NeuralSubgroupModel(nn.Module):
    def __init__(self, in_dim, hid=128, drop=0.2):
        super().__init__()
        # instantiate with the same hid everywhere
        self.backbone = SharedBackbone(in_dim, hid=hid, drop=drop)
        self.gaussian = MeanPoolingGaussianLayer(hid)
    def forward(self, x, race):
        emb = self.backbone(x)
        return self.gaussian(emb, race)


def train_model(model, X, race, y, epochs=NN_epoch):
    optimizer     = optim.Adam(model.parameters(), lr=1e-3)
    y_t           = torch.tensor(y, dtype=torch.float32).view(-1,1).to(device)
    global_losses = []
    subgroup_mse  = {g: [] for g in racial_cols}
    for _ in trange(epochs, desc="Gaussian-NN Training…", delay=0.5, mininterval=0.5):
        model.train()
        optimizer.zero_grad()
        mu, var = model(X, race)                             # [K],[K]
        mu_i  = (race * mu.unsqueeze(0)).sum(1,keepdim=True)  # [N,1]
        var_i = (race * var.unsqueeze(0)).sum(1,keepdim=True) # [N,1]
        loss  = 0.5*((torch.log(var_i+1e-6) + (y_t-mu_i)**2/(var_i+1e-6))).mean()
        loss.backward(); optimizer.step()
        global_losses.append(loss.item())
        y_pred = mu_i.detach().cpu().numpy().flatten()
        y_true = y_t.cpu().numpy().flatten()
        for k,grp in enumerate(racial_cols):
            mask = (race[:,k]==1).cpu().numpy()
            subgroup_mse[grp].append(
                mean_squared_error(y_true[mask], y_pred[mask]) if mask.sum()>0 else np.nan
            )
    return model, global_losses, subgroup_mse

# instantiate and train neural model
neural_model = NeuralSubgroupModel(in_dim=X_all.shape[1], hid=128, drop=0.2).to(device)
neural_model, train_losses, subgroup_mse = train_model(
    neural_model, X_all, race_all, y_all
)

# ========================== Linear mapping model ==========================
print("== (3) Constructing Linear Model ==")
class LinearSubgroupModel:
    def __init__(self, lambda_reg=1.0):
        self.lambda_reg = lambda_reg
    def fit(self, X_np, race_np, y_np):
        K, D = race_np.shape[1], X_np.shape[1]
        self.mu_lin  = np.zeros(K)
        self.var_lin = np.zeros(K)
        for k in range(K):
            idx = np.where(race_np[:,k]==1)[0]
            Xk, yk = X_np[idx], y_np[idx]
            ridge = Ridge(alpha=self.lambda_reg, fit_intercept=True)
            ridge.fit(Xk, yk)
            y_pred      = ridge.predict(Xk)
            self.mu_lin[k]  = y_pred.mean()
            self.var_lin[k] = np.var(yk - y_pred, ddof=1)
    def forward(self, X_tensor, race_tensor):
        mu_t  = torch.tensor(self.mu_lin, dtype=torch.float32, device=device)
        var_t = torch.tensor(self.var_lin, dtype=torch.float32, device=device)
        return mu_t, var_t

# fit linear model on training split
linear_model = LinearSubgroupModel(lambda_reg=1.0)
linear_model.fit(X_tr, r_tr, y_tr)

# ========================== Reward-function generator ==========================
print("== (4) Constructing Reward Functions ==")
def make_reward_funcs(mu_arr, var_arr):
    funcs = []
    for m, v in zip(mu_arr, var_arr):
        s     = np.sqrt(v)
        denom = float(norm.pdf(m, loc=m, scale=s))
        funcs.append(lambda p, m=m, s=s, d=denom:
                     float(norm.pdf(p, loc=m, scale=s)/d))
    return funcs

# extract subgroup params
neural_model.eval()
with torch.no_grad():
    mu_nn_t, var_nn_t = neural_model(X_all, race_all)
mu_nn_k  = mu_nn_t.cpu().numpy()
var_nn_k = var_nn_t.cpu().numpy()

mu_lin_t, var_lin_t = linear_model.forward(X_all, race_all)
mu_lin_k  = mu_lin_t.cpu().numpy()
var_lin_k = var_lin_t.cpu().numpy()

r_funcs_nn  = make_reward_funcs(mu_nn_k, var_nn_k)
r_funcs_lin = make_reward_funcs(mu_lin_k, var_lin_k)

# ========================== (Evaluate & Compare) ==========================
X_te_t    = torch.tensor(X_te, dtype=torch.float32).to(device)
race_te_t = torch.tensor(r_te, dtype=torch.float32).to(device)


print("== (5) Evaluating Neural Net vs. Linear Model ==")
def evaluate_model(mu_k_arr, race_tensor, y_true):
    race_np = race_tensor.cpu().numpy()
    y_pred  = (race_np * mu_k_arr).sum(axis=1)
    rmse    = np.sqrt(mean_squared_error(y_true, y_pred))
    grp_rmse = {}
    for k,grp in enumerate(racial_cols):
        mask = (race_np[:,k]==1)
        grp_rmse[grp] = (np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
                         if mask.sum()>0 else np.nan)
    return y_pred, rmse, grp_rmse

y_pred_nn, rmse_nn, grp_nn = evaluate_model(mu_nn_k, race_te_t, y_te)
y_pred_ln, rmse_ln, grp_ln = evaluate_model(mu_lin_k, race_te_t, y_te)
res_nn = y_te - y_pred_nn   # neural residuals
res_ln = y_te - y_pred_ln   # linear residuals


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics plot
# ─────────────────────────────────────────────────────────────────────────────
from scipy.stats import gaussian_kde
palette = sns.color_palette("tab10", n_colors=K)
inds    = np.arange(K)
width   = 0.3

# (a) Per‐group RMSE bar chart
fig, ax = plt.subplots(figsize=(6,5))
vals_nn = [grp_nn[g] for g in racial_cols]
vals_ln = [grp_ln[g]     for g in racial_cols]
ax.bar(inds - width/2, vals_nn, width, label="Neural", color=palette[:K])
ax.bar(inds + width/2, vals_ln, width, label="Linear", color=palette[:K], alpha=0.6)
ax.set_xticks(inds)
ax.set_xticklabels(racial_cols, rotation=30)
ax.set_ylabel("Test RMSE")
ax.set_title("Per-Group Test RMSE")
ax.legend(frameon=False)
plt.tight_layout()
plt.show()



for i, grp in enumerate(racial_cols):
    ax = axes[i]
    ys_nn, ys_ln = all_ys[i]
    ax.plot(xs, ys_nn, '-',  color=palette[i], label="Neural (mod)")
    ax.plot(xs, ys_ln, '--', color=palette[i], label="Linear")
    ax.set_title(f"{grp} Residuals")
    ax.set_xlabel("y_true - y_pred")
    if i == 0:
        ax.set_ylabel("Density")
    ax.set_xlim(x_min, x_max)
for ax in axes:
    ax.axvline(+rmse_nn, color='blue', linestyle=':', label='±Neural RMSE')
    ax.axvline(-rmse_nn, color='blue', linestyle=':')
    ax.axvline(+rmse_ln, color='orange', linestyle='-.', label='±Linear RMSE')
    ax.axvline(-rmse_ln, color='orange', linestyle='-.')
    ax.legend(frameon=False)

plt.suptitle("Residuals by Subgroup: Neural vs Linear", y=1.02)
plt.tight_layout()
plt.show()


# =====================ACTION DEPENDENT SETUP===================================
print("== (6) Constructing Action Dependent Environment ==")

def sample_random_meta_arms(n):
    """Uniform on simplex, filtered by budget."""
    Ps = []
    while len(Ps) < n:
        p = np.random.dirichlet(np.ones(K))
        if (p * group_sizes).sum() <= B:
            Ps.append(p)
    return np.stack(Ps)  # (n, K)

def evaluate_meta_arm(p, r_funcs):
    """Scalar utility U(p) = Σ_k p[k] · r_funcs[k](p[k])."""
    return sum(p[k] * r_funcs[k](p[k]) for k in range(K))

class IndUCB:
    def __init__(self, N, alpha=2.0):
        self.N      = N
        self.counts = np.zeros(N)
        self.means  = np.zeros(N)
        self.alpha  = alpha
    def select(self, t):
        bonus = self.alpha * np.sqrt(2*np.log(t)/(self.counts+1e-9))
        return int(np.argmax(self.means + bonus))
    def update(self, i, r):
        self.counts[i] += 1
        n = self.counts[i]
        self.means[i] += (r - self.means[i]) / n


class IndCUCB(IndUCB):
    def select(self, t):
        bonus = self.alpha * np.sqrt(3*np.log(t) / (self.counts + 1e-9))
        return int(np.argmax(self.means + bonus))


class IndEXP3:
    def __init__(self, N, eta=0.1):
        self.N       = N
        self.weights = np.ones(N)
        self.eta     = eta
    def select(self):
        w     = np.clip(self.weights, 1e-6, 1e6)
        probs = w / w.sum()
        i     = np.random.choice(self.N, p=probs)
        return i, probs
    def update(self, i, r, probs):
        x = r / (probs[i] + 1e-12)
        self.weights[i] *= np.exp(self.eta * x)



class IndmEXP3(IndEXP3):
    def __init__(self, N):
        super().__init__(N, eta=0.2)


class IndLinUCB:
    def __init__(self, X_raw, alpha=1.0):
        self.H      = X_raw
        self.alpha  = alpha
        self.N, self.D = X_raw.shape
        self.A      = np.eye(self.D)
        self.b      = np.zeros(self.D)
    def select(self, t):
        A_inv = np.linalg.inv(self.A)
        theta = A_inv.dot(self.b)
        means = self.H.dot(theta)
        bonus = self.alpha * np.sqrt(np.sum(self.H.dot(A_inv)*self.H, axis=1))
        return int(np.argmax(means + bonus))
    def update(self, i, r):
        x = self.H[i]
        self.A += np.outer(x, x)
        self.b += r * x




# 3) Meta‐level BO: UCBO
def run_ucbo(r_funcs, mapping_label,
             n_init=n_init, cand_size=cand_size, budget=bo_budget):
    print(f"→ UCBO with {mapping_label} reward mapping: initializing…")
    # 1) INITIAL pool
    Ps = sample_random_meta_arms(n_init)    # (n_init, K)
    Us = np.array([evaluate_meta_arm(p, r_funcs) for p in Ps])
    best_idx = int(Us.argmax())
    best_p   = Ps[best_idx].copy()
    best_U   = Us[best_idx]
    hist     = list(Us)
    print(f"   Initial {n_init} arms evaluated")

    # 2) REMAINING rounds
    rem = budget - n_init
    for i in range(rem):
        # (a) generate local candidates around best_p
        Cs = []
        while len(Cs) < cand_size:
            delta = np.random.uniform(-epsilon, epsilon, size=K)
            p_new = best_p + delta
            p_new = np.clip(p_new, 0, 1)
            if p_new.sum() > 0:
                p_new = p_new / p_new.sum()
                if (p_new * group_sizes).sum() <= B:
                    Cs.append(p_new)
        Cs = np.stack(Cs)  # (cand_size, K)

        # (b) evaluate
        Us_c = np.array([evaluate_meta_arm(p, r_funcs) for p in Cs])
        acq  = Us_c  # here simple UCB‐BO with no extra bonus

        # (c) pick best
        idx_next = int(acq.argmax())
        p_next   = Cs[idx_next]
        U_next   = Us_c[idx_next]

        # (d) update history & best
        hist.append(U_next)
        if U_next > best_U:
            best_U, best_p = U_next, p_next.copy()

        if (i+1) % (rem//5 or 1) == 0:
            print(f"   UCBO {mapping_label}: round {i+1}/{rem}")

    print(f"✓ UCBO with {mapping_label} done")
    return np.array(hist[:budget])

    
class IndividualLinUCBBase(IndLinUCB):
    def __init__(self, X_raw, alpha=1.0):
        super().__init__(X_raw, alpha=alpha)

    def select_from(self, idxs):
        feats = self.H[idxs]
        A_inv = np.linalg.inv(self.A)
        theta = A_inv.dot(self.b)
        bonus = self.alpha * np.sqrt(np.sum(feats.dot(A_inv)*feats, axis=1))
        scores = feats.dot(theta) + bonus
        return idxs[int(np.argmax(scores))]

    # new signature: accept i (index), r (reward), x (feature vector)
    def update(self, i, r, x):
        # same as before, but using the passed-in x
        self.A += np.outer(x, x)
        self.b += r * x



# 5) LinUCBO: meta‐level UCBO + individual LinUCB base‐level
def run_linucbo(r_funcs, mapping_label,
                contexts, race_np,
                n_init=n_init, cand_size=cand_size, budget=bo_budget,
                runs=3):
    """
    Meta‐level UCBO + base‐level LinUCB.
    Respects the initial n_init draws as part of the BO budget.
    """
    print(f"→ LinUCBO with {mapping_label} mapping: initializing…")

    # ─────────────────────────────────────────────────────────────────────────────
    # (a) Meta‐level INITIAL DESIGN
    # ─────────────────────────────────────────────────────────────────────────────
    Ps = sample_random_meta_arms(n_init)   # (n_init, K)
    Us = np.array([evaluate_meta_arm(p, r_funcs) for p in Ps])  # (n_init,)
    best_idx    = int(Us.argmax())
    best_p      = Ps[best_idx].copy()
    best_U      = Us[best_idx]
    hist        = list(Us)                 # keep history of utilities

    # Build full grid once to compute static best
    grid   = torch.cartesian_prod(*[
        torch.arange(epsilon, 1+1e-9, epsilon, device=group_sizes_tensor.device)
        for _ in range(K)
    ]).float()
    usage  = (grid * group_sizes_tensor).sum(dim=1)
    B_t    = torch.tensor(B, dtype=usage.dtype, device=usage.device)
    mask   = torch.isclose(usage, B_t, atol=1e-6)
    meta_np = grid[mask].cpu().numpy()
    static_best_U = max(evaluate_meta_arm(p, r_funcs) for p in meta_np)

    rem = budget - n_init
    results = np.zeros((runs, budget))

    for run_i in range(runs):
        print(f"  run {run_i+1}/{runs}")
        # reset per‐run best
        best_p_run, best_U_run = best_p.copy(), best_U
        # base‐level LinUCB learner with scalar alpha
        indiv_alg = IndividualLinUCBBase(contexts, alpha=2.0)
        cum = 0.0

        # (b) Account for initial design points in regret
        for t0 in range(n_init):
            cum += (static_best_U - Us[t0])
            results[run_i, t0] = cum

        # (c) Remaining BO + allocation rounds
        for t_rel in range(rem):
            t = n_init + t_rel

            # meta‐level local search around best_p_run
            # (we already have the first n_init draws)
            Cs, Us_c = [], []
            while len(Cs) < cand_size:
                delta = np.random.uniform(-epsilon, epsilon, size=K)
                p_new = best_p_run + delta
                p_new = np.clip(p_new, 0, 1)
                if p_new.sum() > 0:
                    p_new = p_new / p_new.sum()
                    if (p_new * group_sizes).sum() <= B:
                        Cs.append(p_new)
            Cs = np.stack(Cs)
            Us_c = np.array([evaluate_meta_arm(p, r_funcs) for p in Cs])
            idx_next = int(np.argmax(Us_c))
            p_t      = Cs[idx_next]
            U_t      = Us_c[idx_next]
            hist.append(U_t)
            if U_t > best_U_run:
                best_U_run, best_p_run = U_t, p_t.copy()

            # base‐level: allocate p_t fraction with LinUCB
            reward_sum = 0.0
            for k in range(K):
                idxs_k = np.where(race_np[:,k] == 1)[0]
                m_k    = int(round(p_t[k] * len(idxs_k)))
                for _ in range(m_k):
                    j   = indiv_alg.select_from(idxs_k)
                    x_j = contexts[j]
                    r_j = r_funcs[k](p_t[k])
                    indiv_alg.update(j, r_j, x_j)
                    reward_sum += r_j

            # cumulative regret
            cum += (static_best_U - reward_sum)
            results[run_i, t] = cum

            if (t+1) % max(1, (budget//5)) == 0:
                print(f" Run t={t+1}/{budget} executed")

        print(f"  ✓ LinUCBO {mapping_label} run {run_i+1} done\n")

    return results



# 5) Master runner combining individual + meta baselines
def run_action_dependent_all(T=200, runs=3):
    # Prepare student-level data
    X_all_np    = X_all.cpu().numpy()       # (N, D)
    race_all_np = race_all.cpu().numpy()    # (N, K)
    N = X_all_np.shape[0]

    results = {}
    for mapping_label, r_funcs in [
        ("Linear",    r_funcs_lin),
        ("Nonlinear", r_funcs_nn),
    ]:
        print(f"\n--- {mapping_label} reward mapping ---")
        # Compute per-student rewards and best possible
        mu_arr = mu_lin_k if mapping_label=="Linear" else mu_nn_k
        mu_i   = race_all_np.dot(mu_arr)       # (N,)
        best_i = mu_i.max()

        # 1A) student-level baselines
        for name, alg in [
            ("UCB",    IndUCB),
            ("CUCB",   IndCUCB),
            ("EXP3",   IndEXP3),
            ("mEXP3",  IndmEXP3),
            ("LinUCB", IndLinUCB),
        ]:
            key = f"{mapping_label}_{name}"
            results[key] = np.zeros((runs, T))
            print(f"→ Running {name}")

            for run_i in range(runs):
                # instantiate correctly
                if name=="UCB":
                    alg = IndUCB(N)
                elif name=="CUCB":
                    alg = IndCUCB(N)
                elif name=="EXP3":
                    alg = IndEXP3(N)
                elif name=="mEXP3":
                    alg = IndmEXP3(N)
                else:  # LinUCB
                    alg = IndLinUCB(X_all_np)

                cum = 0.0

                #setting up my budget constraints
                if isinstance(B, (int, float)):
                    resource_budget = {"global": int(B)}
                    budget_mode = "global"
                else:
                    resource_budget = dict(B)  # {'Scholarship': 10, ...}
                    budget_mode = "multi"

                used_students = set()  # individuals who have already received a resource this round

                for t in range(1, T+1):
                    # ========== Budget Checks ==========
                    if budget_mode == "global":
                        if resource_budget["global"] <= 0:
                            print(f"Global budget exhausted at round {t}")
                            break
                        valid_idxs = list(set(range(N)) - used_students)

                    elif budget_mode == "multi":
                        valid_idxs = []
                        for i in range(N):
                            group_k = race_all_np[i].argmax()
                            group_label = racial_cols[group_k]
                            if resource_budget.get(group_label, 0) > 0:
                                valid_idxs.append(i)
                        valid_idxs = list(set(valid_idxs) - used_students)

                        if not valid_idxs:
                            print(f"All group-specific budgets exhausted or no valid individuals at round {t}")
                            break
                    if name in ("EXP3","mEXP3"):
                        i, probs = alg.select()
                        if i not in valid_idxs:
                            continue  # skip if individual already assigned or budget unavailable
                        r = mu_i[i]
                        alg.update(i, r, probs)
                    else:
                        i = alg.select(t)
                        if i not in valid_idxs:
                            continue
                        r = mu_i[i]
                        alg.update(i, r)
                        # ========== Budget Accounting ==========
                    if budget_mode == "global":
                        resource_budget["global"] -= 1
                    else:
                        group_k = race_all_np[i].argmax()
                        group_label = racial_cols[group_k]
                        resource_budget[group_label] -= 1

                    # ========== Assignment Constraint ==========
                    used_students.add(i)  # record that this student received a resource

                    cum += (best_i - r)
                    results[key][run_i, t-1] = cum

                print(f"  ✓ {name} run {run_i+1}/{runs}")

        # 1B) meta-level baselines
        print(f"--- Meta baselines (UCBO & LinUCBO) ---")
        ucbo_hist = run_ucbo(r_funcs, mapping_label,
                             n_init=n_init, cand_size=cand_size, budget=T)
        linucbo_res = run_linucbo(r_funcs, mapping_label,
                                  contexts=X_all_np,
                                  race_np=race_all_np,
                                  n_init=n_init,
                                  cand_size=cand_size,
                                  budget=T,
                                  runs=runs)

        results[f"{mapping_label}_UCBO"]    = np.tile(ucbo_hist,    (runs,1))
        results[f"{mapping_label}_LinUCBO"] = linucbo_res

    return results


# ==== 2) Execute ====
all_results = run_action_dependent_all(T=T, runs=3)
