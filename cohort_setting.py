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
T             = 8000   # total rounds
runs          = 5     # simulation replications
NN_epoch      = 100   # epochs for NN

#-------------------------- BO parameters -------------------------
n_init        = 3
cand_size     = 30
bo_budget     = 8 
beta          = 2.0

# ----------------  multiple resource‐types & budgets -------------------

# four resource types: 1=Scholarship,2=Loan,3=WorkStudy,4=Waiver
budgets       = {1: 388,    2: 346,   3: 281,    4: 28}
# per‐type cool‐downs (in rounds)
cooldowns     = {1: 1,   2: 1,   3: 1,   4: 1}

# ------------ For population‐replacement every 8 rounds ----------------
cohort_length = 8               # one “program” = 8 rounds
# n_cohorts     = T // cohort_length 
# # we will do stratified‐kfold with n_cohorts splits

# pick *any* number of cross-val folds I like:
n_splits      = 8                # 8-fold CV
# two different RNG seeds, for splitting vs. assignment
seed_split    = 123
seed_assign   = 456
# compute how many cohorts you'll actually simulate
n_cohorts     = int(np.ceil(T / cohort_length))

# ========================== Data Preparation ==========================
print("== (1) Loading & cleaning data ==")
df = pd.read_csv()

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

#---------------------------------Cross Validation-----------------------------

# build full data arrays for cohort‐based training
X_np      = X.values                  # (N, D)
race_np   = race                      # (N, K)
y_np      = y                         # (N,)
N         = X_np.shape[0]
race_idx  = race_np.argmax(axis=1)    # each student’s group 0..K-1

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

# — Cohort‐based model retraining per 8‐round program —
mu_nn_k_all  = np.zeros((n_cohorts, K))
var_nn_k_all = np.zeros((n_cohorts, K))
mu_lin_k_all = np.zeros((n_cohorts, K))
var_lin_k_all = np.zeros((n_cohorts, K))

for c in range(n_cohorts):
    # “test_idx” is the cᵗʰ cohort: we’ll allocate resources to these students
    test_idx  = test_idx_per_cohort[c]
    # train on *all the other* students
    train_idx = np.setdiff1d(np.arange(N), test_idx_per_cohort[c])

    # — train my Neural‐Gaussian on that training set —
    # Prepare training tensors for neural net
    X_train_t   = torch.tensor(X_np[train_idx], dtype=torch.float32).to(device)
    race_train_t= torch.tensor(race_np[train_idx], dtype=torch.float32).to(device)
    y_train     = y_np[train_idx]
    # 1) Neural‐Gaussian model
    model = NeuralSubgroupModel(in_dim=X_np.shape[1], hid=128, drop=0.2).to(device)
    model, _, _ = train_model(model, X_train_t, race_train_t, y_train, epochs=NN_epoch)
    model.eval()
    with torch.no_grad():
        mu_t, var_t = model(torch.tensor(X_np, dtype=torch.float32).to(device),
                            torch.tensor(race_np, dtype=torch.float32).to(device))
    mu_nn_k_all[c]  = mu_t.cpu().numpy()
    var_nn_k_all[c] = var_t.cpu().numpy()

    #  — train your Linear‐Ridge on that same train_idx —
    lin_model = LinearSubgroupModel(lambda_reg=1.0)
    lin_model.fit(X_np[train_idx], race_np[train_idx], y_np[train_idx])
    mu_lin_k_all[c]  = lin_model.mu_lin
    var_lin_k_all[c] = lin_model.var_lin

# now r_funcs must be chosen per cohort when running experiments

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

# downstream, I select the cohort's mu/var when running experiments


# =====================ACTION DEPENDENT SETUP===================================
print("== (6) Constructing Action Dependent Environment ==")

def sample_random_meta_arms(n, budgets, group_sizes):
    """
    Uniform on simplex for each resource, filtered by per-resource budgets.
    Returns (n, R, K) array, R = len(budgets)
    """
    Rs = list(budgets.keys())
    R = len(Rs)
    Ps = []
    while len(Ps) < n:
        P = np.stack([np.random.dirichlet(np.ones(K)) for _ in Rs], axis=0)  # (R, K)
        if all((P[r_idx] * group_sizes).sum() <= budgets[r] for r_idx, r in enumerate(Rs)):
            Ps.append(P)
    return np.array(Ps)  # (n, R, K)

def evaluate_meta_arm(P, r_funcs):
    """
    Utility U(P) = sum_{r,k} P[r,k] * r_funcs[r][k](P[r,k])
    """
    Rs = list(r_funcs.keys())
    return sum(
        P[r_idx, k] * r_funcs[r][k](P[r_idx, k])
        for r_idx, r in enumerate(Rs) for k in range(P.shape[1])
    )

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
             n_init=n_init, cand_size=cand_size, budget=bo_budget, budgets=None):
    """
    UCBO for multiple resource types. 
    r_funcs: dict mapping resource type to list of K reward functions
    budgets: dict mapping resource type to budget
    """
    print(f"→ UCBO with {mapping_label} reward mapping: initializing…")
    Rs = list(budgets.keys())
    R = len(Rs)
    # 1) INITIAL pool
    Ps = sample_random_meta_arms(n_init, budgets, group_sizes)    # (n_init, R, K)
    Us = np.array([
        sum(P[r_idx, k] * r_funcs[r][k](P[r_idx, k])
            for r_idx, r in enumerate(Rs) for k in range(K))
        for P in Ps
    ])
    best_idx = int(Us.argmax())
    best_P   = Ps[best_idx].copy()
    best_U   = Us[best_idx]
    hist     = list(Us)
    print(f"   Initial {n_init} arms evaluated")

    # 2) REMAINING rounds
    rem = budget - n_init
    for i in range(rem):
        # (a) generate local candidates around best_P
        Cs = []
        while len(Cs) < cand_size:
            P_new = best_P + np.random.uniform(-epsilon, epsilon, size=(R, K))
            P_new = np.clip(P_new, 0, 1)
            # renormalize each resource row to sum 1
            P_new /= P_new.sum(axis=1, keepdims=True)
            if all((P_new[r_idx] * group_sizes).sum() <= budgets[r] for r_idx, r in enumerate(Rs)):
                Cs.append(P_new)
        Cs = np.stack(Cs)  # (cand_size, R, K)
        # (b) evaluate
        Us_c = np.array([
            sum(P[r_idx, k] * r_funcs[r][k](P[r_idx, k])
                for r_idx, r in enumerate(Rs) for k in range(K))
            for P in Cs
        ])
        acq  = Us_c  # here simple UCB‐BO with no extra bonus
        # (c) pick best
        idx_next = int(acq.argmax())
        P_next   = Cs[idx_next]
        U_next   = Us_c[idx_next]
        # (d) update history & best
        hist.append(U_next)
        if U_next > best_U:
            best_U, best_P = U_next, P_next.copy()
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
                runs=runs, budgets=None):
    """
    Meta‐level UCBO + base‐level LinUCB for multiple resource types.
    r_funcs: dict mapping resource type to list of K reward functions
    budgets: dict mapping resource type to budget
    """
    print(f"→ LinUCBO with {mapping_label} mapping: initializing…")
    Rs = list(budgets.keys())
    R = len(Rs)
    # (a) Meta‐level INITIAL DESIGN
    Ps = sample_random_meta_arms(n_init, budgets, group_sizes)   # (n_init, R, K)
    Us = np.array([
        sum(P[r_idx, k] * r_funcs[r][k](P[r_idx, k])
            for r_idx, r in enumerate(Rs) for k in range(K))
        for P in Ps
    ])
    best_idx    = int(Us.argmax())
    best_P      = Ps[best_idx].copy()
    best_U      = Us[best_idx]
    hist        = list(Us)
    # For static best, brute-force grid is not practical for R > 1, so skip or approximate
    static_best_U = np.max(Us)
    rem = budget - n_init
    results = np.zeros((runs, budget))
    for run_i in range(runs):
        print(f"  run {run_i+1}/{runs}")
        best_P_run, best_U_run = best_P.copy(), best_U
        indiv_alg = IndividualLinUCBBase(contexts, alpha=2.0)
        cum = 0.0
        for t0 in range(n_init):
            cum += (static_best_U - Us[t0])
            results[run_i, t0] = cum
        for t_rel in range(rem):
            t = n_init + t_rel
            # meta‐level local search around best_P_run
            Cs = []
            while len(Cs) < cand_size:
                P_new = best_P_run + np.random.uniform(-epsilon, epsilon, size=(R, K))
                P_new = np.clip(P_new, 0, 1)
                P_new /= P_new.sum(axis=1, keepdims=True)
                if all((P_new[r_idx] * group_sizes).sum() <= budgets[r] for r_idx, r in enumerate(Rs)):
                    Cs.append(P_new)
            Cs = np.stack(Cs)
            Us_c = np.array([
                sum(P[r_idx, k] * r_funcs[r][k](P[r_idx, k])
                    for r_idx, r in enumerate(Rs) for k in range(K))
                for P in Cs
            ])
            idx_next = int(np.argmax(Us_c))
            P_t      = Cs[idx_next]
            U_t      = Us_c[idx_next]
            hist.append(U_t)
            if U_t > best_U_run:
                best_U_run, best_P_run = U_t, P_t.copy()
            # base‐level: allocate P_t[r,k] fraction with LinUCB
            reward_sum = 0.0
            for r_idx, r in enumerate(Rs):
                for k in range(K):
                    idxs_k = np.where(race_np[:,k] == 1)[0]
                    m_k    = int(round(P_t[r_idx, k] * len(idxs_k)))
                    for _ in range(m_k):
                        j   = indiv_alg.select_from(idxs_k)
                        x_j = contexts[j]
                        r_j = r_funcs[r][k](P_t[r_idx, k])
                        indiv_alg.update(j, r_j, x_j)
                        reward_sum += r_j
            cum += (static_best_U - reward_sum)
            results[run_i, t] = cum
            if (t+1) % max(1, (budget//5)) == 0:
                print(f" Run t={t+1}/{budget} executed")
        print(f"  ✓ LinUCBO {mapping_label} run {run_i+1} done\n")
    return results



# 5) Master runner combining individual + meta baselines
def run_action_dependent_all(T=T, runs=runs, cohort_idx=0):
    # --------------------------------------------------------------------------------
    # NOTE ON BUDGET CONSTRAINT:
    # Previously we assumed a single resource type with a universal budget B.
    # In that setting, BO methods found per‑group sub‑budgets, then base‑level UCB
    # or LinUCB allocated up to that sub‑budget per group. Now we have four
    # resource types (1: Scholarship, 2: Loan, 3: WorkStudy, 4: Waiver) each with
    # its own budget. For non‑BO baselines we simply consume from the
    # `budgets` dict passed in parameters. For BO baselines, the meta‑level BO
    # loop will output an optimal sub‑budget per group per resource type, and
    # the base level must then allocate each resource type within its
    # group‑specific BO sub‑budget.
    # --------------------------------------------------------------------------------
    # Prepare student-level data
    # only consider current cohort for allocation
    cohort_idxs  = test_idx_per_cohort[cohort_idx]
    X_all_np     = X_np[cohort_idxs]       # (n_cohort, D)
    race_all_np  = race_np[cohort_idxs]    # (n_cohort, K)
    N            = len(cohort_idxs)

    # Select cohort-specific μ,σ² for this run
    mu_nn_k = mu_nn_k_all[cohort_idx]
    var_nn_k = var_nn_k_all[cohort_idx]
    mu_lin_k = mu_lin_k_all[cohort_idx]
    var_lin_k = var_lin_k_all[cohort_idx]
    r_funcs_nn  = make_reward_funcs(mu_nn_k, var_nn_k)
    r_funcs_lin = make_reward_funcs(mu_lin_k, var_lin_k)

    results = {}
    for mapping_label, r_funcs, mu_arr in [
        ("Linear",    r_funcs_lin, mu_lin_k),
        ("Nonlinear", r_funcs_nn, mu_nn_k),
    ]:
        print(f"\n--- {mapping_label} reward mapping ---")
        # Compute per-student rewards and best possible
        mu_i   = race_all_np.dot(mu_arr)       # (n_cohort,)
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
            results[key] = np.zeros((runs, cohort_length))
            print(f"→ Running {name}")

            for run_i in range(runs):
                # select only the current cohort of students for allocation
                test_idx = test_idx_per_cohort[cohort_idx]
                # track most recent round each student got each resource type
                last_allocated = {}
                # instantiate correctly
                if name=="UCB":
                    alg_inst = IndUCB(N)
                elif name=="CUCB":
                    alg_inst = IndCUCB(N)
                elif name=="EXP3":
                    alg_inst = IndEXP3(N)
                elif name=="mEXP3":
                    alg_inst = IndmEXP3(N)
                else:  # LinUCB
                    alg_inst = IndLinUCB(X_all_np)

                cum = 0.0

                #setting up my budget constraints
                if isinstance(B, (int, float)):
                    resource_budget = {"global": int(B)}
                    budget_mode = "global"
                else:
                    resource_budget = dict(B)  # {'Scholarship': 10, ...}
                    budget_mode = "multi"

                used_students = set()  # individuals who have already received a resource this round

                # only simulate `cohort_length` rounds for this cohort
                for t_rel in range(1, cohort_length+1):
                    t = t_rel
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
                    # (removed restriction to students in current cohort; already sliced)
                    if name in ("EXP3","mEXP3"):
                        i, probs = alg_inst.select()
                        if i not in valid_idxs:
                            continue  # skip if individual already assigned or budget unavailable
                        r = mu_i[i]
                        alg_inst.update(i, r, probs)
                        resource_type = 1  # TODO: assign actual resource type if needed
                    else:
                        i = alg_inst.select(t)
                        if i not in valid_idxs:
                            continue
                        r = mu_i[i]
                        alg_inst.update(i, r)
                        resource_type = 1  # TODO: assign actual resource type if needed
                    # ========== Budget Accounting ==========
                    if budget_mode == "global":
                        resource_budget["global"] -= 1
                    else:
                        group_k = race_all_np[i].argmax()
                        group_label = racial_cols[group_k]
                        resource_budget[group_label] -= 1

                    # ========== Assignment Constraint ==========
                    used_students.add(i)  # record that this student received a resource
                    # enforce per-resource cooldown: skip any student whose last allocation of this
                    # resource type was within `cooldowns[resource_type]` rounds
                    last_alloc_round = last_allocated.setdefault(i, {})
                    # assume `resource_type` variable holds the chosen type (1–4)
                    if t - last_alloc_round.get(resource_type, -999) <= cooldowns[resource_type]:
                        continue
                    last_allocated[i][resource_type] = t

                    cum += (best_i - r)
                    results[key][run_i, t-1] = cum

                print(f"  ✓ {name} run {run_i+1}/{runs}")

        # 1B) meta-level baselines
        print(f"--- Meta baselines (UCBO & LinUCBO) ---")
        ucbo_hist = run_ucbo(r_funcs, mapping_label,
                             n_init=n_init, cand_size=cand_size, budget=T, budgets=budgets)
        linucbo_res = run_linucbo(r_funcs, mapping_label,
                                  contexts=X_all_np,
                                  race_np=race_all_np,
                                  n_init=n_init,
                                  cand_size=cand_size,
                                  budget=T,
                                  runs=runs,
                                  budgets=budgets)

        results[f"{mapping_label}_UCBO"]    = np.tile(ucbo_hist,    (runs,1))
        results[f"{mapping_label}_LinUCBO"] = linucbo_res

    return results


# ==== 2) Execute ====
all_results = run_action_dependent_all(T=T, runs=runs, cohort_idx=0)


# ─────────────────────────────────────────────────────────────────────────────
# COMPLETE History‐Dependent Experiment (self-contained)
# ─────────────────────────────────────────────────────────────────────────────
from scipy.stats import beta

# Number of subgroups (used elsewhere)
K = race_all.size(1)

# ─────────────────────────────────────────────────────────────────────────────
# 1) Kernelized History Environment (Beta‐kernel)
# ─────────────────────────────────────────────────────────────────────────────
class HistoryEnvBeta:
    def __init__(self, r_funcs, a=2.0, b=5.0):
        """
        r_funcs: list of reward functions per subgroup k
        a, b:    Beta distribution parameters that shape the delay kernel
        """
        self.r_funcs = r_funcs
        self.history = []     # stores past allocation vectors pₜ
        self.a, self.b = a, b

    def update_impact(self, p):
        """
        Given a new allocation vector p (length-K):
          1) Append it to history.
          2) Compute a weight w_i for each past round i via Beta.pdf over normalized time.
          3) Normalize weights to sum to 1.
          4) Return f, the weighted average of all past p's:
               f[k] = Σ_i w_i * p_i[k]
        Intuition:
          • Early rounds (i small) and late rounds (i ≈ H) get less/more weight
            depending on (a,b), implementing a smooth “delay” kernel over time.
        """
        # 1) record this round’s allocation
        self.history.append(p.copy())

        H = len(self.history)
        # 2) normalised time points t_i = (i+0.5)/H for i=0..H-1
        t = (np.arange(H) + 0.5) / H

        # 3) compute unnormalized Beta‐kernel weights at each t_i
        w = beta.pdf(t, self.a, self.b)

        # 4) normalize so that Σ w_i = 1
        w /= w.sum()

        # 5) return the “effective” allocation f as weighted sum of past p’s
        #    this f encodes delayed response: more recent allocations can
        #    contribute more (or less) depending on a,b.
        return sum(w[i] * self.history[i] for i in range(H))

    def play(self, p, f):
        """
        Given the current allocation p and its delayed impact f:
          • For each subgroup k, you allocate fraction p[k] but the
            realized expected reward uses f[k] instead of p[k].
          • So we compute U = Σ_k p[k] * r_funcs[k]( f[k] ).
        This decouples the planning variable p from the “effective”
        fraction f that the delay kernel supplies.
        """
        return sum(p[k] * self.r_funcs[k](f[k]) for k in range(len(p)))




class IndDUCB:
    def __init__(self, N, gamma=0.9, alpha=2.0):
        self.N      = N
        self.gamma  = gamma
        self.alpha  = alpha
        self.counts = np.zeros(N)
        self.means  = np.zeros(N)
    def select(self, t):
        bonus = self.alpha * np.sqrt(2*np.log(t)/(self.counts+1e-9))
        return int(np.argmax(self.means + bonus))
    def update(self, i, r):
        self.counts *= self.gamma
        self.means  *= self.gamma
        self.counts[i] += 1
        n = self.counts[i]
        self.means[i] += (r - self.means[i]) / n

class IndSWUCB:
    def __init__(self, N, window=50, alpha=2.0):
        self.N       = N
        self.window  = window
        self.alpha   = alpha
        self.history = []
    def select(self, t):
        recent = self.history[-self.window:]
        counts = np.zeros(self.N); sums = np.zeros(self.N)
        for i, r in recent:
            counts[i] += 1; sums[i] += r
        means = sums / (counts + 1e-9)
        bonus = self.alpha * np.sqrt(2*np.log(t)/(counts+1e-9))
        return int(np.argmax(means + bonus))
    def update(self, i, r):
        self.history.append((i, r))


# 3) History-Dependent Runner (includes UCBO & LinUCBO)
def run_history_dependent_all(T=T, runs=runs, beta_a=2.0, beta_b=5.0,
                             resource_budget=None, cooldowns=None,
                             cohort_length=8):
    # Prepare raw features & race to NumPy
    X_raw_np  = X_all.cpu().numpy()
    race_np   = race_all.cpu().numpy()

    # setup budget mode
    if resource_budget is None:
        resource_budget_local = {"global": B}
        budget_mode = "global"
    else:
        resource_budget_local = resource_budget.copy()
        budget_mode = "multi"

    # prepare cooldown tracker: remaining rounds before re-allocation
    if cooldowns is None:
        cooldowns = {"global": 1}
    cooldown_tracker = {}  # maps student (i) to remaining cooldown

    # determine number of students
    N = race_np.shape[0]

    results = {}
    for mapping_label, r_funcs in [
        ("Linear",    r_funcs_lin),
        ("Nonlinear", r_funcs_nn),
    ]:
        print(f"\n=== History-Dependent ({mapping_label}) ===")

        # 3A) Student-level baselines
        for name, Alg in [
            ("UCB",   IndUCB),
            ("LinUCB",IndLinUCB),
            ("EXP3",  IndEXP3),
            ("DUCB",  IndDUCB),
            ("SWUCB", IndSWUCB),
        ]:
            key = f"{mapping_label}_{name}"
            results[key] = np.zeros((runs, T))
            print(f"→ {name}")
            for run_i in range(runs):
                env = HistoryEnvBeta(r_funcs, a=beta_a, b=beta_b)
                used_students = set()
                cooldown_tracker.clear()
                if name == "LinUCB":
                    alg = Alg(X_raw_np, alpha=1.0)
                else:
                    alg = Alg(race_np.shape[0])

                cum = 0.0
                for t in range(1, T+1):
                    # at start of cohort, reset history & trackers
                    if (t-1) % cohort_length == 0 and t > 1:
                        env.history.clear()
                        used_students.clear()
                        cooldown_tracker.clear()

                    # decrement cooldowns
                    for i_cd in list(cooldown_tracker):
                        cooldown_tracker[i_cd] -= 1
                        if cooldown_tracker[i_cd] <= 0:
                            del cooldown_tracker[i_cd]

                    if budget_mode == "global":
                        if resource_budget_local["global"] <= 0:
                            print(f"Global budget exhausted at round {t}")
                            break
                        valid_idxs = list(set(range(N)) - used_students)
                    else:
                        # per-group budgets
                        valid_idxs = [
                            i for i in range(N)
                            if resource_budget_local.get(
                                racial_cols[race_np[i].argmax()], 0
                            ) > 0 and i not in used_students
                        ]
                        if not valid_idxs:
                            print(f"No valid students available at round {t}")
                            break

                    # select an arm, retry until a valid one is found
                    if name == "EXP3":
                        while True:
                            i, probs = alg.select()
                            if i in valid_idxs:
                                break
                    else:
                        while True:
                            i = alg.select(t)
                            if i in valid_idxs:
                                break

                    p = np.zeros(K)
                    p[int(race_np[i].argmax())] = 1.0

                    f = env.update_impact(p)
                    r = env.play(p, f)

                    if name == "EXP3":
                        alg.update(i, r, probs)
                    else:
                        alg.update(i, r)

                    # deduct budget
                    if budget_mode == "global":
                        resource_budget_local["global"] -= 1
                    else:
                        grp = racial_cols[race_np[i].argmax()]
                        resource_budget_local[grp] -= 1
                    # record assignment
                    used_students.add(i)
                    # enforce cooldown for student i
                    cooldown_rounds = cooldowns.get("global", 1)
                    cooldown_tracker[i] = cooldown_rounds

                    cum += r
                    results[key][run_i, t-1] = cum

                print(f"  ✓ run {run_i+1}/{runs}")
            print()

        # 3B) Meta-level baselines
        print(f"→ UCBO ({mapping_label})")
        ucbo_hist = run_ucbo(r_funcs, mapping_label, budget=T)
        print(f"→ LinUCBO ({mapping_label})")
        linucbo_res = run_linucbo(
            r_funcs, mapping_label,
            contexts = X_raw_np,
            race_np   = race_np,
            budget    = T,
            runs      = runs
        )

        results[f"{mapping_label}_UCBO"]    = np.tile(ucbo_hist,    (runs,1))
        results[f"{mapping_label}_LinUCBO"] = linucbo_res

    return results

