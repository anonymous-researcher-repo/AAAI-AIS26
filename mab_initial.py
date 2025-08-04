import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from tqdm import trange
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
import warnings

warnings.filterwarnings("ignore")
sns.set(style="whitegrid", context="talk")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)
print("Using device:", device)

# ========================== Parameters ==========================
base_arms = 5
K = base_arms
epsilon = 0.1
T = 100          # reduced for faster iteration
runs = 20
B = 500          # total resource budget
# ========================== Data Preparation ==========================
print("Loading and cleaning data...")
df = pd.read_csv('/resource_alloc/RA-data.csv')
df = df.rename(columns={
    'Race_Asian, Hawaii/Pac. Islander': 'Asian',
    'Race_Black or African American': 'Black',
    'Race_Hispanic': 'Hispanic',
    'Race_More than one race': 'MR',
    'Race_White': 'White'
})
# filter and drop unwanted columns
racial_cols = ['Asian','Black','Hispanic','MR','White'][:base_arms]
df = df[df[racial_cols].sum(axis=1) > 0]
# create binary resource column: 1 if any resource >0, else 0
df['resource_binary'] = (df['resource'] > 0).astype(int)
df = df.drop(columns=[
    'Unnamed: 0','gender_Male','Race_Amer. Indian/Alaska Native',
    'F3_Highest level of education','resource'  # drop original multi-level resource
]).dropna()
print(f"Final cleaned dataset: {df.shape}")

group_sizes = df[racial_cols].sum(axis=0).astype(int).tolist()
group_sizes_tensor = torch.tensor(group_sizes, dtype=torch.float32, device=device)

# features include all except target and group indicators
feature_cols = [c for c in df.columns if c not in ['attainment','F3_GPA(all)']+racial_cols]
X = df[feature_cols]
y = df['F3_GPA(all)'].values
race = df[racial_cols].values

print("Standardizing features...")
num_cols = X.select_dtypes(include='number').columns
scaler = StandardScaler().fit(X[num_cols])
X[num_cols] = scaler.transform(X[num_cols])

print("Train/test split...")
X_tr, X_te, y_tr, y_te, r_tr, r_te = train_test_split(
    X.values, y, race,
    test_size=0.2, random_state=0,
    stratify=race.argmax(axis=1)
)
X_all = torch.tensor(np.vstack([X_tr, X_te]), dtype=torch.float32).to(device)
race_all = torch.tensor(np.vstack([r_tr, r_te]), dtype=torch.float32).to(device)
y_all = np.hstack([y_tr, y_te])
print(f"Combined sample size: {X_all.shape[0]}")

# ========================== Neural Network Model ==========================
class SharedBackbone(nn.Module):
    def __init__(self, in_dim, hid=128, drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hid, hid), nn.ReLU(), nn.Dropout(drop)
        )
    def forward(self, x):
        return self.net(x)

class MeanPoolingGaussianLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim,64), nn.ReLU(), nn.Linear(64,2))
    def forward(self, emb, race):
        grouped    = emb.unsqueeze(1) * race.unsqueeze(-1)  # [N,K,H]
        pooled_sum = grouped.sum(dim=0)                    # [K,H]
        counts     = race.sum(dim=0, keepdim=True).T       # [K,1]
        pooled     = pooled_sum / (counts + 1e-6)          # [K,H]
        out        = self.mlp(pooled)                      # [K,2]
        return out[:,0], torch.exp(out[:,1])

class SubgroupModel(nn.Module):
    def __init__(self, backbone, gaussian):
        super().__init__()
        self.backbone = backbone
        self.gaussian = gaussian
    def forward(self, x, race):
        emb = self.backbone(x)
        return self.gaussian(emb, race)

def train_model(model, X, race, y, epochs=20):
    print(f"Training neural net for {epochs} epochs...")
    opt = optim.Adam(model.parameters(), lr=1e-3)
    y_t = torch.tensor(y, dtype=torch.float32).reshape(-1,1).to(device)
    for _ in trange(epochs, desc="NN Training"):
        model.train()
        opt.zero_grad()
        mu, var = model(X, race)
        loss = 0.5 * ((torch.log(var+1e-6)) + ((y_t.mean()-mu)**2)/(var+1e-6)).mean()
        loss.backward()
        opt.step()
    print("NN training complete.")
    return model

input_dim = X_all.shape[1]
model = SubgroupModel(SharedBackbone(input_dim), MeanPoolingGaussianLayer(128)).to(device)
model = train_model(model, X_all, race_all, y_all, epochs=20)

# ========================== Extract Subgroup Parameters ==========================
print("Extracting μ_k and σ_k per subgroup...")
model.eval()
with torch.no_grad():
    mu_all, var_all = model(X_all, race_all)
mu_k = mu_all.cpu().numpy()
sigma_k = np.sqrt(var_all.cpu().numpy())

# build reward functions R_k(p) = norm.pdf(p; m_k, s_k) / denom
r_funcs = []
for k in range(K):
    m,s = mu_k[k], sigma_k[k]
    denom = norm.pdf(m, loc=m, scale=s)
    r_funcs.append(lambda p, m=m, s=s, d=denom: float(norm.pdf(p, loc=m, scale=s)/d))
print("Subgroup reward funcs ready.")

# ========================== Helper: Grid & Refinement ==========================
def build_grid(eps):
    levels = torch.arange(eps, 1+1e-9, eps, device=device)
    grid   = torch.cartesian_prod(*([levels]*K))
    mask_s = torch.isclose(grid.sum(dim=1), torch.tensor(1.0, device=device), atol=1e-6)
    usage  = (grid * group_sizes_tensor.unsqueeze(0)).sum(1)
    mask_b = usage <= B
    return grid[mask_s & mask_b]

def refine_locally(meta, stats, eps, t, M=20):
    # UCB scoring on current (meta, stats)
    counts = np.array([stats[tuple(p.tolist())][0] for p in meta])
    means  = np.array([stats[tuple(p.tolist())][1]/(c if c>0 else 1)
                       for p,c in zip(meta,counts)])
    bonus  = np.sqrt(2*np.log(t+1)/(counts+1e-9))
    scores = means + bonus
    winners= [meta[i] for i in np.argsort(scores)[-M:]]
    # local grid around winners
    new_eps = eps/2
    offsets = [-new_eps,0.0,+new_eps]
    cand    = set()
    for w in winners:
        w = w.cpu().tolist()
        for delta in product(offsets, repeat=K):
            q = [max(0,min(1,w[k]+delta[k])) for k in range(K)]
            tot = sum(q)
            if abs(tot-1.0)<1e-3 and sum(q[i]*group_sizes[i] for i in range(K))<=B:
                q = [v/tot for v in q]
                cand.add(tuple(round(v,6) for v in q))
    new_meta = torch.tensor(list(cand), device=device)
    new_stats= {tuple(p.tolist()): stats.get(tuple(p.tolist()), [0,0.0]) for p in new_meta}
    return new_meta, new_stats

# ========================== Environments ==========================
class ActionEnv:
    def __init__(self, r_funcs): self.r_funcs = r_funcs
    def play(self, p):
        return {k: p[k].item()*self.r_funcs[k](p[k].item()) for k in range(K)}

class HistoryEnv:
    def __init__(self, r_funcs, gamma):
        self.r_funcs, self.gamma = r_funcs, gamma
        self.history = []
    def update_impact(self, p):
        self.history.append(p)
        H = len(self.history)
        ws= torch.tensor([self.gamma**(H-1-i) for i in range(H)], device=device)
        return sum(ws[i]*self.history[i] for i in range(H))/ws.sum()
    def play(self, p, f):
        return {k: p[k].item()*self.r_funcs[k](f[k].item()) for k in range(K)}

# ========================== Baseline Bandit Algorithms ==========================
class ActionDependentUCB:
    def __init__(self, meta_arms, sigma=1e-6):
        self.meta, self.M = meta_arms, meta_arms.size(0)
        self.counts = torch.zeros(self.M, device=device)
        self.means  = torch.zeros(self.M, device=device)
    def select(self, t):
        bonus = torch.sqrt(2*torch.log(torch.tensor(t+1.0,device=device))/(self.counts+1e-9))
        idx   = torch.argmax(self.means + bonus).item()
        return idx, self.meta[idx]
    def update(self, idx, rewards):
        r = sum(rewards.values())
        self.counts[idx] += 1
        n = self.counts[idx]
        self.means[idx] += (r - self.means[idx].item())/n

class EXP3:
    def __init__(self, meta_arms, eta=0.1):
        self.meta   = meta_arms
        self.M      = meta_arms.size(0)
        self.eta    = eta
        self.weights= torch.ones(self.M,device=device)
    def select(self):
        w     = torch.clamp(torch.nan_to_num(self.weights,nan=1.0,posinf=1e6),1e-6,1e6)
        probs = w/w.sum()
        idx   = torch.multinomial(probs,1).item()
        return idx, self.meta[idx], probs
    def update(self, idx, reward, probs):
        x    = reward/(probs[idx]+1e-12)
        expo = torch.clamp(self.eta*x, max=85.0)
        self.weights[idx]=torch.clamp(self.weights[idx]*torch.exp(expo),1e-6,1e6)

class mEXP3:
    def __init__(self, meta_arms, eta=0.1):
        self.exp3 = EXP3(meta_arms,eta)
    def select(self): return self.exp3.select()
    def update(self,idx,r,probs): self.exp3.update(idx,r,probs)

class CUCB(ActionDependentUCB):
    pass

class DiscountedUCB:
    def __init__(self,K,gamma,ξ=1.0):
        self.K,self.gamma,self.ξ = K,gamma,ξ
        self.history = []
    def select(self,t=None):
        H=len(self.history)
        if H==0: return 0,[float('inf')]*self.K
        ws = np.array([self.gamma**(H-1-s) for s in range(H)])
        vals=[]
        for k in range(self.K):
            picks=np.array([1 if a==k else 0 for a,_ in self.history])
            rews =np.array([r for _,r in self.history])
            N=(ws*picks).sum(); R=(ws*picks*rews).sum()
            mu=R/(N+1e-9); bonus=np.sqrt(self.ξ*np.log(max(1,H))/(N+1e-9))
            vals.append(mu+bonus)
        arm=int(np.argmax(vals))
        return arm,vals
    def update(self,arm,rew): self.history.append((arm,rew))

class SWUCB:
    def __init__(self,K,window,ξ=1.0):
        self.K,self.window,self.ξ=K,window,ξ
        self.history=[]
    def select(self,t):
        last=self.history[-self.window:]
        vals=[]
        for k in range(self.K):
            picks=np.array([1 if a==k else 0 for a,_ in last])
            rews =np.array([r for _,r in last])
            N=picks.sum(); mu=(picks*rews).sum()/(N+1e-9)
            bonus=np.sqrt(self.ξ*np.log(min(t,self.window))/(N+1e-9))
            vals.append(mu+bonus)
        arm=int(np.argmax(vals)); return arm,vals
    def update(self,arm,rew): self.history.append((arm,rew))

# ========================== Bayesian Optimization Core ==========================
def sample_random_meta_arms(n):
    """Uniformly sample n valid _numpy_ meta‐arms from simplex under budget"""
    Ps=[]
    while len(Ps)<n:
        p = np.random.dirichlet(np.ones(K))
        if (p*group_sizes).sum()<=B: Ps.append(p)
    return np.array(Ps)

def evaluate_meta_arm(p):
    """Compute U(p) = sum_k p[k]*mu_k via the pretrained NN on sampled individuals."""
    # copy
    X_mod = X.values.copy()
    race_arr = race.copy()
    # randomly assign resource_binary per p
    for k in range(K):
        idxs = np.where(race_arr[:,k]==1)[0]
        m = int(np.round(p[k]*len(idxs)))
        chosen = np.random.choice(idxs, size=m, replace=False)
        # zero all, then mark chosen
        X_mod[idxs, feature_cols.index('resource_binary')] = 0
        X_mod[chosen, feature_cols.index('resource_binary')] = 1
    # predict subgroup mu
    Xt = torch.tensor(X_mod, dtype=torch.float32).to(device)
    racet = torch.tensor(race_arr, dtype=torch.float32).to(device)
    mu_k_tensor,_ = model(Xt, racet)  # only mu needed
    mu_k = mu_k_tensor.detach().cpu().numpy()
    return float((p * mu_k).sum())

# initialize BO
N_init = 10
Ps = sample_random_meta_arms(N_init)
Ys = np.array([evaluate_meta_arm(p) for p in Ps])

kernel = ConstantKernel(1.0)*Matern(length_scale=1.0, nu=2.5) + WhiteKernel(1e-6)
gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
gp.fit(Ps, Ys)

def acquisition_ucb(Ps_batch, beta=2.0):
    mu, sigma = gp.predict(Ps_batch, return_std=True)
    return mu + beta * sigma

# run BO
n_iters = 50
for i in range(n_iters):
    # optimize acquisition by sampling 1000 random meta‐arms and picking best
    candidates = sample_random_meta_arms(1000)
    scores     = acquisition_ucb(candidates)
    p_next     = candidates[np.argmax(scores)]
    y_next     = evaluate_meta_arm(p_next)
    # update GP
    Ps = np.vstack([Ps, p_next])
    Ys = np.append(Ys, y_next)
    gp.fit(Ps, Ys)

print("BO complete.")

# ========================== Runners & Plotting ==========================
def plot_with_CI(ax, data, label, color):
    m = data.mean(axis=0); s = data.std(axis=0)
    ax.plot(m, label=label, color=color)
    ax.fill_between(np.arange(len(m)), m-2*s, m+2*s, alpha=0.2, color=color)

# ... Include run_action_dependent, run_action_dependent_adaptive, run_history_dependent as above ...
# ... Then call and plot Figure 2 and Figure 3 ...

# ========================== Static Action‐Dependent Runner ==========================
def run_action_dependent(T, runs):
    algs = {
        "UCB":   UCB_K(K),
        "EXP3":  EXP3_K(K),
        "DUCB":  DiscountedUCB_K(K, gamma=0.5),
        "SWUCB": SlidingUCB_K(K, window=20),
    }
    results = {name: np.zeros((runs, T)) for name in algs}
    best    = max(r_funcs[k](1.0) for k in range(K))  # one-hot means p_k=1.0

    for name, alg in algs.items():
        print(f"[AD] {name}")
        for run in range(runs):
            env = ActionEnv(r_funcs)
            cum = 0.0
            for t in range(1, T+1):
                # EXP3 already returns (arm, p, probs)
                if name == "EXP3":
                    arm, p, probs = alg.select()
                    rew           = sum(env.play(p).values())
                    alg.update(arm, rew, probs)

                # DUCB/SWUCB return (arm, vals) -> we need to convert arm -> one-hot p
                elif name in ("DUCB", "SWUCB"):
                    arm, _ = alg.select(t)
                    p      = meta_arms[arm]
                    rew    = sum(env.play(p).values())
                    alg.update(arm, rew)

                # UCB_K returns (arm, p)
                else:  # "UCB"
                    arm, p = alg.select(t)
                    rew    = sum(env.play(p).values())
                    alg.update(arm, rew)

                cum += best - rew
                results[name][run, t-1] = cum

    return results


# ========================== Static History‐Dependent Runner ==========================
def run_history_dependent(gamma, runs, T):
    algs = {
        "UCB":   UCB_K(K),
        "EXP3":  EXP3_K(K),
        "DUCB":  DiscountedUCB_K(K, gamma),
        "SWUCB": SlidingUCB_K(K, window=20),
    }
    results = {n: np.zeros((runs, T)) for n in algs}
    best    = max(r_funcs[k](1.0) for k in range(K))

    for name, alg in algs.items():
        print(f"[HD γ={gamma}] {name}")
        for run in range(runs):
            env = HistoryEnv(r_funcs, gamma)
            cum = 0.0
            for t in range(1, T+1):
                if name == "EXP3":
                    arm, p, probs = alg.select()
                    f             = env.update_impact(p)
                    rew           = sum(env.play(p, f).values())
                    alg.update(arm, rew, probs)

                elif name in ("DUCB", "SWUCB"):
                    arm, _ = alg.select(t)
                    p      = meta_arms[arm]
                    f      = env.update_impact(p)
                    rew    = sum(env.play(p, f).values())
                    alg.update(arm, rew)

                else:  # "UCB"
                    arm, p = alg.select(t)
                    f       = env.update_impact(p)
                    rew     = sum(env.play(p, f).values())
                    alg.update(arm, rew)

                cum += best - rew
                results[name][run, t-1] = cum

    return results


# ========================== Plotting ==========================
def plot_with_CI(ax, data, label, color):
    m = data.mean(axis=0)
    s = data.std(axis=0)
    ax.plot(m, label=label, color=color)
    ax.fill_between(range(len(m)), m-2*s, m+2*s, alpha=0.2, color=color)

# === Figure 2: Action‐Dependent + History‐Dependent ===
print("Running experiments…")
fig, axes = plt.subplots(1, 4, figsize=(24,5))

# Action‐Dependent
resA = run_action_dependent(T, runs)
for name, col in zip(resA, ("r","orange","g","b")):
    plot_with_CI(axes[0], resA[name], name, col)
axes[0].set(title="(a) Action‐Dep", xlabel="Round", ylabel="Cum Regret")
axes[0].legend()

# History‐Dependent for several γ
for i, gamma in enumerate((0.2,0.5,0.8), start=1):
    resH = run_history_dependent(gamma, runs, T)
    for name, col in zip(resH, ("r","orange","g","b")):
        plot_with_CI(axes[i], resH[name], name, col)
    axes[i].set(title=f"({chr(ord('a')+i)}) γ={gamma}", xlabel="Round")
    axes[i].legend()

plt.tight_layout()
plt.show()
