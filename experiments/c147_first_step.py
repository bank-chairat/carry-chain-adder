import csv, numpy as np, torch, torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch.func import functional_call, jvp
torch.set_num_threads(2); n = 8
def bits(x, k): return ((x[:, None] >> np.arange(k)) & 1).astype(np.float32)
A, B = np.meshgrid(np.arange(256), np.arange(256)); A = A.ravel(); B = B.ravel()
X = np.concatenate([bits(A, n), bits(B, n)], 1); Y = bits(A + B, n + 1)
perm = np.random.default_rng(500).permutation(len(A)); tr, te = perm[:len(A) // 2], perm[len(A) // 2:]
Xtr, Xte, ytr = torch.tensor(X[tr]), torch.tensor(X[te]), torch.tensor(Y[tr]); f4 = 2 * Y[te][:, 4] - 1
T = {}
for d in ('c83_a', 'c83_b'):
    for r in csv.DictReader(open(f'{d}/init_align_joint_n8.csv')): T[int(r['s'])] = int(r['time'])
for s in range(40, 60): T[s] = int(np.load(f'c127/traj_s{s}.npz')['t4'])
S = list(range(60)) + list(range(100, 500)); P = {}
for s in S:
    torch.manual_seed(10000 + s); L, d = [], 16
    for _ in range(3): L += [nn.Linear(d, 128), nn.ReLU()]; d = 128
    net = nn.Sequential(*L, nn.Linear(d, n + 1)); params = dict(net.named_parameters())
    z = net(Xtr); r = (torch.sigmoid(z) - ytr).detach()
    gall = torch.autograd.grad((z * r).sum() / len(tr), list(params.values()), retain_graph=True)
    g4 = torch.autograd.grad((z[:, 4] * r[:, 4]).sum() / len(tr), list(params.values()))
    def z4(p): return functional_call(net, p, (Xte,))[:, 4]
    out = []
    for g in (gall, [torch.sign(x) for x in gall], [torch.sign(x) for x in g4]):
        tang = {k: -v for k, v in zip(params, g)}; _, dz = jvp(z4, (params,), (tang,))
        dz = dz.detach().numpy().astype(np.float64); out.append(float(np.corrcoef(dz, f4)[0, 1]) if dz.std() > 0 else 0.0)
    P[s] = out
J = {s: float(np.load(f'c139/init_s{s}.npz')['J'][-1]) for s in S}; new = list(range(100, 500)); old = list(range(60))
for k, nm in enumerate(('P_GD', 'P_Adam', 'P_4 (alone, Adam sign)')):
    lab = [s for s in new if J[s] >= 0.2 or J[s] < 0.05]
    print(f'{nm:24s} Spearman with J500 (s 100-499) {spearmanr([P[s][k] for s in new], [J[s] for s in new]).statistic:+.2f}; escape AUC {roc_auc_score([1 if J[s] >= 0.2 else 0 for s in lab], [P[s][k] for s in lab]):.2f}; Spearman with T (s 0-59) {spearmanr([P[s][k] for s in old], [T[s] for s in old]).statistic:+.2f}; median value {np.median([P[s][k] for s in new]):+.4f}')
np.save('c147_P.npy', np.array([[s] + P[s] for s in S]))
