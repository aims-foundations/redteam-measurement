import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


data = pd.read_parquet('harmbench/responses.parquet', engine='pyarrow')

data[['attack', 'judge']] = data['test_condition'].str.extract(
    r'attack=([^;]+);judge=([^;]+)'
)
items = pd.read_parquet('harmbench/_contrib/items.parquet') #data['category'] is available alongside data['attack'] and data['judge'], so we can filter (data[data.category=='standard']) 
data = data.merge(items[['item_id', 'category']], on='item_id', how='left')
# data = data[data.category != 'copyright']
# Rows are already one-per-(model, behavior, attack, judge); both judges kept.

data = data.drop(columns=(['test_condition','benchmark_id','trial','correct_answer', 'trace']))
data = data[data['response'].isin([0.0, 1.0])]

# distinct ids (plural = a list of many)
theta_ids = data['subject_id'].unique()   # models
z_ids     = data['item_id'].unique()      # behaviors
v_ids = data['attack'].unique()           # attack algorithms
j_ids = data['judge'].unique()            # judges (harmbench, advbench)
M, N, K, J = len(theta_ids), len(z_ids), len(v_ids), len(j_ids)

# id-string -> position (a lookup map)
theta_id_to_idx = {s: i for i, s in enumerate(theta_ids)}
z_id_to_idx     = {it: i for i, it in enumerate(z_ids)}
v_id_to_idx = {a: i for i, a in enumerate(v_ids)}
j_id_to_idx = {judge: i for i, judge in enumerate(j_ids)}

# per-observation index arrays (these index into theta_hat / z_hat)
theta_idx = torch.tensor(data['subject_id'].map(theta_id_to_idx).values)
z_idx     = torch.tensor(data['item_id'].map(z_id_to_idx).values)
v_idx = torch.tensor(data['attack'].map(v_id_to_idx).values)
j_idx = torch.tensor(data['judge'].map(j_id_to_idx).values)
y         = torch.tensor(data['response'].values, dtype=torch.float)

#80/20 train/test split on observation positions 
n_obs = len(y)
train_pos, test_pos = train_test_split(
    range(n_obs), test_size=0.20, random_state=1, shuffle=True
)

theta_idx_tr = theta_idx[train_pos]
z_idx_tr     = z_idx[train_pos]
v_idx_tr      = v_idx[train_pos]
j_idx_tr     = j_idx[train_pos]
y_tr         = y[train_pos]

theta_idx_te = theta_idx[test_pos]
z_idx_te     = z_idx[test_pos]
v_idx_te = v_idx[test_pos]
j_idx_te     = j_idx[test_pos]
y_te         = y[test_pos]

theta_hat = torch.zeros(M, requires_grad=True)
z_hat     = torch.zeros(N, requires_grad=True)
v_hat = torch.zeros(K, requires_grad=True)
j_hat = torch.zeros(J, requires_grad=True) # per-judge effect; ONE per judge (J judges)
# b_hat = torch.ones(K, requires_grad=True) # per-attacker discrimination; ONE per attacker, like v_hat

# class balance: fraction of 1s (jailbroken) in each set
# print("train mean response:", y_tr.mean().item())
# print("test  mean response:", y_te.mean().item())

# print("first 10 train positions:", train_pos[:10])
# print("first 10 test positions:", test_pos[:10])
# print("first 10 train responses:", y_tr[:10])

def log_likelihood(t_idx, z_i, v_i, j_i, y_):
    # p = torch.sigmoid(b_hat[v_i]*theta_hat[t_idx] + z_hat[z_i] + v_hat[v_i] + j_hat[j_i]).clamp(1e-9, 1-1e-9)
    p = torch.sigmoid(theta_hat[t_idx] + z_hat[z_i] + v_hat[v_i] + j_hat[j_i]).clamp(1e-9, 1-1e-9)
    return (y_*torch.log(p) +(1-y_)*torch.log(1-p)).sum().item()

def auc(t_idx, z_i, v_i, j_i, y_):
    # p = torch.sigmoid(b_hat[v_i]*theta_hat[t_idx] + z_hat[z_i] + v_hat[v_i] + j_hat[j_i])
    p = torch.sigmoid(theta_hat[t_idx] + z_hat[z_i] + v_hat[v_i] + j_hat[j_i])
    return roc_auc_score(y_.numpy(), p.detach().numpy())

train_ll_history = []
test_ll_history  = []
train_auc_history = []
test_auc_history  = []


lr = 0.001
reg = 0.05
prev_loss = float('inf')
# opt = torch.optim.Adam([theta_hat, z_hat, v_hat, j_hat, b_hat], lr=lr)
opt = torch.optim.Adam([theta_hat, z_hat, v_hat, j_hat], lr=lr)

for i in range(10000):
    with torch.no_grad():
        #record the change of log-likelihood throughout the loop
        train_ll_history.append(log_likelihood(theta_idx_tr, z_idx_tr, v_idx_tr, j_idx_tr, y_tr))
        test_ll_history.append(log_likelihood(theta_idx_te, z_idx_te, v_idx_te, j_idx_te, y_te))
        #record the change of AUC throughout the loop
        train_auc_history.append(auc(theta_idx_tr, z_idx_tr, v_idx_tr, j_idx_tr, y_tr))
        test_auc_history.append(auc(theta_idx_te, z_idx_te, v_idx_te, j_idx_te, y_te))

    # p_hat = torch.sigmoid(b_hat[v_idx_tr]*theta_hat[theta_idx_tr] + z_hat[z_idx_tr] + v_hat[v_idx_tr] + j_hat[j_idx_tr])
    p_hat = torch.sigmoid(theta_hat[theta_idx_tr] + z_hat[z_idx_tr] + v_hat[v_idx_tr] + j_hat[j_idx_tr])

    loss = torch.nn.functional.binary_cross_entropy(p_hat, y_tr)  # = mean negative log-likelihood

    opt.zero_grad()
    loss.backward()
    opt.step()

    with torch.no_grad(): # identifiability centering
        theta_hat -= theta_hat.mean()
        v_hat     -= v_hat.mean()
       # z_hat     -= z_hat.mean()
        j_hat     -= j_hat.mean()
        # b_hat     -= b_hat.mean()

    if abs(prev_loss - loss.item()) < 1e-6:
        break
    
    prev_loss = loss.item()
    
#TRAIN fit (data the model was fitted on) 
# p_hat_tr = torch.sigmoid(b_hat[v_idx_tr]*theta_hat[theta_idx_tr] + z_hat[z_idx_tr] + v_hat[v_idx_tr] + j_hat[j_idx_tr])
p_hat_tr = torch.sigmoid(theta_hat[theta_idx_tr] + z_hat[z_idx_tr] + v_hat[v_idx_tr] + j_hat[j_idx_tr])
ll_tr = (y_tr*torch.log(p_hat_tr.clamp(1e-9,1-1e-9)) + (1-y_tr)*torch.log((1-p_hat_tr).clamp(1e-9,1-1e-9))).sum()

print("Train log-likelihood:", ll_tr.item())
print("Train AUC:", roc_auc_score(y_tr.numpy(), p_hat_tr.detach().numpy()))

#TEST evaluation
# p_hat_te = torch.sigmoid(b_hat[v_idx_te]*theta_hat[theta_idx_te] + z_hat[z_idx_te] + v_hat[v_idx_te] + j_hat[j_idx_te])
p_hat_te = torch.sigmoid(theta_hat[theta_idx_te] + z_hat[z_idx_te] + v_hat[v_idx_te] + j_hat[j_idx_te])
ll_te = (y_te*torch.log(p_hat_te.clamp(1e-9,1-1e-9)) + (1-y_te)*torch.log((1-p_hat_te).clamp(1e-9,1-1e-9))).sum()

print("Test log-likelihood:", ll_te.item())
print("Test AUC:", roc_auc_score(y_te.numpy(), p_hat_te.detach().numpy()))


#Draw the graph of log-likelihood
plt.figure()
plt.plot(train_ll_history, label="Train")
plt.plot(test_ll_history,  label="Test")
plt.xlabel("Iteration")
plt.ylabel("Log-likelihood")
plt.title("Log-likelihood per iteration")
plt.legend()
plt.savefig("harmbench_ll_curve.png", dpi=150)
plt.show()

#Draw the graph of AUC (separate figure)
plt.figure()
plt.plot(train_auc_history, label="Train", linewidth=3)
plt.plot(test_auc_history,  label="Test", linestyle="--")
plt.xlabel("Iteration")
plt.ylabel("AUC")
plt.title("AUC per iteration")
plt.legend()
plt.savefig("harmbench_auc_curve.png", dpi=150)
plt.show()

