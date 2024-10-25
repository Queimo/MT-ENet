import sys

import botorch.models.transforms.outcome
import botorch.models.transforms.utils
sys.path.append('..')
import botorch.models.transforms
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import numpy as np
import torch
from mtevi.mtevi import *
from mtevi.utils import *
from models import *
import math

use_gpu = lambda x=True: torch.set_default_tensor_type(torch.cuda.FloatTensor 
                                             if torch.cuda.is_available() and x 
                                             else torch.FloatTensor)
#use_gpu()


def plot_other(fig, ax, X, Y, X_outliers, Y_outliers, test_x_gp, pred_y, std_y, train_boundary_A, train_boundary_B):
    ax.scatter(X,Y, color='black')
    ax.scatter(X_outliers, Y_outliers, color='red', marker='x')
    ax.plot(test_x_gp, pred_y, linewidth=4)
    ax.fill_between(np.linspace(*train_boundary_A,1000), -3, 10, color='blue', alpha=.20)
    ax.fill_between(np.linspace(*train_boundary_B,1000), -3, 10, color='blue', alpha=.20)
    ax.set_ylim(bottom=-3, top=10)
    ax.vlines(6, -3, 10, linestyle=':', color='green', linewidth=5)
    return fig, ax

def plot_confidence(fig, ax, pred_y, std_y, test_x: np.array, freedom_y=30, conf_list=[0.9, 0.7, 0.5, 0.3]):
    for conf in conf_list:
        conf = confidence_interval(np.array(pred_y),
                    np.array(std_y),
                    np.array(freedom_y), conf)
        ax.fill_between(np.squeeze(test_x), conf[0], conf[1], color='red', alpha=.15)
    return fig, ax 

def plot_fig(model, fig, ax, output='mtnet_test.png'):
    true_y = []
    pred_y = []
    std_y = []
    epi_y = []
    alea_y = []
    freedom_y = []

    for x in test_x:
        x = torch.Tensor([x])
        true_y = true_y + list(y.cpu().numpy().flatten())
        gamma, nu, alpha, beta = model(x.float())
        std_y += list(np.sqrt((beta*(1+nu)/(alpha*nu)).cpu().detach().numpy().flatten()))
        alea_y += list(np.sqrt((beta/(alpha)).cpu().detach().numpy().flatten()))
        epi_y += list(np.sqrt((beta/((alpha)*nu)).cpu().detach().numpy().flatten()))
        freedom_y += list((2*alpha).cpu().detach().numpy().flatten())
        pred_y = pred_y + list(gamma.cpu().detach().numpy().flatten())
    
    fig, ax = plot_other(fig, ax, X, Y, X_outliers, Y_outliers, test_x, pred_y, std_y, train_boundary_A, train_boundary_B)
    fig, ax = plot_confidence(fig, ax, pred_y, std_y, test_x, freedom_y) 

    return fig, ax

np.random.seed(0)
torch.manual_seed(0)
X = np.concatenate((np.linspace(-3, 6, 195), np.linspace(6, 10, 5)))
rand_num = [6,3]
Y = np.sin(X*4)**3 + (X**2)/10 + np.random.randn(*X.shape)*0.1 + 2*np.random.normal(scale=np.abs(7-np.abs(X))*0.05)

X = np.expand_dims(X, 1)
dim = len(X[0])

sparse_idx_A = 197
sparse_idx_B = 199

s = 40
e = 130

X_t = np.concatenate((X[s:e], X[sparse_idx_A:sparse_idx_B]))
X_v = np.concatenate((X[:s], X[e:sparse_idx_A], X[sparse_idx_B:]))
Y_t = np.concatenate((Y[s:e], Y[sparse_idx_A:sparse_idx_B]))
Y_v = np.concatenate((Y[:s], Y[e:sparse_idx_A], Y[sparse_idx_B:]))

#outliers 30 points
X_outliers = np.linspace(-1,3,30)

density_multiplier = 1
X_outliers = np.concatenate([X_outliers]*density_multiplier)

Y_outliers = np.sin(X_outliers*4)**3 + (X_outliers**2)/10 + np.random.randn(*X_outliers.shape)*0.1 + 2*np.random.normal(scale=np.abs(7-np.abs(X_outliers))*0.05) + 5

X_outliers = np.expand_dims(X_outliers, 1)  # Expand dimensions to match X_t

X_t = np.concatenate((X_t, X_outliers))
Y_t = np.concatenate((Y_t, Y_outliers))

Y_t = np.expand_dims(Y_t, 1)
Y_v = np.expand_dims(Y_v, 1)


epochs = 100

data = []
for i in range(len(X_t)):
    data.append((X_t[i], Y_t[i]))
data_v = []
for i in range(len(X_v)):
    data_v.append((X_v[i], Y_v[i]))
    
train_loader = torch.utils.data.DataLoader(data, batch_size=128, shuffle=True)
valid_loader = torch.utils.data.DataLoader(data_v, batch_size=256, shuffle=True)

train_boundary_A = (float(X[s]), float(X[e]))
train_boundary_B = (float(X[sparse_idx_A]), float(X[sparse_idx_B]))

test_x = np.linspace(-3,10,1000)

fig = plt.figure(figsize=(10,10))
plt.scatter(X,Y, marker='.')
plt.savefig("synthetic_result/raw_data.png")

##################################################
### MT evi net  ##################################
##################################################
##################################################
model = EvidentialNetwork(dim)
objective = EvidentialnetMarginalLikelihood()
objective_mse = torch.nn.MSELoss(reduction='none')
reg = EvidenceRegularizer(factor=0.0001)

gamma_history = []
nu_history = []
alpha_history = []
beta_history = []

mse_history = []
mse_history_v = []
nll_history = []
nll_history_v = []

total_mse = 0.
total_valid_mse = 0.
total_nll = 0.
total_valid_nll = 0.

model.train()

opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1E-3)
scheduler = torch.optim.lr_scheduler.StepLR(opt, 1, gamma=0.99)
for epoch in range(epochs):
    model.train()
    for x, y in train_loader:
        gamma, nu, alpha, beta = model(x.float())
        y = y.float()
        opt.zero_grad()
        
        nll = (objective(gamma,nu,alpha,beta,y)).mean()
        nll += (reg(gamma, nu, alpha, beta, y)).mean()
        total_nll += nll.item()

        mse = modified_mse(gamma, nu, alpha, beta, y).mean() 
        total_mse += mse.item()
        loss = nll + mse
        loss.backward()
        opt.step()
    
    cur_mse = total_mse/len(train_loader)
    cur_nll = total_nll/len(train_loader)
    
    model.eval()
    
    for x_v, y_v in valid_loader:
        gamma, nu, alpha, beta = model(x_v.float())
        loss_v = objective_mse(gamma, y_v).mean()
        total_valid_mse += loss_v.item()
        
        loss_v = objective(gamma, nu, alpha, beta, y_v).mean()
        total_valid_nll += loss_v.item()
        
    cur_valid_mse = total_valid_mse/len(valid_loader)
    cur_valid_nll = total_valid_nll/len(valid_loader)
    
    print("Epoch {}: Train loss [{:.5f}] Val loss [{:.5f}] Train NLL [{:.5f}] Val NLL [{:.5f}]".format(
        epoch+1, cur_mse, cur_valid_mse, cur_nll, cur_valid_nll))
    mse_history_v.append(cur_valid_mse)
    nll_history_v.append(cur_valid_nll)
    total_mse = 0.
    total_valid_mse = 0.
    total_nll = 0.
    total_valid_nll = 0.
    scheduler.step()
fig, axs = plt.subplots(figsize=(10,10), ncols=2, nrows=2)
axs = axs.flatten()
ax = axs[0]
plot_fig(model, fig, ax)
ax.set_title("MT Evidential Network")
##########################################################################
######## GP regression    ################################################
##########################################################################
##########################################################################
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF
krn = 1.0*Matern(length_scale_bounds='fixed', nu=0.05)
gpr = GaussianProcessRegressor(kernel=krn, random_state=1000).fit(X_t,Y_t)
test_x = np.expand_dims(np.linspace(-3,10,1000), axis=1)

pred_y, std_y = gpr.predict(test_x, return_std=True)
pred_y = pred_y.flatten()



ax = axs[1]
fig, ax = plot_other(fig, ax, X, Y, X_outliers, Y_outliers, test_x, pred_y, std_y, train_boundary_A, train_boundary_B)
fig, ax = plot_confidence(fig, ax, pred_y, std_y, test_x, 30, [0.9, 0.7, 0.5, 0.3])
ax.set_title("GP Regression")
##########################################################################
######## botorchGP regression ############################################
##########################################################################
##########################################################################
import botorch
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import RBFKernel, ScaleKernel
import matplotlib.pyplot as plt

# Convert data to torch tensors
X_t_tensor = torch.tensor(X_t, dtype=torch.double)
Y_t_tensor = torch.tensor(Y_t, dtype=torch.double)

model = SingleTaskGP(X_t_tensor, Y_t_tensor, outcome_transform=botorch.models.transforms.outcome.Standardize(m=1))

# Fit the model
mll = ExactMarginalLogLikelihood(model.likelihood, model)
botorch.fit_gpytorch_mll(mll)

# Test data
test_x = torch.tensor(np.linspace(-3, 10, 1000).reshape(-1, 1), dtype=torch.double)

# Make predictions
model.eval()
with torch.no_grad():
    posterior = model.posterior(test_x)
    pred_y = posterior.mean.squeeze().numpy()
    std_y = posterior.variance.sqrt().squeeze().numpy()

# Plot predictions and confidence intervals
ax = axs[2]
fig, ax = plot_other(fig, ax, X, Y, X_outliers, Y_outliers, test_x, pred_y, std_y, train_boundary_A, train_boundary_B)
fig, ax = plot_confidence(fig, ax, pred_y, std_y, test_x, 30, [0.9, 0.7, 0.5, 0.3])
ax.set_title("BoTorch GP Regression")


##################################################
### vanilla evi net  #############################
##################################################
##################################################
model = EvidentialNetwork(dim)
objective = EvidentialnetMarginalLikelihood()
objective_mse = torch.nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1E-3)
scheduler = torch.optim.lr_scheduler.StepLR(opt, 1, gamma=0.99)

total_mse = 0.
total_valid_mse = 0.
total_nll = 0.
total_valid_nll = 0.

for epoch in range(epochs):
    model.train()
    for x, y in train_loader:
        opt.zero_grad()
        gamma, nu, alpha, beta = model(x.float())
        
        loss = objective(gamma, nu, alpha, beta, y).mean()
        loss += reg(gamma, nu, alpha, beta, y).mean()
        nll_history.append(loss.item())
        total_nll += loss.item()
        
        loss.backward()
        opt.step()
    
    cur_mse = total_mse/len(train_loader)
    cur_nll = total_nll/len(train_loader)
    
    model.eval()
    
    for x_v, y_v in valid_loader:
        gamma, nu, alpha, beta = model(x_v.float())
        loss_v = objective_mse(gamma, y_v)
        
        mse_history_v.append(loss_v.item())
        total_valid_mse += loss_v.item()
        
        loss_v = objective(gamma, nu, alpha, beta, y_v).mean()
        nll_history_v.append(loss_v.item())
        total_valid_nll += loss_v.item()
        
    cur_valid_mse = total_valid_mse/len(valid_loader)
    cur_valid_nll = total_valid_nll/len(valid_loader)
    
    print("Epoch {}: Train loss [{:.5f}] Val loss [{:.5f}] Train NLL [{:.5f}] Val NLL [{:.5f}]".format(
        epoch+1, cur_mse, cur_valid_mse, cur_nll, cur_valid_nll))
    
    total_mse = 0.
    total_valid_mse = 0.
    total_nll = 0.
    total_valid_nll = 0.
    scheduler.step()
##################################################
true_y = []
pred_y = []
std_y = []
epi_y = []
alea_y = []
freedom_y = []

for x in test_x:
    x = torch.Tensor([x])
    true_y = true_y + list(y.cpu().numpy().flatten())
    gamma, nu, alpha, beta = model(x.float())
    std_y += list(np.sqrt((beta*(1+nu)/(alpha*nu)).cpu().detach().numpy().flatten()))
    alea_y += list(np.sqrt((beta/(alpha)).cpu().detach().numpy().flatten()))
    epi_y += list(np.sqrt((beta/((alpha)*nu)).cpu().detach().numpy().flatten()))
    freedom_y += list((2*alpha).cpu().detach().numpy().flatten())
    pred_y = pred_y + list(gamma.cpu().detach().numpy().flatten())
    
ax = axs[3]
fig, ax = plot_other(fig, ax, X, Y, X_outliers, Y_outliers, test_x, pred_y, std_y, train_boundary_A, train_boundary_B)
fig, ax = plot_confidence(fig, ax, pred_y, std_y, test_x, 30, [0.9, 0.7, 0.5, 0.3])
ax.set_title("Vanilla Evidential Network")

#save the figure
fig.tight_layout()
fig.savefig("synthetic_result/full_comp.png")

##########################################################################
##### MC-Dropout #########################################################
##########################################################################
##########################################################################
# mse_history = []
# mse_history_mc = []
# nll_history_mc = []

# total_mse = 0.
# total_valid_mse = 0.
# total_valid_nll = 0.

# model = Network(dim)

# objective_mse = torch.nn.MSELoss()
# opt = torch.optim.Adam(model.parameters(), lr=0.005)#, weight_decay=1E-4)
# scheduler = torch.optim.lr_scheduler.StepLR(opt, 1, gamma=0.99)
# model.train()
# for epoch in range(epochs):
#     for x, y in train_loader:
#         model.train()
#         opt.zero_grad()
#         mu = model(x.float())

#         loss = objective_mse(mu, y.float())
#         mse_history.append(loss.item())
#         total_mse += loss.item()
#         loss.backward()
#         opt.step()

#         model.eval()
#     cur_mse = total_mse/len(train_loader)
#     model.eval()
#     for x_v, y_v in valid_loader:
#         mu, std = model.forward_s(x_v.float(), s=5)
#         loss_v = objective_mse(mu, y_v)
#         total_valid_mse += loss_v.item()
#         mu = mu.cpu().detach().numpy()
#         std = std.cpu().detach().numpy()
#         y_v = y_v.cpu().detach().numpy() 
#         loss_v = -log_likelihood(y_v, mu, std).mean()
#         total_valid_nll += loss_v.item()

#     cur_valid_mse = total_valid_mse/len(valid_loader)
#     cur_valid_nll = total_valid_nll/len(valid_loader)

#     print("Epoch {}: Train loss [{:.5f}] Val loss [{:.5f}] Val NLL [{:.5f}]".format(
#         epoch+1, cur_mse, cur_valid_mse, cur_valid_nll))

#     if cur_valid_nll != math.nan:
#         mse_history_mc.append(cur_valid_mse)
#         nll_history_mc.append(cur_valid_nll)
#     total_mse = 0.
#     total_valid_mse = 0.
#     total_nll = 0.
#     total_valid_nll = 0.
#     scheduler.step()
#     #############################################################################


# true_y = []
# pred_y = []
# std_y = []
# model.eval()
# for x in test_x:
#     x = torch.Tensor([x])
#     true_y = true_y + list(y.cpu().numpy().flatten())
#     mu, std = model.forward_s(x.float())
#     pred_y += list(mu.cpu().detach().numpy())
#     std_y += list(std.cpu().detach().numpy())
#  
# fig, ax = plt.subplots(figsize=(10,10))
# fig, ax = plot_other(fig, ax, X, Y, X_outliers, Y_outliers, test_x, pred_y, std_y, train_boundary_A, train_boundary_B)
# fig, ax = plot_confidence(fig, ax, pred_y, std_y, test_x, 30, [0.9, 0.7, 0.5, 0.3])
# plt.tight_layout()
# plt.savefig("synthetic_result/mcdrop.png")
