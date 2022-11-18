import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt


import functools
import os
import time
import operator
from functools import reduce
from functools import partial
from timeit import default_timer
from utilities3 import *
import scipy

import theseus as th

import deepxde as dde

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


from torch.utils.data import Dataset

import deepxde as dde

print("\n=============================")
print("torch.cuda.is_available(): " + str(torch.cuda.is_available()))
if torch.cuda.is_available():
    print("torch.cuda.get_device_name(0): " + str(torch.cuda.get_device_name(0)))
print("=============================\n")


class BurgersDeepONetDataset(Dataset):
    def __init__(self, path) -> None:
        super().__init__()
        dataloader = MatReader(path, to_cuda=False)
        self.ic_vals = dataloader.read_field("input")  # Initial conditions
        self.solutions = dataloader.read_field("output")  # Solution
        self.gamma = dataloader.read_field("gamma")
        self.sigma = dataloader.read_field("sigma")
        self.tau = dataloader.read_field("tau")
        self.tspan = dataloader.read_field("tspan")

        print("Loaded data")
        nx = self.ic_vals.shape[1]
        nt = self.tspan.shape[1]

        self.grid = torch.from_numpy(np.mgrid[0 : 1 : 1 / nx, 0 : 1 : 1 / nt]).permute(
            1, 2, 0
        )  # Make shape (nx, nt, 2)

        self.solutions = self.solutions.permute(
            0, 2, 1
        )  # Need to transpose x, t for some reason

    def __len__(self):
        return len(self.ic_vals)

    def __getitem__(self, index):
        return (self.ic_vals[index].float(), self.grid.float()), self.solutions[
            index, ..., None
        ].float()


def reduce_last_dim(mat, w):
    return (mat * w[:, None]).sum(-1)
    


class LitDeepOnet(pl.LightningModule):
    def __init__(self, lr, nx, n_samples=2500, viscosity=1e-2, constrain=False, N_basis=600):
        super().__init__()
        self.lr = lr
        self.nx = nx
        self.viscosity = viscosity
        self.n_samples = n_samples
        self.constrain = constrain
        self.N_basis = N_basis

        self.trunk_net = nn.Sequential(
            nn.Linear(2, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128 if not self.constrain else N_basis),
        )

        self.branch_net = nn.Sequential(
            nn.Linear(nx, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
        )

        self.save_hyperparameters()


    def fit_linear_combo(self, ic_vals, grid):
        # Build least squares objective

        # Forward pass on the basis functions
        batch_size = grid.shape[0]
        idx = torch.stack([torch.randperm(self.nx * self.nx)[:self.n_samples] for i in range(batch_size)], 0)
        flat_grid = grid.reshape(batch_size, -1, 2)

        # Make this better
        flat_sampled_grid = torch.stack([
            flat_grid[k][idx[k]]
            for k in range(batch_size)
        ], 0)
        flat_sampled_grid = flat_sampled_grid.reshape(-1, 2).detach() # Need to make it flat to compute jacobians

        flat_sampled_grid.requires_grad = True
        pred = self.trunk_net(flat_sampled_grid)

        residual, dy_x, dy_t, dy_xx = self.pde(flat_sampled_grid, pred)
        residual = residual.reshape(batch_size, self.n_samples, self.N_basis)
        dy_x =  dy_x.reshape(batch_size, self.n_samples, self.N_basis)
        dy_t = dy_t.reshape(batch_size, self.n_samples, self.N_basis)
        dy_xx = dy_xx.reshape(batch_size, self.n_samples, self.N_basis)
        pred = pred.reshape(batch_size, self.n_samples, self.N_basis)

        def quad_error_fn(optim_vars, aux_vars):
            # TODO: Add IC / BC 
            y, dy_x, dy_t, dy_xx = aux_vars 
            w, = optim_vars
            y_tensor, dy_x_tensor, dy_t_tensor, dy_xx_tensor = map(functools.partial(reduce_last_dim, w=w.tensor), (y.tensor, dy_x.tensor, dy_t.tensor, dy_xx.tensor))
            residual_tensor = dy_t_tensor + y_tensor * dy_x_tensor - self.viscosity / np.pi * dy_xx_tensor

            return residual_tensor

        def ridge_error_fn(optim_vars, aux_vars):
            del aux_vars 
            w, = optim_vars
            # TODO: ridge scale as a model parameter
            return 1e-7 * w.tensor.reshape(batch_size, self.N_basis)
        
        objective = th.Objective()
        objective.device = pred.device
        w_th = th.Vector(self.N_basis, name='w')  # Linear combination weight 
        y_th = th.Variable(pred, 'y')
        dy_x_th = th.Variable(dy_x, 'dy_x')
        dy_t_th = th.Variable(dy_t, 'dy_t')
        dy_xx_th = th.Variable(dy_xx, 'dy_xx') 

        optim_vars = [w_th]
        aux_vars = [y_th, dy_x_th, dy_t_th, dy_xx_th]

        cost_function = th.AutoDiffCostFunction(
            optim_vars, quad_error_fn, dim=self.n_samples,  # TODO: Update to n_samples + n_ic + n_bc
             aux_vars=aux_vars, name="pde_residual_cost_fn"
        )
        ridge_function = th.AutoDiffCostFunction(
            optim_vars, quad_error_fn, dim=self.N_basis,  # TODO: Update to n_samples + n_ic + n_bc
             aux_vars=aux_vars, name="ridge"
        )
        objective = th.Objective()
        objective.to(pred.device)
        objective.add(cost_function)
        # objective.add(ridge_function)

        optimizer = th.LevenbergMarquardt(
            objective,
            max_iterations=50,
            step_size=0.5,
        )
        theseus_optim = th.TheseusLayer(optimizer)

        theseus_inputs = {
            "dy_x": dy_x,
            "dy_t": dy_t,
            "dy_xx": dy_xx,
            "y": pred,
            "w": torch.rand(batch_size, self.N_basis, device=pred.device)
        }

        updated_inputs, info = theseus_optim.forward(theseus_inputs)

        w = updated_inputs['w']
        return w


    def flat_forward(self, ic_vals, grid):
        """
        Args:
          ic_vals: shape (batch, nx)
          grid: shape (-1, 2)
        Returns:
          pred: shape (-1, 1)
        """
        grid = grid.reshape(-1, self.nx, self.nx, 2)
        batch_size = grid.shape[0]
        if self.constrain:
            trunk_rep = self.trunk_net(grid).reshape(batch_size, -1, self.N_basis)
            w = self.fit_linear_combo(ic_vals, grid)  # This is the branch rep
            return reduce_last_dim(trunk_rep, w).reshape(-1, 1)

        else:
            trunk_rep = self.trunk_net(grid)
            branch_rep = self.branch_net(ic_vals)[:, None, None]
            output = (trunk_rep * branch_rep).sum(-1, keepdim=True)
        return output.reshape(-1, 1)

    def forward(self, ic_vals, grid):
        """
        Args:
          ic_vals: shape (batch, nx)
          grid: shape (batch, nx, nt, 2)
        Returns:
          pred: shape (batch, nx, nt, 1)
        """
        trunk_rep = self.trunk_net(grid)
        branch_rep = self.branch_net(ic_vals)[:, None, None]
        return (trunk_rep * branch_rep).sum(-1, keepdim=True)

    def pde(self, x, y):
        N_basis = y.shape[-1]
        if N_basis == 1:
            dy_x = dde.grad.jacobian(y, x, i=0, j=0)
            dy_t = dde.grad.jacobian(y, x, i=0, j=1)
            dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        else:
            dy_x = torch.stack([dde.grad.jacobian(y, x, i=i, j=0) for i in range(N_basis)], -1).squeeze(1)
            dy_t = torch.stack([dde.grad.jacobian(y, x, i=i, j=1) for i in range(N_basis)], -1).squeeze(1)
            dy_xx = torch.stack([dde.grad.hessian(y, x, component=k, i=0, j=0) for k in range(N_basis)], -1).squeeze(1)
        
        dde.gradients.clear()
        return dy_t + y * dy_x - self.viscosity / np.pi * dy_xx, dy_x, dy_t, dy_xx

    def periodic_bc_loss(self, y, dy_x):
        batch_size = y.shape[0]
        zero_order = (y[:, 0] - y[:, -1]).reshape(batch_size, -1)
        first_order = (dy_x[:, 0] - dy_x[:, -1]).reshape(batch_size, -1)
        dim = zero_order.shape[-1]
        return (
            1
            / dim
            * (
                nn.MSELoss()(zero_order, torch.zeros_like(zero_order))
                + nn.MSELoss()(first_order, torch.zeros_like(first_order))
            )
        )

    def training_step(self, batch, batch_idx):
        (ic_vals, grid), target = batch
        grid_shape = grid.shape
        pred_shape = grid_shape[:-1] + (1,)
        grid = grid.reshape(-1, 2)
        with torch.enable_grad():
            grid.requires_grad = True
            # Required to compute the PDE residual correctly
            pred = self.flat_forward(ic_vals, grid)
            # TODO: Compute IC/BC/RES losses
            residual, dy_x, dy_t, dy_xx = self.pde(grid, pred)
        batch_size = grid_shape[0]
        residual = residual.reshape(batch_size, -1)
        residual_loss = (
            1 / residual.shape[-1] * nn.MSELoss()(residual, torch.zeros_like(residual))
        )

        bc_loss = self.periodic_bc_loss(
            pred.reshape(*pred_shape), dy_x.reshape(*pred_shape)
        )

        ic_loss = nn.MSELoss()(
            ic_vals, pred.reshape(pred_shape)[:, :, 0].reshape(batch_size, -1)
        )

        batch_size = pred.shape[0]
        mse_loss = nn.MSELoss()(
            pred.reshape(batch_size, -1), target.reshape(batch_size, -1)
        )

        self.log("train_ic_loss", ic_loss.item())
        self.log("train_bc_loss", bc_loss.item())
        self.log("train_mse_loss", mse_loss.item())
        self.log("train_residual_loss", residual_loss.item())
        train_loss = residual_loss + ic_loss + bc_loss
        self.log("train_loss", train_loss.item())
        return train_loss
        return mse_loss

    def validation_step(self, batch, batch_idx):
        (ic_vals, grid), target = batch
        grid_shape = grid.shape
        pred_shape = grid_shape[:-1] + (1,)
        grid = grid.reshape(-1, 2)
        with torch.enable_grad():
            grid.requires_grad = True
            # Required to compute the PDE residual correctly
            pred = self.flat_forward(ic_vals, grid)
            # TODO: Compute IC/BC/RES losses
            residual, dy_x, dy_t, dy_xx = self.pde(grid, pred)
        batch_size = grid_shape[0]
        residual = residual.reshape(batch_size, -1)
        residual_loss = nn.MSELoss()(residual, torch.zeros_like(residual))

        bc_loss = self.periodic_bc_loss(
            pred.reshape(*pred_shape), dy_x.reshape(*pred_shape)
        )

        ic_loss = (
            1
            / pred.shape[-1]
            * nn.MSELoss()(
                ic_vals, pred.reshape(pred_shape)[:, :, 0].reshape(batch_size, -1)
            )
        )

        batch_size = pred.shape[0]
        mse_loss = nn.MSELoss()(
            pred.reshape(batch_size, -1), target.reshape(batch_size, -1)
        )

        self.log("val_ic_loss", ic_loss.item())
        self.log("val_bc_loss", bc_loss.item())
        self.log("val_mse_loss", mse_loss.item())
        self.log("val_residual_loss", residual_loss.item())

        self.figure_dir = os.path.join(self.trainer.log_dir, "figures")
        os.makedirs(self.figure_dir, exist_ok=True)
        fig, axes = plt.subplots(ncols=3)

        vmin = min(pred.reshape(pred_shape)[0].min(), target[0].min()).cpu()
        vmax = max(pred.reshape(pred_shape)[0].min(), target[0].max()).cpu()

        axes[0].imshow(
            pred.reshape(pred_shape)[0].reshape(self.nx, self.nx).cpu(),
            vmin=vmin,
            vmax=vmax,
            cmap="RdBu",
        )
        axes[0].set_title("Pred")
        axes[1].imshow(
            target[0].reshape(self.nx, self.nx).cpu(), vmin=vmin, vmax=vmax, cmap="RdBu"
        )
        axes[1].set_title("Target")
        axes[2].imshow(
            (pred.reshape(pred_shape) - target)[0].reshape(self.nx, self.nx).cpu(),
            vmin=-max(abs(vmin), abs(vmax)),
            vmax=max(abs(vmin), abs(vmax)),
            cmap="RdBu",
        )
        axes[2].set_title("Difference")
        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.figure_dir, f"prediction_epoch_{self.current_epoch}")
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


def periodic(x):
    x *= 2 * np.pi
    return torch.concatenate(
        [torch.cos(x), torch.sin(x), torch.cos(2 * x), torch.sin(2 * x)], 1
    )


def DeepONet_main(train_data_res, save_index, if_constrain=False):
    """
    Parameters
    ----------
    train_data_res : resolution of the training data
    save_index : index of the saving folder
    """

    ################################################################
    #  configurations
    ################################################################

    s = train_data_res
    # sub = 2**6 #subsampling rate
    sub = 2**13 // s  # subsampling rate (step size)

    batch_size = 2
    learning_rate = 1e-3

    epochs = 500  # default 500
    step_size = 100  # default 100
    gamma = 0.5

    modes = 16
    width = 64

    ################################################################
    # read training data
    ################################################################

    train_set = BurgersDeepONetDataset(
        "/home/ubuntu/deeponet-fno/data/burgers/burgers_train_1000.mat"
    )
    # TODO: generate and change path
    val_set = BurgersDeepONetDataset(
        "/home/ubuntu/deeponet-fno/data/burgers/Burgers_test_10.mat"
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator("cuda"),
    )
    test_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False
    )

    model = LitDeepOnet(lr=learning_rate, nx=101, n_samples=750, N_basis=75, constrain=True)

    trainer = pl.Trainer(
        accelerator="gpu",
        gpus=1,
        callbacks=[EarlyStopping(monitor="val_mse_loss", mode="min", patience=3)],
        check_val_every_n_epoch=5,
    )
    trainer.fit(
        model=model, train_dataloaders=train_loader, val_dataloaders=test_loader
    )


if __name__ == "__main__":

    training_data_resolution = 128
    save_index = 0

    DeepONet_main(training_data_resolution, save_index)
