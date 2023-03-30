import argparse

from torch.func import jacfwd, vmap, jacrev

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


import functools
import os
from utilities3 import *


import theseus as th

import deepxde as dde

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


from torch.utils.data import Dataset

import deepxde as dde

import wandb
from pytorch_lightning.loggers import WandbLogger

# torch.set_default_dtype(torch.float64)

print("\n=============================")
print("torch.cuda.is_available(): " + str(torch.cuda.is_available()))
if torch.cuda.is_available():
    print("torch.cuda.get_device_name(0): " + str(torch.cuda.get_device_name(0)))
print("=============================\n")


class BurgersDeepONetDataset(Dataset):
    def __init__(self, path, n_samples_interior) -> None:
        super().__init__()
        self.n_samples_interior = n_samples_interior

        dataloader = MatReader(path, to_cuda=False)
        self.ic_vals = dataloader.read_field("input")  # Initial conditions
        self.solutions = dataloader.read_field("output")  # Solution
        self.gamma = dataloader.read_field("gamma")
        self.sigma = dataloader.read_field("sigma")
        self.tau = dataloader.read_field("tau")
        self.tspan = dataloader.read_field("tspan")

        print("Loaded data")
        self.nx = self.ic_vals.shape[1]
        self.nt = self.tspan.shape[1]

        self.grid = torch.from_numpy(
            np.mgrid[0 : 1 : 1 / self.nx, 0 : 1 : 1 / self.nt]
        ).permute(
            1, 2, 0
        )  # Make shape (nx, nt, 2)

        self.solutions = self.solutions.permute(
            0, 2, 1
        )  # Need to transpose x, t for some reason

    def __len__(self):
        return len(self.ic_vals)

    def __getitem__(self, index):
        # get random index:
        random_grid_index = np.random.randint(
            0, self.nx * self.nt, size=(self.n_samples_interior,)
        )
        sampled_grid = self.grid.reshape(-1, 2)[random_grid_index]

        return (
            # ic: values and x grid
            self.ic_vals[index],  # (nx,)
            self.grid[:, 0, :],  # (nx, 2)
            # bcs: t grid at x=0 and x=1
            torch.stack(
                [torch.zeros(self.nt, device=self.grid.device), self.grid[0, :, 1]],
                dim=1,
            ),  # (nt, 2)
            torch.stack(
                [torch.ones(self.nt, device=self.grid.device), self.grid[-1, :, 1]],
                dim=1,
            ),  # (nt, 2)
            # interior: sampled points for x and t
            sampled_grid,  # (n_samples_interior, 2)
            # full solution and grid
            self.solutions[index].reshape(-1, 1),  # (nx * nt, 1)
            self.grid.reshape(-1, 2),  # (nx * nt, 2)
        )


def reduce_last_dim(mat, w):
    return (mat * w[:, None]).sum(-1)


class ConstrainedModel(nn.Module):
    def __init__(
        self,
        N_basis,
        n_layers,
        viscosity,
        ridge,
        nsamples_constraint,
        nsamples_residual,
        max_iterations=200,
    ):
        super().__init__()
        self.N_basis = N_basis
        self.viscosity = viscosity
        self.ridge = ridge
        self.nsamples_constraint = nsamples_constraint
        self.nsamples_residual = nsamples_residual
        self.max_iterations = max_iterations

        self.trunk_net = nn.Sequential(
            nn.Linear(2, 100), *([nn.Tanh(), nn.Linear(100, 100)] * n_layers)
        )

        self.save_hyperparameters()

    def burgers(self, y, dy_x, dy_t, dy_xx):
        return dy_t + y * dy_x - self.viscosity * dy_xx

    def compute_all_necessary_partials_and_values(
        self, ic_grid, sampled_grid, bc_grid_start, bc_grid_end
    ):
        """Compute all necessary partials and values for the loss function.

        Args:
            ic_grid (torch.Tensor): Grid of points for the initial condition
            sampled_grid (torch.Tensor): Grid of points for the interior
            bc_grid_start (torch.Tensor): Grid of points for the boundary condition at x=0
            bc_grid_end (torch.Tensor): Grid of points for the boundary condition at x=1

        Returns:
            tuple: (y_interior, dy_x, dy_t, dy_xx, ic_pred, bc_pred_start, bc_pred_end, bc_dy_x_start, bc_dy_x_end)
        """
        dy = vmap(jacfwd(self.trunk_net))(sampled_grid)
        dy_x = dy[:, 0]
        dy_t = dy[:, 1]

        dy_xx = vmap(jacrev(jacfwd(self.trunk_net)))(sampled_grid)[:, 0, 0]

        y_interior = self.trunk_net(sampled_grid)

        # Compute predictions on IC:
        ic_pred = self.trunk_net(ic_grid)

        # Compute predictions on BC:
        bc_pred_start = self.trunk_net(bc_grid_start)
        bc_pred_end = self.trunk_net(bc_grid_end)

        # Compute derivatives on BC:
        bc_dy_x_start = vmap(jacfwd(self.trunk_net))(bc_grid_start)[:, 0]
        bc_dy_x_end = vmap(jacfwd(self.trunk_net))(bc_grid_end)[:, 0]

        # Concat predictions and derivatives:
        bc_start = torch.cat([bc_pred_start, bc_dy_x_start], dim=1)
        bc_end = torch.cat([bc_pred_end, bc_dy_x_end], dim=1)
        bc_diff = bc_start - bc_end

        return (
            ic_pred,
            bc_diff,
            y_interior,
            dy_x,
            dy_t,
            dy_xx,
        )

    def fit_last_layer(
        self, ic_vals, ic_grid, sampled_grid, bc_grid_start, bc_grid_end
    ):
        """
        Builds and solves the constrained optimization problem.

        Args:
            ic_vals: Initial conditions. Shape (nx,)
            ic_grid: Grid for the initial conditions. Shape (nx, 2)
            sampled_grid: Sampled points for the interior. Shape (n_samples_interior, 2)
            bc_grid_start: Boundary conditions grid at x=0. Shape (nt, 2)
            bc_grid_end: Boundary conditions grid at x=1. Shape (nt, 2)

        Returns:
            w: The weights for the last layer. Shape (N_basis,)
        """

        # Compute partials and predictions on interior
        (
            ic_pred,
            bc_diff,
            y_interior,
            dy_x,
            dy_t,
            dy_xx,
        ) = self.compute_all_necessary_partials_and_values(
            ic_grid, sampled_grid, bc_grid_start, bc_grid_end
        )

        # We now have all the variables we need to setup the
        # nonlinear least squares problem.
        # We use theseus to solve it.

        # Setup loss functions

        def pde_residual_error_fn(optim_vars, aux_vars):
            y, dy_x, dy_t, dy_xx = aux_vars
            (w,) = optim_vars
            y_tensor, dy_x_tensor, dy_t_tensor, dy_xx_tensor = map(
                functools.partial(reduce_last_dim, w=w.tensor),
                (y.tensor, dy_x.tensor, dy_t.tensor, dy_xx.tensor),
            )
            residual_tensor = (
                dy_t_tensor + y_tensor * dy_x_tensor - self.viscosity * dy_xx_tensor
            )

            return residual_tensor

        def ic_error_fn(optim_vars, aux_vars):
            pred_ic, ic_vals = aux_vars
            (w,) = optim_vars
            y_tensor = reduce_last_dim(pred_ic.tensor, w.tensor)
            return y_tensor - ic_vals.tensor

        def bc_error_fn(optim_vars, aux_vars):
            (bc_diff,) = aux_vars
            (w,) = optim_vars
            diff_tensor = reduce_last_dim(bc_diff.tensor, w.tensor)
            return diff_tensor

        def ridge_error_fn(optim_vars, aux_vars):
            del aux_vars
            (w,) = optim_vars
            return w.tensor

        objective = th.Objective()

        w_th = th.Vector(
            self.N_basis, name="w", dtype=torch.double
        )  # Linear combination weight

        y_th = th.Variable(y_interior.double(), "y")
        dy_x_th = th.Variable(dy_x.double(), "dy_x")
        dy_t_th = th.Variable(dy_t.double(), "dy_t")
        dy_xx_th = th.Variable(dy_xx.double(), "dy_xx")

        ic_vals_th = th.Variable(ic_vals.double(), "ic_vals")
        ic_vals_pred_th = th.Variable(ic_pred.double(), "ic_vals_pred")

        bc_diff_th = th.Variable(bc_diff.double(), "bc_diff")

        optim_vars = [w_th]

        pde_cost_function = th.AutoDiffCostFunction(
            optim_vars,
            pde_residual_error_fn,
            dim=self.n_samples_constraint,
            aux_vars=[y_th, dy_x_th, dy_t_th, dy_xx_th],
            name="pde_residual_cost_fn",
            cost_weight=th.ScaleCostWeight(1.0, dtype=torch.double),
        )
        ic_cost_function = th.AutoDiffCostFunction(
            optim_vars,
            ic_error_fn,
            dim=self.nx,
            aux_vars=[ic_vals_pred_th, ic_vals_th],
            name="ic_cost_fn",
            cost_weight=th.ScaleCostWeight(20.0, dtype=torch.double),
        )
        bc_cost_function = th.AutoDiffCostFunction(
            optim_vars,
            bc_error_fn,
            dim=self.nx * 2,
            aux_vars=[bc_diff_th],
            name="bc_cost_fn",
            cost_weight=th.ScaleCostWeight(1.0, dtype=torch.double),
        )

        ridge_cost_function = th.AutoDiffCostFunction(
            optim_vars,
            ridge_error_fn,
            dim=self.N_basis,
            aux_vars=[],
            name="ridge_cost_fn",
            cost_weight=th.ScaleCostWeight(self.ridge, dtype=torch.double),
        )
        objective = th.Objective(dtype=torch.double)
        objective.add(pde_cost_function)
        objective.add(ic_cost_function)
        objective.add(bc_cost_function)
        objective.add(ridge_cost_function)

        objective.to(ic_pred.device)

        optimizer = th.LevenbergMarquardt(
            objective,
            max_iterations=self.max_iterations,
            step_size=0.5,
        )

        theseus_optim = th.TheseusLayer(optimizer)

        theseus_inputs = {
            "dy_x": dy_x.double(),
            "dy_t": dy_t.double(),
            "dy_xx": dy_xx.double(),
            "bc_diff": bc_diff.double(),
            "ic_vals": ic_vals.double(),
            "ic_vals_pred": ic_pred.double(),
            "y": y_interior.double(),
            "w": torch.rand(self.N_basis, device=ic_pred.device, dtype=torch.double),
        }

        updated_inputs, info = theseus_optim.forward(theseus_inputs)
        # print(info)

        w = updated_inputs["w"].float()
        return w


    def forward_with_w(self, sampled_grid, w):
        trunk_rep = self.trunk_net(sampled_grid)
        y = reduce_last_dim(trunk_rep, w)
        return y

    def forward(self, ic_vals, ic_grid, bc_grid_start, bc_grid_end, interior_grid, full_grid):
        """
        Forward pass of the constrained network.

        Args:
            ic_vals: Initial condition values. Shape: (batch_size, nx)
            ic_grid: Initial condition grid. Shape: (batch_size, nx, 2)
            bc_grid_start: Boundary condition grid at x=0. Shape: (batch_size, nt, 2)
            bc_grid_end: Boundary condition grid at x=1. Shape: (batch_size, nt, 2)
            interior_grid: Interior grid. Shape: (batch_size, n_samples, 2)
            full_grid: Full grid. Shape: (batch_size, nx * nt, 2)
        """
        # points for fitting w:
        constraint_grid = interior_grid[:, :self.n_samples_constraint]
        # points for computing residual loss:
        residual_grid = interior_grid[:, -self.n_samples_residual:]

        # get w for each datapoint in the batch:
        # w shape: (batch_size, N_basis)
        w = vmap(self.fit_last_layer)(ic_vals, ic_grid, constraint_grid, bc_grid_start, bc_grid_end)

        # Prediction, for eval and plotting only
        with torch.no_grad():
            y_full = vmap(self.forward_with_w)(full_grid, w)

        # Compute residual for the interior grid
        (
            ic_pred,
            bc_diff,
            y_interior,
            dy_x,
            dy_t,
            dy_xx,
        ) = self.compute_all_necessary_partials_and_values(
            ic_grid, residual_grid, bc_grid_start, bc_grid_end
        )

        # Reduce the partials using w
        y_interior = reduce_last_dim(y_interior, w)
        dy_x = reduce_last_dim(dy_x, w)
        dy_t = reduce_last_dim(dy_t, w)
        dy_xx = reduce_last_dim(dy_xx, w)

        # Compute residual loss
        pde_residual = self.burgers(y_interior, dy_x, dy_t, dy_xx)

        return (
            y_full,
            ic_pred,
            bc_diff,
            pde_residual
        )

class DeepONet(nn.Module):
    def __init__(
        self,
        n_layers,
        viscosity,
        nsamples_residual,
        nx
    ):
        super().__init__()
        self.viscosity = viscosity
        self.nsamples_residual = nsamples_residual
        self.nx = nx
        self.n_layers = n_layers

        self.trunk_net = nn.Sequential(
            nn.Linear(2, 100), *([nn.Tanh(), nn.Linear(100, 100)] * n_layers)
        )

        self.branch_net = nn.Sequential(
            nn.Linear(self.nx, 100), *([nn.Tanh(), nn.Linear(100, 100)] * n_layers)
        )

    def burgers(self, y, dy_x, dy_t, dy_xx):
        return dy_t + y * dy_x - self.viscosity * dy_xx

    def compute_predictions(self, ic_vals, grid):
        w = self.branch_net(ic_vals)[None]
        trunk_rep = self.trunk_net(grid)
        y = (trunk_rep * w).sum(-1, keepdims=True)
        return y
    
    # @torch.compile
    def compute_all_necessary_partials_and_values(self, ic_vals, ic_grid, sampled_grid, bc_grid_start, bc_grid_end):
        # Inner vmap is over the grid points
        # Outer vmap is over the batch
        dy = vmap(vmap(jacrev(self.compute_predictions, argnums=1), (None, 0)))(ic_vals, sampled_grid)
        dy_x = dy[:, 0]
        dy_t = dy[:, 1]

        dy_xx = vmap(vmap(jacfwd(jacrev(self.compute_predictions, argnums=1), argnums=1), (None, 0)))(ic_vals, sampled_grid)[:, 0, 0]

        y_interior = vmap(vmap(self.compute_predictions, (None, 0)))(ic_vals, sampled_grid)

        # Compute predictions on IC:
        ic_pred = vmap(vmap(self.compute_predictions, (None, 0)))(ic_vals, ic_grid)

        # Compute predictions on BC:
        bc_pred_start = vmap(vmap(self.compute_predictions, (None, 0)))(ic_vals, bc_grid_start)
        bc_pred_end = vmap(vmap(self.compute_predictions, (None, 0)))(ic_vals, bc_grid_end)

        # Compute derivatives on BC:
        bc_dy_x_start = vmap(vmap(jacrev(self.compute_predictions, argnums=1), (None, 0)))(ic_vals, bc_grid_start)[..., 0]
        bc_dy_x_end = vmap(vmap(jacrev(self.compute_predictions, argnums=1), (None, 0)))(ic_vals, bc_grid_end)[..., 0]

        # Concat predictions and derivatives:
        bc_start = torch.cat([bc_pred_start, bc_dy_x_start], dim=1)
        bc_end = torch.cat([bc_pred_end, bc_dy_x_end], dim=1)
        bc_diff = bc_start - bc_end

        return (
            ic_pred,
            bc_diff,
            y_interior,
            dy_x,
            dy_t,
            dy_xx,
        ) 


    def forward(self, ic_vals, ic_grid, bc_grid_start, bc_grid_end, interior_grid, full_grid):
        """
        Forward pass of the unconstrained network.

        Args:
            ic_vals: Initial condition values. Shape: (batch_size, nx)
            ic_grid: Initial condition grid. Shape: (batch_size, nx, 2)
            bc_grid_start: Boundary condition grid at x=0. Shape: (batch_size, nt, 2)
            bc_grid_end: Boundary condition grid at x=1. Shape: (batch_size, nt, 2)
            interior_grid: Interior grid. Shape: (batch_size, n_samples, 2)
            full_grid: Full grid. Shape: (batch_size, nx * nt, 2)
        """
        # points for computing residual loss:
        residual_grid = interior_grid[:, -self.nsamples_residual:]

        # Prediction, for visualization/evaluation only
        # with torch.no_grad():
        #     y_full = vmap(self.compute_predictions)(ic_vals, full_grid)

        y_full = torch.zeros(1, device=residual_grid.device)

        # Compute residual for the interior grid
        (
            ic_pred,
            bc_diff,
            y_interior,
            dy_x,
            dy_t,
            dy_xx,
        ) = self.compute_all_necessary_partials_and_values(
            ic_vals, ic_grid, residual_grid, bc_grid_start, bc_grid_end
        )

        # Compute residual loss
        pde_residual = self.burgers(y_interior, dy_x, dy_t, dy_xx)

        return (
            y_full,
            ic_pred,
            bc_diff,
            pde_residual
        )
    

class BurgersLitModel(pl.LightningModule):
    def __init__(
        self,
        lr=1e-3,
        lr_scheduler_step=2000,
        lr_scheduler_factor=0.9,
        nx=101,
        nt=101,
        n_samples_constraint=750,
        n_samples_residual=250,
        viscosity=1e-2,
        constrain=False,
        N_basis=600,
        n_layers=8,
        max_iterations=200,
        ridge=1e-4,
        log_freq = 20
    ):
        super().__init__()
        self.lr = lr
        self.nx = nx
        self.nt = nt
        self.viscosity = viscosity
        self.n_samples_constraint = n_samples_constraint
        self.n_samples_residual = n_samples_residual
        self.constrain = constrain
        self.N_basis = N_basis
        self.max_iterations = max_iterations
        self.ridge = ridge
        self.lr_scheduler_step = lr_scheduler_step
        self.lr_scheduler_factor = lr_scheduler_factor
        self.log_freq = log_freq

        if constrain:
            self.model = ConstrainedModel(
                N_basis,
                n_layers,
                viscosity,
                ridge,
                n_samples_constraint,
                n_samples_residual,
                max_iterations=200
            )

        else:
            self.model = DeepONet(
                n_layers,
                viscosity,
                n_samples_residual,
                nx
            )

        self.save_hyperparameters()

    def periodic_bc_loss(self, bc_diff):
        dim = bc_diff.shape[-1] / 2
        return 1 / dim * nn.MSELoss()(bc_diff, torch.zeros_like(bc_diff))
    
    def relative_error(self, y, target):
        return torch.mean(torch.linalg.norm(y - target) / torch.linalg.norm(target))
        
    def residual_loss(self, pde_residual):
        return torch.mean(pde_residual ** 2)
    
    def initial_condition_loss(self, ic_pred, ic_vals):
        return torch.mean((ic_pred.flatten() - ic_vals.flatten()) ** 2)

    def mse_loss(self, y, target):
        residual = y.flatten() - target.flatten()
        return torch.mean(residual ** 2)

    def forward(self, batch):
        """
        Computes all losses for training and logging.

        Args:
            batch: A batch of data. See `BurgersDataModule` for details.
        """
        (
            ic_vals,
            ic_grid,
            bc_grid_start,
            bc_grid_end,
            interior_grid,
            target,
            full_grid,
        ) = batch

        (
            y_full,
            ic_pred,
            bc_diff,
            pde_residual
        ) = self.model(ic_vals, ic_grid, bc_grid_start, bc_grid_end, interior_grid, full_grid)

        # Compute losses
        ic_loss = self.initial_condition_loss(ic_pred, ic_vals)
        bc_loss = self.periodic_bc_loss(bc_diff)
        residual_loss = self.residual_loss(pde_residual)
        relative_error = self.relative_error(y_full, target)
        mse_loss = self.mse_loss(y_full, target)

        return {
            "ic_loss": ic_loss,
            "bc_loss": bc_loss,
            "residual_loss": residual_loss,
            "relative_error": relative_error,
            "mse_loss": mse_loss,
        }


    def training_step(self, batch, batch_idx):
        losses = self.forward(batch)
        train_loss = 20 * losses["ic_loss"] + losses["bc_loss"] + losses["residual_loss"]

        log_dict = {"train_" + k: v.item() for k, v in losses.items()}
        log_dict["train_loss"] = train_loss.item()

        for k, v in log_dict.items():
            self.log(k, v)


        # if batch_idx % self.log_freq == 0:
        #     self.figure_dir = os.path.join(self.trainer.log_dir, "figures")
        #     os.makedirs(self.figure_dir, exist_ok=True)
        #     fig, axes = plt.subplots(ncols=3)

        #     vmin = min(pred.reshape(pred_shape)[0].min(), target[0].min()).cpu()
        #     vmax = max(pred.reshape(pred_shape)[0].min(), target[0].max()).cpu()

        #     dividers = list(map(make_axes_locatable, axes))

        #     im0 = axes[0].imshow(
        #         pred.detach().reshape(pred_shape)[0].reshape(self.nx, self.nx).cpu(),
        #         vmin=vmin,
        #         vmax=vmax,
        #         cmap="RdBu",
        #     )
        #     cax0 = dividers[0].append_axes("right", size="5%", pad=0.05)
        #     fig.colorbar(im0, cax=cax0, orientation="vertical")

        #     axes[0].set_title("Pred")
        #     im1 = axes[1].imshow(
        #         target[0].reshape(self.nx, self.nx).cpu(),
        #         vmin=vmin,
        #         vmax=vmax,
        #         cmap="RdBu",
        #     )

        #     cax1 = dividers[1].append_axes("right", size="5%", pad=0.05)
        #     fig.colorbar(im0, cax=cax1, orientation="vertical")

        #     axes[1].set_title("Target")
        #     diff = (
        #         (pred.reshape(pred_shape) - target)
        #         .detach()[0]
        #         .reshape(self.nx, self.nx)
        #         .cpu()
        #     )
        #     im2 = axes[2].imshow(
        #         diff,
        #         vmin=-abs(diff).max(),
        #         vmax=abs(diff).max(),
        #         cmap="RdBu",
        #     )

        #     cax2 = dividers[2].append_axes("right", size="5%", pad=0.05)
        #     fig.colorbar(im2, cax=cax2, orientation="vertical")

        #     axes[2].set_title("Difference")
        #     for ax in axes:
        #         ax.axis("off")

        #     plt.tight_layout()
        #     plt.savefig(
        #         os.path.join(
        #             self.figure_dir,
        #             f"prediction_train_epoch_{self.current_epoch}_batch_{batch_idx}",
        #         )
        #     )
        #     self.logger.log_image("train_plot", [fig])
        #     plt.close()

        self.lr_scheduler.step()

        return train_loss

    def validation_step(self, batch, batch_idx):  
        losses = self.forward(batch)
        val_loss = 20 * losses["ic_loss"] + losses["bc_loss"] + losses["residual_loss"]

        log_dict = {"val_" + k: v.item() for k, v in losses.items()}
        log_dict["val_loss"] = val_loss.item()

        for k, v in log_dict.items():
            self.log(k, v)

        # self.figure_dir = os.path.join(self.trainer.log_dir, "figures")
        # os.makedirs(self.figure_dir, exist_ok=True)
        # if batch_idx % 20 == 0:
        #     fig, axes = plt.subplots(ncols=3)

        #     vmin = min(pred.reshape(pred.shape)[0].min(), target[0].min()).cpu()
        #     vmax = max(pred.reshape(pred.shape)[0].min(), target[0].max()).cpu()

        #     dividers = list(map(make_axes_locatable, axes))

        #     im0 = axes[0].imshow(
        #         pred.detach().reshape(pred.shape)[0].reshape(self.nx, self.nx).cpu(),
        #         vmin=vmin,
        #         vmax=vmax,
        #         cmap="RdBu",
        #     )
        #     cax0 = dividers[0].append_axes("right", size="5%", pad=0.05)
        #     fig.colorbar(im0, cax=cax0, orientation="vertical")

        #     axes[0].set_title("Pred")
        #     im1 = axes[1].imshow(
        #         target[0].reshape(self.nx, self.nx).cpu(),
        #         vmin=vmin,
        #         vmax=vmax,
        #         cmap="RdBu",
        #     )

        #     cax1 = dividers[1].append_axes("right", size="5%", pad=0.05)
        #     fig.colorbar(im1, cax=cax1, orientation="vertical")

        #     axes[1].set_title("Target")
        #     diff = (
        #         (pred.reshape(pred.shape) - target)
        #         .detach()[0]
        #         .reshape(self.nx, self.nx)
        #         .cpu()
        #     )
        #     im2 = axes[2].imshow(
        #         diff,
        #         vmin=-abs(diff).max(),
        #         vmax=abs(diff).max(),
        #         cmap="RdBu",
        #     )

        #     cax2 = dividers[2].append_axes("right", size="5%", pad=0.05)
        #     fig.colorbar(im2, cax=cax2, orientation="vertical")

        #     axes[2].set_title("Difference")
        #     for ax in axes:
        #         ax.axis("off")

        #     plt.tight_layout()
        #     plt.savefig(
        #         os.path.join(
        #             self.figure_dir,
        #             f"prediction_val_epoch_{self.current_epoch}_batch_{batch_idx}",
        #         )
        #     )
        #     self.logger.log_image("val_plot", [fig])
        #     plt.close()
        return val_loss

    # def test_step(self, batch, batch_idx):
        # (ic_vals, grid), target = batch
        # batch_size = grid.shape[0]
        # grid_shape = grid.shape
        # pred_shape = grid_shape[:-1] + (1,)
        # grid = grid.reshape(-1, 2)
        # # Sample a subset of the grid
        # if self.constrain:
        #     pred, residual, w = self.flat_forward(ic_vals, grid)
        # else:
        #     with torch.enable_grad():
        #         grid.requires_grad = True
        #         # Required to compute the PDE residual correctly
        #         pred = self.flat_forward(ic_vals, grid)
        #         # TODO: Compute IC/BC/RES losses
        #         residual, dy_x, dy_t, dy_xx = self.pde(grid, pred)
        # batch_size = grid_shape[0]
        # residual = residual.reshape(batch_size, -1)
        # residual_loss = (
        #     1 / residual.shape[-1] * nn.MSELoss()(residual, torch.zeros_like(residual))
        # )

        # return pred.reshape(*pred_shape), target, residual_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.lr_scheduler_step, gamma=self.lr_scheduler_factor
        )
        return [optimizer], [{"scheduler": self.lr_scheduler, "interval": "step"}]


def DeepONet_main():
    """
    Parameters
    ----------
    train_data_res : resolution of the training data
    save_index : index of the saving folder
    """

    ################################################################
    #  configurations
    ################################################################

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=5, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument(
        "--lr_scheduler_step",
        type=float,
        default=2000,
        help="Number of training steps before decaying the learning rate.",
    )
    parser.add_argument(
        "--lr_scheduler_factor", type=float, default=0.9, help="Factor for lr decrease."
    )
    parser.add_argument(
        "--ridge",
        type=float,
        default=1e-4,
        help="ridge regularization for nonlinear solver",
    )
    parser.add_argument("--epochs", type=int, default=500, help="number of epochs")

    parser.add_argument(
        "--nsamples_constraint",
        type=int,
        default=750,
        help="Number of samples for the interior constraint",
    )
    parser.add_argument(
        "--nsamples_residual",
        type=int,
        default=1750,
        help="Number of samples for the residual loss.",
    )
    parser.add_argument(
        "--Nbasis",
        type=int,
        default=75,
        help="Number of basis functions. If 1, reduces to PhysicsInformed DeepONets.",
    )
    parser.add_argument(
        "--ngpus",
        type=int,
        default=1,
        help="Number of gpus to use. For now only 1 seems to work.",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=50,
        help="Max interations for nonlinear LS solver",
    )
    parser.add_argument("--log_every_n_steps", type=int, default=1)
    parser.add_argument(
        "--viscosity", default=1e-2, type=float, help="viscosity of the fluid"
    )

    args = parser.parse_args()

    print(args, "\n")

    wandb_logger = WandbLogger(project="PDEs-Burgers", save_dir="logs")
    wandb_logger.experiment.config.update(vars(args))

    ################################################################
    # read training data
    ################################################################

    train_set = BurgersDeepONetDataset(
        # "/home/negroni/deeponet-fno/data/burgers/Burgers_train_1000_nov22.mat"
        "/home/negroni/deeponet-fno/data/burgers/Burgers_train_1000_visc_0.01_Feb23.mat",
        n_samples_interior=args.nsamples_residual + args.nsamples_constraint,
    )

    val_set = BurgersDeepONetDataset(
        # "/home/negroni/deeponet-fno/data/burgers/Burger_test_50_nov22.mat"
        "/home/negroni/deeponet-fno/data/burgers/Burgers_test_50_visc_0.01_Feb23.mat",
        n_samples_interior=args.nsamples_residual + args.nsamples_constraint,
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch,
        shuffle=True,
        pin_memory=True,
        num_workers=40,
    )

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch, shuffle=False, pin_memory=True
    )

    model = BurgersLitModel(
        lr=args.lr,
        lr_scheduler_step=args.lr_scheduler_step,
        lr_scheduler_factor=args.lr_scheduler_factor,
        n_samples_constraint=args.nsamples_constraint,
        N_basis=args.Nbasis,
        constrain=args.Nbasis > 1,
        max_iterations=args.max_iterations,
        viscosity=args.viscosity,
        n_samples_residual=args.nsamples_residual,
        ridge=args.ridge,
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=5,
        monitor="val_mse_loss",
        every_n_train_steps=30 if args.Nbasis > 1 else None,
        mode="min",
        filename="model-{epoch:02d}-{train_loss:.6f}",
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.ngpus,
        strategy="ddp" if args.ngpus > 1 else None,
        precision=64,
        max_epochs=args.epochs,
        enable_checkpointing=True,
        log_every_n_steps=args.log_every_n_steps,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=1,
        logger=wandb_logger,
    )

    trainer.fit(
        model=model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )


if __name__ == "__main__":

    DeepONet_main()
