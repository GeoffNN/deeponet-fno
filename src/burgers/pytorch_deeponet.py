import argparse

import matplotlib.pyplot as plt
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

# torch.set_default_dtype(torch.float64)

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
    def __init__(self, lr, nx, n_samples=2500, viscosity=1e-2, constrain=False, N_basis=600, max_iterations=200, ridge=1e-4):
        super().__init__()
        self.lr = lr
        self.nx = nx
        self.viscosity = viscosity
        self.n_samples = n_samples
        self.constrain = constrain
        self.N_basis = N_basis
        self.max_iterations = max_iterations
        self.ridge = ridge

        self.trunk_net = nn.Sequential(
            nn.Linear(2, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 100 if not self.constrain else N_basis),
        )

        self.branch_net = nn.Sequential(
            nn.Linear(nx, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
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

        # compute forward on IC 
        ic_vals_pred = self.trunk_net(grid[:, :, 0])

        # Compute pred and dy_x on the boundary
        boundary_grid = torch.stack(
            [grid[:, 0], grid[:, -1]], 1
            ).reshape(-1, 2).detach() # (b, 2, nt, 2)
        boundary_grid.requires_grad = True

        pred_boundaries = self.trunk_net(boundary_grid)
        residual_bc, dy_x_bc, dy_t_bc, dy_xx_bc = self.pde(boundary_grid, pred_boundaries)

        pred_boundaries = pred_boundaries.reshape(batch_size, 2, self.nx, -1)
        dy_x_bc = dy_x_bc.reshape_as(pred_boundaries)

        bc_diff = torch.concat([
            (pred_boundaries[:, 0] - pred_boundaries[:, -1]).reshape(batch_size, self.nx, self.N_basis),
            (dy_x_bc[:, 0] - dy_x_bc[:, -1]).reshape(batch_size, self.nx, self.N_basis)
        ], -2)

        def pde_residual_error_fn(optim_vars, aux_vars):
            y, dy_x, dy_t, dy_xx = aux_vars 
            w, = optim_vars
            y_tensor, dy_x_tensor, dy_t_tensor, dy_xx_tensor = map(functools.partial(reduce_last_dim, w=w.tensor), (y.tensor, dy_x.tensor, dy_t.tensor, dy_xx.tensor))
            residual_tensor = dy_t_tensor + y_tensor * dy_x_tensor - self.viscosity / np.pi * dy_xx_tensor

            return residual_tensor

        def ic_error_fn(optim_vars, aux_vars):
            ic_vals_pred, ic_vals = aux_vars 
            w, = optim_vars
            y_tensor = reduce_last_dim(ic_vals_pred.tensor, w.tensor)
            return (y_tensor - ic_vals.tensor)

        def bc_error_fn(optim_vars, aux_vars):
            bc_diff, = aux_vars 
            w, = optim_vars
            diff_tensor = reduce_last_dim(bc_diff.tensor, w.tensor)
            return diff_tensor

        def ridge_error_fn(optim_vars, aux_vars):
            del aux_vars
            w, = optim_vars
            return self.ridge * w.tensor
        
        objective = th.Objective()
        objective.device = pred.device
        w_th = th.Vector(self.N_basis, name='w')  # Linear combination weight 
        y_th = th.Variable(pred, 'y')
        dy_x_th = th.Variable(dy_x, 'dy_x')
        dy_t_th = th.Variable(dy_t, 'dy_t')
        dy_xx_th = th.Variable(dy_xx, 'dy_xx') 

        ic_vals_th = th.Variable(ic_vals, 'ic_vals')
        ic_vals_pred_th = th.Variable(ic_vals_pred, 'ic_vals_pred')

        bc_diff_th =th.Variable(bc_diff, 'bc_diff')
    
        optim_vars = [w_th]
        aux_vars = [y_th, dy_x_th, dy_t_th, dy_xx_th, ic_vals_th]

        pde_cost_function = th.AutoDiffCostFunction(
            optim_vars, pde_residual_error_fn, dim=self.n_samples, 
             aux_vars=[y_th, dy_x_th, dy_t_th, dy_xx_th], name="pde_residual_cost_fn"
        )
        ic_cost_function = th.AutoDiffCostFunction(
            optim_vars, ic_error_fn, dim=self.nx, 
             aux_vars=[ic_vals_pred_th, ic_vals_th], name="ic_cost_fn"
        )
        bc_cost_function = th.AutoDiffCostFunction(
            optim_vars, bc_error_fn, dim=self.nx * 2,
             aux_vars=[bc_diff_th], name="bc_cost_fn"
        )

        ridge_cost_function = th.AutoDiffCostFunction(
            optim_vars, ridge_error_fn, dim=self.N_basis,
             aux_vars=[], name="ridge_cost_fn"
        )
        objective = th.Objective()
        objective.add(pde_cost_function)
        objective.add(ic_cost_function)
        objective.add(bc_cost_function)
        objective.add(ridge_cost_function)
        objective.to(pred.device)

        optimizer = th.LevenbergMarquardt(
            objective,
            max_iterations=self.max_iterations,
            step_size=0.5,
        )
        theseus_optim = th.TheseusLayer(optimizer)

        theseus_inputs = {
            "dy_x": dy_x,
            "dy_t": dy_t,
            "dy_xx": dy_xx,
            "bc_diff": bc_diff,
            "ic_vals": ic_vals,
            "ic_vals_pred": ic_vals_pred,
            "y": pred,
            "w": torch.rand(batch_size, self.N_basis, device=pred.device)
        }


        updated_inputs, info = theseus_optim.forward(theseus_inputs)
        print(info)

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

    def periodic_bc_diff(self, y, dy_x):
        # y, dy_x are the full predictions, w/ shape (b, nx, nt, 1)
        batch_size = y.shape[0]
        zero_order = (y[:, 0] - y[:, -1]).reshape(batch_size, -1)
        first_order = (dy_x[:, 0] - dy_x[:, -1]).reshape(batch_size, -1)

        return torch.concat(
            [zero_order, first_order], -1
        )

    def periodic_bc_loss(self, y, dy_x):
        diff = self.periodic_bc_diff(y, dy_x)
        dim = diff.shape[-1] / 2
        return (
            1
            / dim
            * (
                nn.MSELoss()(diff, torch.zeros_like(diff))
                + nn.MSELoss()(diff, torch.zeros_like(diff))
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

        if batch_idx % 20 == 0:
            self.figure_dir = os.path.join(self.trainer.log_dir, "figures")
            os.makedirs(self.figure_dir, exist_ok=True)
            fig, axes = plt.subplots(ncols=3)

            vmin = min(pred.reshape(pred_shape)[0].min(), target[0].min()).cpu()
            vmax = max(pred.reshape(pred_shape)[0].min(), target[0].max()).cpu()

            axes[0].imshow(
                pred.detach().reshape(pred_shape)[0].reshape(self.nx, self.nx).cpu(),
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
                (pred.reshape(pred_shape) - target).detach()[0].reshape(self.nx, self.nx).cpu(),
                vmin=-max(abs(vmin), abs(vmax)),
                vmax=max(abs(vmin), abs(vmax)),
                cmap="RdBu",
            )
            axes[2].set_title("Difference")
            for ax in axes:
                ax.axis("off")

            plt.tight_layout()
            plt.savefig(
                os.path.join(self.figure_dir, f"prediction_train_epoch_{self.current_epoch}_batch_{batch_idx}")
            )
            plt.close()


        return train_loss

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

        plt.close()

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

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=3, help='batch size')
    parser.add_argument("--lr", type=float, default=1e-3, help='learning rate')
    parser.add_argument("--ridge", type=float, default=1e-4, help='ridge regularization for nonlinear solver')
    parser.add_argument("--epochs", type=int, default=500, help='number of epochs')
    parser.add_argument("--nsamples", type=int, default=500, help="Number of samples for the interior constraint")
    parser.add_argument("--Nbasis", type=int, default=75, help="Number of basis functions. If 1, reduces to PhysicsInformed DeepONets.")
    parser.add_argument("--ngpus", type=int, default=1, help="Number of gpus to use. For now only 1 seems to work.")
    parser.add_argument("--max_iterations", type=int, default=30, help="Max interations for nonlinear LS solver")
    parser.add_argument("--log_every_n_steps", type=int, default=1)

    args = parser.parse_args()

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
        batch_size=args.batch,
        shuffle=True,
        generator=torch.Generator("cuda"),
    )
    test_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch, shuffle=False
    )

    model = LitDeepOnet(lr=args.lr, nx=101, n_samples=args.nsamples, N_basis=args.Nbasis, constrain=args.Nbasis > 1, max_iterations=args.max_iterations)

    trainer = pl.Trainer(
        accelerator="gpu",
        gpus=args.ngpus,
        # precision=64,
        log_every_n_steps=args.log_every_n_steps,
        callbacks=[EarlyStopping(monitor="val_mse_loss", mode="min", patience=3)],
        check_val_every_n_epoch=1,
    )
    trainer.fit(
        model=model, train_dataloaders=train_loader, val_dataloaders=test_loader
    )


if __name__ == "__main__":

    training_data_resolution = 128
    save_index = 0

    DeepONet_main(training_data_resolution, save_index)
