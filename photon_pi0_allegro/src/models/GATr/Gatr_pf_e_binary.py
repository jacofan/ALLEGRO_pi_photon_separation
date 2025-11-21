from os import path
import sys

# sys.path.append(
#     path.abspath("/afs/cern.ch/work/m/mgarciam/private/geometric-algebra-transformer/")
# )
# sys.path.append(path.abspath("/mnt/proj3/dd-23-91/cern/geometric-algebra-transformer/"))
from time import time
from gatr import GATr, SelfAttentionConfig, MLPConfig
from gatr.interface import embed_point, extract_scalar, extract_point, embed_scalar
from torch_scatter import scatter_add, scatter_mean, scatter_max
import torch
import torch.nn as nn
from src.utils.save_features import save_features
from src.logger.plotting_tools import PlotCoordinates
import numpy as np
from typing import Tuple, Union, List
import dgl
from src.logger.plotting_tools import PlotCoordinates
from src.models.mlp_readout_layer import MLPReadout

import lightning as L
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from src.layers.inference_oc import create_and_store_graph_output
from xformers.ops.fmha import BlockDiagonalMask
import os
import wandb

import torch.nn.functional as F
import pandas as pd

from sklearn.utils.class_weight import compute_class_weight

# This is not a wrapper: this is the model!!!
class ExampleWrapper(L.LightningModule):
    """Example wrapper around a GATr model.

    Expects input data that consists of a point cloud: one 3D point for each item in the data.
    Returns outputs that consists of one scalar number for the whole dataset.

    Parameters
    ----------
    blocks : int
        Number of transformer blocks
    hidden_mv_channels : int
        Number of hidden multivector channels
    hidden_s_channels : int
        Number of hidden scalar channels
    """

    def __init__(
        self,
        args,
        dev,
    ):
        super().__init__()
        self.strict_loading = False
        self.input_dim = 3      # input points have 3 coord., why? they should be 4 
        self.output_dim = 2     # output has 2 values (binary classification?)

        self.loss_final = 0
        self.number_b = 0
        self.args = args
        self.m = nn.Sigmoid()  # sigmoid activation function (binary class.)

        hidden_mv_channels = 16
        hidden_s_channels = 16
        blocks = 8 
        self.gatr = GATr(
            in_mv_channels = 1,
            out_mv_channels = 1,
            hidden_mv_channels = hidden_mv_channels,
            in_s_channels = 3,
            out_s_channels = 1,
            hidden_s_channels = hidden_s_channels,
            num_blocks = blocks,
            attention = SelfAttentionConfig(),  # Use default parameters for attention
            mlp = MLPConfig(),  # Use default parameters for MLP
        )
        self.ScaledGooeyBatchNorm2_1 = nn.BatchNorm1d(self.input_dim, momentum = 0.1) #batch normalization
        self.loss_crit = nn.BCELoss()   # binary cross entropy loss criterio


        self.readout = "sum"   # sum pooling: add features over all points
        self.MLP_layer = MLPReadout(16, 2)   # gives the final output

    ## def obtain_loss_weighted(self, labels_true):
    ## 
    ##     self.loss_crit = nn.BCELoss()

    def forward(self, g, step_count, mask, labels, eval="", return_train=False):
        """Forward pass.

        Parameters
        ----------
        inputs : torch.Tensor with shape (*batch_dimensions, num_points, 3)
            Point cloud input data

        Returns
        -------
        outputs : torch.Tensor with shape (*batch_dimensions, 1)
            Model prediction: a single scalar for the whole point cloud.
        """

        # node features → transformer → node embeddings → sum pooling → hg → MLP → output

        # g is a graph object, .ndata to access the node features. every node has a feature vector h
        # tensor of hit position x,y,z 
        position = g.ndata["h"][:,1:]   
        # tensor of hit energy
        # view reshapes the tensor
        energy = g.ndata["h"][:,0].view(-1, 1)   

        #print("inputs: ", position)
        #print(position.size())
        #print("inputs_scalar: ", energy)
        #print(energy.size())

        # batch normalization for inputs (energy is not normalized)
        # ACHTUNG: seems not working! Changing  names now works
        inputs = self.ScaledGooeyBatchNorm2_1(position) 

        #print("scaled: ", inputs)

        #dim: (batch_size*num_points, 16)
        embedded_inputs = embed_point(inputs) + embed_scalar(energy)
        #print("embedded_inputs: ", embedded_inputs)
        #print("size embedded_inputs: ", embedded_inputs.size())

        # dim: (batch_size*num_points, 1, 16)
        embedded_inputs = embedded_inputs.unsqueeze(-2)  


        #print("unsq_embedded_inputs: ", embedded_inputs)
        #print("size unsq_embedded_inputs: ", embedded_inputs.size())

        
        # Pass data through GATr, _ is thre attention weight (ignored)
        embedded_outputs, _ = self.gatr(
            embedded_inputs, scalars=None, attention_mask=mask
        )  # (..., num_points, 1, 16)
        
        # new embedding for each node, assigned to the new vector "h_"
        #dim: (batch_size*num_points, 16)
        g.ndata["h_"] = embedded_outputs[:, 0, :]
        #print("g.ndata[h_]: ", g.ndata["h_"])
        #print("size: ", g.ndata["h_"].size())
        #dim: (batch_size, 16), after sum pooling
        hg = dgl.sum_nodes(g, "h_")
        #print("hg: ", hg)
        #print("size: ", hg.size())

        #Multi-layer perceptron, final decision head of the model
        all_features = self.MLP_layer(hg)

        #print("all features: ", all_features)
        return all_features

    def build_attention_mask(self, g):
        """Construct attention mask from pytorch geometric batch.

        Parameters
        ----------
        inputs : torch_geometric.data.Batch
            Data batch.

        Returns
        -------
        attention_mask : xformers.ops.fmha.BlockDiagonalMask
            Block-diagonal attention mask: within each sample, each token can attend to each other
            token.
        """
        batch_numbers = obtain_batch_numbers(g)

        return (
            BlockDiagonalMask.from_seqlens(
                torch.bincount(batch_numbers.long()).tolist()
            ),
            batch_numbers,
        )

    def training_step(self, batch, batch_idx):
        y = batch[1]        # labels
        batch_g = batch[0]  # graph input data
        # initial_time = time() useless
        mask, labels = self.build_attention_mask(batch_g)
        #print("y: ", y)
        #print("batch_g: ", batch_g)
        h_np = batch_g.ndata["h"].cpu().numpy()
        #print("content ", h_np[:10])

        if self.trainer.is_global_zero:
            model_output = self(batch_g, batch_idx, mask, labels)
        else:
            model_output = self(batch_g, 1, mask, labels)


        loss = self.loss_crit(
                self.m(model_output),
                1.0 * F.one_hot(y.view(-1).long(), num_classes=2),
            )
        
        
        if self.trainer.is_global_zero:
            wandb.log({"loss": loss.item()})
            acc = torch.mean(
                1.0 * (model_output.argmax(axis=1) == y.view(-1))
            )
            wandb.log({"accuracy": acc.item()})
        self.loss_final = loss.item() + self.loss_final
        self.number_b = self.number_b + 1
        # print("output: ", model_output)
        del model_output
        return loss

    def validation_step(self, batch, batch_idx):
        
        self.validation_step_outputs = []
        y = batch[1]
        batch_g = batch[0]
        mask, labels = self.build_attention_mask(batch_g)
        model_output = self(batch_g, 1, mask, labels)
        #if y == 0 (pi0) then [1,0], if y == 1 (photon) then [0,1]
        loss = self.loss_crit(
            #torch.sigmoid(model_output),
            self.m(model_output),
            1.0 * F.one_hot(y.view(-1).long(), num_classes=2),
        )
        ##for i in range(len(y)):
        ##    true_label = y[i].item()
        ##    col0_score = model_output[i, 0].item()
        ##    col1_score = model_output[i, 1].item()
        ##    
        ##    # Determine what true_label represents
        ##    true_class = "pi0" if true_label == 0 else "photon"
        ##    
        ##    print(f"Sample {i}:")
        ##    print(f"  True label: {true_label} ({true_class})")
        ##    print(f"  Column 0 score: {col0_score:6.3f}")
        ##    print(f"  Column 1 score: {col1_score:6.3f}")

        model_output1 = model_output
        if self.args.predict:
            # d = {
            #     "pi": model_output1.detach().cpu()[:, 0].view(-1),
            #     # "pi0": model_output1.detach().cpu()[:, 1].view(-1),
            #     "e": model_output1.detach().cpu()[:, 1].view(-1),
            #     "muon": model_output1.detach().cpu()[:, 2].view(-1),
            #     "rho": model_output1.detach().cpu()[:, 3].view(-1),
            #     "labels_true": labels_true.detach().cpu().view(-1),
            #     # "energy": y.E.detach().cpu().view(-1),
            # }
            d = {
                "pi0": model_output1.detach().cpu()[:, 0].view(-1),
                "photon": model_output1.detach().cpu()[:, 1].view(-1),
                "labels_true": y.detach().cpu().view(-1),
            }
            df = pd.DataFrame(data=d)
            self.eval_df.append(df)



        # if self.trainer.is_global_zero:
        # print(model_output)
        # print(labels_true)
        wandb.log({"loss_val": loss.item()})
        acc = torch.mean(1.0 * (model_output.argmax(axis=1) == y.view(-1)))
        # print(acc)
        wandb.log({"accuracy val ": acc.item()})

        # if self.trainer.is_global_zero:
        wandb.log(
            {
                "conf_mat": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=y.view(-1).detach().cpu().numpy(),
                    preds=model_output.argmax(axis=1).view(-1).detach().cpu().numpy(),
                    class_names=["0", "1"],
                )
            }
        )

        # Apply softmax to get probabilities
        probs = torch.softmax(model_output, dim=1)
    
        # Log ROC curve
        wandb.log(
            {
                "roc_curve": wandb.plot.roc_curve(
                    y_true=y.view(-1).detach().cpu().numpy(),
                    y_probas=probs.detach().cpu().numpy(),
                    labels=["pi0", "photon"]  # Or ["0", "1"] if you prefer, but check the order
                )
            }   
        )


        del loss
        del model_output

    def on_train_epoch_end(self):

        self.log("train_loss_epoch", self.loss_final / self.number_b)

    def on_train_epoch_start(self):
        # if self.trainer.is_global_zero and self.current_epoch == 0:
        #     self.stat_dict = {}
        self.make_mom_zero()

    def on_validation_epoch_start(self):
        self.make_mom_zero()
        self.eval_df = []

    def on_validation_epoch_end(self):
        if self.args.predict:
            df_batch1 = pd.concat(self.eval_df)
            df_batch1.to_pickle(self.args.model_prefix + "/model_output_eval_logits.pt")

    def make_mom_zero(self):
        if self.current_epoch > 1 or self.args.predict:
            print("making momentum 0")
            self.ScaledGooeyBatchNorm2_1.momentum = 0

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=1e-3
        )
        print("Optimizer params:", filter(lambda p: p.requires_grad, self.parameters()))
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, patience=3),
                "interval": "epoch",
                "monitor": "train_loss_epoch",
                "frequency": 1
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }


def obtain_batch_numbers(g):
    graphs_eval = dgl.unbatch(g)
    number_graphs = len(graphs_eval)
    batch_numbers = []
    for index in range(0, number_graphs):
        gj = graphs_eval[index]
        num_nodes = gj.number_of_nodes()
        batch_numbers.append(index * torch.ones(num_nodes))
        num_nodes = gj.number_of_nodes()
    batch = torch.cat(batch_numbers, dim=0)
    return batch