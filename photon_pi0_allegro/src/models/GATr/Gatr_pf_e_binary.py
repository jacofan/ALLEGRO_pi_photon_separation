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
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

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

            #multivector channels: 16-components geometric algebra objects, to capture spatial infos
            hidden_mv_channels = hidden_mv_channels,

            # scalar channels: no geometric structure, should be used to capture non geometric properties (energy?)
            ## in_s_channels = 3,
            ## out_s_channels = 1,
            in_s_channels = None,
            out_s_channels = None,

            hidden_s_channels = hidden_s_channels,
            num_blocks = blocks,
            attention = SelfAttentionConfig(),  # Use default parameters for attention
            mlp = MLPConfig(),  # Use default parameters for MLP
        )
        self.ScaledGooeyBatchNorm2_1 = nn.BatchNorm1d(self.input_dim, momentum = 0.1) #batch normalization
        self.loss_crit = nn.BCELoss()   # binary cross entropy loss criterio


        self.readout = "sum"   # sum pooling: add features over all points
        self.MLP_layer = MLPReadout(16, 2)   # gives the final output decision

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
        inputs = self.ScaledGooeyBatchNorm2_1(position) 

        #dim: (batch_size*num_points, 16)
        embedded_inputs = embed_point(inputs) + embed_scalar(energy)

        #print("embedded_inputs: ", embedded_inputs)
        #print("size embedded_inputs: ", embedded_inputs.size())

        scalars = torch.zeros((inputs.shape[0], 1))

        # dim: (batch_size*num_points, 1, 16)
        embedded_inputs = embedded_inputs.unsqueeze(-2)  

        #print("unsq_embedded_inputs: ", embedded_inputs)
        #print("size unsq_embedded_inputs: ", embedded_inputs.size())

        # Pass data through GATr, _ is the attention weight (ignored)
        embedded_outputs, _ = self.gatr(
            embedded_inputs, scalars = scalars, attention_mask=mask
        )  # (..., num_points, 1, 16)
        
        # new embedding for each node, assigned to the new vector "h_"
        #dim: (batch_size*num_points, 16)
        g.ndata["h_"] = embedded_outputs[:, 0, :]

        #print("g.ndata[h_]: ", g.ndata["h_"])
        #print("size: ", g.ndata["h_"].size())
        
        #dim: (batch_size, 16), after sum pooling of all the embeddings
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

        # a batch is grouping several events 
        y = batch[1]        # labels
        batch_g = batch[0]  # graph input data

        # this mask allows to connect different hits (since they are part of the same cluster)
        # mask is a block diagonal matrix telling the cumulative starting position of each graph's node sequence in the batch
        mask, labels = self.build_attention_mask(batch_g)

        #print("mask: ", mask)
        #print("labels: ", labels)
        # print("y: ", y)
        # print("batch_g: ", batch_g)

        h_np = batch_g.ndata["h"].cpu().numpy()
        #print("h_np ", h_np)
        #print("h_np size ", h_np.shape)

        # this is the "forward" step!
        #not clear why need to differentiate
        if self.trainer.is_global_zero:
            model_output = self(batch_g, batch_idx, mask, labels)
        else:
            model_output = self(batch_g, 1, mask, labels)

        # Can't we just do:    
        #model_output = self(batch_g, batch_idx, mask, labels)

        # loss calculation
        loss = self.loss_crit(
                self.m(model_output),
                1.0 * F.one_hot(y.view(-1).long(), num_classes=2),
            )
        
        #print("model_output: ", model_output)
        #print("model_output_argmax: ", model_output.argmax(axis=1))
        #print("y: ", y)
        #print("y.view(-1): ", y.view(-1))


        if self.trainer.is_global_zero:
            wandb.log({"loss": loss.item()})

            # accuracy computation: model_output.argmax(axis=1) picks the index of largest value per row (the predicted class)
            # then, this is compared with true labels, giving a boolean tensor, which is multiplied by one (true = 1, false = 0) and then averaged
            acc = torch.mean(1.0 * (model_output.argmax(axis=1) == y.view(-1)))
            wandb.log({"accuracy": acc.item()})

        self.loss_final = loss.item() + self.loss_final
        self.number_b = self.number_b + 1
        # print("output: ", model_output)
        del model_output
        return loss

    def validation_step(self, batch, batch_idx):
        
        # self.validation_step_outputs = [] useless?
        y = batch[1] #labels
        batch_g = batch[0] # graph input data

        mask, labels = self.build_attention_mask(batch_g)

        # forward step
        model_output_val = self(batch_g, 1, mask, labels)

        # accumulate predictions (all ranks!)
        self.val_logits.append(model_output_val.detach().cpu())
        self.val_labels.append(y.view(-1).detach().cpu())        

        #if y == 0 (pi0) then [1,0], if y == 1 (photon) then [0,1]
        loss = self.loss_crit(
            #torch.sigmoid(model_output), this was used, but I changed it for consistency with training step
            self.m(model_output_val),
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

        if self.args.predict:
            d = {
                "pi0": model_output_val.detach().cpu()[:, 0].view(-1),
                "photon": model_output_val.detach().cpu()[:, 1].view(-1),
                "labels_true": y.detach().cpu().view(-1),
            }

            df = pd.DataFrame(data=d)
            #print("dataframe: ", df)
            self.eval_df.append(df)

        if self.trainer.is_global_zero:

            # print(model_output)
            # print(labels_true)
            
            # in this way, wandb is showing only the last loss/accuracy of the epoch
            wandb.log({"loss_validation": loss.item()})

            acc = torch.mean(1.0 * (model_output_val.argmax(axis=1) == y.view(-1)))
            # print(acc)
            wandb.log({"accuracy_validation ": acc.item()})

            # if self.trainer.is_global_zero:
        ##    wandb.log(
        ##        {
        ##            "confusion matrix": wandb.plot.confusion_matrix(
        ##                probs=None,
        ##                y_true=y.view(-1).detach().cpu().numpy(),
        ##                preds=model_output_val.argmax(axis=1).view(-1).detach().cpu().numpy(),
        ##                #class_names=["0", "1"],
        ##            )
        ##        }
        ##    )
##
        ##    # Apply softmax to get probabilities
        ##    probs = torch.softmax(model_output_val, dim=1)
##
        ##    # y true and y tensor are the same object
        ##    #print("y_true_valid: ", y.view(-1).detach().cpu().numpy())
        ##    #print("y_probas_valid ", probs.detach().cpu().numpy())
##
        ##    # Log ROC curve
        ##    wandb.log(
        ##        {
        ##            "roc curve": wandb.plot.roc_curve(
        ##                y_true=y.view(-1).detach().cpu().numpy(),
        ##                y_probas=probs.detach().cpu().numpy(),
        ##                #labels=["0", "1"]  # Or ["0", "1"] if you prefer, but check the order
        ##            )
        ##        }   
        ##    )


        del loss
        del model_output_val

    # hook called at the end of every training epoch
    def on_train_epoch_end(self):
        # total loss across batches in the epoch / number of batches 
        self.log("train loss epoch", self.loss_final / self.number_b, sync_dist=True)


    def on_train_epoch_start(self):
        # if self.trainer.is_global_zero and self.current_epoch == 0:
        #     self.stat_dict = {}
        self.make_mom_zero()

    def on_validation_epoch_start(self):
        self.make_mom_zero()
        self.eval_df = []

        self.val_labels = []
        self.val_logits = []

    def on_validation_epoch_end(self):
        if self.args.predict:
            df_batch1 = pd.concat(self.eval_df)
            df_batch1.to_pickle(self.args.model_prefix + "/model_output_eval_logits.pt")

        # concatenate different batches 
        logits = torch.cat(self.val_logits)     # local
        labels = torch.cat(self.val_labels)     # local

        # gather from all ranks, = processes across different gpus
        logits = self.all_gather(logits)
        labels = self.all_gather(labels)

        # only rank 0 (the main process) logs metrics
        if self.trainer.is_global_zero:

            # reshape after gather: (world_size, N, num_classes)
            logits = logits.reshape(-1, logits.shape[-1])
            labels = labels.reshape(-1)

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=1)

            # returns the index of the maximum value
            preds = probs.argmax(dim=1)

            # print("y_true ", labels.detach().cpu().numpy())
            # print("y_probas ", probs.detach().cpu().numpy())
            # print("y_probas[:, 0] ", probs.detach().cpu().numpy()[:, 0])
            # print("preds ", preds.detach().cpu().numpy())

            y_truth = labels.detach().cpu().numpy()
            y_probs_pi0_gamma = probs.detach().cpu().numpy() #first column pi0, second column gamma
            y_preds = preds.detach().cpu().numpy()

            y_probs_pi0 = y_probs_pi0_gamma[:, 0]
            y_probs_gamma = y_probs_pi0_gamma[:, 1]

            wandb.log({
                "confusion_matrix": wandb.plot.confusion_matrix(
                    y_true=y_truth,
                    preds=y_preds,
                )
            })

            # Calculate AUC
            # in y_true pi0 = 0, gamma = 1 

            auc = roc_auc_score(y_truth, y_probs_gamma)  
            wandb.log({"auc": auc})

            wandb.log({
                "roc_curve": wandb.plot.roc_curve(
                    y_true=y_truth,
                    y_probas=y_probs_pi0_gamma,
                    labels=["pi0", "gamma"],
                )
            })

            fpr, tpr, thresholds = roc_curve(y_truth, y_probs_gamma)

            # Get the current epoch number for the title/filename
            current_epoch = self.current_epoch 

            # 3. Plot the ROC Curve
            plt.figure(figsize=(12, 10))

            # Plot ROC curve
            plt.plot(
                fpr,
                tpr,
                color='red',
                lw=2,
                label=f'ROC curve (AUC = {auc:.4f})',
            )

            # Plot baseline
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

            plt.xlim([0.0, 1.05])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate (FPR)')
            plt.ylabel('True Positive Rate (TPR)')
            plt.title(f'ROC Curve - Epoch {current_epoch}')
            plt.legend(loc='upper left', fontsize='small')
            plt.grid(True)

            # 4. Save the Plot
            # Save the plot with a unique filename for each epoch
            filename = f"roc_curve_4gpus_epoch_{current_epoch:03d}.png"
            plt.savefig(filename)
            plt.close()


    def make_mom_zero(self):
        if self.current_epoch > 1 or self.args.predict:
            print("Making momentum 0")
            self.ScaledGooeyBatchNorm2_1.momentum = 0
            
    # Optimizer is Adam, scheduler is ReduceLROnPlateau
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.args.start_lr
        )
        print("Optimizer params:", filter(lambda p: p.requires_grad, self.parameters()))
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, patience=3),
                "interval": "epoch",
                "monitor": "train loss epoch",
                "frequency": 1
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }

# return a tensor containing numbers from0 to batch_dim
# to connect different hits (=graphs) in the same cluster ("number" in the batch)
def obtain_batch_numbers(g):
    graphs_eval = dgl.unbatch(g)
    #print("graphs_eval: ", graphs_eval)
    number_graphs = len(graphs_eval)
    batch_numbers = []
    for index in range(0, number_graphs):
        gj = graphs_eval[index]
        num_nodes = gj.number_of_nodes()
        batch_numbers.append(index * torch.ones(num_nodes))
        num_nodes = gj.number_of_nodes()
    batch = torch.cat(batch_numbers, dim=0)
    #print("batch: ", batch)
    #print("batch size: ", batch.size())

    return batch