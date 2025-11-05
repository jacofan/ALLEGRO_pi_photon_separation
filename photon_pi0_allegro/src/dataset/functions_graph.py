import numpy as np
import torch
import dgl
from src.dataset.functions_data import (
    calculate_distance_to_boundary,
)
import time
from src.dataset.functions_particles import concatenate_Particles_GT, Particles_GT
from src.dataset.utils_hits import create_noise_label
from src.dataset.dataclasses import Hits





def create_graph(
    output,
    for_training =True 
):
    prediction = not for_training
    graph_empty = False
   
    hits = Hits.from_data(
    output,
    )

    g = dgl.graph(([], []))
    g.add_nodes(hits.hit_features.shape[0])
    g.ndata["h"] = hits.hit_features.float() 
    y = torch.max(torch.Tensor(output["y_event_type"])).view(-1)

    return [g, y], graph_empty


def graph_batch_func(list_graphs):
    """collator function for graph dataloader

    Args:
        list_graphs (list): list of graphs from the iterable dataset

    Returns:
        batch dgl: dgl batch of graphs
    """
    list_graphs_g = [el[0] for el in list_graphs]
    list_y = [el[1] for el in list_graphs]
    ys = torch.cat(list_y, dim=0)
    # ys = torch.reshape(ys, [-1, list_y[0].shape[1]])
    
    bg = dgl.batch(list_graphs_g)
    # reindex particle number
    return bg, ys

