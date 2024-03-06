import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

import config
from gnn import GatedGraphNeuralNetwork, AdjacencyList


class GNNEncoder(nn.Module):
    def __init__(self, gnn_hidden_size, num_edge_types, device=config.device, dropout=config.dropout,
                 gnn_layer_timesteps=8):
        super(GNNEncoder, self).__init__()
        self.gnn_hidden_size = gnn_hidden_size
        self.num_edge_types = num_edge_types
        self.dropout = dropout
        self.device = device
        self.gnn = GatedGraphNeuralNetwork(hidden_size=self.gnn_hidden_size, num_edge_types=self.num_edge_types,
                                           layer_timesteps=[gnn_layer_timesteps], residual_connections={},
                                           state_to_message_dropout=self.dropout,
                                           rnn_dropout=self.dropout)
        # self.gnn_output_layer = nn.Linear(self.gnn_hidden_size, self.gnn_hidden_size * 2, bias=False)

    def forward(self, node_embedding, node_lens, node_as_output, edge_prt2ch,
                edge_prev2next, edge_align, edge_com2sub):
        batch_node_vec = []
        batch_graph_vec = []
        batch_size = node_embedding.shape[0]
        # process one graph one time
        for i in range(batch_size):
            adj_list_type1 = AdjacencyList(node_num=node_lens[i], adj_list=edge_prt2ch[i],
                                           device=self.device)

            adj_list_type2 = AdjacencyList(node_num=node_lens[i], adj_list=edge_prev2next[i],
                                           device=self.device)

            adj_list_type3 = AdjacencyList(node_num=node_lens[i], adj_list=edge_align[i],
                                           device=self.device)
            adj_list_type4 = AdjacencyList(node_num=node_lens[i], adj_list=edge_com2sub[i],
                                           device=self.device)

            node_representations = self.gnn.compute_node_representations(
                initial_node_representation=node_embedding[i, :, :],
                adjacency_lists=[adj_list_type1,
                                 adj_list_type2,
                                 adj_list_type3,
                                 adj_list_type4])
            # update the embedding
            # node_representations = self.gnn_output_layer(node_representations)

            node_representations = node_representations[:node_lens[i], :]
            node_representations = node_representations[node_as_output[i], :]
            batch_graph_vec.append(torch.tanh(node_representations.mean(dim=0).unsqueeze(0)))
            batch_node_vec.append(node_representations)

        batch_graph_vec = torch.cat(batch_graph_vec, dim=0)
        lens = [x.shape[0] for x in batch_node_vec]
        batch_node_vec = pad_sequence(batch_node_vec)
        node_mask = torch.Tensor(batch_size, 1, max(lens)) == 999999
        for i in range(batch_size):
            node_mask[i, :, :lens[i]] = True
        # self.node_mask = (self.node_value != pad).unsqueeze(-2)
        # batch_node_vec, type, tensor    size: batch_size, num_node, hidden_size
        # batch_graph_vec, type, tensor,  size: batch_size, hidden_size
        return batch_node_vec.permute(1, 0, 2), node_mask, batch_graph_vec
