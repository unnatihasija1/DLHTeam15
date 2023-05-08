import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
import numpy as np
import math

import inspect

from torch_geometric.utils import scatter_, softmax, add_self_loops, degree
from torch_geometric.nn.inits import glorot, zeros, uniform

from build_tree import build_stage_one_edges, build_stage_two_edges, build_cominbed_edges
from build_tree import build_icd9_tree, build_atc_tree


class OntologyEmbedding(nn.Module):
    def __init__(self, voc, build_tree_func,
                 in_channels=100, out_channels=20, heads=5):
        super(OntologyEmbedding, self).__init__()

        # initial tree edges
        res, graph_voc = build_tree_func(list(voc.idx2word.values()))
        stage_one_edges = build_stage_one_edges(res, graph_voc)
        stage_two_edges = build_stage_two_edges(res, graph_voc)

        self.edges1 = torch.tensor(stage_one_edges)
        self.edges2 = torch.tensor(stage_two_edges)
        self.graph_voc = graph_voc

        # construct model
        assert in_channels == heads * out_channels
        #self.g = GATConv(in_channels=in_channels,
        #                 out_channels=out_channels,
        #                 heads=heads)
        #self.g = GCNConv(in_channels=in_channels,
        #                 out_channels=out_channels,
        #                 heads=heads)
                         
        self.g = GTNConv(in_channels=in_channels,
                        out_channels=out_channels,
                        heads=heads, dropout=0.1) 

        # tree embedding
        num_nodes = len(graph_voc.word2idx)
        self.embedding = nn.Parameter(torch.Tensor(num_nodes, in_channels))

        # idx mapping: FROM leaf node in graphvoc TO voc
        self.idx_mapping = [self.graph_voc.word2idx[word]
                            for word in voc.idx2word.values()]

        self.init_params()

    def get_all_graph_emb(self):
        emb = self.embedding
        emb = self.g(self.g(emb, self.edges1.to(emb.device)),
                     self.edges2.to(emb.device))
        return emb

    def forward(self):
        """
        :param idxs: [N, L]
        :return:
        """
        emb = self.embedding

        emb = self.g(self.g(emb, self.edges1.to(emb.device)),
                     self.edges2.to(emb.device))

        return emb[self.idx_mapping]

    def init_params(self):
        glorot(self.embedding)


class MessagePassing(nn.Module):
    r"""Base class for creating message passing layers

    .. math::
        \mathbf{x}_i^{\prime} = \gamma_{\mathbf{\Theta}} \left( \mathbf{x}_i,
        \square_{j \in \mathcal{N}(i)} \, \phi_{\mathbf{\Theta}}
        \left(\mathbf{x}_i, \mathbf{x}_j,\mathbf{e}_{i,j}\right) \right),

    where :math:`\square` denotes a differentiable, permutation invariant
    function, *e.g.*, sum, mean or max, and :math:`\gamma_{\mathbf{\Theta}}`
    and :math:`\phi_{\mathbf{\Theta}}` denote differentiable functions such as
    MLPs.
    See `here <https://rusty1s.github.io/pytorch_geometric/build/html/notes/
    create_gnn.html>`__ for the accompanying tutorial.

    """
    def __init__(self, aggr='mean'):
    #def __init__(self, aggr='add'):
        super(MessagePassing, self).__init__()

        self.message_args = inspect.getargspec(self.message)[0][1:]
        self.update_args = inspect.getargspec(self.update)[0][2:]

    def propagate(self, aggr, edge_index, **kwargs):
        r"""The initial call to start propagating messages.
        Takes in an aggregation scheme (:obj:`"add"`, :obj:`"mean"` or
        :obj:`"max"`), the edge indices, and all additional data which is
        needed to construct messages and to update node embeddings."""

        assert aggr in ['add', 'mean', 'max']
        kwargs['edge_index'] = edge_index

        size = None
        message_args = []
        for arg in self.message_args:
            if arg[-2:] == '_i':
                tmp = kwargs[arg[:-2]]
                size = tmp.size(0)
                message_args.append(tmp[edge_index[0]])
            elif arg[-2:] == '_j':
                tmp = kwargs[arg[:-2]]
                size = tmp.size(0)
                message_args.append(tmp[edge_index[1]])
            else:
                message_args.append(kwargs[arg])

        update_args = [kwargs[arg] for arg in self.update_args]

        out = self.message(*message_args)
        out = scatter_(aggr, out, edge_index[0], dim_size=size)
        out = self.update(out, *update_args)

        return out

    def message(self, x_j):  # pragma: no cover
        r"""Constructs messages in analogy to :math:`\phi_{\mathbf{\Theta}}`
        for each edge in :math:`(i,j) \in \mathcal{E}`.
        Can take any argument which was initially passed to :meth:`propagate`.
        In addition, features can be lifted to the source node :math:`i` and
        target node :math:`j` by appending :obj:`_i` or :obj:`_j` to the
        variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`."""

        return x_j

    def update(self, aggr_out):  # pragma: no cover
        r"""Updates node embeddings in analogy to
        :math:`\gamma_{\mathbf{\Theta}}` for each node
        :math:`i \in \mathcal{V}`.
        Takes in the output of aggregation as first argument and any argument
        which was initially passed to :meth:`propagate`."""

        return aggr_out

class GTNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True, negative_slope=0.05, dropout=0.2, bias=True):
        super(GTNConv, self).__init__(aggr='add')  # "Add" aggregation (step 3).

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        #self.weight = nn.Linear(in_channels, heads * out_channels)
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, 4*out_channels))
        #self.lin_r = nn.Linear(in_channels, heads * out_channels, bias=False)

        self.att = nn.Parameter(torch.Tensor(1, heads, 2*out_channels))
        #self.att_r = nn.Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        #glorot(self.weight)
        #glorot(self.lin_r.weight)
        #glorot(self.att_l)
        #glorot(self.att_r)
        zeros(self.bias)

    def forward(self, x, edge_index):
        """"""
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Linearly transform node feature matrix.
        #x_l = self.lin_l(x).view(-1, self.heads, self.out_channels)
        #x_r = self.lin_r(x).view(-1, self.heads, self.out_channels)

        # Step 2: Compute normalization.
        #edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 3: Start propagating messages.
        # propagate_type: (x: Tensor, norm: Tensor)
        #out_l = self.propagate(edge_index, x=x_l, norm=norm)
        #out_r = self.propagate(edge_index, x=x_r, norm=norm)
        #out = self.propagate(edge_index, x=x, norm=norm)
        out = self.propagate('add', edge_index, x=x, num_nodes=x.size(0))

        # Step 4: Concatenate multi-head embeddings.
        # if self.concat:
            # out = torch.cat([out_l, out_r], dim=-1)  # [N, heads, 2 * out_channels]
        # else:
            # out = out_l + out_r  # [N, heads, out_channels]

        # Step 5: Apply bias.
        if self.bias is not None:
            out += self.bias

        # Step 6: Apply non-linearity.
        out = F.leaky_relu(out, self.negative_slope)

        return out

class GCNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, bias=True):
        super(GCNConv, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, 4*out_channels))
        # Initialize the parameters.
        stdv = 1. / math.sqrt(out_channels)
        self.weight.data.uniform_(-stdv, stdv)
        #self.weight = nn.Parameter(torch.Tensor(in_channels, heads * out_channels))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        else:
            self.register_parameter('bias', None)

        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))
        #self.reset_parameters()
    
    def forward(self, x, edge_index):
        
        num_nodes = x.shape[0]
        adj_matrix = np.zeros((num_nodes, num_nodes))
        
        np.fill_diagonal(adj_matrix, 1)
        adj_matrix[edge_index[0].cpu(),edge_index[1].cpu()] = 1
        adj_matrix = torch.FloatTensor(adj_matrix)
        #print(adj_matrix)
        
        degree = torch.sum(adj_matrix, dim=1)
        degree_matrix = torch.diag(degree)
        
        #print("degree matrix", degree_matrix)
        
        d_sqrt_inv = torch.sqrt(torch.inverse(degree_matrix))
        normalized_adj_matrix = torch.mm(torch.mm(d_sqrt_inv, adj_matrix), d_sqrt_inv)
        
        # Calculate output
        device = normalized_adj_matrix.device
        x = x.to(device)
        self.weight = torch.nn.Parameter(torch.FloatTensor(self.weight.size()))
        self.weight = self.weight.to(device)
        ret = torch.mm(normalized_adj_matrix, x).mm(self.weight)
        #ret = torch.mm(normalized_adj_matrix, x).mm(self.weight)
        
        return ret
        
class GATConv(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{j} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions. (default:
            :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
        attentions are averaged instead of concatenated. (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0,
                 bias=True):
        super(GATConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = nn.Parameter(
            torch.Tensor(in_channels, heads * out_channels))
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index):
        """"""
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))
        x = torch.mm(x, self.weight).view(-1, self.heads, self.out_channels)
        return self.propagate('add', edge_index, x=x, num_nodes=x.size(0))

    def message(self, x_i, x_j, edge_index, num_nodes):
        # Compute attention coefficients.
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        #alpha = F.relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0], num_nodes)

        alpha = F.dropout(alpha, p=self.dropout)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            #aggr_out = aggr_out.sum(dim=1)
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads, dropout, bias=True):
        super(GCNLayer, self).__init__()
        # self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        # if bias:
            # self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        # else:
            # self.bias = None
        # self.reset_parameters()
        
        self.weight = nn.Parameter(
            torch.Tensor(in_channels, heads*out_channels))
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, adj, x):
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            output += self.bias
        return F.relu(output)
class ConcatEmbeddings(nn.Module):
    """Concat rx and dx ontology embedding for easy access
    """

    def __init__(self, config, dx_voc, rx_voc):
        super(ConcatEmbeddings, self).__init__()
        # special token: "[PAD]", "[CLS]", "[MASK]"
        self.special_embedding = nn.Parameter(
            torch.Tensor(config.vocab_size - len(dx_voc.idx2word) - len(rx_voc.idx2word), config.hidden_size))
        self.rx_embedding = OntologyEmbedding(rx_voc, build_atc_tree,
                                              config.hidden_size, config.graph_hidden_size,
                                              config.graph_heads)
        self.dx_embedding = OntologyEmbedding(dx_voc, build_icd9_tree,
                                              config.hidden_size, config.graph_hidden_size,
                                              config.graph_heads)
        self.init_params()

    def forward(self, input_ids):
        device = self.special_embedding.device
        rx = self.rx_embedding().to(device)
        dx = self.dx_embedding().to(device)
        emb = torch.cat((self.special_embedding, rx, dx), dim=0)

        return emb[input_ids]

    def init_params(self):
        glorot(self.special_embedding)


class FuseEmbeddings(nn.Module):
    """Construct the embeddings from ontology, patient info and type embeddings.
    """

    def __init__(self, config, dx_voc, rx_voc):
        super(FuseEmbeddings, self).__init__()
        self.ontology_embedding = ConcatEmbeddings(config, dx_voc, rx_voc)
        self.type_embedding = nn.Embedding(2, config.hidden_size)
        # self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

    def forward(self, input_ids, input_types=None, input_positions=None):
        """
        :param input_ids: [B, L]
        :param input_types: [B, L]
        :param input_positions:
        :return:
        """
        # return self.ontology_embedding(input_ids)
        ontology_embedding = self.ontology_embedding(
            input_ids) + self.type_embedding(input_types)
        return ontology_embedding
