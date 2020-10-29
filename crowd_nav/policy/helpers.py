import torch
import torch.nn as nn
from torch.nn import Parameter


def mlp(input_dim, mlp_dims, last_relu=False):
    layers = []
    mlp_dims = [input_dim] + mlp_dims
    for i in range(len(mlp_dims) - 1):
        layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
        # nn.init.orthogonal(layers[-1].weight)
        if i != len(mlp_dims) - 2 or last_relu:
            layers.append(nn.LeakyReLU(negative_slope=-0.2))
    net = nn.Sequential(*layers)
    return net

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.orthogonal(self.W.data, gain=1.414)
        self.a = Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.orthogonal(self.a.data, gain=1.414)
        self.bias = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.orthogonal(self.bias.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(negative_slope=-0.2)

    def forward(self, input, adj):

        # shape of input is batch_size, graph_size,feature_dims
        # shape of adj is batch_size, graph_size, graph_size
        assert len(input.shape) == 3
        assert len(adj.shape) == 3
        # map input to h
        h = torch.matmul(input, self.W)
        N = h.size()[1]
        batch_size = h.size()[0]
        a_input = torch.cat([h.repeat(1, 1, N).view(batch_size, N * N, -1), h.repeat(1, N, 1)],
                            dim=-1).view(batch_size, N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = nn.functional.softmax(attention, dim=2)
        attention = nn.functional.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        h_prime = h_prime + self.bias
        return nn.functional.elu(h_prime)

class GraphAttentionLayerSim(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayerSim, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.similarity_function = 'embedded_gaussian'
        self.W_a = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W_a.data, gain=1.414)
        self.bias = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_uniform_(self.bias.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(negative_slope=-0.2)


    def forward(self, input, adj):

        # shape of input is batch_size, graph_size,feature_dims
        # shape of adj is batch_size, graph_size, graph_size
        assert len(input.shape) == 3
        assert len(adj.shape) == 3
        # map input to h
        e = self.leakyrelu(self.compute_similarity_matrix(input))
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        attention = nn.functional.softmax(attention, dim=2)
        h_prime = torch.matmul(attention, input)
        h_prime = h_prime + self.bias
        return nn.functional.elu(h_prime)

    def compute_similarity_matrix(self, X):
        if self.similarity_function == 'embedded_gaussian':
            A = torch.matmul(torch.matmul(X, self.W_a), X.permute(0, 2, 1))
        elif self.similarity_function == 'gaussian':
            A = torch.matmul(X, X.permute(0, 2, 1))
        elif self.similarity_function == 'cosine':
            X = torch.matmul(X, self.W_a)
            A = torch.matmul(X, X.permute(0, 2, 1))
            magnitudes = torch.norm(A, dim=2, keepdim=True)
            norm_matrix = torch.matmul(magnitudes, magnitudes.permute(0, 2, 1))
            A = torch.div(A, norm_matrix)
        elif self.similarity_function == 'squared':
            A = torch.matmul(X, X.permute(0, 2, 1))
            squared_A = A * A
            A = squared_A / torch.sum(squared_A, dim=2, keepdim=True)
        elif self.similarity_function == 'equal_attention':
            A= (torch.ones(X.size(1), X.size(1)) / X.size(1)).expand(X.size(0), X.size(1), X.size(1))
        elif self.similarity_function == 'diagonal':
            A = (torch.eye(X.size(1), X.size(1))).expand(X.size(0), X.size(1), X.size(1))
        else:
            raise NotImplementedError
        return A

class GAT(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.nheads = nheads
        self.attentions = [GraphAttentionLayerSim(in_feats, hid_feats, dropout=dropout, alpha=alpha, concat=True)
                           for _ in range(self.nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = mlp(hid_feats * nheads, [out_feats], last_relu=True)
        self.add_module('out_gat', self.out_att)

    def forward(self, x, adj):
        assert len(x.shape) == 3
        assert len(adj.shape) == 3
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = self.out_att(x)
        return x