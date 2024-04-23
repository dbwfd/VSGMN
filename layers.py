import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha,k1,k2,is_att_dp=True,is_gcnWh=False, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.k1=k1
        self.k2=k2
        self.concat = concat
        self.is_att_dp = is_att_dp
        self.is_gcnWh = is_gcnWh

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj,adj_eye):
        if self.is_gcnWh:
            Wh = self.get_gcnWh(adj=adj,h=h)
        else:
            Wh = torch.matmul(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        if self.is_att_dp:
            e = self._prepare_attentional_mechanism_input_dotproduct(Wh)
        else:
            e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15 * torch.ones_like(e)

        e = torch.where(adj_eye > 0, e, zero_vec)

        attention = F.softmax(e, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.permute(0,2,1)

        # 筛选邻居
        # a, index1 = torch.sort(e, dim=-1)
        # a[:,:, 0:85 - self.k2] = 0
        # b, index2 = torch.sort(index1)
        # e = torch.gather(a, dim=-1, index=index2)

        return self.leakyrelu(e)
    def _prepare_attentional_mechanism_input_dotproduct(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)

        out1 = Wh
        out_normal = F.normalize(out1, p=2, dim=-1)
        attscore = torch.matmul(out_normal, torch.permute(out_normal, (0, 2, 1)))
        # 筛选邻居
        a, index1 = torch.sort(attscore, dim=-1)
        a[:, 0:85 - 10] = 0
        # a[:, -self.config.n_num:] = F.softmax(a[:, -self.config.n_num:], dim=-1)
        b, index2 = torch.sort(index1)
        attscore = torch.gather(a, dim=-1, index=index2)
        # attscore[attscore < 0.766] = 0

        return attscore
    def get_gcnWh(self,adj,h):
        adj=adj
        L=self.getLaplace(adj)


        out= torch.matmul(L,h) #b,att_num,v_dim ->b,att_num,v_dim
        out = torch.matmul(out, self.W) #k+s+u,v_dim ->k+s+u,gdim1


        return out

    def getLaplace(self,adj,isNormalize=True):
        #需要修改
        n,_=adj.size()
        #单位矩阵
        i = torch.eye(n).cuda()

        # 度矩阵
        ldiag = torch.sum(adj, dim=-1)
        """
        2d对角线嵌入3d，需要修改！！！！
        """
        degree = torch.diag_embed(ldiag)


        laplace = adj + degree


        if isNormalize:
            ldiag = torch.sum(adj, dim=-1) ** (-1 / 2)
            degree = torch.diag_embed(ldiag)
            laplace = i - torch.matmul(torch.matmul(degree, adj), degree)



        return laplace



    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj.nonzero().t()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'