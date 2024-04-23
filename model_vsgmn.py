import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers import GraphAttentionLayer

device = torch.device("cuda:0")


class VSGMN(nn.Module):
    def __init__(self, config, att, init_w2v_att, init_visual_prototype, seenclass, unseenclass,
                 is_bias=True, bias=1, is_conservative=True):
        super(VSGMN, self).__init__()
        self.config = config
        self.dim_f = config.dim_f
        self.dim_v = config.dim_v
        self.nclass = config.num_class
        self.seenclass = seenclass
        self.unseenclass = unseenclass
        self.is_bias = is_bias
        self.is_conservative = is_conservative
        self.is_episode = config.is_episode
        self.is_sampleVP = config.is_sampleVP
        self.is_attgat = config.is_attgat
        self.is_att_dp = config.is_att_dp
        self.is_wd = config.is_wd
        self.is_gcnWh = config.is_gcnWh
        self.eta1 = config.eta1
        self.eta2 = config.eta2
        self.is_unseen_constrain = config.is_unseen_constrain
        self.is_inter_graph = config.is_inter_graph
        self.is_complete_graph_loss = config.is_complete_graph_loss
        self.is_normdif = config.is_normdif
        self.is_prenorm = config.is_prenorm
        self.is_edge_feature=config.is_edge_feature
        # class-level semantic vectors
        self.att = nn.Parameter(F.normalize(att), requires_grad=False)
        # GloVe features for attributes name
        self.V = nn.Parameter(F.normalize(init_w2v_att), requires_grad=True)
        self.Vfix = nn.Parameter(F.normalize(init_w2v_att), requires_grad=False)
        self.Vp = nn.Parameter(F.normalize(init_visual_prototype), requires_grad=False)  # 50xv
        # for self-calibration
        self.bias = nn.Parameter(torch.tensor(bias), requires_grad=False)
        self.test_bias=nn.Parameter(torch.tensor(config.bias), requires_grad=False)
        mask_bias = np.ones((1, self.nclass))
        mask_bias[:, self.seenclass.cpu().numpy()] *= -1
        self.mask_bias = nn.Parameter(torch.tensor(
            mask_bias, dtype=torch.float), requires_grad=False)
        # mapping
        self.W_1 = nn.Parameter(nn.init.normal_(
            torch.empty(self.dim_v, config.tf_common_dim)), requires_grad=True)
        # transformer
        self.transformer = Transformer(
            ec_layer=config.tf_ec_layer,
            dc_layer=config.tf_dc_layer,
            dim_com=config.tf_common_dim,
            dim_feedforward=config.tf_dim_feedforward,
            dropout=config.tf_dropout,
            SAtt=config.tf_SAtt,
            heads=config.tf_heads,
            aux_embed=config.tf_aux_embed)
        # for loss computation
        self.log_softmax_func = nn.LogSoftmax(dim=1)
        self.weight_ce = nn.Parameter(torch.eye(self.nclass), requires_grad=False)

        # for p-branch gcn
        self.W_p1 = nn.Linear(config.num_attribute, config.num_attribute)
        self.W_p2 = nn.Linear(config.num_attribute, config.num_attribute)
        # for s-branch gcn
        self.W_s1 = nn.Linear(300, 300)
        self.W_s2 = nn.Linear(300, 300)
        # for s-branch gcn
        self.W_sp1 = nn.Linear(config.num_attribute, config.num_attribute)
        self.W_sp2 = nn.Linear(config.num_attribute, config.num_attribute)

        if self.config.is_gmlayer:
            if self.is_edge_feature:
                self.W_gmv1 = nn.Linear(2 * config.num_attribute+config.gm_edge_dim, config.gm_hidden_size * config.num_attribute)
                self.W_gmv2 = nn.Linear(2 * config.num_attribute+config.gm_edge_dim, config.gm_hidden_size * config.num_attribute)
                self.W_gms1 = nn.Linear(2 * config.num_attribute+config.gm_edge_dim, config.gm_hidden_size * config.num_attribute)
                self.W_gms2 = nn.Linear(2 * config.num_attribute+config.gm_edge_dim, config.gm_hidden_size * config.num_attribute)
            else:
                self.W_gmv1 = nn.Linear(2 * config.num_attribute, config.gm_hidden_size * config.num_attribute)
                self.W_gmv2 = nn.Linear(2 * config.num_attribute, config.gm_hidden_size * config.num_attribute)
                self.W_gms1 = nn.Linear(2 * config.num_attribute, config.gm_hidden_size * config.num_attribute)
                self.W_gms2 = nn.Linear(2 * config.num_attribute, config.gm_hidden_size * config.num_attribute)

            if self.is_inter_graph:
                self.W_upv1 = nn.Linear((2 + config.gm_hidden_size) * config.num_attribute, config.num_attribute)
                self.W_upv2 = nn.Linear((2 + config.gm_hidden_size) * config.num_attribute, config.num_attribute)
                self.W_ups1 = nn.Linear((2 + config.gm_hidden_size) * config.num_attribute, config.num_attribute)
                self.W_ups2 = nn.Linear((2 + config.gm_hidden_size) * config.num_attribute, config.num_attribute)
            else:
                self.W_upv1 = nn.Linear((1 + config.gm_hidden_size) * config.num_attribute, config.num_attribute)
                self.W_upv2 = nn.Linear((1 + config.gm_hidden_size) * config.num_attribute, config.num_attribute)
                self.W_ups1 = nn.Linear((1 + config.gm_hidden_size) * config.num_attribute, config.num_attribute)
                self.W_ups2 = nn.Linear((1 + config.gm_hidden_size) * config.num_attribute, config.num_attribute)

        if self.config.is_layernorm:
            self.layernorm_v1 = nn.LayerNorm(config.gm_hidden_size * config.num_attribute)
            self.layernorm_v2 = nn.LayerNorm(config.gm_hidden_size * config.num_attribute)
            self.layernorm_s1 = nn.LayerNorm(config.gm_hidden_size * config.num_attribute)
            self.layernorm_s2 = nn.LayerNorm(config.gm_hidden_size * config.num_attribute)

            self.layernorm_upv1 = nn.LayerNorm(config.num_attribute)
            self.layernorm_upv2 = nn.LayerNorm(config.num_attribute)
            self.layernorm_ups1 = nn.LayerNorm(config.num_attribute)
            self.layernorm_ups2 = nn.LayerNorm(config.num_attribute)

        if self.is_attgat:
            self.attgat = GAT(nfeat=512, nhid=config.gat_hidden, nclass=1, dropout=config.gat_dropout,
                              alpha=config.alpha, nheads=config.gat_heads, k1=config.k1, k2=config.k2,
                              is_att_dp=self.is_att_dp, is_gcnWh=self.is_gcnWh)
            self.W_2 = nn.Parameter(nn.init.normal_(
                torch.empty(self.dim_v, config.tf_common_dim)), requires_grad=True)
        if self.is_complete_graph_loss:
            self.aggregator = GraphAggregator(node_hidden_sizes=[config.node_hidden_size],
                                              graph_transform_sizes=[config.graph_transform_size],
                                              input_size=[config.num_attribute], gated=True)

    def forward(self, input, batch_label=None, from_img=False, is_training=False):
        Fs = self.resnet101(input) if from_img else input
        # transformer-based visual-to-semantic embedding
        # classification

        att = self.att
        Vp = self.Vp
        if is_training and self.is_sampleVP:
            att = torch.cat((self.att[batch_label], self.att[self.unseenclass]), dim=0)
            unseenvp = self.Vp[self.unseenclass]
            Vp = unseenvp
        if self.is_episode and is_training:
            Vp = torch.cat((self.Vp[batch_label], self.Vp[self.unseenclass]), dim=0)
            att = torch.cat((self.att[batch_label], self.att[self.unseenclass]), dim=0)

        Trans_out_embed = self.forward_feature_transformer(Fs)
        Trans_vP_embed = self.forward_feature_transformer(Vp)
        if is_training and self.is_sampleVP:
            Trans_vP_embed = torch.cat((Trans_out_embed, Trans_vP_embed), dim=0)
        if self.config.is_gmlayer:
            prototype_visual_hide, prototype_visual, prototype_semantic_hide, prototype_semantic = self.inter_branch_gm(
                Trans_vP_embed, att)
        else:
            prototype_visual_hide, prototype_visual, prototype_semantic_hide, prototype_semantic = self.inter_branch(
                Trans_vP_embed, att)
        if self.is_complete_graph_loss:
            graphs_embed = self.complete_graph_branch(prototype_visual, prototype_semantic)
        # S_pp=torch.einsum('bh,jh->bj', Trans_out_embed,F.normalize(prototype_semantic,p=2,dim=-1)) #b,num_class

        # classification

        package = {'pred': self.forward_attribute(Trans_out_embed),
                   'embed': Trans_out_embed}
        package['S_pp'] = package['pred']
        package['S_pp_origin'] = package['pred']
        package['S_pp_test'] = package['pred']-self.vec_bias+self.vec_test_bias

        package['prototype_hide'] = prototype_semantic_hide
        package['prototype'] = prototype_semantic
        package['vprototype_hide'] = prototype_visual_hide
        package['vprototype'] = prototype_visual
        #package['vprototype_origin_embed']= Trans_vP_embed
        #package['att_batch_vp'] = att

        if self.is_complete_graph_loss:
            package['graphs'] = graphs_embed
        return package

    def forward_feature_transformer(self, Fs):
        # visual
        if len(Fs.shape) == 4:
            shape = Fs.shape
            Fs = Fs.reshape(shape[0], shape[1], shape[2] * shape[3])
        Fs = F.normalize(Fs, dim=1)
        # attributes
        V_n = F.normalize(self.V) if self.config.normalize_V else self.V
        # locality-augmented visual features
        Trans_out = self.transformer(Fs, V_n)
        if self.is_attgat:
            Trans_out_gatb = self.att_branch_mutiheads(Trans_out)
        # embedding to semantic space
        embed = torch.einsum('iv,vf,bif->bi', V_n, self.W_1, Trans_out)
        if self.is_attgat:
            embed_gatb = torch.einsum('iv,vf,bif->bi', V_n, self.W_2, Trans_out_gatb)
            embed = self.eta1 * embed + self.eta2 * embed_gatb

        return embed

    def forward_attribute(self, embed):
        embed = torch.einsum('ki,bi->bk', self.att, embed)
        self.vec_bias = self.mask_bias * self.bias
        self.vec_test_bias=self.mask_bias * self.test_bias
        embed = embed + self.vec_bias

        return embed

    def p_branch(self, att, edgeweight=None):
        V_P = att
        out1 = self.W_p1(V_P)
        out_normal = F.normalize(out1, p=2, dim=-1)
        attscore = torch.matmul(out_normal, out_normal.T)
        # 筛选邻居
        attscore[attscore < 0.766] = 0
        attscore = torch.softmax(10 * attscore, dim=-1)
        out1 = torch.matmul(attscore, V_P)

        out2 = self.W_p2(out1)
        out_normal = F.normalize(out2, p=2, dim=-1)
        attscore = torch.matmul(out_normal, out_normal.T)
        # 筛选邻居
        attscore[attscore < 0.766] = 0
        attscore = torch.softmax(10 * attscore, dim=-1)
        out2 = torch.matmul(attscore, out1)
        return out1, out2

    def s_branch(self, v_embed):

        out1 = self.W_sp1(v_embed)
        out_normal = F.normalize(out1, p=2, dim=-1)
        attscore = torch.matmul(out_normal, out_normal.T)
        # 筛选邻居
        attscore[attscore < self.config.k3] = 0
        attscore = torch.softmax(self.config.theta * attscore, dim=-1)
        out1 = torch.matmul(attscore, v_embed)

        out2 = self.W_sp2(out1)
        out_normal = F.normalize(out2, p=2, dim=-1)
        attscore = torch.matmul(out_normal, out_normal.T)
        # 筛选邻居
        attscore[attscore < self.config.k3] = 0
        attscore = torch.softmax(self.config.theta * attscore, dim=-1)
        out2 = torch.matmul(attscore, out1)
        return out1, out2

    def inter_branch(self, v_embed, s_embed):

        if self.config.is_layernorm and self.is_prenorm:
            v_embed = self.layernorm1(v_embed)
            s_embed = self.layernorm1(s_embed)
        v_out1 = self.W_sp1(v_embed)
        s_out1 = self.W_p1(s_embed)

        v_out_normal = F.normalize(v_out1, p=2, dim=-1)
        s_out_normal = F.normalize(s_out1, p=2, dim=-1)
        v_attscore = torch.matmul(v_out_normal, v_out_normal.T)
        s_attscore = torch.matmul(s_out_normal, s_out_normal.T)

        v_attscore[v_attscore < self.config.k3] = 0
        if self.is_unseen_constrain:
            v_attscore[:, -len(self.unseenclass):] = 0
        s_attscore[s_attscore < self.config.k3] = 0
        v_attscore = torch.softmax(self.config.theta * v_attscore, dim=-1)
        s_attscore = torch.softmax(self.config.theta * s_attscore, dim=-1)

        v_out1 = torch.matmul(v_attscore, v_embed)
        s_out1 = torch.matmul(s_attscore, s_embed)
        if self.is_inter_graph:
            if self.is_normdif:
                v_out1 = (1 - self.config.k4) * v_out1 + self.config.k4 * (
                            F.normalize(v_embed, p=2, dim=-1) - F.normalize(s_embed, p=2, dim=-1))
                s_out1 = (1 - self.config.k4) * s_out1 + self.config.k4 * (
                            F.normalize(s_embed, p=2, dim=-1) - F.normalize(v_embed, p=2, dim=-1))
            else:
                v_out1 = (1 - self.config.k4) * v_out1 + self.config.k4 * (v_embed - s_embed)
                s_out1 = (1 - self.config.k4) * s_out1 + self.config.k4 * (s_embed - v_embed)

        if self.config.is_layernorm and self.is_prenorm:
            v_out1 = self.layernorm2(v_out1)
            s_out1 = self.layernorm2(s_out1)
        if self.config.is_layernorm and not self.is_prenorm:
            v_out1 = self.layernorm1(v_out1)
            s_out1 = self.layernorm1(s_out1)

        v_out2 = self.W_sp2(v_out1)
        s_out2 = self.W_p2(s_out1)

        v_out_normal = F.normalize(v_out2, p=2, dim=-1)
        s_out_normal = F.normalize(s_out2, p=2, dim=-1)
        v_attscore = torch.matmul(v_out_normal, v_out_normal.T)
        s_attscore = torch.matmul(s_out_normal, s_out_normal.T)

        v_attscore[v_attscore < self.config.k3] = 0
        if self.is_unseen_constrain:
            v_attscore[:, -len(self.unseenclass):] = 0
        s_attscore[s_attscore < self.config.k3] = 0
        v_attscore = torch.softmax(self.config.theta * v_attscore, dim=-1)
        s_attscore = torch.softmax(self.config.theta * s_attscore, dim=-1)

        v_out2 = torch.matmul(v_attscore, v_out1)
        s_out2 = torch.matmul(s_attscore, s_out1)
        if self.is_inter_graph:
            if self.is_normdif:
                v_out2 = (1 - self.config.k4) * v_out2 + self.config.k4 * (
                            F.normalize(v_out1, p=2, dim=-1) - F.normalize(s_out1, p=2, dim=-1))
                s_out2 = (1 - self.config.k4) * s_out2 + self.config.k4 * (
                            F.normalize(s_out1, p=2, dim=-1) - F.normalize(v_out1, p=2, dim=-1))
            else:
                v_out2 = (1 - self.config.k4) * v_out2 + self.config.k4 * (v_out1 - s_out1)
                s_out2 = (1 - self.config.k4) * s_out2 + self.config.k4 * (s_out1 - v_out1)
        if self.config.is_layernorm and not self.is_prenorm:
            v_out2 = self.layernorm2(v_out2)
            s_out2 = self.layernorm2(s_out2)
        return v_out1, v_out2, s_out1, s_out2

    def inter_branch_gm(self, v_embed, s_embed):
        difference_v1, difference_v2, difference_s1, difference_s2 = None, None, None, None
        if self.is_inter_graph:
            difference_v1 = v_embed - s_embed
            difference_s1 = s_embed - v_embed

        v_message1 = self.message_pass(v_embed, self.W_gmv1, self.layernorm_v1)
        v_out1 = self.node_update(v_embed, v_message1, difference=difference_v1, updatenet=self.W_upv1,
                                  layernorm=self.layernorm_upv1)
        s_message1 = self.message_pass(s_embed, self.W_gms1, self.layernorm_s1)
        s_out1 = self.node_update(s_embed, s_message1, difference=difference_s1, updatenet=self.W_ups1,
                                  layernorm=self.layernorm_ups1)

        if self.is_inter_graph:
            difference_v2 = v_out1 - s_out1
            difference_s2 = s_out1 - v_out1

        v_message2 = self.message_pass(v_out1, self.W_gmv2, self.layernorm_v2)
        v_out2 = self.node_update(v_out1, v_message2, difference=difference_v2, updatenet=self.W_upv2,
                                  layernorm=self.layernorm_upv2)

        s_message2 = self.message_pass(s_out1, self.W_gms2, self.layernorm_s2)
        s_out2 = self.node_update(s_out1, s_message2, difference=difference_s2, updatenet=self.W_ups2,
                                  layernorm=self.layernorm_ups2)

        return v_out1, v_out2, s_out1, s_out2

    def message_pass(self, embed, message_net, layernorm):

        d_y = torch.unsqueeze(embed, dim=1)
        d_x = torch.unsqueeze(embed, dim=0)
        d_y = d_y.repeat(1, embed.shape[0], 1)
        d_x = d_x.repeat(embed.shape[0], 1, 1)
        edge_feature = torch.matmul(embed, embed.T)
        if self.is_unseen_constrain:
            d_y = torch.unsqueeze(embed, dim=1)
            d_x = torch.unsqueeze(embed[:embed.shape[0]-len(self.unseenclass),:], dim=0)
            d_y = d_y.repeat(1, embed.shape[0]-len(self.unseenclass), 1)
            d_x = d_x.repeat(embed.shape[0], 1, 1)
            edge_feature = torch.matmul(embed, embed[:embed.shape[0]-len(self.unseenclass),:].T)
        if self.is_edge_feature:
            d_z = torch.unsqueeze(edge_feature, dim=2)
            d_z = d_z.repeat(1, 1, embed.shape[1])
            d = torch.cat([d_y, d_x, d_z], dim=-1)  # num_class, 2 * num_attribute
        else:
            d = torch.cat([d_y, d_x], dim=-1)


        #d[:,-len(self.unseenclass):,:]=0
        d = message_net(d)
        d = F.relu(d)
        out = torch.sum(d, dim=1)
        out = layernorm(out)

        return out

    def node_update(self, nodes, message, difference=None, updatenet=None, layernorm=None):
        if difference is None:
            nodes = torch.cat([nodes, message], dim=-1)
        else:
            nodes = torch.cat([nodes, message, difference], dim=-1)

        out = updatenet(nodes)
        out = F.relu(out)
        out = layernorm(out)

        return out

    def complete_graph_branch(self, visual_graph, semantic_graph):
        nodes = torch.concat((visual_graph, semantic_graph), dim=0)
        idx = torch.cat([torch.zeros(semantic_graph.shape[0], dtype=torch.int64).to(self.config.device),
                         torch.ones(semantic_graph.shape[0], dtype=torch.int64).to(self.config.device)], dim=0)

        graph_embeddings = self.aggregator(nodes, idx, 2)
        return graph_embeddings

    def att_branch(self, v_embed):

        out1 = self.W_s1(v_embed)
        out_normal = F.normalize(out1, p=2, dim=-1)
        attscore = torch.einsum('bif,bjf->bij', out_normal, out_normal)

        # 筛选邻居
        attscore[attscore < 0.766] = 0
        attscore = torch.softmax(10 * attscore, dim=-1)
        out1 = torch.matmul(attscore, v_embed)

        # out2 = self.W_s2(out1)
        # out_normal = F.normalize(out2, p=2, dim=-1)
        # attscore = torch.einsum('bif,bjf->bij', out_normal,out_normal)
        # # 筛选邻居
        # attscore[attscore < 0.766] = 0
        # attscore = torch.softmax(10 * attscore, dim=-1)
        # out2 = torch.matmul(attscore, out1)
        return out1

    def att_pbranch_mutiheads(self, v_embed):
        adj, adj_eye = self.getadj(v_embed)
        embed = self.attgat(v_embed, adj, adj_eye)
        return embed

    def att_sbranch_mutiheads(self, v_embed):
        adj, adj_eye = self.getadj(v_embed)
        embed = self.attgat(v_embed, adj, adj_eye)
        return embed

    def att_branch_mutiheads(self, v_embed):
        adj, adj_eye = self.getadj(v_embed)
        embed = self.attgat(v_embed, adj, adj_eye)
        return embed

    def getadj(self, h, isKGConstrain=True):

        v = self.V.detach()
        adj = torch.matmul(v, v.T)
        if isKGConstrain:

            k1 = self.config.k1
            # 计算前n个相似度
            a, index1 = torch.sort(adj, dim=-1)
            a[:, 0:self.config.num_attribute - k1] = 0
            b, index2 = torch.sort(index1)
            adj = torch.gather(a, dim=1, index=index2)
            a = adj
            b = adj.T
            adj = torch.where(a == b, a, a + b)

        zeromask = adj <= 0
        onemask = adj > 0
        adj[zeromask] = 0
        adj_eye = adj.clone()
        adj[np.eye(self.config.num_attribute, dtype=np.bool)] = 0

        return adj, adj_eye

    def compute_loss_Self_Calibrate(self, in_package):
        S_pp = in_package['pred']
        if self.is_episode:
            batch_class = in_package['batch_class']
            S_pp1 = S_pp[:, batch_class]
            S_pp_unseen = S_pp[:, self.unseenclass]
            S_pp = torch.cat((S_pp1, S_pp_unseen), 1)
            assert S_pp.size(1) == len(batch_class) + len(self.unseenclass)

        Prob_all = F.softmax(S_pp, dim=-1)
        if self.is_episode:
            Prob_unseen = Prob_all[:, -len(self.unseenclass):]
        else:
            Prob_unseen = Prob_all[:, self.unseenclass]
        assert Prob_unseen.size(1) == len(self.unseenclass)
        mass_unseen = torch.sum(Prob_unseen, dim=1)
        loss_pmp = -torch.log(torch.mean(mass_unseen))
        return loss_pmp

    def compute_aug_cross_entropy(self, in_package):
        Labels = in_package['batch_label']
        S_pp = in_package['S_pp']

        if self.is_bias:
            S_pp = S_pp - self.vec_bias

        if not self.is_conservative:
            S_pp = S_pp[:, self.seenclass]
            Labels = Labels[:, self.seenclass]
            assert S_pp.size(1) == len(self.seenclass)
        if self.is_episode:
            batch_class = in_package['batch_class']
            S_pp1 = S_pp[:, batch_class]
            S_pp_unseen = S_pp[:, self.unseenclass]
            S_pp = torch.cat((S_pp1, S_pp_unseen), 1)
            Labels1 = Labels[:, batch_class]
            Labels_unseen = Labels[:, self.unseenclass]
            Labels = torch.cat((Labels1, Labels_unseen), 1)
            assert S_pp.size(1) == len(batch_class) + len(self.unseenclass)

        Prob = self.log_softmax_func(S_pp)

        loss = -torch.einsum('bk,bk->b', Prob, Labels)
        loss = torch.mean(loss)
        return loss

    def compute_reg_loss(self, in_package):
        tgt = torch.matmul(in_package['batch_label'], self.att)
        embed = in_package['embed']
        loss_reg = F.mse_loss(embed, tgt, reduction='mean')

        return loss_reg

    def discrepancy(self, out1, out2):
        return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))

    def discrepancy_slice_wasserstein(self, p1, p2):
        p1 = F.softmax(p1, dim=-1)
        p2 = F.softmax(p2, dim=-1)
        s = p1.shape
        if s[1] > 1:
            # proj = torch.randn(s[1], 128)
            proj = torch.randn(s[1], 128).to(device)
            proj *= torch.rsqrt(torch.sum(torch.mul(proj, proj), 0, keepdim=True))
            p1 = torch.matmul(p1, proj)
            p2 = torch.matmul(p2, proj)
        p1 = torch.topk(p1, s[0], dim=0)[0]
        p2 = torch.topk(p2, s[0], dim=0)[0]
        dist = p1 - p2
        wdist = torch.mean(torch.mul(dist, dist))

        return wdist

    def compute_kl_loss(self, in_package):
        v_prototype1 = F.normalize(in_package['vprototype_hide'], p=2, dim=-1)
        v_prototype2 = F.normalize(in_package['vprototype'], p=2, dim=-1)
        s_prototype1 = F.normalize(in_package['prototype_hide'], p=2, dim=-1)
        s_prototype2 = F.normalize(in_package['prototype'], p=2, dim=-1)

        v1_cos = torch.matmul(v_prototype1, v_prototype1.T)
        v2_cos = torch.matmul(v_prototype2, v_prototype2.T)
        s1_cos = torch.matmul(s_prototype1, s_prototype1.T)
        s2_cos = torch.matmul(s_prototype2, s_prototype2.T)

        # if self.is_episode:
        #     batch_class=in_package['batch_class']
        #     v1_cos1 = v1_cos[:, batch_class]
        #     v1_cos_unseen = v1_cos[:, self.unseenclass]
        #     v1_cos = torch.cat((v1_cos1, v1_cos_unseen), 1)
        #
        #     v2_cos1 = v2_cos[:, batch_class]
        #     v2_cos_unseen = v2_cos[:, self.unseenclass]
        #     v2_cos = torch.cat((v2_cos1, v2_cos_unseen), 1)
        #
        #     s1_cos1 = s1_cos[:, batch_class]
        #     s1_cos_unseen = s1_cos[:, self.unseenclass]
        #     s1_cos = torch.cat((s1_cos1, s1_cos_unseen), 1)
        #
        #     s2_cos1 = s2_cos[:, batch_class]
        #     s2_cos_unseen = s2_cos[:, self.unseenclass]
        #     s2_cos = torch.cat((s2_cos1, s2_cos_unseen), 1)

        v1_prob = torch.log_softmax(v1_cos, dim=-1)
        v2_prob = torch.log_softmax(v2_cos, dim=-1)
        s1_prob = torch.softmax(s1_cos, dim=-1)
        s2_prob = torch.softmax(s2_cos, dim=-1)

        loss_kl = F.kl_div(v1_prob, s1_prob, reduction='sum') + F.kl_div(v2_prob, s2_prob, reduction='sum')
        return loss_kl

    def compute_kl_class_loss(self, in_package):
        batch_label = in_package['batch_label_origin']
        normalize_embed = F.normalize(in_package['embed'], p=2, dim=-1)

        normalize_att = F.normalize(self.att, p=2, dim=-1)[batch_label]

        embed_cos = torch.matmul(normalize_embed, normalize_att.T)
        att_cos = torch.matmul(normalize_att, normalize_att.T)

        # if self.is_episode:
        #     batch_class=in_package['batch_class']
        #     v1_cos1 = v1_cos[:, batch_class]
        #     v1_cos_unseen = v1_cos[:, self.unseenclass]
        #     v1_cos = torch.cat((v1_cos1, v1_cos_unseen), 1)
        #
        #     v2_cos1 = v2_cos[:, batch_class]
        #     v2_cos_unseen = v2_cos[:, self.unseenclass]
        #     v2_cos = torch.cat((v2_cos1, v2_cos_unseen), 1)
        #
        #     s1_cos1 = s1_cos[:, batch_class]
        #     s1_cos_unseen = s1_cos[:, self.unseenclass]
        #     s1_cos = torch.cat((s1_cos1, s1_cos_unseen), 1)
        #
        #     s2_cos1 = s2_cos[:, batch_class]
        #     s2_cos_unseen = s2_cos[:, self.unseenclass]
        #     s2_cos = torch.cat((s2_cos1, s2_cos_unseen), 1)

        embed_prob = torch.log_softmax(embed_cos, dim=-1)
        att_prob = torch.softmax(att_cos, dim=-1)

        loss_kl = F.kl_div(embed_prob, att_prob, reduction='sum')
        return loss_kl

    def compute_kl_loss_nogat(self, in_package):
        v_prototype1 = F.normalize(in_package['vprototype_origin_embed'], p=2, dim=-1)
        s_prototype1 = F.normalize(in_package['att_batch_vp'], p=2, dim=-1)

        v1_cos = torch.matmul(v_prototype1, v_prototype1.T)
        s1_cos = torch.matmul(s_prototype1, s_prototype1.T)
        v1_prob = torch.log_softmax(v1_cos, dim=-1)
        s1_prob = torch.softmax(s1_cos, dim=-1)

        loss_kl = F.kl_div(v1_prob, s1_prob, reduction='sum')
        return loss_kl

    def compute_complete_loss(self, in_package, type='l2'):
        graphs = in_package['graphs']
        graphs = F.normalize(graphs, p=2, dim=-1)
        if type == 'l2':
            loss = F.mse_loss(graphs[0], graphs[1], reduction='sum')
        elif type == 'cos':
            loss = 1 - torch.matmul(graphs[0], graphs[1].T)
        return loss

    def compute_wd_loss(self, in_package):
        v_prototype1 = F.normalize(in_package['vprototype_hide'], p=2, dim=-1)
        v_prototype2 = F.normalize(in_package['vprototype'], p=2, dim=-1)
        s_prototype1 = F.normalize(in_package['prototype_hide'], p=2, dim=-1)
        s_prototype2 = F.normalize(in_package['prototype'], p=2, dim=-1)

        v1_cos = torch.matmul(v_prototype1, v_prototype1.T)
        v2_cos = torch.matmul(v_prototype2, v_prototype2.T)
        s1_cos = torch.matmul(s_prototype1, s_prototype1.T)
        s2_cos = torch.matmul(s_prototype2, s_prototype2.T)

        # if self.is_episode:
        #     batch_class=in_package['batch_class']
        #     v1_cos1 = v1_cos[:, batch_class]
        #     v1_cos_unseen = v1_cos[:, self.unseenclass]
        #     v1_cos = torch.cat((v1_cos1, v1_cos_unseen), 1)
        #
        #     v2_cos1 = v2_cos[:, batch_class]
        #     v2_cos_unseen = v2_cos[:, self.unseenclass]
        #     v2_cos = torch.cat((v2_cos1, v2_cos_unseen), 1)
        #
        #     s1_cos1 = s1_cos[:, batch_class]
        #     s1_cos_unseen = s1_cos[:, self.unseenclass]
        #     s1_cos = torch.cat((s1_cos1, s1_cos_unseen), 1)
        #
        #     s2_cos1 = s2_cos[:, batch_class]
        #     s2_cos_unseen = s2_cos[:, self.unseenclass]
        #     s2_cos = torch.cat((s2_cos1, s2_cos_unseen), 1)

        loss_wd = self.discrepancy_slice_wasserstein(v1_cos, s1_cos) + self.discrepancy_slice_wasserstein(v2_cos,
                                                                                                          s2_cos)
        return loss_wd

    def compute_loss(self, in_package):
        if len(in_package['batch_label'].size()) == 1:
            in_package['batch_label_origin'] = torch.tensor(in_package['batch_label']).to(self.config.device)
            in_package['batch_label'] = self.weight_ce[in_package['batch_label']]

        loss_CE = self.compute_aug_cross_entropy(in_package)
        loss_cal = self.compute_loss_Self_Calibrate(in_package)
        loss_reg = self.compute_reg_loss(in_package)
        # loss_reg_att = self.compute_att_reg_loss(in_package)
        loss_kl = self.compute_kl_loss(in_package)
        #loss_kl=self.compute_kl_loss_nogat(in_package)

        # loss_kl_class=self.compute_kl_class_loss(in_package)
        loss = loss_CE + self.config.lambda_ * loss_cal + self.config.lambda_reg * loss_reg + self.config.lambda_kl * loss_kl
        # loss = loss_CE + self.config.lambda_reg * loss_reg + self.config.lambda_kl * loss_kl
        out_package = {'loss': loss, 'loss_CE': loss_CE, 'loss_cal': loss_cal,
                       'loss_reg': loss_reg, 'loss_kl': loss_kl, 'loss_kl_class': loss_kl}
        if self.is_complete_graph_loss:
            loss_cpl = self.compute_complete_loss(in_package, type='l2')
            loss += self.config.lambda_complete_loss * loss_cpl
            out_package['loss_cpl'] = loss_cpl
        return out_package


class Transformer(nn.Module):
    def __init__(self, ec_layer=1, dc_layer=1, dim_com=300,
                 dim_feedforward=2048, dropout=0.1, heads=1,
                 in_dim_cv=2048, in_dim_attr=300, SAtt=True,
                 aux_embed=True):
        super(Transformer, self).__init__()
        # input embedding
        self.embed_cv = nn.Sequential(nn.Linear(in_dim_cv, dim_com))
        if aux_embed:
            self.embed_cv_aux = nn.Sequential(nn.Linear(in_dim_cv, dim_com))
        self.embed_attr = nn.Sequential(nn.Linear(in_dim_attr, dim_com))
        # transformer encoder
        self.transformer_encoder = MultiLevelEncoder_woPad(N=ec_layer,
                                                           d_model=dim_com,
                                                           h=1,
                                                           d_k=dim_com,
                                                           d_v=dim_com,
                                                           d_ff=dim_feedforward,
                                                           dropout=dropout)
        # transformer decoder
        decoder_layer = TransformerDecoderLayer(d_model=dim_com,
                                                nhead=heads,
                                                dim_feedforward=dim_feedforward,
                                                dropout=dropout,
                                                SAtt=SAtt)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=dc_layer)

    def forward(self, f_cv, f_attr):
        # linearly map to common dim
        h_cv = self.embed_cv(f_cv.permute(0, 2, 1))
        h_attr = self.embed_attr(f_attr)
        h_attr_batch = h_attr.unsqueeze(0).repeat(f_cv.shape[0], 1, 1)
        # visual encoder
        memory = self.transformer_encoder(h_cv).permute(1, 0, 2)
        # attribute-visual decoder
        out = self.transformer_decoder(h_attr_batch.permute(1, 0, 2), memory)
        return out.permute(1, 0, 2)


class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048,
                 dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(EncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadGeometryAttention(d_model, d_k, d_v, h, dropout,
                                                identity_map_reordering=identity_map_reordering,
                                                attention_module=attention_module,
                                                attention_module_kwargs=attention_module_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.lnorm = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(
            d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, relative_geometry_weights,
                attention_mask=None, attention_weights=None, pos=None):
        q, k = (queries + pos, keys +
                pos) if pos is not None else (queries, keys)
        att = self.mhatt(q, k, values, relative_geometry_weights,
                         attention_mask, attention_weights)
        att = self.lnorm(queries + self.dropout(att))
        ff = self.pwff(att)
        return ff


class MultiLevelEncoder_woPad(nn.Module):
    def __init__(self, N, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048,
                 dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(MultiLevelEncoder_woPad, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)])

        self.WGs = nn.ModuleList(
            [nn.Linear(64, 1, bias=True) for _ in range(h)])

    def forward(self, input, attention_mask=None, attention_weights=None, pos=None):
        relative_geometry_embeddings = BoxRelationalEmbedding(
            input, grid_size=(14, 14))
        flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(
            -1, 64)
        box_size_per_head = list(relative_geometry_embeddings.shape[:3])
        box_size_per_head.insert(1, 1)
        relative_geometry_weights_per_head = [layer(
            flatten_relative_geometry_embeddings).view(box_size_per_head) for layer in self.WGs]
        relative_geometry_weights = torch.cat(
            (relative_geometry_weights_per_head), 1)
        relative_geometry_weights = F.relu(relative_geometry_weights)
        out = input
        for layer in self.layers:
            out = layer(out, out, out, relative_geometry_weights,
                        attention_mask, attention_weights, pos=pos)
        return out


class TransformerDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", SAtt=True):
        super(TransformerDecoderLayer, self).__init__(d_model, nhead,
                                                      dim_feedforward=dim_feedforward,
                                                      dropout=dropout,
                                                      activation=activation)
        self.SAtt = SAtt

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        if self.SAtt:
            tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
            tgt = self.norm1(tgt + self.dropout1(tgt2))
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


def get_relative_pos(x, batch_size, norm_len):
    x = x.view(1, -1, 1).expand(batch_size, -1, -1)
    return x / norm_len


def get_grids_pos(batch_size, seq_len, grid_size=(7, 7)):
    assert seq_len == grid_size[0] * grid_size[1]
    x = torch.arange(0, grid_size[0]).float().to(device)
    y = torch.arange(0, grid_size[1]).float().to(device)
    # x = torch.arange(0, grid_size[0]).float()
    # y = torch.arange(0, grid_size[1]).float()
    px_min = x.view(-1, 1).expand(-1, grid_size[0]).contiguous().view(-1)
    py_min = y.view(1, -1).expand(grid_size[1], -1).contiguous().view(-1)
    px_max = px_min + 1
    py_max = py_min + 1
    rpx_min = get_relative_pos(px_min, batch_size, grid_size[0])
    rpy_min = get_relative_pos(py_min, batch_size, grid_size[1])
    rpx_max = get_relative_pos(px_max, batch_size, grid_size[0])
    rpy_max = get_relative_pos(py_max, batch_size, grid_size[1])
    return rpx_min, rpy_min, rpx_max, rpy_max


def BoxRelationalEmbedding(f_g, dim_g=64, wave_len=1000, trignometric_embedding=True,
                           grid_size=(7, 7)):
    batch_size, seq_len = f_g.size(0), f_g.size(1)
    x_min, y_min, x_max, y_max = get_grids_pos(batch_size, seq_len, grid_size)
    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    w = (x_max - x_min) + 1.
    h = (y_max - y_min) + 1.
    delta_x = cx - cx.view(batch_size, 1, -1)
    delta_x = torch.clamp(torch.abs(delta_x / w), min=1e-3)
    delta_x = torch.log(delta_x)
    delta_y = cy - cy.view(batch_size, 1, -1)
    delta_y = torch.clamp(torch.abs(delta_y / h), min=1e-3)
    delta_y = torch.log(delta_y)
    delta_w = torch.log(w / w.view(batch_size, 1, -1))
    delta_h = torch.log(h / h.view(batch_size, 1, -1))
    matrix_size = delta_h.size()
    delta_x = delta_x.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_y = delta_y.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_w = delta_w.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_h = delta_h.view(batch_size, matrix_size[1], matrix_size[2], 1)
    position_mat = torch.cat((delta_x, delta_y, delta_w, delta_h), -1)
    if trignometric_embedding == True:
        feat_range = torch.arange(dim_g / 8).to(device)
        # feat_range = torch.arange(dim_g / 8)
        dim_mat = feat_range / (dim_g / 8)
        dim_mat = 1. / (torch.pow(wave_len, dim_mat))
        dim_mat = dim_mat.view(1, 1, 1, -1)
        position_mat = position_mat.view(
            batch_size, matrix_size[1], matrix_size[2], 4, -1)
        position_mat = 100. * position_mat
        mul_mat = position_mat * dim_mat
        mul_mat = mul_mat.view(batch_size, matrix_size[1], matrix_size[2], -1)
        sin_mat = torch.sin(mul_mat)
        cos_mat = torch.cos(mul_mat)
        embedding = torch.cat((sin_mat, cos_mat), -1)
    else:
        embedding = position_mat
    return (embedding)


class ScaledDotProductGeometryAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, dropout=.1, comment=None):
        super(ScaledDotProductGeometryAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.init_weights()
        self.comment = comment

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, box_relation_embed_matrix,
                attention_mask=None, attention_weights=None):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries).view(b_s, nq, self.h,
                                    self.d_k).permute(0, 2, 1, 3)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)
        v = self.fc_v(values).view(b_s, nk, self.h,
                                   self.d_v).permute(0, 2, 1, 3)
        att = torch.matmul(q, k) / np.sqrt(self.d_k)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        w_g = box_relation_embed_matrix
        w_a = att
        w_mn = - w_g + w_a
        w_mn = torch.softmax(w_mn, -1)
        att = self.dropout(w_mn)
        out = torch.matmul(att, v).permute(
            0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out = self.fc_o(out)
        return out


class MultiHeadGeometryAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, dropout=.1, identity_map_reordering=False,
                 can_be_stateful=False, attention_module=None,
                 attention_module_kwargs=None, comment=None):
        super(MultiHeadGeometryAttention, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.attention = ScaledDotProductGeometryAttention(
            d_model=d_model, d_k=d_k, d_v=d_v, h=h, comment=comment)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.can_be_stateful = can_be_stateful
        if self.can_be_stateful:
            self.register_state('running_keys', torch.zeros((0, d_model)))
            self.register_state('running_values', torch.zeros((0, d_model)))

    def forward(self, queries, keys, values, relative_geometry_weights,
                attention_mask=None, attention_weights=None):
        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            keys = self.running_keys
            self.running_values = torch.cat([self.running_values, values], 1)
            values = self.running_values
        if self.identity_map_reordering:
            q_norm = self.layer_norm(queries)
            k_norm = self.layer_norm(keys)
            v_norm = self.layer_norm(values)
            out = self.attention(q_norm, k_norm, v_norm, relative_geometry_weights,
                                 attention_mask, attention_weights)
            out = queries + self.dropout(torch.relu(out))
        else:
            out = self.attention(queries, keys, values, relative_geometry_weights,
                                 attention_mask, attention_weights)
            out = self.dropout(out)
            out = self.layer_norm(queries + out)
        return out


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, dropout=.1, identity_map_reordering=False):
        super(PositionWiseFeedForward, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input):
        if self.identity_map_reordering:
            out = self.layer_norm(input)
            out = self.fc2(self.dropout_2(F.relu(self.fc1(out))))
            out = input + self.dropout(torch.relu(out))
        else:
            out = self.fc2(self.dropout_2(F.relu(self.fc1(input))))
            out = self.dropout(out)
            out = self.layer_norm(input + out)
        return out


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, k1, k2, is_att_dp=False, is_gcnWh=False):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions1 = [
            GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, k1=k1, k2=k2, is_att_dp=is_att_dp,
                                is_gcnWh=is_gcnWh, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions1):
            self.add_module('attention1_{}'.format(i), attention)

        # self.attentions2 = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
        #                    range(nheads)]
        # for i, attention in enumerate(self.attentions2):
        #     self.add_module('attention2_{}'.format(i), attention)

        # self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj, adj_eye):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj, adj_eye) for att in self.attentions1], dim=-1)
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = torch.cat([att(x, adj,adj_eye) for att in self.attentions2], dim=-1)
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.elu(self.out_att(x, adj))
        return x


class GraphAggregator(nn.Module):
    """This module computes graph representations by aggregating from parts."""

    def __init__(self,
                 node_hidden_sizes,
                 graph_transform_sizes=None,
                 input_size=None,
                 gated=True,
                 aggregation_type='sum',
                 name='graph-aggregator'):
        """Constructor.

        Args:
          node_hidden_sizes: the hidden layer sizes of the node transformation nets.
            The last element is the size of the aggregated graph representation.

          graph_transform_sizes: sizes of the transformation layers on top of the
            graph representations.  The last element of this list is the final
            dimensionality of the output graph representations.

          gated: set to True to do gated aggregation, False not to.

          aggregation_type: one of {sum, max, mean, sqrt_n}.
          name: name of this module.
        """
        super(GraphAggregator, self).__init__()

        self._node_hidden_sizes = node_hidden_sizes
        self._graph_transform_sizes = graph_transform_sizes
        self._graph_state_dim = node_hidden_sizes[-1]
        self._input_size = input_size
        #  The last element is the size of the aggregated graph representation.
        self._gated = gated
        self._aggregation_type = aggregation_type
        self._aggregation_op = None
        self.MLP1, self.MLP2 = self.build_model()

    def build_model(self):
        node_hidden_sizes = self._node_hidden_sizes
        if self._gated:
            node_hidden_sizes[-1] = self._graph_state_dim * 2

        layer = []
        layer.append(nn.Linear(self._input_size[0], node_hidden_sizes[0]))
        for i in range(1, len(node_hidden_sizes)):
            layer.append(nn.ReLU())
            layer.append(nn.Linear(node_hidden_sizes[i - 1], node_hidden_sizes[i]))
        MLP1 = nn.Sequential(*layer)

        if (self._graph_transform_sizes is not None and
                len(self._graph_transform_sizes) > 0):
            layer = []
            layer.append(nn.Linear(self._graph_state_dim, self._graph_transform_sizes[0]))
            for i in range(1, len(self._graph_transform_sizes)):
                layer.append(nn.ReLU())
                layer.append(nn.Linear(self._graph_transform_sizes[i - 1], self._graph_transform_sizes[i]))
            MLP2 = nn.Sequential(*layer)

        return MLP1, MLP2

    def forward(self, node_states, graph_idx, n_graphs):
        """Compute aggregated graph representations.

        Args:
          node_states: [n_nodes, node_state_dim] float tensor, node states of a
            batch of graphs concatenated together along the first dimension.
          graph_idx: [n_nodes] int tensor, graph ID for each node.
          n_graphs: integer, number of graphs in this batch.

        Returns:
          graph_states: [n_graphs, graph_state_dim] float tensor, graph
            representations, one row for each graph.
        """

        node_states_g = self.MLP1(node_states)

        if self._gated:
            gates = torch.sigmoid(node_states_g[:, :self._graph_state_dim])
            node_states_g = node_states_g[:, self._graph_state_dim:] * gates

        graph_states = unsorted_segment_sum(node_states_g, graph_idx, n_graphs)

        if self._aggregation_type == 'max':
            # reset everything that's smaller than -1e5 to 0.
            graph_states *= torch.FloatTensor(graph_states > -1e5)
        # transform the reduced graph states further

        if (self._graph_transform_sizes is not None and
                len(self._graph_transform_sizes) > 0):
            graph_states = self.MLP2(graph_states)

        return graph_states


def unsorted_segment_sum(data, segment_ids, num_segments):
    """
    Computes the sum along segments of a tensor. Analogous to tf.unsorted_segment_sum.

    :param data: A tensor whose segments are to be summed.
    :param segment_ids: The segment indices tensor.
    :param num_segments: The number of segments.
    :return: A tensor of same data type as the data argument.
    """

    assert all([i in data.shape for i in segment_ids.shape]), "segment_ids.shape should be a prefix of data.shape"

    # Encourage to use the below code when a deterministic result is
    # needed (reproducibility). However, the code below is with low efficiency.

    # tensor = torch.zeros(num_segments, data.shape[1], device=data.device)
    # for index in range(num_segments):
    #     tensor[index, :] = torch.sum(data[segment_ids == index, :], dim=0)
    # return tensor

    if len(segment_ids.shape) == 1:
        s = torch.prod(torch.tensor(data.shape[1:], device=data.device)).long()
        segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *data.shape[1:])

    assert data.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

    shape = [num_segments] + list(data.shape[1:])
    tensor = torch.zeros(*shape, device=data.device).scatter_add(0, segment_ids, data)
    tensor = tensor.type(data.dtype)
    return tensor


if __name__ == '__main__':
    pass
