"""Torch modules for graph attention networks(GAT)."""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn

import dgl
from dgl import function as fn
from dgl.nn.pytorch import edge_softmax
from dgl._ffi.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair



class myGATConv(nn.Module):
    """
    Adapted from
    https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/gatconv.html#GATConv
    """

    def __init__(
        self,
        edge_feats,
        num_etypes,
        in_feats,
        out_feats,
        num_heads,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
        bias=False,
        alpha=0.0,
    ):
        super(myGATConv, self).__init__()
        self._edge_feats = edge_feats
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.edge_emb = nn.Embedding(num_etypes, edge_feats)
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
        self.fc_e = nn.Linear(edge_feats, edge_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_e = nn.Parameter(th.FloatTensor(size=(1, num_heads, edge_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer("res_fc", None)
        self.reset_parameters()
        self.activation = activation
        self.bias = bias
        if bias:
            self.bias_param = nn.Parameter(th.zeros((1, num_heads, out_feats)))
        self.alpha = alpha

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_e.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, e_feat, res_attn=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )

            h_src = h_dst = self.feat_drop(feat)
            feat_src = feat_dst = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
            if graph.is_block:
                feat_dst = feat_src[: graph.number_of_dst_nodes()]
            e_feat = self.edge_emb(e_feat)
            e_feat = self.fc_e(e_feat).view(-1, self._num_heads, self._edge_feats)
            ee = (e_feat * self.attn_e).sum(dim=-1).unsqueeze(-1)
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({"ft": feat_src, "el": el})
            graph.dstdata.update({"er": er})
            graph.edata.update({"ee": ee})
            graph.apply_edges(fn.u_add_v("el", "er", "e"))
            e = self.leaky_relu(graph.edata.pop("e") + graph.edata.pop("ee"))
            # compute softmax
            graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))
            if res_attn is not None:
                graph.edata["a"] = graph.edata["a"] * (1 - self.alpha) + res_attn * self.alpha
            # message passing
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            # bias
            if self.bias:
                rst = rst + self.bias_param
            # activation
            if self.activation:
                rst = self.activation(rst)
            return rst, graph.edata.pop("a").detach()


class myHeteroGATConvSample(nn.Module):
    def __init__(
        self,
        edge_feats,
        num_etypes,
        in_feats,
        out_feats,
        num_heads,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
        bias=False,
        alpha=0.0,
        share_weight=False,
        device=None,
    ):
        super(myHeteroGATConvSample, self).__init__()
        self._edge_feats = edge_feats
        self._num_heads = num_heads
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._in_src_feats = self._in_dst_feats = in_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._shared_weight=share_weight
        # self.edge_emb = nn.Embedding(num_etypes, edge_feats)
        self.edge_emb = nn.Parameter(th.FloatTensor(size=(num_etypes, edge_feats)))
        if not share_weight:
            self.fc = self.weight = nn.ModuleDict({
                    name: nn.Linear(in_feats[name], out_feats * num_heads, bias=False) for name in in_feats
            })
        else:
            in_dim = None
            for name in in_feats:
                if in_dim:
                    assert in_dim == in_feats[name]
                else:
                    in_dim = in_feats[name]
            self.fc = nn.Linear(in_dim, out_feats * num_heads, bias=False)
        self.fc_e = nn.Linear(edge_feats, edge_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_e = nn.Parameter(th.FloatTensor(size=(1, num_heads, edge_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._shared_weight:
                in_dim = None
                for name in in_feats:
                    if in_dim:
                        assert in_dim == in_feats[name]
                    else:
                        in_dim = in_feats[name]
                if  in_dim != num_heads * out_feats:
                    self.res_fc = nn.Linear(in_dim, num_heads * out_feats, bias=False)
                else:
                    self.res_fc = Identity()
            else:
                self.res_fc = nn.ModuleDict()
                for ntype in in_feats.keys():
                    if self._in_dst_feats[ntype] != num_heads * out_feats:
                        self.res_fc[ntype] = nn.Linear(self._in_dst_feats[ntype], num_heads * out_feats, bias=False)
                    else:
                        self.res_fc[ntype] = Identity()
        else:
            self.register_buffer("res_fc", None)
        self.reset_parameters()
        self.activation = activation
        self.bias = bias
        if bias:
            self.bias_param = nn.Parameter(th.zeros((1, num_heads, out_feats)))
        self.alpha = alpha
        self.device = device

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        if self._shared_weight:
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            for name in self.fc:
                nn.init.xavier_normal_(self.fc[name].weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_e.weight, gain=gain)
        nn.init.normal_(self.edge_emb, 0, 1)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, src_nfeat, res_attn=None):
        with graph.local_scope():
            funcs = {}
            ntypes_src = []
            ntypes_dst = []
            etypes = []
            for ntype in src_nfeat:
                if graph.number_of_nodes(ntype)>0:
                    ntypes_src.append(ntype)
                if graph.num_dst_nodes(ntype)>0:
                    ntypes_dst.append(ntype)
            for src, etype, dst in graph.canonical_etypes:
                if graph.number_of_edges(etype)>0:
                    etypes.append((src, etype, dst))


            dst_nfeat = {}

            for ntype in ntypes_dst:
                if self.res_fc is not None:
                    if graph.is_block:
                        graph.dstnodes[ntype].data['h'] = src_nfeat[ntype][:graph.number_of_dst_nodes(ntype)]
                    else:
                        graph.dstnodes[ntype].data['h'] = src_nfeat[ntype]

            for ntype in ntypes_src:
                src_nfeat[ntype] = self.feat_drop(src_nfeat[ntype])
                if self._shared_weight:
                    src_nfeat[ntype] = self.fc(src_nfeat[ntype]).view(-1, self._num_heads, self._out_feats)
                else:
                    src_nfeat[ntype] = self.fc[ntype](src_nfeat[ntype]).view(-1, self._num_heads, self._out_feats)
                if graph.is_block:
                    if graph.number_of_dst_nodes(ntype) > 0:
                        dst_nfeat[ntype] = src_nfeat[ntype][:graph.number_of_dst_nodes(ntype)]
                else:
                    dst_nfeat[ntype] = src_nfeat[ntype]
                graph.srcnodes[ntype].data['ft'] = src_nfeat[ntype]

            for src, etype, dst in graph.canonical_etypes:
                if graph.number_of_edges(etype) <= 0:
                    graph.edges[etype].data["a"] = th.zeros((0, self._num_heads, 1)).to(self.device)
                    continue
                feat_src = src_nfeat[src]
                feat_dst = dst_nfeat[dst]
                el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
                graph.srcnodes[src].data['el'] = el
                er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
                graph.dstnodes[dst].data['er'] = er
                e_feat = self.edge_emb[int(etype)].unsqueeze(0)
                e_feat = self.fc_e(e_feat).view(-1, self._num_heads, self._edge_feats)
                ee = (e_feat * self.attn_e).sum(dim=-1).unsqueeze(-1).expand(graph.number_of_edges(etype), self._num_heads, 1)
                graph.apply_edges(fn.u_add_v("el", "er", "e"), etype=etype)
                graph.edges[etype].data["a"] = self.leaky_relu(graph.edges[etype].data.pop("e") + ee)

            hg = dgl.to_homogeneous(graph, edata=["a"])
            a = self.attn_drop(edge_softmax(hg, hg.edata.pop("a")))
            e_t = hg.edata['_TYPE']

            for src, etype, dst in etypes:
                t = graph.get_etype_id(etype)
                graph.edges[etype].data["a"] = a[e_t == t]
                if res_attn is not None:
                    if graph.is_block:
                        graph.edges[etype].data["a"] = graph.edges[etype].data["a"] * (1 - self.alpha) + res_attn[
                            etype][:graph.number_of_edges(etype)] * self.alpha
                    else:
                        graph.edges[etype].data["a"] = graph.edges[etype].data["a"] * (1 - self.alpha) + res_attn[
                            etype] * self.alpha
                funcs[etype] = (fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))

            graph.multi_update_all(funcs, 'sum')
            rst = graph.dstdata["ft"]
            # graph.edata.pop("el")
            # graph.edata.pop("er")
            # residual
            if self.res_fc is not None:
                for ntype in ntypes_dst:
                    if self._shared_weight:
                        rst[ntype] = self.res_fc(graph.dstnodes[ntype].data['h']).view(
                            graph.dstnodes[ntype].data['h'].shape[0], -1, self._out_feats) + rst[ntype]
                    else:
                        rst[ntype] = self.res_fc[ntype](graph.dstnodes[ntype].data['h']).view(
                            graph.dstnodes[ntype].data['h'].shape[0], -1, self._out_feats) + rst[ntype]
            # bias
            if self.bias:
                for ntype in ntypes_dst:
                    rst[ntype] = rst[ntype] + self.bias_param
            # activation
            if self.activation:
                for ntype in ntypes_dst:
                    rst[ntype] = self.activation(rst[ntype])
            res_attn = {e: graph.edges[e].data["a"].detach() for s, e, d in etypes}
            graph.edata.pop("a")
            return rst, res_attn


class myHeteroGATConv(nn.Module):
    def __init__(
        self,
        edge_feats,
        num_etypes,
        in_feats,
        out_feats,
        num_heads,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
        bias=False,
        alpha=0.0,
        share_weight=False,
    ):
        super(myHeteroGATConv, self).__init__()
        self._edge_feats = edge_feats
        self._num_heads = num_heads
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._in_src_feats = self._in_dst_feats = in_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._shared_weight=share_weight
        # self.edge_emb = nn.Embedding(num_etypes, edge_feats)
        self.edge_emb = nn.Parameter(th.FloatTensor(size=(num_etypes, edge_feats)))
        if not share_weight:
            self.fc = self.weight = nn.ModuleDict({
                    name: nn.Linear(in_feats[name], out_feats * num_heads, bias=False) for name in in_feats
            })
        else:
            in_dim = None
            for name in in_feats:
                if in_dim:
                    assert in_dim == in_feats[name]
                else:
                    in_dim = in_feats[name]
            self.fc = nn.Linear(in_dim, out_feats * num_heads, bias=False)
        self.fc_e = nn.Linear(edge_feats, edge_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_e = nn.Parameter(th.FloatTensor(size=(1, num_heads, edge_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._shared_weight:
                in_dim = None
                for name in in_feats:
                    if in_dim:
                        assert in_dim == in_feats[name]
                    else:
                        in_dim = in_feats[name]
                if  in_dim != num_heads * out_feats:
                    self.res_fc = nn.Linear(in_dim, num_heads * out_feats, bias=False)
                else:
                    self.res_fc = Identity()
            else:
                self.res_fc = nn.ModuleDict()
                for ntype in in_feats.keys():
                    if self._in_dst_feats[ntype] != num_heads * out_feats:
                        self.res_fc[ntype] = nn.Linear(self._in_dst_feats[ntype], num_heads * out_feats, bias=False)
                    else:
                        self.res_fc[ntype] = Identity()
        else:
            self.register_buffer("res_fc", None)
        self.reset_parameters()
        self.activation = activation
        self.bias = bias
        if bias:
            self.bias_param = nn.Parameter(th.zeros((1, num_heads, out_feats)))
        self.alpha = alpha

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        if self._shared_weight:
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            for name in self.fc:
                nn.init.xavier_normal_(self.fc[name].weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_e.weight, gain=gain)
        nn.init.normal_(self.edge_emb, 0, 1)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, nfeat, res_attn=None):
        with graph.local_scope():
            funcs = {}

            for ntype in graph.ntypes:
                h = self.feat_drop(nfeat[ntype])
                if self._shared_weight:
                    feat = self.fc(h).view(-1, self._num_heads, self._out_feats)
                else:
                    feat = self.fc[ntype](h).view(-1, self._num_heads, self._out_feats)
                graph.nodes[ntype].data['ft'] = feat
                if self.res_fc is not None:
                    graph.nodes[ntype].data['h'] = h

            for src, etype, dst in graph.canonical_etypes:
                feat_src = graph.nodes[src].data['ft']
                feat_dst = graph.nodes[dst].data['ft']
                el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
                graph.nodes[src].data['el'] = el
                er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
                graph.nodes[dst].data['er'] = er
                e_feat = self.edge_emb[int(etype)].unsqueeze(0)
                e_feat = self.fc_e(e_feat).view(-1, self._num_heads, self._edge_feats)
                ee = (e_feat * self.attn_e).sum(dim=-1).unsqueeze(-1).expand(graph.number_of_edges(etype), self._num_heads, 1)
                graph.apply_edges(fn.u_add_v("el", "er", "e"), etype=etype)
                graph.edges[etype].data["a"] = self.leaky_relu(graph.edges[etype].data.pop("e") + ee)

            hg = dgl.to_homogeneous(graph, edata=["a"])
            a = self.attn_drop(edge_softmax(hg, hg.edata.pop("a")))
            e_t = hg.edata['_TYPE']

            for src, etype, dst in graph.canonical_etypes:
                t = graph.get_etype_id(etype)
                graph.edges[etype].data["a"] = a[e_t == t]
                if res_attn is not None:
                    graph.edges[etype].data["a"] = graph.edges[etype].data["a"] * (1 - self.alpha) + res_attn[etype] * self.alpha
                funcs[etype] = (fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))

            graph.multi_update_all(funcs, 'sum')
            rst = graph.ndata.pop('ft')
            graph.edata.pop("el")
            graph.edata.pop("er")
            # residual
            if self.res_fc is not None:
                for ntype in graph.ntypes:
                    if self._shared_weight:
                        rst[ntype] = self.res_fc(graph.nodes[ntype].data['h']).view(graph.nodes[ntype].data['h'].shape[0], -1, self._out_feats) + rst[ntype]
                    else:
                        rst[ntype] = self.res_fc[ntype](graph.nodes[ntype].data['h']).view(
                            graph.nodes[ntype].data['h'].shape[0], -1, self._out_feats) + rst[ntype]
            # bias
            if self.bias:
                for ntype in graph.ntypes:
                    rst[ntype] = rst[ntype] + self.bias_param
            # activation
            if self.activation:
                for ntype in graph.ntypes:
                    rst[ntype] = self.activation(rst[ntype])
            res_attn = {e: graph.edges[e].data["a"].detach() for e in graph.etypes}
            graph.edata.pop("a")
            return rst, res_attn







