import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import dgl

from conv import myHeteroGATConv

import torch.nn.functional as F

import argparse
from tqdm import tqdm
import copy
from sklearn.metrics import f1_score, roc_auc_score
import pickle
from dgl.nn import GraphConv
import pandas as pd
import os


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def accuracy(y_pred, y_true):
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[1].type_as(y_true)
    correct = preds.eq(y_true).double()
    correct = correct.sum().item()
    return correct / len(y_true)


class Attention(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(Attention, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        self.tanh = nn.Tanh()
        self.att = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)

        self.softmax = nn.Softmax(dim=-1)
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

    def forward(self, embeds):
        beta = []
        attn_curr = self.attn_drop(self.att)
        for embed in embeds:
            sp = self.tanh(self.fc(embed)).mean(dim=0)
            beta.append(attn_curr.matmul(sp.t()))
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)
        z = 0
        for i in range(len(embeds)):
            z += embeds[i]*beta[i]
        return z


class GCN(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, h_feats)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h


class SimpleHeteroHGN(nn.Module):
    r"""The Simple-HGN model from the `"Are we really making much progress? Revisiting, benchmarking, and refining heterogeneous graph neural networks"`_ paper

    Args:
        num_features (int) : Number of input features.
        num_classes (int) : Number of classes.
        hidden_size (int) : The dimension of node representation.
        dropout (float) : Dropout rate for model training.
    """

    def __init__(
        self,
        edge_dim,
        num_etypes,
        in_dims,
        num_hidden,
        num_classes,
        num_layers,
        heads,
        feat_drop,
        attn_drop,
        negative_slope,
        residual,
        alpha,
        target_ntype,
        shared_weight=False,
    ):
        super(SimpleHeteroHGN, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        # self.device = torch.device("cuda:1" if torch.cuda.is_available() and use_cuda else "cpu")
        self.device = torch.device(f"cuda:{args.gpu}")
        self.g = None
        self.g_cs = []
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = F.elu

        # contrastive gcn
        self.gcn = GCN(in_dims[target_ntype], num_hidden * heads[-2])
        self.att = Attention(num_hidden * heads[-2], 0)

        # input projection (no residual)
        self.gat_layers.append(
            myHeteroGATConv(
                edge_dim,
                num_etypes,
                in_dims,
                num_hidden,
                heads[0],
                feat_drop,
                attn_drop,
                negative_slope,
                False,
                self.activation,
                alpha=alpha,
            )
        )
        # hidden layers
        for l in range(1, num_layers):  # noqa E741
            # due to multi-head, the in_dim = num_hidden * num_heads
            in_dims = {n: num_hidden * heads[l - 1] for n in in_dims}
            self.gat_layers.append(
                myHeteroGATConv(
                    edge_dim,
                    num_etypes,
                    in_dims,
                    num_hidden,
                    heads[l],
                    feat_drop,
                    attn_drop,
                    negative_slope,
                    residual,
                    self.activation,
                    alpha=alpha,
                    share_weight=shared_weight,
                )
            )
        # output projection
        in_dims = num_hidden * heads[-2]
        self.fc = nn.Linear(in_dims, num_classes)
        self.epsilon = torch.FloatTensor([1e-12]).to(self.device)

    def forward(self, X, target_ntype, return_feature=False, contrastive=False):  # features_list, e_feat):
        h = X  # torch.cat(h, 0)
        res_attn = None
        if contrastive:
            hs = []
            for g in self.g_cs:
                hs.append(self.gcn(g, h[target_ntype]))
            if len(hs) > 1:
                h = self.att(hs)
            else:
                h = hs[0]
        else:
            for l in range(self.num_layers):  # noqa E741
                h, res_attn = self.gat_layers[l](self.g, h, res_attn=res_attn)
                h = {n: h[n].flatten(1) for n in h}
            h = h[target_ntype]
        # output projection
        logits = self.fc(h)
        # This is an equivalent replacement for tf.l2_normalize, see https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/math/l2_normalize for more information.
        logits = logits / (torch.max(torch.norm(logits, dim=1, keepdim=True), self.epsilon))
        if return_feature:
            return logits, h
        return logits

    def contrastive_loss(self, f1, f2, t=1.0, pos_mask=None):
        f1_norm = torch.norm(f1, dim=-1, keepdim=True)
        f2_norm = torch.norm(f2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(f1, f2.t())
        dot_denominator = torch.mm(f1_norm, f2_norm.t()) + 1e-8
        sim = torch.exp(dot_numerator / dot_denominator / t)
        sim = sim / (torch.sum(sim, dim=1).view(-1, 1) + 1e-8)
        loss = -torch.log(sim.mul(pos_mask).sum(dim=-1)).mean()

        return loss

    def loss(self, x, target_ntype, target_node, label, lam=1, contrastive=False, pos_mask=None):
        logits, h1 = self.forward(x, target_ntype, return_feature=True)
        y = logits[target_node]
        s_loss = self.cross_entropy_loss(y, label)
        if contrastive:
            _, h2 = self.forward(x, target_ntype, return_feature=True, contrastive=True)
            c_loss = self.contrastive_loss(h1, h2, t=args.t, pos_mask=pos_mask)
        else:
            c_loss = 0
        loss = s_loss + lam*c_loss
        return loss, c_loss, s_loss

    def evaluate(self, x, target_ntype, target_node, label):
        logits = self.forward(x, target_ntype)
        y = logits[target_node]
        loss = self.cross_entropy_loss(y, label)
        acc = accuracy(y, label)
        macro_f1 = f1_score(y_pred=y.argmax(1).cpu().numpy(), y_true=label.cpu().numpy(), average='macro')
        micro_f1 = f1_score(y_pred=y.argmax(1).cpu().numpy(), y_true=label.cpu().numpy(), average='micro')
        y = F.softmax(y, dim=-1)
        if y.shape[1] == 2:
            y = y[:, 1]
        auc = roc_auc_score(y_score=y.detach().cpu().numpy(), y_true=label.cpu().numpy(), multi_class='ovr')

        return loss.item(), acc, macro_f1, micro_f1, auc


def train(model, optimizer, target_ntype,train_node, train_label, valid_node, valid_label, test_node, test_label, max_epoch, max_patience, pos_mask):
    patience = 0
    best_score = 0
    max_score = 0
    min_loss = np.inf

    x = model.g.ndata.pop("nfeat")

    for epoch in range(max_epoch):
        model.train()
        optimizer.zero_grad()
        loss, c_loss, s_loss = model.loss(x, target_ntype, train_node, train_label, lam=args.lam, contrastive=args.contrastive, pos_mask=pos_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
        optimizer.step()
        if epoch % args.log_epoch == 0:
            print(f"Epoch: {epoch}\tSupervised Loss: {s_loss:.4f}\t\tContrastive Loss: {c_loss:.4f}", end='\t')
        model.eval()
        logits = model.forward(x, target_ntype)
        train_acc = accuracy(logits[train_node], train_label)
        train_loss = model.cross_entropy_loss(logits[train_node], train_label).cpu().item()
        val_acc = accuracy(logits[valid_node], valid_label)
        val_loss = model.cross_entropy_loss(logits[valid_node], valid_label).cpu().item()
        if epoch % args.log_epoch == 0:
            print(f"Train: {train_acc:.3f}, {train_loss:.3f}, Val: {val_acc:.3f}, {val_loss:.3f}")
        if val_loss <= min_loss or val_acc >= max_score:
            if val_acc >= best_score:
                # best_loss = val_loss
                best_score = val_acc
                best_model = copy.deepcopy(model.state_dict())
            min_loss = np.min((min_loss, val_loss))
            max_score = np.max((max_score, val_acc))
            patience = 0
        else:
            patience += 1
            if patience == max_patience:
                model.load_state_dict(best_model)
                break
        if args.contrastive and epoch >= args.warm_epoch and (epoch - args.warm_epoch) % args.adjust_epoch == 0:
            # adjust pos mask
            Z = F.softmax(logits, dim=-1).detach().cpu().numpy()
            H = 1 + (Z * np.log(Z)).sum(1) / np.log(Z.shape[1])
            th = H[_info["train_index"].long()].mean()
            H[_info["train_index"].long()] = 1.0
            mask = (H > th)
            pred = Z.argmax(1)
            pred[_info["train_index"].long()] = _info["train_label"].numpy()
            node_index = np.arange(Z.shape[0])
            df = pd.DataFrame({"index": node_index[mask], "class": pred[mask]})
            e = pd.merge(df, df, on='class', how='inner')
            e = e[e['index_x'] != e['index_y']]
            e = (e['index_x'].to_numpy(), e['index_y'].to_numpy())
            N = Z.shape[0]
            pos_mask = sp.coo_matrix((np.ones_like(e[0]), e), shape=(N, N)) + sp.eye(N)
            pos_mask = torch.Tensor(pos_mask.todense()).bool().to(model.device)

    model.eval()
    _, test_acc, test_macro_f1, test_micro_f1, auc = model.evaluate(x, target_ntype, test_node, test_label)
    print(f"Test ACC = {test_acc}\t Macro-F1 = {test_macro_f1}\t Micro-F1 = {test_micro_f1}\t AUC = {auc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--slope', type=float, default=0.05)
    parser.add_argument('--edge-dim', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epoch', type=int, default=5000)
    parser.add_argument('--warm_epoch', type=int, default=100)
    parser.add_argument('--log_epoch', type=int, default=100)
    parser.add_argument('--adjust_epoch', type=int, default=50)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--weight-decay', default=5e-5, type=float)
    parser.add_argument('--seed', default=6, type=int)
    parser.add_argument('--gpu', default=1, type=int)
    parser.add_argument('--lam', default=3, type=float)
    parser.add_argument('--t', default=1.0, type=float)
    parser.add_argument('--view', default=["ratio_0.001", "mam", "mdm"], type=str, nargs='*')
    parser.add_argument('--contrastive', action="store_true", default=False)
    parser.add_argument('--dataset', default="imdb", type=str)
    parser.add_argument('--num', type=int, default=20, help="training sample num")
    args = parser.parse_args()

    set_seed(args.seed)

    gs, info = dgl.load_graphs(f"data/{args.dataset}/processed/{args.dataset}_contrastive_graph.bin")
    g = gs[0]
    g_cs = []
    pos_mask = None
    in_dim = {n: g.nodes[n].data['nfeat'].shape[1] for n in g.ntypes}
    num_nodes = g.number_of_nodes()
    edge_type_num = len(g.etypes)

    heads = [args.num_heads] * args.num_layers

    _info = torch.load(f"data/{args.dataset}/processed/index_{args.num}.bin")
    if args.dataset == 'dblp':
        num_classes = 4
        target_ntype = "author"
    elif args.dataset == 'acm':
        num_classes = 3
        target_ntype = "paper"
    elif args.dataset == 'imdb':
        num_classes = 3
        target_ntype = "movie"
    else:
        raise NotImplementedError

    if args.contrastive:
        with open(f"data/{args.dataset}/processed/multi_view_all.pkl", 'rb') as f:
            view_edges = pickle.load(f)
        for v in args.view:
            view_edge = view_edges[v]
            g_c = dgl.graph((view_edge[0], view_edge[1]), num_nodes=g.number_of_nodes(target_ntype))
            g_c = g_c.remove_self_loop()
            g_c = g_c.add_self_loop()
            g_cs.append(g_c)

        df = pd.DataFrame({"index": _info["train_index"].numpy(), "class":_info["train_label"].numpy()})
        e = pd.merge(df, df, on='class', how='inner')
        e = e[e['index_x'] != e['index_y']]
        e = (e['index_x'].to_numpy(), e['index_y'].to_numpy())
        N = g.number_of_nodes(target_ntype)
        pos_mask = sp.coo_matrix((np.ones_like(e[0]), e), shape=(N, N)) + sp.eye(N)
        pos_mask = torch.Tensor(pos_mask.todense()).bool()

    model = SimpleHeteroHGN(args.edge_dim, edge_type_num, in_dim, args.hidden_size, num_classes, args.num_layers,
                      heads, args.dropout, args.dropout, args.slope, True, 0.05, target_ntype, shared_weight=True)
    model = model.to(model.device)
    model.g = g.to(model.device)
    if args.contrastive:
        model.g_cs = [g_c.to(model.device) for g_c in g_cs]
        pos_mask = pos_mask.to(model.device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr,  weight_decay=args.weight_decay)

    train(model, opt, target_ntype, _info["train_index"].long().to(model.device),
      _info["train_label"].long().to(model.device), _info["valid_index"].long().to(model.device),
      _info["valid_label"].long().to(model.device),  _info["test_index"].long().to(model.device),
      _info["test_label"].long().to(model.device), args.epoch, args.patience, pos_mask)

