import numpy as np
import pandas as pd
import dgl
import torch
import torch.nn as nn
from dgl import function as fn
import argparse
from sklearn.cluster import KMeans
import pickle
import scipy.sparse as sp
import torch
import sklearn
import os


def nei_to_edge(nei):
    edge_src = []
    edge_dst = []
    for i, n in enumerate(nei):
        edge_src += [i]*len(n)
        edge_dst += list(n)
    return [edge_src, edge_dst]


def make_index(t=[20, 40, 60]):
    label = np.load(f"data/{dataset}/raw/labels.npy")
    for i in t:
        info = {}
        info["label"] = torch.Tensor(label)
        train = np.load(f"data/{dataset}/raw/train_" + str(i) + ".npy")
        test = np.load(f"data/{dataset}/raw/test_" + str(i) + ".npy")
        val = np.load(f"data/{dataset}/raw/val_" + str(i) + ".npy")
        info["train_index"] = torch.Tensor(train)
        info["test_index"] = torch.Tensor(test)
        info["valid_index"] = torch.Tensor(val)
        info["train_label"] = torch.Tensor(label[train])
        info["test_label"] = torch.Tensor(label[test])
        info["valid_label"] = torch.Tensor(label[val])

        test_mask = np.zeros_like(label).astype(bool)
        test_mask[test] = True
        train_mask = np.zeros_like(label).astype(bool)
        train_mask[train] = True
        val_mask = np.zeros_like(label).astype(bool)
        val_mask[val] = True
        train_inductive = train_mask[~test_mask].nonzero()[0]
        val_inductive = val_mask[~test_mask].nonzero()[0]
        info["train_index_inductive"] = torch.Tensor(train_inductive)
        info["valid_index_inductive"] = torch.Tensor(val_inductive)
        info["train_label_inductive"] = torch.Tensor(label[train_mask])
        info["valid_label_inductive"] = torch.Tensor(label[val_mask])

        torch.save(info, f"data/{dataset}/processed/index_{i}.bin")


def make_acm_contrastive_graph():
    nei_a = np.load("data/acm/raw/nei_a.npy", allow_pickle=True)
    nei_s = np.load("data/acm/raw/nei_s.npy", allow_pickle=True)
    edge_a = nei_to_edge(nei_a)
    edge_s = nei_to_edge(nei_s)

    type_num = [4019, 7167, 60]

    edge_dict = {("paper", "0", "paper"): (range(type_num[0]), range(type_num[0])),
                 ("author", "1", "author"): (range(type_num[1]), range(type_num[1])),
                 ("subject", "2", "subject"): (range(type_num[2]), range(type_num[2])),
                 ("paper", "3", "author"): (edge_a[0], edge_a[1]),
                 ("author", "4", "paper"): (edge_a[1], edge_a[0]),
                 ("paper", "5", "subject"): (edge_s[0], edge_s[1]),
                 ("subject", "6", "paper"): (edge_s[1], edge_s[0]),
                 }

    g = dgl.heterograph(edge_dict)
    # extract bag-of-word representations of plot keywords for each movie
    # X is a sparse matrix
    paper_X = sp.load_npz("data/acm/raw/p_feat.npz")
    author_X = sp.load_npz("data/acm/raw/a_feat.npz")

    g.nodes["paper"].data["nfeat"] = torch.Tensor(paper_X.todense())
    g.nodes["author"].data["nfeat"] = torch.Tensor(author_X.todense())

    funcs = {"5": (fn.copy_u("nfeat", "e"), fn.mean("e", "nfeat"))}

    g.multi_update_all(funcs, 'sum')

    dgl.save_graphs("data/acm/processed/acm_contrastive_graph.bin", [g], {})


def make_acm_inductive_graph(ratio=[20, 40, 60]):
    nei_a = np.load("data/acm/raw/nei_a.npy", allow_pickle=True)
    nei_s = np.load("data/acm/raw/nei_s.npy", allow_pickle=True)
    edge_a = nei_to_edge(nei_a)
    edge_s = nei_to_edge(nei_s)

    type_num = [4019, 7167, 60]

    edge_dict = {("paper", "0", "paper"): (range(type_num[0]), range(type_num[0])),
                 ("author", "1", "author"): (range(type_num[1]), range(type_num[1])),
                 ("subject", "2", "subject"): (range(type_num[2]), range(type_num[2])),
                 ("paper", "3", "author"): (edge_a[0], edge_a[1]),
                 ("author", "4", "paper"): (edge_a[1], edge_a[0]),
                 ("paper", "5", "subject"): (edge_s[0], edge_s[1]),
                 ("subject", "6", "paper"): (edge_s[1], edge_s[0]),
                 }

    g = dgl.heterograph(edge_dict)
    # extract bag-of-word representations of plot keywords for each movie
    # X is a sparse matrix
    paper_X = sp.load_npz("data/acm/raw/p_feat.npz").todense()
    author_X = sp.load_npz("data/acm/raw/a_feat.npz").todense()

    g.nodes["paper"].data["nfeat"] = torch.Tensor(paper_X)
    g.nodes["author"].data["nfeat"] = torch.Tensor(author_X)

    funcs = {"5": (fn.copy_u("nfeat", "e"), fn.mean("e", "nfeat"))}

    g.multi_update_all(funcs, 'sum')

    for r in ratio:
        test_index = np.load(f"data/acm/raw/test_{r}.npy")
        test_num = test_index.shape[0]
        test_mask = np.zeros(type_num[0]).astype(bool)
        test_mask[test_index] = True
        edge_a = nei_to_edge(nei_a[~test_mask])
        edge_s = nei_to_edge(nei_s[~test_mask])

        edge_dict = {("paper", "0", "paper"): (range(type_num[0]-test_num), range(type_num[0]-test_num)),
                     ("author", "1", "author"): (range(type_num[1]), range(type_num[1])),
                     ("subject", "2", "subject"): (range(type_num[2]), range(type_num[2])),
                     ("paper", "3", "author"): (edge_a[0], edge_a[1]),
                     ("author", "4", "paper"): (edge_a[1], edge_a[0]),
                     ("paper", "5", "subject"): (edge_s[0], edge_s[1]),
                     ("subject", "6", "paper"): (edge_s[1], edge_s[0]),
                     }

        gi = dgl.heterograph(edge_dict)

        gi.nodes["paper"].data["nfeat"] = torch.Tensor(paper_X[~test_mask])
        gi.nodes["author"].data["nfeat"] = torch.Tensor(author_X)

        funcs = {"5": (fn.copy_u("nfeat", "e"), fn.mean("e", "nfeat"))}

        gi.multi_update_all(funcs, 'sum')

        dgl.save_graphs(f"data/acm/processed/acm_inductive_graph_{r}.bin", [gi, g], {})


def make_dblp_contrastive_graph():
    pa = np.loadtxt("data/dblp/raw/pa.txt").astype(np.int32)
    pc = np.loadtxt("data/dblp/raw/pc.txt").astype(np.int32)
    pt = np.loadtxt("data/dblp/raw/pt.txt").astype(np.int32)

    type_num = [4057, 14328, 7723, 20]

    edge_dict = {("author", "0", "author"): (range(type_num[0]), range(type_num[0])),
                 ("paper", "1", "paper"): (range(type_num[1]), range(type_num[1])),
                 ("term", "2", "term"): (range(type_num[2]), range(type_num[2])),
                 ("conference", "3", "conference"): (range(type_num[3]), range(type_num[3])),
                 ("paper", "4", "author"): (pa[:, 0], pa[:, 1]),
                 ("author", "5", "paper"): (pa[:, 1], pa[:, 0]),
                 ("paper", "6", "term"): (pt[:, 0], pt[:, 1]),
                 ("term", "7", "paper"): (pt[:, 1], pt[:, 0]),
                 ("paper", "8", "conference"): (pc[:, 0], pc[:, 1]),
                 ("conference", "9", "paper"): (pc[:, 1], pc[:, 0]),
                 }

    g = dgl.heterograph(edge_dict)

    # extract bag-of-word representations of plot keywords for each movie
    # X is a sparse matrix
    paper_X = sp.load_npz("data/dblp/raw/p_feat.npz")
    author_X = sp.load_npz("data/dblp/raw/a_feat.npz")

    g.nodes["paper"].data["nfeat"] = torch.Tensor(paper_X.todense())
    g.nodes["author"].data["nfeat"] = torch.Tensor(author_X.todense())


    funcs = {"6": (fn.copy_u("nfeat", "e"), fn.mean("e", "nfeat")),
             "8": (fn.copy_u("nfeat", "e"), fn.mean("e", "nfeat"))}

    g.multi_update_all(funcs, 'sum')

    dgl.save_graphs("data/dblp/processed/dblp_contrastive_graph.bin", [g], {})


def make_dblp_inductive_graph(ratio=[20, 40, 60]):
    pa = np.loadtxt("data/dblp/raw/pa.txt").astype(np.int32)
    pc = np.loadtxt("data/dblp/raw/pc.txt").astype(np.int32)
    pt = np.loadtxt("data/dblp/raw/pt.txt").astype(np.int32)

    type_num = [4057, 14328, 7723, 20]

    pa = sp.coo_matrix((np.ones(pa.shape[0]), (pa[:, 0], pa[:, 1])), shape=(type_num[1], type_num[0]))
    pc = sp.coo_matrix((np.ones(pc.shape[0]), (pc[:, 0], pc[:, 1])), shape=(type_num[1], type_num[3]))
    pt = sp.coo_matrix((np.ones(pt.shape[0]), (pt[:, 0], pt[:, 1])), shape=(type_num[1], type_num[2]))

    edge_dict = {("author", "0", "author"): (range(type_num[0]), range(type_num[0])),
                 ("paper", "1", "paper"): (range(type_num[1]), range(type_num[1])),
                 ("term", "2", "term"): (range(type_num[2]), range(type_num[2])),
                 ("conference", "3", "conference"): (range(type_num[3]), range(type_num[3])),
                 ("paper", "4", "author"): (pa.row, pa.col),
                 ("author", "5", "paper"): (pa.col, pa.row),
                 ("paper", "6", "term"): (pt.row, pt.col),
                 ("term", "7", "paper"): (pt.col, pt.row),
                 ("paper", "8", "conference"): (pc.row, pc.col),
                 ("conference", "9", "paper"): (pc.col, pc.row),
                 }

    g = dgl.heterograph(edge_dict)

    # extract bag-of-word representations of plot keywords for each movie
    # X is a sparse matrix
    paper_X = sp.load_npz("data/dblp/raw/p_feat.npz").todense()
    author_X = sp.load_npz("data/dblp/raw/a_feat.npz").todense()

    g.nodes["paper"].data["nfeat"] = torch.Tensor(paper_X)
    g.nodes["author"].data["nfeat"] = torch.Tensor(author_X)


    funcs = {"6": (fn.copy_u("nfeat", "e"), fn.mean("e", "nfeat")),
             "8": (fn.copy_u("nfeat", "e"), fn.mean("e", "nfeat"))}

    g.multi_update_all(funcs, 'sum')

    for r in ratio:
        test_index = np.load(f"data/dblp/raw/test_{r}.npy")
        test_num = test_index.shape[0]
        test_mask = np.zeros(type_num[0]).astype(bool)
        test_mask[test_index] = True
        pa_i = sp.coo_matrix(pa.todense()[:, ~test_mask])

        edge_dict = {("author", "0", "author"): (range(type_num[0]-test_num), range(type_num[0]-test_num)),
                     ("paper", "1", "paper"): (range(type_num[1]), range(type_num[1])),
                     ("term", "2", "term"): (range(type_num[2]), range(type_num[2])),
                     ("conference", "3", "conference"): (range(type_num[3]), range(type_num[3])),
                     ("paper", "4", "author"): (pa_i.row, pa_i.col),
                     ("author", "5", "paper"): (pa_i.col, pa_i.row),
                     ("paper", "6", "term"): (pt.row, pt.col),
                     ("term", "7", "paper"): (pt.col, pt.row),
                     ("paper", "8", "conference"): (pc.row, pc.col),
                     ("conference", "9", "paper"): (pc.col, pc.row),
                     }
        g_i = dgl.heterograph(edge_dict)

        g_i.nodes["paper"].data["nfeat"] = torch.Tensor(paper_X)
        g_i.nodes["author"].data["nfeat"] = torch.Tensor(author_X[~test_mask])

        funcs = {"6": (fn.copy_u("nfeat", "e"), fn.mean("e", "nfeat")),
                 "8": (fn.copy_u("nfeat", "e"), fn.mean("e", "nfeat"))}

        g_i.multi_update_all(funcs, 'sum')
        dgl.save_graphs(f"data/dblp/processed/dblp_inductive_graph_{r}.bin", [g_i, g], {})


def make_imdb_contrastive_graph():
    ma = sp.load_npz("data/imdb/raw/ma.npz")
    md = sp.load_npz("data/imdb/raw/md.npz")

    type_num = [4278, 2081, 5257]

    edge_dict = {("movie", "0", "movie"): (range(type_num[0]), range(type_num[0])),
                 ("director", "1", "director"): (range(type_num[1]), range(type_num[1])),
                 ("actor", "2", "actor"): (range(type_num[2]), range(type_num[2])),
                 ("movie", "3", "director"): (md.row, md.col),
                 ("director", "4", "movie"): (md.col, md.row),
                 ("movie", "5", "actor"): (ma.row, ma.col),
                 ("actor", "6", "movie"): (ma.col, ma.row),
                 }

    g = dgl.heterograph(edge_dict)

    # extract bag-of-word representations of plot keywords for each movie
    # X is a sparse matrix
    movie_X = np.load("data/imdb/raw/m_feat.npy")

    g.nodes["movie"].data["nfeat"] = torch.Tensor(movie_X)

    funcs = {"3": (fn.copy_u("nfeat", "e"), fn.mean("e", "nfeat")),
             "5": (fn.copy_u("nfeat", "e"), fn.mean("e", "nfeat"))}

    g.multi_update_all(funcs, 'sum')

    dgl.save_graphs("data/imdb/processed/imdb_contrastive_graph.bin", [g], {})


def make_imdb_inductive_graph(ratio=[20, 40, 60]):
    ma = sp.load_npz("data/imdb/raw/ma.npz")
    md = sp.load_npz("data/imdb/raw/md.npz")

    type_num = [4278, 2081, 5257]

    edge_dict = {("movie", "0", "movie"): (range(type_num[0]), range(type_num[0])),
                 ("director", "1", "director"): (range(type_num[1]), range(type_num[1])),
                 ("actor", "2", "actor"): (range(type_num[2]), range(type_num[2])),
                 ("movie", "3", "director"): (md.row, md.col),
                 ("director", "4", "movie"): (md.col, md.row),
                 ("movie", "5", "actor"): (ma.row, ma.col),
                 ("actor", "6", "movie"): (ma.col, ma.row),
                 }

    g = dgl.heterograph(edge_dict)

    # extract bag-of-word representations of plot keywords for each movie
    # X is a sparse matrix
    movie_X = np.load("data/imdb/raw/m_feat.npy")

    g.nodes["movie"].data["nfeat"] = torch.Tensor(movie_X)

    funcs = {"3": (fn.copy_u("nfeat", "e"), fn.mean("e", "nfeat")),
             "5": (fn.copy_u("nfeat", "e"), fn.mean("e", "nfeat"))}

    g.multi_update_all(funcs, 'sum')

    for r in ratio:
        test_index = np.load(f"data/imdb/raw/test_{r}.npy")
        test_num = test_index.shape[0]
        test_mask = np.zeros(type_num[0]).astype(bool)
        test_mask[test_index] = True
        ma_i = sp.coo_matrix(ma.todense()[~test_mask])
        md_i = sp.coo_matrix(md.todense()[~test_mask])

        edge_dict = {("movie", "0", "movie"): (range(type_num[0]-test_num), range(type_num[0]-test_num)),
                     ("director", "1", "director"): (range(type_num[1]), range(type_num[1])),
                     ("actor", "2", "actor"): (range(type_num[2]), range(type_num[2])),
                     ("movie", "3", "director"): (md_i.row, md_i.col),
                     ("director", "4", "movie"): (md_i.col, md_i.row),
                     ("movie", "5", "actor"): (ma_i.row, ma_i.col),
                     ("actor", "6", "movie"): (ma_i.col, ma_i.row),
                     }

        g_i = dgl.heterograph(edge_dict)
        g_i.nodes["movie"].data["nfeat"] = torch.Tensor(movie_X[~test_mask])

        funcs = {"3": (fn.copy_u("nfeat", "e"), fn.mean("e", "nfeat")),
                 "5": (fn.copy_u("nfeat", "e"), fn.mean("e", "nfeat"))}

        g_i.multi_update_all(funcs, 'sum')

        dgl.save_graphs(f"data/imdb/processed/imdb_inductive_graph_{r}.bin", [g_i, g], {})


def make_contrastive_graph():
    if dataset == 'acm':
        make_acm_contrastive_graph()
    elif dataset == 'dblp':
        make_dblp_contrastive_graph()
    elif dataset == 'imdb':
        make_imdb_contrastive_graph()
    else:
        raise NotImplementedError


def make_inductive_contrastive_graph():
    if dataset == 'acm':
        make_acm_inductive_graph()
    elif dataset == 'dblp':
        make_dblp_inductive_graph()
    elif dataset == 'imdb':
        make_imdb_inductive_graph()
    else:
        raise NotImplementedError


def get_target_X():
    if dataset == 'acm':
        return sp.load_npz("data/acm/raw/p_feat.npz").todense()
    elif dataset == 'dblp':
        return sp.load_npz("data/dblp/raw/a_feat.npz").todense()
    elif dataset == 'imdb':
        return np.load("data/imdb/raw/m_feat.npy")
    else:
        raise NotImplementedError


def make_all_view(paths=["pap", "psp"]):
    edges = {}
    for p in paths:
        path = sp.load_npz(f"data/{dataset}/raw/{p}.npz").tocoo()
        r = path.row
        c = path.col
        edges[p] = (r, c)
    X = get_target_X()
    X = sklearn.preprocessing.normalize(X, axis=1)
    sim = np.matmul(X, X.T)
    sim_sort = np.sort(sim.flatten())
    N = X.shape[0]
    for ratio in [0.005, 0.001]:
        th = sim_sort[-int(N * N * ratio)]
        if th >= 1.0:
            break
        m = sim > th
        print(f"ratio_{ratio}: th: {th} ratio: {m.sum()}/{N * N} = {m.sum() / N / N}")
        row, col = np.nonzero(m)
        edges[f"ratio_{ratio}"] = (row, col)
    with open(f"data/{dataset}/processed/multi_view_all.pkl", 'wb') as w:
        pickle.dump(edges, w)


def make_all_view_inductive(paths=["pap", "psp"], t=[20, 40, 60]):
    X = get_target_X()
    X = sklearn.preprocessing.normalize(X, axis=1)

    for i in t:
        test_index = np.load(f"data/{dataset}/raw/test_{i}.npy")
        test_num = test_index.shape[0]
        test_mask = np.zeros(X.shape[0]).astype(bool)
        test_mask[test_index] = True

        edges = {}
        for p in paths:
            path = sp.load_npz(f"data/{dataset}/raw/{p}.npz").todense()[~test_mask][:, ~test_mask]
            path = sp.coo_matrix(path)
            r = path.row
            c = path.col
            edges[p] = (r, c)
        sim = np.matmul(X[~test_mask], X[~test_mask].T)
        sim_sort = np.sort(sim.flatten())
        N = X.shape[0] - test_num
        for ratio in [0.005, 0.001]:
            th = sim_sort[-int(N * N * ratio)]
            if th >= 1.0:
                break
            m = sim > th
            print(f"ratio_{ratio}: th: {th} ratio: {m.sum()}/{N * N} = {m.sum() / N / N}")
            row, col = np.nonzero(m)
            edges[f"ratio_{ratio}"] = (row, col)
        with open(f"data/{dataset}/processed/multi_view_all_inductive_{i}.pkl", 'wb') as w:
            pickle.dump(edges, w)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='acm', type=str)
    args = parser.parse_args()
    dataset = args.dataset

    if not os.path.exists(f"data/{dataset}/processed"):
        os.makedirs(f"data/{dataset}/processed")
    if dataset == "acm":
        ps = ["pap", "psp"]
    elif dataset == "dblp":
        ps = ["apa", "apcpa", "aptpa"]
    elif dataset == "imdb":
        ps = ["mam", "mdm"]
    else:
        raise NotImplementedError

    make_index([20, 40, 60])
    make_contrastive_graph()
    make_inductive_contrastive_graph()
    make_all_view(ps)
    make_all_view_inductive(ps, [20, 40, 60])
    print("process finish")
