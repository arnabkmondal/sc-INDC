from time import time
import os
import math

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn import metrics
import h5py
import scanpy as sc
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib import gridspec
import seaborn as sns
from munkres import Munkres

font = font_manager.FontProperties(family='Times New Roman',
                                   # weight='bold',
                                   style='normal', size=16)


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment as linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(ind[0], ind[1])]) * 1.0 / y_pred.size


def compute_accuracy(y_pred, y_t, tot_cl):
    # compute the accuracy using Hungarian algorithm
    m = Munkres()
    mat = np.zeros((tot_cl, tot_cl))
    for i in range(tot_cl):
        for j in range(tot_cl):
            mat[i][j] = np.sum(np.logical_and(y_pred == i, y_t == j))
    indexes = m.compute(-mat)

    corresp = []
    for i in range(tot_cl):
        corresp.append(indexes[i][1])

    pred_corresp = [corresp[int(predicted)] for predicted in y_pred]
    acc = np.sum(pred_corresp == y_t) / float(len(y_t))
    return acc


class ZINBLoss(nn.Module):
    def __init__(self):
        super(ZINBLoss, self).__init__()

    def forward(self, x, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0):
        eps = 1e-10
        scale_factor = scale_factor[:, None]
        mean = mean * scale_factor

        t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
        t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (x * (torch.log(disp + eps) - torch.log(mean + eps)))
        nb_final = t1 + t2

        nb_case = nb_final - torch.log(1.0 - pi + eps)
        zero_nb = torch.pow(disp / (disp + mean + eps), disp)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)

        if ridge_lambda > 0:
            ridge = ridge_lambda * torch.square(pi)
            result += ridge

        result = torch.mean(result)
        return result


class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()

    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)


class DispAct(nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()

    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)


def buildEncNetwork(layers, z_dim, activation="relu"):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i - 1], layers[i]))
        net.append(nn.BatchNorm1d(layers[i], eps=2e-5))
        if activation == "relu":
            net.append(nn.ReLU())
        elif activation == "sigmoid":
            net.append(nn.Sigmoid())
    net.append(nn.Linear(layers[-1], z_dim))
    return nn.Sequential(*net)


def buildDecNetwork(layers, x_dim, activation="relu"):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i - 1], layers[i]))
        net.append(nn.BatchNorm1d(layers[i], eps=2e-5))
        if activation == "relu":
            net.append(nn.ReLU())
        elif activation == "sigmoid":
            net.append(nn.Sigmoid())
    net.append(nn.Linear(layers[-1], x_dim))
    return nn.Sequential(*net)


class scRNADataset(torch.utils.data.Dataset):
    # new dataset class that allows to get the sample indices of mini-batch
    def __init__(self, x, count_x, size_factor, y):
        self.x = x
        self.count_x = count_x
        self.size_factor = size_factor
        self.y = y

    def __getitem__(self, index):
        x_data, count_x_data, sf_data, target = self.x[index], self.count_x[index], self.size_factor[index], \
                                                self.y[index]
        return x_data, count_x_data, sf_data, target, index

    def __len__(self):
        return len(self.x)


class ClusterNet(nn.Module):
    def __init__(self, input_dim, z_dim, n_clusters, encodeLayer=[], decodeLayer=[],
                 activation="relu", sigma=1., alpha=1., gamma=1., ml_weight=1., cl_weight=1.):
        super(ClusterNet, self).__init__()
        self.z_dim = z_dim
        self.n_clusters = n_clusters
        self.activation = activation
        self.sigma = sigma
        self.alpha = alpha
        self.gamma = gamma
        self.ml_weight = ml_weight
        self.cl_weight = cl_weight
        self.encoder = buildEncNetwork([input_dim] + encodeLayer, z_dim=z_dim, activation=activation)
        self.decoder = buildDecNetwork([z_dim] + decodeLayer, x_dim=input_dim, activation=activation)
        self.linear_probe = nn.Linear(self.z_dim, self.n_clusters)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def entropy(self, p):
        if (len(p.size())) == 2:
            return - torch.sum(p * torch.log(p + 1e-8)) / float(len(p))
        elif (len(p.size())) == 1:
            return - torch.sum(p * torch.log(p + 1e-8))
        else:
            raise NotImplementedError

    def Compute_entropy(self, y_hat):
        # compute the entropy and the conditional entropy
        p = F.softmax(y_hat, dim=1)
        p_ave = torch.sum(p, dim=0) / len(y_hat)
        return self.entropy(p), self.entropy(p_ave)

    def kl(self, p, q):
        # compute KL divergence between p and q
        return torch.sum(p * torch.log((p + 1e-8) / (q + 1e-8))) / float(len(p))

    def distance(self, y0, y1):
        # compute KL divergence between the outputs of the newtrok
        return self.kl(F.softmax(y0, dim=1), F.softmax(y1, dim=1))

    def forward(self, x):
        z = self.encoder(x + torch.randn_like(x) * self.sigma)
        x_hat = self.decoder(z)

        z0 = self.encoder(x)
        y_hat = self.linear_probe(z0)
        return z0, y_hat, x_hat

    def encodeBatch(self, X, batch_size=256):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()

        encoded = []
        num = X.shape[0]
        num_batch = int(math.ceil(1.0 * X.shape[0] / batch_size))
        for batch_idx in range(num_batch):
            xbatch = X[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]
            inputs = Variable(xbatch)
            z, _, _ = self.forward(inputs)
            encoded.append(z.data)

        encoded = torch.cat(encoded, dim=0)
        return encoded

    def vat(self, x, eps_list, xi=10, Ip=1):
        # compute the regularized penality [eq. (4) & eq. (6), 1]
        use_cuda = torch.cuda.is_available()
        with torch.no_grad():
            _, y, _ = self.forward(Variable(x))
        d = torch.randn((x.size()[0], x.size()[1]))
        d = F.normalize(d, p=2, dim=1)
        for ip in range(Ip):
            d_var = Variable(d)
            if use_cuda:
                d_var = d_var.cuda()
            d_var.requires_grad_(True)
            _, y_p, _ = self.forward(x + xi * d_var)
            kl_loss = self.distance(y, y_p)
            kl_loss.backward(retain_graph=True)
            d = d_var.grad
            d = F.normalize(d, p=2, dim=1)
        d_var = d
        if use_cuda:
            d_var = d_var.cuda()
        eps = 0.25 * eps_list
        eps = eps.view(-1, 1)
        _, y_2, _ = self.forward(x + eps * d_var)
        return self.distance(y, y_2)

    def loss_unlabeled(self, x, eps_list):
        # to use enc_aux_noubs
        L = self.vat(x, eps_list)
        return L

    def upload_nearest_dist(self, dataset):
        # Import the range of local perturbation for VAT
        nearest_dist = np.loadtxt(f'./dataset/{args.nb_genes}/{args.ds}/10th_neighbor.txt').astype(np.float32)
        return nearest_dist

    def transform(self, x):
        std = 1.0
        mean = 0.0
        n_mask_per_example = 3
        mask_width = 100
        device = x.device

        x = x + torch.randn(x.size()).to(device) * std + mean

        mask = torch.ones_like(x)
        indices_weights = np.random.random((mask.shape[0], n_mask_per_example + 1))
        number_of_ones = mask.shape[1] - mask_width * n_mask_per_example
        ones_sizes = np.round(indices_weights[:, :n_mask_per_example].T
                              * (number_of_ones / np.sum(indices_weights, axis=-1))).T.astype(np.int32)
        ones_sizes[:, 1:] += mask_width
        zeros_start_indices = np.cumsum(ones_sizes, axis=-1)
        for sample_idx in range(len(mask)):
            for zeros_idx in zeros_start_indices[sample_idx]:
                mask[sample_idx, zeros_idx: zeros_idx + mask_width] = 0
        x = x * mask

        return x

    def pre_train(self, train_loader, test_loader, batch_size=256, lr=0.001, epochs=400):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        print("Pretraining stage")
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),
                               lr=lr, amsgrad=True)
        nearest_dist = torch.from_numpy(self.upload_nearest_dist(args.ds))
        nearest_dist = nearest_dist.cuda()
        
        for epoch in range(epochs):
            for batch_idx, (x_batch, x_raw_batch, sf_batch, target_batch, index) in enumerate(train_loader):
                self.train()
                x_tensor = Variable(x_batch).cuda()
                index_tensor = Variable(index).cuda()
                _, y_pred, x_tensor_hat = self.forward(x_tensor)
                recon_loss = torch.nn.MSELoss()(x_tensor_hat, x_tensor)

                aver_entropy, entropy_aver = self.Compute_entropy(y_pred)
                r_mutual_i = aver_entropy - 4 * entropy_aver
                loss_ul = self.loss_unlabeled(x_tensor, nearest_dist[index_tensor])

                loss = loss_ul + 0.1 * r_mutual_i + 2.0 * recon_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print('Epoch [{}/{}], Recon loss:{:.4f}, Total loss:{:.4f}'.
                      format(batch_idx + 1, epoch + 1, recon_loss.item(), loss.item()))

            # """
            # statistics
            self.eval()
            p_pred = np.zeros((len(trainset), self.n_clusters))
            y_pred = np.zeros(len(trainset))
            y_t = np.zeros(len(trainset))
            with torch.no_grad():
                for i, data in enumerate(test_loader, 0):
                    x_batch, x_raw_batch, sf_batch, target_batch, index = data
                    if use_cuda:
                        x_tensor = Variable(x_batch).cuda()
                        target_batch = Variable(target_batch).cuda()
                    _, outputs, _ = self.forward(x_tensor)
                    outputs = F.softmax(outputs, dim=1)
                    y_pred[i * batch_size:(i + 1) * batch_size] = torch.argmax(outputs, dim=1).cpu().numpy()
                    p_pred[i * batch_size:(i + 1) * batch_size, :] = outputs.detach().cpu().numpy()
                    y_t[i * batch_size:(i + 1) * batch_size] = target_batch.cpu().numpy()
            acc = compute_accuracy(y_pred, y_t, self.n_clusters)
            print("epoch: ", epoch + 1,
                  "\t NMI = {:.4f}".format(metrics.normalized_mutual_info_score(y_t, y_pred)),
                  "\t ARI = {:.4f}".format(metrics.adjusted_rand_score(y_t, y_pred)), "\t acc = {:.4f} ".
                  format(acc))
            # """


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def empty_safe(fn, dtype):
    def _fn(x):
        if x.size:
            return fn(x)
        return x.astype(dtype)

    return _fn


decode = empty_safe(np.vectorize(lambda _x: _x.decode("utf-8")), str)


def read_clean(data):
    assert isinstance(data, np.ndarray)
    if data.dtype.type is np.bytes_:
        data = decode(data)
    if data.size == 1:
        data = data.flat[0]
    return data


def dict_from_group(group):
    assert isinstance(group, h5py.Group)
    d = dotdict()
    for key in group:
        if isinstance(group[key], h5py.Group):
            value = dict_from_group(group[key])
        else:
            value = read_clean(group[key][...])
        d[key] = value
    return d


def normalize(adata, copy=True, highly_genes=None, filter_min_counts=True,
              size_factors=True, normalize_input=True, logtrans_input=True):
    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError
    norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    assert 'n_count' not in adata.obs, norm_error
    if adata.X.size < 50e6:  # check if adata.X is integer only if array is small
        if sp.sparse.issparse(adata.X):
            assert (adata.X.astype(int) != adata.X).nnz == 0, norm_error
        else:
            assert np.all(adata.X.astype(int) == adata.X), norm_error

    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)
    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata
    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0
    if logtrans_input:
        sc.pp.log1p(adata)
    if highly_genes != None:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5,
                                    n_top_genes=highly_genes, subset=True)
    if normalize_input:
        sc.pp.scale(adata)
    return adata


def preprocess(X, nb_genes=500):
    X = np.ceil(X).astype(np.uint32)
    count_X = X
    print(X.shape, count_X.shape, f"keeping {nb_genes} genes")
    adata = sc.AnnData(X)

    adata = normalize(adata,
                      copy=True,
                      highly_genes=nb_genes,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)
    X_normalized = adata.X.astype(np.float32)
    return X_normalized, adata.raw[:, adata.var_names].X, adata.obs['size_factors']  # .reshape(-1, 1)


if __name__ == "__main__":

    # setting the hyper parameters
    import argparse

    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--maxiter', default=150, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--update_interval', default=1, type=int)

    parser.add_argument('-nz', '--z_dim', type=int, default=16)
    parser.add_argument('-ng', '--nb_genes', type=int, default=3000)
    parser.add_argument('-d', '--ds', type=str, default='mouse_ES_cell')
    parser.add_argument('-e', '--expt', type=str, default='run1')

    args = parser.parse_args()

    nb_genes = args.nb_genes
    z_dim = args.z_dim
    dataset = args.ds
    DATA_ROOT = '../BYOL/byol-pytorch-master/datasets'
    save_dir = f'./Results/{args.nb_genes}/{args.ds}/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    df_column = ["dataset",
                 "ARI", "NMI", "Silhouette", "Calinski", "Acc",
                 "time", "z_dim", "nb_genes", "batch_size"]
    df = pd.DataFrame(columns=df_column)
    df_entry = {"dataset": dataset, "batch_size": args.batch_size}

    data_mat = h5py.File(f"{DATA_ROOT}/{dataset}/data.h5", "r")
    try:
        X = np.array(data_mat['X'])
        Y = np.array(data_mat['Y'])
        unique_class = np.unique(Y)
    except:
        exprs_handle = data_mat["exprs"]
        if isinstance(exprs_handle, h5py.Group):
            X = sp.sparse.csr_matrix((exprs_handle["data"][...], exprs_handle["indices"][...],
                                      exprs_handle["indptr"][...]), shape=exprs_handle["shape"][...])
        else:
            X = exprs_handle[...].astype(np.float32)
        if isinstance(X, np.ndarray):
            X = np.array(X)
        else:
            X = np.array(X.toarray())
        obs = pd.DataFrame(dict_from_group(data_mat["obs"]), index=decode(data_mat["obs_names"][...]))
        Y = np.array(obs["cell_type1"])
        unique_class = np.unique(Y)
        class_dict = {}
        i = 0
        for cls in unique_class:
            class_dict.update({cls: i})
            i = i + 1
        for i, label in enumerate(Y):
            Y[i] = class_dict[label]
        Y = Y.astype(np.int32)

    print(np.where(X == 0)[0].shape[0] / (X.shape[0] * X.shape[1]))
    cluster_number = np.unique(Y).shape[0]

    X, raw_X, sf = preprocess(X, nb_genes=nb_genes)

    trainset = scRNADataset(X, raw_X, sf, Y)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    X_cuda = torch.tensor(X).cuda()

    sd = 2.5

    model = ClusterNet(input_dim=X.shape[1], z_dim=args.z_dim, n_clusters=cluster_number,
                       encodeLayer=[512, 512], decodeLayer=[512, 512], sigma=sd)

    print(str(model))

    tic = time()

    model.pre_train(trainloader, testloader, batch_size=args.batch_size, lr=0.0008, epochs=args.epochs)

    latent = model.encodeBatch(X_cuda)
    kmeans = KMeans(cluster_number, n_init=20)
    y_pred = kmeans.fit_predict(latent.data.cpu().numpy())

    toc = time()

    z_embedded = TSNE(n_components=2).fit_transform(latent.data.cpu().numpy())
    sil = np.round(metrics.silhouette_score(latent.data.cpu().numpy(), y_pred), 5)
    cal = np.round(metrics.calinski_harabasz_score(latent.data.cpu().numpy(), y_pred), 5)
    print('K-means: SIL= %.4f, CAL= %.4f' % (sil, cal))
    df_entry.update({"Silhouette": sil, "Calinski": cal})
    if Y is not None:
        acc = np.round(cluster_acc(Y, y_pred), 5)
        nmi = np.round(metrics.normalized_mutual_info_score(Y, y_pred), 5)
        ari = np.round(metrics.adjusted_rand_score(Y, y_pred), 5)
        print('K-means: ACC= %.4f, NMI= %.4f, ARI= %.4f' % (acc, nmi, ari))
        df_entry.update({"Acc": acc, "NMI": nmi, "ARI": ari})

    exec_time = toc - tic
    print('Training time: %d seconds.' % exec_time)
    df_entry.update({"time": exec_time / 60, "z_dim": args.z_dim, "nb_genes": args.nb_genes})

    df = df.append(df_entry, ignore_index=True)

    with open(f'./Results/{args.nb_genes}/{args.expt}.csv', 'a') as f:
        df.to_csv(f, index=False, header=f.tell() == 0)

    cm = metrics.confusion_matrix(Y, y_pred)
    cm_argmax = cm.argmax(axis=0)
    y_pred_ = np.array([cm_argmax[i] for i in y_pred])

    n_cols = 3
    nb_rows = 1

    fig = plt.figure(figsize=(15, 4 * nb_rows))
    gs = gridspec.GridSpec(nb_rows, n_cols, width_ratios=[1, 1, 1])

    dataset_names = {
        '10X_PBMC': '10X PBMC',
        'mouse_ES_cell': 'Mouse ES Cell',
        'worm_neuron_cell': 'Worm Neuron Cell',
        'mouse_bladder_cell': 'Mouse Bladder Cell',
        'Quake_Smart-seq2_Trachea': 'QS Trachea',
        'Quake_Smart-seq2_Diaphragm': 'QS Diaphragm',
        'Quake_10x_Spleen': 'Q Spleen',
        'Quake_10x_Bladder': 'Q Bladder',
        'Quake_Smart-seq2_Lung': 'QS Lung',
        'Quake_10x_Limb_Muscle': 'Q Limb Muscle',
        'Quake_Smart-seq2_Limb_Muscle': 'QS Limb Muscle',
        'Romanov': 'Romanov',
        'Adam': 'Adam',
        'Muraro': 'Muraro',
        'Young': 'Young',
    }

    # df = pd.DataFrame(z_embedded, columns=['x', 'y']).assign(category=Y)
    # df.to_csv(f'./scINDC_{dataset}_lat{z_dim}.csv', index=None)

    # ax = plt.subplot(nb_rows, n_cols, 1)
    ax = plt.subplot(gs[0])
    p = sns.scatterplot(x=z_embedded[:, 0], y=z_embedded[:, 1], hue=y_pred, ax=ax, legend=None,
                        palette="deep", style=y_pred)
    ax.set_xlabel("")
    ax.set_ylabel("")
    # ax.legend([])
    ax.set_title(f"Proposed Method + KM \n(ARI: {ari}, NMI: {nmi})", fontweight="bold",
                 fontdict={'font': 'Times New Roman', 'size': 16})
    sns.despine()

    # ax = plt.subplot(nb_rows, n_cols, 2)
    ax = plt.subplot(gs[1])
    p = sns.scatterplot(x=z_embedded[:, 0], y=z_embedded[:, 1], hue=Y, ax=ax,
                        palette="deep", style=Y, legend='full')
    ax.legend(loc='center right', bbox_to_anchor=(3.25, 0.5), ncol=1,
              title=dataset_names[args.ds], title_fontsize=16, prop=font, fancybox=True, shadow=False)
    legend = p.legend_
    for i, label in enumerate(unique_class):
        legend.get_texts()[i].set_text(label)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(f"Ground Truth\n({cluster_number} clusters)", fontweight="bold",
                 fontdict={'font': 'Times New Roman', 'size': 16})
    sns.despine()

    # ax = plt.subplot(nb_rows, n_cols, 3)
    ax = plt.subplot(gs[2])
    p = sns.scatterplot(x=z_embedded[:, 0], y=z_embedded[:, 1], hue=y_pred_, ax=ax, legend=None,
                        palette="deep", style=y_pred_)
    ax.set_xlabel("")
    ax.set_ylabel("")
    # ax.legend([])
    ax.set_title(f"Proposed Method + KM \n(ACC: {acc})", fontweight="bold",
                 fontdict={'font': 'Times New Roman', 'size': 16})
    sns.despine()

    fig.savefig(f"{save_dir}/{dataset_names[args.ds]}-visualization.pdf", bbox_inches='tight')
