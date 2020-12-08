import torch
import numpy as np
from torch.autograd import Function
from PythonGraphPers_withCompInfo import PyPers, PyPersCC, PyPersRev, PyPersCCRev, PyPersAll
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import timeit
# import faiss


def pairwise_distance(point_cloud_refer, point_cloud_query):
    """Compute pairwise distance of a point cloud.
    Args:
      point_cloud: tensor (num_points, num_dims)
    Returns:
      pairwise distance: (num_points, num_points)
    """
    point_cloud_transpose = torch.transpose(point_cloud_refer, 0, 1)

    point_cloud_inner = torch.matmul(point_cloud_query, point_cloud_transpose)
    point_cloud_inner = -2 * point_cloud_inner

    point_cloud_query_square = torch.sum(point_cloud_query**2, dim=-1, keepdim=True)
    point_cloud_refer_square = torch.sum(point_cloud_refer**2, dim=-1, keepdim=True)
    point_cloud_refer_square = torch.transpose(point_cloud_refer_square, 0, 1)

    return point_cloud_query_square + point_cloud_inner + point_cloud_refer_square


def knn(adj_matrix, k=20):
    """Get KNN based on the pairwise distance.
    Args:
      pairwise distance: (batch_size, num_points, num_points)
      k: int
    Returns:
      nearest neighbors: (batch_size, num_points, k)
    """
    neg_adj = -adj_matrix
    dists, nn_idx = torch.topk(neg_adj, k=k)
    return nn_idx, dists


def calc_knn_graph(feats_point_cloud, k=2, refer_trunk_size=50000, query_trunk_size=10000):
    """
    Since GPU knn is memory intensive, so we split the query and reference data points into several trunks.
    Each time, we process a trunk of data (in other words, a batch of data).

    refer_trunk_size: The trunk size for the reference points.
    query_trunk_size: The trunk size for the query points.
    """
    with torch.no_grad():
        num_refer_trunk = feats_point_cloud.size(0) // refer_trunk_size
        remain_refer = feats_point_cloud.size(0) - num_refer_trunk * refer_trunk_size

        num_query_trunk = feats_point_cloud.size(0) // query_trunk_size
        remain_query = feats_point_cloud.size(0) - num_query_trunk * query_trunk_size

        knnG = []
        for i in range(num_query_trunk):
            curr_query = feats_point_cloud[i*query_trunk_size:(i+1)*query_trunk_size]

            curr_dist = []
            for j in range(num_refer_trunk):
                curr_refer = feats_point_cloud[j*refer_trunk_size:(j+1)*refer_trunk_size]
                adj_matrix = pairwise_distance(curr_refer, curr_query)
                adj_matrix = -adj_matrix

                curr_dist.append(adj_matrix)

            if remain_refer > 0:
                curr_refer = feats_point_cloud[num_refer_trunk * refer_trunk_size:]
                adj_matrix = pairwise_distance(curr_refer, curr_query)
                adj_matrix = -adj_matrix

                curr_dist.append(adj_matrix)

            curr_dist = torch.cat(curr_dist, 1)
            knnG.append(torch.topk(curr_dist, k=k+1)[1])

        # if there remain some data points ...
        if remain_query > 0:
            curr_query = feats_point_cloud[num_query_trunk * query_trunk_size:]

            curr_dist = []
            for j in range(num_refer_trunk):
                curr_refer = feats_point_cloud[j * refer_trunk_size:(j + 1) * refer_trunk_size]
                adj_matrix = pairwise_distance(curr_refer, curr_query)
                adj_matrix = -adj_matrix

                curr_dist.append(adj_matrix)

            if remain_refer > 0:
                curr_refer = feats_point_cloud[num_refer_trunk * refer_trunk_size:]
                adj_matrix = pairwise_distance(curr_refer, curr_query)
                adj_matrix = -adj_matrix

                curr_dist.append(adj_matrix)

            curr_dist = torch.cat(curr_dist, 1)
            knnG.append(torch.topk(curr_dist, k=k + 1)[1])

        knnG = torch.cat(knnG, 0)
        knnG_list = knnG.cpu().numpy().tolist()

    return knnG_list


# -- function for computing topo weights
def calc_topo_weights_with_components_idx(ntrain, prob_all, feats_point_cloud, ori_label, pred_label,
                                          use_log=False, nclass=10, k=2, cp_opt=3,
                                          refer_trunk_size=50000, query_trunk_size=10000):
    """
    Since GPU knn is memory intensive, so we split the query and reference data points into several trunks.
    Each time, we process a trunk of data (in other words, a batch of data).

    refer_trunk_size: The trunk size for the reference points.
    query_trunk_size: The trunk size for the query points.

    nclass: The number of class.
    cp_opt: Should always be set to 3 here. Just use it as a black box. The underlying reason is rooted in the C++ code
        for computing the largest connected component (which was originally written for computing the persistent homology).
    """
    # -- first, compute the knn graph --
    print('computing knn graph')
    start = timeit.default_timer()
    
    knnG_list = calc_knn_graph(feats_point_cloud, k=k, refer_trunk_size=refer_trunk_size, query_trunk_size=query_trunk_size)

    stop = timeit.default_timer()
    print('Finish computing knn graph. Consume time: ', stop - start)

    # -- next, compute phi functions, which is related to persistent homology --
    data_selected = set()  # whether a data has been selected
    tot_num_comp = 0
    tot_comp_nvert = 0
    tot_num_pt2fix = 0

    topo_wt = np.zeros((ntrain, nclass))
    idx_of_small_comps = set()

    start = timeit.default_timer()
    for j in range(nclass):
        tmp_prob_curr = prob_all[:, j]
        tmp_prob_all = prob_all.copy()
        tmp_prob_all[:, j] = -1.0
        tmp_prob_alt = np.amax(tmp_prob_all, axis=1)
        tmp_best_alt = np.argmax(tmp_prob_all, axis=1)
        if use_log:
            phi = np.log(tmp_prob_alt) - np.log(tmp_prob_curr)
        else:
            phi = tmp_prob_alt - tmp_prob_curr

        phi_list = list(phi.ravel())

        # Compute persistence
        skip1D = 1
        levelset_val = 0 + np.finfo('float32').eps
        relevant_vlist = PyPersAll(phi_list, knnG_list, ntrain, levelset_val, skip1D, j, ori_label, pred_label)

        assert len(relevant_vlist) == 6
        assert relevant_vlist[0][0] == len(relevant_vlist[1])

        tot_comp_nvert = tot_comp_nvert + relevant_vlist[0][0]
        tot_num_comp = tot_num_comp + relevant_vlist[0][2]
        tot_num_pt2fix = tot_num_pt2fix + len(relevant_vlist[2 + cp_opt])

        curr_comp_nvert = relevant_vlist[0][0]
        curr_ncomp = relevant_vlist[0][2]

        # relevant_vlist[2] -- comp vert list
        # relevant_vlist[3] -- birth vert list
        # relevant_vlist[4] -- crit vert list
        # relevant_vlist[5] -- rob crit vert list
        assert curr_comp_nvert == len(relevant_vlist[2])
        assert curr_ncomp <= len(relevant_vlist[2])  # less and equal
        assert curr_ncomp == len(relevant_vlist[3])
        assert curr_ncomp <= len(relevant_vlist[4])  # less and equal
        assert curr_ncomp >= len(relevant_vlist[5])

        if curr_ncomp == 0:
            print('WARNING: No extra components, skip to the next label.')
            continue

        selected_vidx = relevant_vlist[2 + cp_opt]
        selected_vidx = list(set(selected_vidx).difference(data_selected))
        data_selected = data_selected.union(set(selected_vidx))

        topo_wt[selected_vidx, j] = -1.0
        topo_wt[selected_vidx, tmp_best_alt[selected_vidx]] = 1.0

        idx_of_small_comps = idx_of_small_comps.union(relevant_vlist[2])

    idx_of_small_comps = list(idx_of_small_comps)

    return topo_wt, idx_of_small_comps
