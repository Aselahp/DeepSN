# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch

from torch import nn
from torch_scatter import scatter_add
from torch_geometric.utils import degree
from main.model.orthogonal import Orthogonal

import itertools
import math
import torch_sparse


def remove_duplicate_edges(edge_index):
    processed_edges = set()
    new_edge_index = []

    for e in range(edge_index.size(1)):
        source, target = sorted((edge_index[0, e].item(), edge_index[1, e].item()))
        if (source, target) in processed_edges:
            continue
        processed_edges.add((source, target))
        new_edge_index.append([source, target])
    print(f"Removed {edge_index.size(1) - len(new_edge_index)} edges")
    return torch.tensor(new_edge_index, dtype=torch.long).t()


def build_sheaf_laplacian(N, K, edge_index, maps):
    """
    Builds a sheaf laplacian given the edge_index and the restriction maps

    Args:
        N: The number of nodes in the graph
        K: The dimensionality of the Stalks
        edge_index: Edge index of the graph without duplicate edges. We assume that edge i has orientation
            edge_index[0, i] --> edge_index[1, i].
        maps: Tensor of shape [edge_index.size(1), 2 (source/target), K, K] containing the restriction maps of the sheaf
    Returns:
        (index, value): The sheaf Laplacian as a sparse matrix of size (N*K, N*K)
    """
    E = edge_index.size(1)
    index = []
    values = []

    for e in range(E):
        source = edge_index[0, e]
        target = edge_index[1, e]

        top_x = e * K
        # Generate the positions in the block matrix
        top_y = source * K
        for i, j in itertools.product(range(K), range(K)):
            index.append([top_x + i, top_y + j])
            values.append(-maps[e, 0, i, j])

        top_y = target * K
        for i, j in itertools.product(range(K), range(K)):
            index.append([top_x + i, top_y + j])
            values.append(maps[e, 1, i, j])

    index = torch.tensor(index, dtype=torch.long).T
    values = torch.tensor(values)

    index_t, values_t = torch_sparse.transpose(index, values, E * K, N * K)
    index, value = torch_sparse.spspmm(index_t, values_t, index, values, N * K, E * K, N * K, coalesced=True)
    return torch_sparse.coalesce(index, value, N * K, N * K)


def sym_matrix_pow(matrix: torch.Tensor, p: float) -> torch.Tensor:
    r"""
    Power of a matrix using Eigen Decomposition.
    Args:
        matrix: a batch of matrices
        p: power
    Returns:
        Power of a matrix
    """
    vals, vecs = torch.linalg.eigh(matrix)
    vals[vals > 0] = vals[vals > 0].pow(p)
    matrix_pow = vecs @ torch.diag(vals) @ vecs.T
    return matrix_pow


def build_norm_sheaf_laplacian(N, K, edge_index, maps, augmented=True):
    """
    Builds a normalised sheaf laplacian given the edge_index and the restriction maps.

    Args:
        N: The number of nodes in the graph
        K: The dimensionality of the Stalks
        edge_index: Edge index of the graph without duplicate edges. We assume that edge i has orientation
            edge_index[0, i] --> edge_index[1, i].
        maps: Tensor of shape [edge_index.size(1), 2 (source/target), K, K] containing the restriction maps of the sheaf
        augmented: Use D* = D + I instead of D.
    Returns:
        (index, value): The normalised sheaf Laplacian as a sparse matrix of size (N*K, N*K)
    """
    index, values = build_sheaf_laplacian(N, K, edge_index, maps)
    block_diag_indices = []
    block_diag_values = []

    for i in range(N):
        low = i * K
        high = low + K

        mask1 = torch.logical_and(low <= index[0, :], index[0, :] < high)
        mask2 = torch.logical_and(low <= index[1, :], index[1, :] < high)
        mask = torch.logical_and(mask1, mask2)

        d_index = index[:, mask]
        d_values = values[mask]
        d_index = d_index - low

        Dv = torch.sparse_coo_tensor(d_index, d_values).to_dense()
        assert list(Dv.size()) == [K, K]
        if augmented:
            Dv = Dv + torch.eye(K, K)
        Dv_sqrt_inv = sym_matrix_pow(Dv, -0.5).to_sparse()

        block_diag_indices.append(Dv_sqrt_inv.indices() + low)
        block_diag_values.append(Dv_sqrt_inv.values())

    D_sqrt_inv_idx = torch.cat(block_diag_indices, dim=1)
    D_sqrt_val = torch.cat(block_diag_values, dim=0)

    tmp_idx, tmp_val = torch_sparse.spspmm(D_sqrt_inv_idx, D_sqrt_val, index, values, N * K, N * K, N * K,
                                           coalesced=True)
    index, value = torch_sparse.spspmm(tmp_idx, tmp_val, D_sqrt_inv_idx, D_sqrt_val, N * K, N * K, N * K,
                                       coalesced=True)
    return torch_sparse.coalesce(index, value, N * K, N * K)


def build_sheaf_difussion_matrix(N, K, edge_index, maps, augmented=True, return_laplacian=False):
    """
    Builds the difussion matrix P := I - D*^{-1/2}LD*^{-1/2}, where D* = D + I

    Args:
        N: The number of nodes in the graph
        K: The dimensionality of the Stalks
        edge_index: Edge index of the graph without duplicate edges. We assume that edge i has orientation
            edge_index[0, i] --> edge_index[1, i].
        maps: Tensor of shape [edge_index.size(1), 2 (source/target), K, K] containing the restriction maps of the sheaf
        augmented: Use the augmented sheaf Laplacian.
        return_laplacian: Also returns the Laplacian as a second argument.
    Returns:
        (index, value): The difussion matrix associated with the normalised sheaf Laplacian.
    """
    L_index, L_val = build_norm_sheaf_laplacian(N, K, edge_index, maps, augmented=augmented)

    I_index = torch.arange(0, N * K).view(1, -1).tile(2, 1)
    I_val = torch.ones((N * K,))

    index = torch.cat((L_index, I_index), dim=1)
    value = torch.cat((-L_val, I_val), dim=0)

    P_index, P_val = torch_sparse.coalesce(index, value, N * K, N * K, op='add')
    if return_laplacian:
        L_index, L_val = torch_sparse.coalesce(L_index, L_val, N * K, N * K, op='add')
        return (P_index, P_val), (L_index, L_val)
    return P_index, P_val


def dirichlet_energy(L, f, size):
    """Returns the Dirichlet energy of the signal f under the sheaf Laplacian L."""
    right = torch_sparse.spmm(L[0], L[1], size, size, f)
    energy = f.t() @ right
    return energy.item()


def get_edge_index_dict(edge_index, undirected=True):
    """Computes a dictionary mapping the undirected edges in edge_index to an ID."""
    assert edge_index.size(1) % 2 == 0

    E = edge_index.size(1)
    edge_idx_dict = dict()
    next_id = 0

    for e in range(E):
        source = edge_index[0, e].item()
        target = edge_index[1, e].item()
        if undirected:
            edge = tuple(sorted([source, target]))
        else:
            edge = tuple([source, target])

        # Generate or retrieve the edge index
        if edge not in edge_idx_dict:
            edge_idx_dict[edge] = next_id
            next_id += 1

    return edge_idx_dict


def compute_incidence_index(edge_index, d):
    """Computes the indices of a sheaf coboundary matrix from the edge_index of the graph."""
    assert edge_index.size(1) % 2 == 0

    edge_idx_dict = get_edge_index_dict(edge_index)
    index = []

    for edge in range(edge_index.size(1)):
        source = edge_index[0, edge].item()
        target = edge_index[1, edge].item()
        edge_key = tuple(sorted([source, target]))

        top_x = edge_idx_dict[edge_key] * d
        top_y = source * d
        for i, j in itertools.product(range(d), range(d)):
            index.append([top_x + i, top_y + j])

    incidence_index = torch.tensor(index, dtype=torch.long).T
    assert list(incidence_index.size()) == [2, edge_index.size(1) * (d ** 2)]
    return incidence_index


def build_dense_laplacian(size, edge_index, maps, d, normalised=False, diagonal_maps=False, values=None,
                          edge_weights=None):
    """Builds a sheaf laplacian from a given graph using naive dense computations (used for testing)."""
    assert edge_index.size(1) % 2 == 0
    if diagonal_maps:
        assert len(maps.size()) == 2
        assert maps.size(1) == d

    E = edge_index.size(1) // 2
    N = size
    Delta = torch.zeros(size=(E*d, N*d), dtype=torch.float64)
    undirected_edge_idx_dict = get_edge_index_dict(edge_index)
    directed_edge_idx_dict = get_edge_index_dict(edge_index, undirected=False)

    for e in range(edge_index.size(1)):
        source = edge_index[0, e].item()
        target = edge_index[1, e].item()
        edge_key = tuple(sorted([source, target]))

        # Generate the positions in the block matrix
        top_x = undirected_edge_idx_dict[edge_key] * d
        top_y = source * d

        orient = -1 if edge_key[0] == source else 1
        if edge_weights is not None:
            factor1_idx, factor2_idx = (
                directed_edge_idx_dict[(source, target)], directed_edge_idx_dict[(target, source)])
            assert edge_weights[factor1_idx] == edge_weights[factor2_idx]
            maps[e] = maps[e] * edge_weights[factor1_idx]
        if diagonal_maps:
            diag_idx = torch.arange(0, d)
            Delta[top_x + diag_idx, top_y + diag_idx] = orient * maps[e]
        else:
            Delta[top_x: top_x+d, top_y: top_y+d] = orient * maps[e]

    # Compute non-normalised Laplacian.
    L_dense = Delta.T @ Delta

    if values is not None:
        # Append extra entries to the diagonal of the parallel transport maps and update the stalk dimension.
        L_dense, d = append_diag_maps_to_existent_laplacian(size, d, L_dense, edge_index, values)

    if not normalised:
        return L_dense

    # Build normalised Laplacian.
    D_sqrt_inv = torch.zeros((N*d, N*d), dtype=torch.float64)
    for i in range(N):
        low = i * d
        high = low + d

        D_i = L_dense[low:high, low:high]
        D_i = D_i + torch.eye(d)
        D_i_sqrt_inv = sym_matrix_pow(D_i, -0.5)
        D_sqrt_inv[low:high, low:high] = D_i_sqrt_inv

    return D_sqrt_inv @ L_dense @ D_sqrt_inv


def append_diag_maps_to_existent_laplacian(size, learnable_d, L, edge_index, values):
    extra_d = len(values)
    total_d = learnable_d + extra_d

    deg = degree(edge_index[0], num_nodes=size, dtype=L.dtype)
    values = torch.tensor(values, dtype=L.dtype)
    new_L = torch.zeros((size*(extra_d + learnable_d), size*(extra_d + learnable_d)), dtype=L.dtype)

    for idx in range(edge_index.size(1)):
        i, j = edge_index[0][idx], edge_index[1][idx]
        assert i != j

        # Add to the new Laplacian the entries of the existent Laplacian
        new_low_i, new_high_i = i * total_d, i * total_d + learnable_d
        new_low_j, new_high_j = j * total_d, j * total_d + learnable_d

        low_i, high_i = i * learnable_d, i * learnable_d + learnable_d
        low_j, high_j = j * learnable_d, j * learnable_d + learnable_d

        new_L[new_low_i:new_high_i, new_low_j:new_high_j] = L[low_i:high_i, low_j:high_j]

        # Append extra entries to each parallel transport map
        extra_diag_idx = torch.arange(learnable_d, total_d)
        new_L[new_low_i + extra_diag_idx, new_low_j + extra_diag_idx] = values

    for i in range(size):
        # Add to the new Laplacian diagonal, the diagonal entries of the existent Laplacian
        new_low_i, new_high_i = i * total_d, i * total_d + learnable_d
        low_i, high_i = i * learnable_d, i * learnable_d + learnable_d
        new_L[new_low_i:new_high_i, new_low_i:new_high_i] = L[low_i:high_i, low_i:high_i]

        # Append the degree on the diagonal for the extra entries
        extra_diag_idx = torch.arange(learnable_d, total_d)
        new_L[i * total_d + extra_diag_idx, i * total_d + extra_diag_idx] = deg[i]

    return new_L, total_d


def compute_left_right_map_index(edge_index, full_matrix=False):
    """Computes indices for lower triangular matrix or full matrix"""
    edge_to_idx = dict()
    for e in range(edge_index.size(1)):
        source = edge_index[0, e].item()
        target = edge_index[1, e].item()
        edge_to_idx[(source, target)] = e

    left_index, right_index = [], []
    row, col = [], []
    for e in range(edge_index.size(1)):
        source = edge_index[0, e].item()
        target = edge_index[1, e].item()
        if source < target or full_matrix:
            left_index.append(e)
            right_index.append(edge_to_idx[(target, source)])

            row.append(source)
            col.append(target)

    left_index = torch.tensor(left_index, dtype=torch.long, device=edge_index.device)
    right_index = torch.tensor(right_index, dtype=torch.long, device=edge_index.device)
    left_right_index = torch.vstack([left_index, right_index])

    row = torch.tensor(row, dtype=torch.long, device=edge_index.device)
    col = torch.tensor(col, dtype=torch.long, device=edge_index.device)
    new_edge_index = torch.vstack([row, col])

    if full_matrix:
        assert len(left_index) == edge_index.size(1)
    else:
        assert len(left_index) == edge_index.size(1) // 2

    return left_right_index, new_edge_index


def compute_learnable_laplacian_indices(size, edge_index, learned_d, total_d):
    assert torch.all(edge_index[0] < edge_index[1])

    row, col = edge_index
    device = edge_index.device
    row_template = torch.arange(0, learned_d, device=device).view(1, -1, 1).tile(1, 1, learned_d)
    col_template = torch.transpose(row_template, dim0=1, dim1=2)

    non_diag_row_indices = (row_template + total_d*row.reshape(-1, 1, 1)).reshape(1, -1)
    non_diag_col_indices = (col_template + total_d*col.reshape(-1, 1, 1)).reshape(1, -1)
    non_diag_indices = torch.cat((non_diag_row_indices, non_diag_col_indices), dim=0)

    diag = torch.arange(0, size, device=device)
    diag_row_indices = (row_template + total_d*diag.reshape(-1, 1, 1)).reshape(1, -1)
    diag_col_indices = (col_template + total_d*diag.reshape(-1, 1, 1)).reshape(1, -1)
    diag_indices = torch.cat((diag_row_indices, diag_col_indices), dim=0)

    return diag_indices, non_diag_indices


def compute_learnable_diag_laplacian_indices(size, edge_index, learned_d, total_d):
    assert torch.all(edge_index[0] < edge_index[1])
    row, col = edge_index
    device = edge_index.device
    row_template = torch.arange(0, learned_d, device=device).view(1, -1)
    col_template = row_template.clone()

    non_diag_row_indices = (row_template + total_d*row.unsqueeze(1)).reshape(1, -1)
    non_diag_col_indices = (col_template + total_d*col.unsqueeze(1)).reshape(1, -1)
    non_diag_indices = torch.cat((non_diag_row_indices, non_diag_col_indices), dim=0)

    diag = torch.arange(0, size, device=device)
    diag_row_indices = (row_template + total_d*diag.unsqueeze(1)).reshape(1, -1)
    diag_col_indices = (col_template + total_d*diag.unsqueeze(1)).reshape(1, -1)
    diag_indices = torch.cat((diag_row_indices, diag_col_indices), dim=0)

    return diag_indices, non_diag_indices


def compute_fixed_diag_laplacian_indices(size, edge_index, learned_d, total_d):
    assert torch.all(edge_index[0] < edge_index[1])
    row, col = edge_index
    device = edge_index.device
    row_template = torch.arange(learned_d, total_d, device=device).view(1, -1)
    col_template = row_template.clone()

    non_diag_row_indices = (row_template + total_d*row.unsqueeze(1)).reshape(1, -1)
    non_diag_col_indices = (col_template + total_d*col.unsqueeze(1)).reshape(1, -1)
    non_diag_indices = torch.cat((non_diag_row_indices, non_diag_col_indices), dim=0)

    diag = torch.arange(0, size, device=device)
    diag_row_indices = (row_template + total_d*diag.unsqueeze(1)).reshape(1, -1)
    diag_col_indices = (col_template + total_d*diag.unsqueeze(1)).reshape(1, -1)
    diag_indices = torch.cat((diag_row_indices, diag_col_indices), dim=0)

    return diag_indices, non_diag_indices


def batched_sym_matrix_pow(matrices: torch.Tensor, p: float) -> torch.Tensor:
    r"""
    Power of a matrix using Eigen Decomposition.
    Args:
        matrices: A batch of matrices.
        p: Power.
        positive_definite: If positive definite
    Returns:
        Power of each matrix in the batch.
    """
    # vals, vecs = torch.linalg.eigh(matrices)
    # SVD is much faster than  vals, vecs = torch.linalg.eigh(matrices) for large batches.
    vecs, vals, _ = torch.linalg.svd(matrices)
    good = vals > vals.max(-1, True).values * vals.size(-1) * torch.finfo(vals.dtype).eps
    vals = vals.pow(p).where(good, torch.zeros((), device=matrices.device, dtype=matrices.dtype))
    matrix_power = (vecs * vals.unsqueeze(-2)) @ torch.transpose(vecs, -2, -1)
    return matrix_power


def mergesp(index1, value1, index2, value2):
    """Merges two sparse matrices with disjoint indices into one."""
    assert index1.dim() == 2 and index2.dim() == 2
    assert value1.dim() == 1 and value2.dim() == 1
    assert index1.size(1) == value1.numel()
    assert index2.size(1) == value2.numel()
    assert index1.size(0) == 2 and index2.size(0) == 2

    index = torch.cat([index1, index2], dim=1)
    val = torch.cat([value1, value2])
    return index, val


def get_random_edge_weights(edge_index):
    edge_dict = get_edge_index_dict(edge_index, undirected=False)
    edge_weights = torch.FloatTensor(size=(edge_index.size(1), 1)).uniform_(0.0, 1.0)

    # Make the edge weights symmetric
    for i in range(edge_index.size(1)):
        v = edge_index[0, i].item()
        u = edge_index[1, i].item()
        edge_weights[edge_dict[(v, u)]] = edge_weights[edge_dict[(u, v)]]
    return edge_weights


def get_2d_oracle_rotation_angles(edge_index, y, theta=None):
    """Returns the class rotation angles for an oracle 2D orthogonal sheaf."""
    assert y.min() == 0
    if theta is None:
        # This is to be multiplied by 2pi during the construction of the orthogonal matrix
        # in the Connection Laplacian builder.
        theta = 2.0 * math.pi / (y.max() + 1)

    angles = torch.empty(edge_index.size(1), dtype=torch.float32)
    for i in range(edge_index.size(1)):
        v = edge_index[0, i].item()
        u = edge_index[1, i].item()
        cdiff = abs(float(y[u].item() - y[v].item()))
        if v < u:
            angles[i] = theta * cdiff / 2.0
        else:
            angles[i] = -theta * cdiff / 2.0
    assert angles.max() < 2 * math.pi
    return angles.view(-1, 1)


def get_1d_oracle_maps(edge_index, y):
    """Returns the maps for an oracle 2D orthogonal sheaf."""
    maps = torch.empty(edge_index.size(1), dtype=edge_index.dtype)
    for i in range(edge_index.size(1)):
        v = edge_index[0, i].item()
        u = edge_index[1, i].item()
        if v < u or y[v].item() == y[u].item():
            maps[i] = 1.0
        else:
            maps[i] = -1.0
    return maps.view(-1, 1)





class LaplacianBuilder(nn.Module):

    def __init__(self, size, edge_index, d, normalised=False, deg_normalised=False, add_hp=False, add_lp=False,
                 augmented=True):
        super(LaplacianBuilder, self).__init__()
        assert not (normalised and deg_normalised)

        self.d = d
        self.final_d = d
        if add_hp:
            self.final_d += 1
        if add_lp:
            self.final_d += 1
        self.size = size
        self.edges = edge_index.size(1) // 2
        self.edge_index = edge_index
        self.normalised = normalised
        self.deg_normalised = deg_normalised
        self.device = edge_index.device
        self.add_hp = add_hp
        self.add_lp = add_lp
        self.augmented = augmented

        # Preprocess the sparse indices required to compute the Sheaf Laplacian.
        self.full_left_right_idx, _ = compute_left_right_map_index(edge_index, full_matrix=True)
        self.left_right_idx, self.vertex_tril_idx = compute_left_right_map_index(edge_index)
        if self.add_lp or self.add_hp:
            self.fixed_diag_indices, self.fixed_tril_indices = compute_fixed_diag_laplacian_indices(
                size, self.vertex_tril_idx, self.d, self.final_d)
        self.deg = degree(self.edge_index[0], num_nodes=self.size)

    def get_fixed_maps(self, size, dtype):
        assert self.add_lp or self.add_hp

        fixed_diag, fixed_non_diag = [], []
        if self.add_lp:
            fixed_diag.append(self.deg.view(-1, 1))
            fixed_non_diag.append(torch.ones(size=(size, 1), device=self.device, dtype=dtype))
        if self.add_hp:
            fixed_diag.append(self.deg.view(-1, 1))
            fixed_non_diag.append(-torch.ones(size=(size, 1), device=self.device, dtype=dtype))

        fixed_diag = torch.cat(fixed_diag, dim=1)
        fixed_non_diag = torch.cat(fixed_non_diag, dim=1)

        assert self.fixed_tril_indices.size(1) == fixed_non_diag.numel()
        assert self.fixed_diag_indices.size(1) == fixed_diag.numel()

        return fixed_diag, fixed_non_diag

    def scalar_normalise(self, diag, tril, row, col):
        if tril.dim() > 2:
            assert tril.size(-1) == tril.size(-2)
            assert diag.dim() == 2
        d = diag.size(-1)

        if self.augmented:
            diag_sqrt_inv = (diag + 1).pow(-0.5)
        else:
            diag_sqrt_inv = diag.pow(-0.5)
            diag_sqrt_inv.masked_fill_(diag_sqrt_inv == float('inf'), 0)
        diag_sqrt_inv = diag_sqrt_inv.view(-1, 1, 1) if tril.dim() > 2 else diag_sqrt_inv.view(-1, d)
        left_norm = diag_sqrt_inv[row]
        right_norm = diag_sqrt_inv[col]
        non_diag_maps = left_norm * tril * right_norm

        diag_sqrt_inv = diag_sqrt_inv.view(-1, 1, 1) if diag.dim() > 2 else diag_sqrt_inv.view(-1, d)
        diag_maps = diag_sqrt_inv**2 * diag

        return diag_maps, non_diag_maps

    def append_fixed_maps(self, size, diag_indices, diag_maps, tril_indices, tril_maps):
        if not self.add_lp and not self.add_hp:
            return (diag_indices, diag_maps), (tril_indices, tril_maps)

        fixed_diag, fixed_non_diag = self.get_fixed_maps(size, tril_maps.dtype)
        tril_row, tril_col = self.vertex_tril_idx

        # Normalise the fixed parts.
        if self.normalised:
            fixed_diag, fixed_non_diag = self.scalar_normalise(fixed_diag, fixed_non_diag, tril_row, tril_col)
        fixed_diag, fixed_non_diag = fixed_diag.view(-1), fixed_non_diag.view(-1)
        # Combine the learnable and fixed parts.
        tril_indices, tril_maps = mergesp(self.fixed_tril_indices, fixed_non_diag, tril_indices, tril_maps)
        diag_indices, diag_maps = mergesp(self.fixed_diag_indices, fixed_diag, diag_indices, diag_maps)

        return (diag_indices, diag_maps), (tril_indices, tril_maps)

    def create_with_new_edge_index(self, edge_index):
        assert edge_index.max() <= self.size
        new_builder = self.__class__(
            self.size, edge_index, self.d,
            normalised=self.normalised, deg_normalised=self.deg_normalised, add_hp=self.add_hp, add_lp=self.add_lp,
            augmented=self.augmented)
        new_builder.train(self.training)
        return new_builder


class DiagLaplacianBuilder(LaplacianBuilder):
    """Learns a a Sheaf Laplacian with diagonal restriction maps"""

    def __init__(self, size, edge_index, d, normalised=False, deg_normalised=False, add_hp=False, add_lp=False,
                 augmented=True):
        super(DiagLaplacianBuilder, self).__init__(
            size, edge_index, d, normalised, deg_normalised, add_hp, add_lp, augmented)

        self.diag_indices, self.tril_indices = compute_learnable_diag_laplacian_indices(
            size, self.vertex_tril_idx, self.d, self.final_d)

    def normalise(self, diag, tril, row, col):
        if self.normalised:
            d_sqrt_inv = (diag + 1).pow(-0.5) if self.augmented else diag.pow(-0.5)
            left_norm, right_norm = d_sqrt_inv[row], d_sqrt_inv[col]
            tril = left_norm * tril * right_norm
            diag = d_sqrt_inv * diag * d_sqrt_inv
        elif self.deg_normalised:
            deg_sqrt_inv = (self.deg + 1).pow(-0.5) if self.augmented else self.deg.pow(-0.5)
            deg_sqrt_inv = deg_sqrt_inv.unsqueeze(-1)
            deg_sqrt_inv.masked_fill_(deg_sqrt_inv == float('inf'), 0)
            left_norm, right_norm = deg_sqrt_inv[row], deg_sqrt_inv[col]
            tril = left_norm * tril * right_norm
            diag = deg_sqrt_inv * diag * deg_sqrt_inv
        return diag, tril

    def forward(self, maps):
        assert len(maps.size()) == 2
        assert maps.size(1) == self.d
        left_idx, right_idx = self.left_right_idx
        tril_row, tril_col = self.vertex_tril_idx
        row, _ = self.edge_index

        # Compute the un-normalised Laplacian entries.
        left_maps = torch.index_select(maps, index=left_idx, dim=0)
        right_maps = torch.index_select(maps, index=right_idx, dim=0)
        tril_maps = -left_maps * right_maps
        saved_tril_maps = tril_maps.detach().clone()
        diag_maps = scatter_add(maps**2, row, dim=0, dim_size=self.size)

        # Normalise the entries if the normalised Laplacian is used.
        diag_maps, tril_maps = self.normalise(diag_maps, tril_maps, tril_row, tril_col)
        tril_indices, diag_indices = self.tril_indices, self.diag_indices
        tril_maps, diag_maps = tril_maps.view(-1), diag_maps.view(-1)

        # Append fixed diagonal values in the non-learnable dimensions.
        (diag_indices, diag_maps), (tril_indices, tril_maps) = self.append_fixed_maps(
            len(left_maps), diag_indices, diag_maps, tril_indices, tril_maps)

        # Add the upper triangular part
        triu_indices = torch.empty_like(tril_indices)
        triu_indices[0], triu_indices[1] = tril_indices[1], tril_indices[0]
        non_diag_indices, non_diag_values = mergesp(tril_indices, tril_maps, triu_indices, tril_maps)

        # Merge diagonal and non-diagonal
        edge_index, weights = mergesp(non_diag_indices, non_diag_values, diag_indices, diag_maps)

        return (edge_index, weights), saved_tril_maps


class NormConnectionLaplacianBuilder(LaplacianBuilder):
    """Learns a a Sheaf Laplacian with diagonal restriction maps"""

    def __init__(self, size, edge_index, d, add_hp=False, add_lp=False, orth_map=None, augmented=True):
        super(NormConnectionLaplacianBuilder, self).__init__(
            size, edge_index, d, add_hp=add_hp, add_lp=add_lp, normalised=True, augmented=augmented)
        self.orth_transform = Orthogonal(d=self.d, orthogonal_map=orth_map)
        self.orth_map = orth_map

        _, self.tril_indices = compute_learnable_laplacian_indices(
            size, self.vertex_tril_idx, self.d, self.final_d)
        self.diag_indices, _ = compute_learnable_diag_laplacian_indices(
            size, self.vertex_tril_idx, self.d, self.final_d)

    def create_with_new_edge_index(self, edge_index):
        assert edge_index.max() <= self.size
        new_builder = self.__class__(
            self.size, edge_index, self.d, add_hp=self.add_hp, add_lp=self.add_lp, augmented=self.augmented,
            orth_map=self.orth_map)
        new_builder.train(self.training)
        return new_builder

    def normalise(self, diag, tril, row, col):
        if tril.dim() > 2:
            assert tril.size(-1) == tril.size(-2)
            assert diag.dim() == 2
        d = diag.size(-1)

        if self.augmented:
            diag_sqrt_inv = (diag + 1).pow(-0.5)
        else:
            diag_sqrt_inv = diag.pow(-0.5)
            diag_sqrt_inv.masked_fill_(diag_sqrt_inv == float('inf'), 0)
        diag_sqrt_inv = diag_sqrt_inv.view(-1, 1, 1) if tril.dim() > 2 else diag_sqrt_inv.view(-1, d)
        left_norm = diag_sqrt_inv[row]
        right_norm = diag_sqrt_inv[col]
        non_diag_maps = left_norm * tril * right_norm

        diag_sqrt_inv = diag_sqrt_inv.view(-1, 1, 1) if diag.dim() > 2 else diag_sqrt_inv.view(-1, d)
        diag_maps = diag_sqrt_inv**2 * diag

        return diag_maps, non_diag_maps

    def forward(self, map_params, edge_weights=None):
        if edge_weights is not None:
            assert edge_weights.size(1) == 1
        assert len(map_params.size()) == 2
        if self.orth_map in ["matrix_exp", "cayley"]:
            assert map_params.size(1) == self.d * (self.d + 1) // 2
        else:
            assert map_params.size(1) == self.d * (self.d - 1) // 2

        _, full_right_idx = self.full_left_right_idx
        left_idx, right_idx = self.left_right_idx
        tril_row, tril_col = self.vertex_tril_idx
        tril_indices, diag_indices = self.tril_indices, self.diag_indices
        row, _ = self.edge_index

        # Convert the parameters to orthogonal matrices.
        maps = self.orth_transform(map_params)
        if edge_weights is None:
            diag_maps = self.deg.unsqueeze(-1)
        else:
            diag_maps = scatter_add(edge_weights ** 2, row, dim=0, dim_size=self.size)
            maps = maps * edge_weights.unsqueeze(-1)

        # Compute the transport maps.
        left_maps = torch.index_select(maps, index=left_idx, dim=0)
        right_maps = torch.index_select(maps, index=right_idx, dim=0)
        tril_maps = -torch.bmm(torch.transpose(left_maps, -1, -2), right_maps)
        saved_tril_maps = tril_maps.detach().clone()

        # Normalise the entries if the normalised Laplacian is used.
        diag_maps, tril_maps = self.scalar_normalise(diag_maps, tril_maps, tril_row, tril_col)
        tril_maps, diag_maps = tril_maps.view(-1), diag_maps.expand(-1, self.d).reshape(-1)

        # Append fixed diagonal values in the non-learnable dimensions.
        (diag_indices, diag_maps), (tril_indices, tril_maps) = self.append_fixed_maps(
            len(left_maps), diag_indices, diag_maps, tril_indices, tril_maps)

        # Add the upper triangular part
        triu_indices = torch.empty_like(tril_indices)
        triu_indices[0], triu_indices[1] = tril_indices[1], tril_indices[0]
        non_diag_indices, non_diag_values = mergesp(tril_indices, tril_maps, triu_indices, tril_maps)

        # Merge diagonal and non-diagonal
        edge_index, weights = mergesp(non_diag_indices, non_diag_values, diag_indices, diag_maps)

        return (edge_index, weights), saved_tril_maps


class GeneralLaplacianBuilder(LaplacianBuilder):
    """Learns a multi-dimensional Sheaf Laplacian from data."""

    def __init__(self, size, edge_index, d, normalised=False, deg_normalised=False,
                 add_hp=False, add_lp=False, augmented=True):
        super(GeneralLaplacianBuilder, self).__init__(size, edge_index, d,
                                                      normalised=normalised, deg_normalised=deg_normalised,
                                                      add_hp=add_hp, add_lp=add_lp, augmented=augmented)

        # Preprocess the sparse indices required to compute the Sheaf Laplacian.
        self.diag_indices, self.tril_indices = compute_learnable_laplacian_indices(
            size, self.vertex_tril_idx, self.d, self.final_d)

    def normalise(self, diag_maps, non_diag_maps, tril_row, tril_col):
        if self.normalised:
            # Normalise the entries if the normalised Laplacian is used.
            if self.training:
                # During training, we perturb the matrices to ensure they have different singular values.
                # Without this, the gradients of batched_sym_matrix_pow, which uses SVD are non-finite.
                eps = torch.FloatTensor(self.d).uniform_(-0.001, 0.001).to(device=self.device)
            else:
                eps = torch.zeros(self.d, device=self.device)

            to_be_inv_diag_maps = diag_maps + torch.diag(1. + eps).unsqueeze(0) if self.augmented else diag_maps
            d_sqrt_inv = batched_sym_matrix_pow(to_be_inv_diag_maps, -0.5)
            assert torch.all(torch.isfinite(d_sqrt_inv))
            left_norm = d_sqrt_inv[tril_row]
            right_norm = d_sqrt_inv[tril_col]
            non_diag_maps = (left_norm @ non_diag_maps @ right_norm).clamp(min=-1, max=1)
            diag_maps = (d_sqrt_inv @ diag_maps @ d_sqrt_inv).clamp(min=-1, max=1)
            assert torch.all(torch.isfinite(non_diag_maps))
            assert torch.all(torch.isfinite(diag_maps))
        elif self.deg_normalised:
            # These are general d x d maps so we need to divide by 1 / sqrt(deg * d), their maximum possible norm.
            deg_sqrt_inv = (self.deg * self.d + 1).pow(-1/2) if self.augmented else (self.deg * self.d + 1).pow(-1/2)
            deg_sqrt_inv = deg_sqrt_inv.view(-1, 1, 1)
            left_norm = deg_sqrt_inv[tril_row]
            right_norm = deg_sqrt_inv[tril_col]
            non_diag_maps = left_norm * non_diag_maps * right_norm
            diag_maps = deg_sqrt_inv * diag_maps * deg_sqrt_inv
        return diag_maps, non_diag_maps

    def forward(self, maps):
        left_idx, right_idx = self.left_right_idx
        tril_row, tril_col = self.vertex_tril_idx
        tril_indices, diag_indices = self.tril_indices, self.diag_indices
        row, _ = self.edge_index

        # Compute transport maps.
        #assert torch.all(torch.isfinite(maps))
        left_maps = torch.index_select(maps, index=left_idx, dim=0)
        right_maps = torch.index_select(maps, index=right_idx, dim=0)
        tril_maps = -torch.bmm(torch.transpose(left_maps, dim0=-1, dim1=-2), right_maps)
        saved_tril_maps = tril_maps.detach().clone()
        diag_maps = torch.bmm(torch.transpose(maps, dim0=-1, dim1=-2), maps)
        diag_maps = scatter_add(diag_maps, row, dim=0, dim_size=self.size)

        # Normalise the transport maps.
        diag_maps, tril_maps = self.normalise(diag_maps, tril_maps, tril_row, tril_col)
        diag_maps, tril_maps = diag_maps.view(-1), tril_maps.view(-1)

        # Append fixed diagonal values in the non-learnable dimensions.
        (diag_indices, diag_maps), (tril_indices, tril_maps) = self.append_fixed_maps(
            len(left_maps), diag_indices, diag_maps, tril_indices, tril_maps)

        # Add the upper triangular part.
        triu_indices = torch.empty_like(tril_indices)
        triu_indices[0], triu_indices[1] = tril_indices[1], tril_indices[0]
        non_diag_indices, non_diag_values = mergesp(tril_indices, tril_maps, triu_indices, tril_maps)

        # Merge diagonal and non-diagonal
        edge_index, weights = mergesp(non_diag_indices, non_diag_values, diag_indices, diag_maps)

        return (edge_index, weights), saved_tril_maps

