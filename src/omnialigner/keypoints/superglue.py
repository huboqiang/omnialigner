from copy import deepcopy
from typing import List, Tuple

import torch
from torch import nn

from omnialigner.dtypes import Tensor_image_NCHW, KeypointDetectorMeta
from omnialigner.keypoints.xfeat import warp_corners_and_draw_matches
from omnialigner.plotting import keypoint_viz as viz

def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert(nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def remove_borders(keypoints, scores, border: int, height: int, width: int):
    """ Removes keypoints too close to the border """
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]


def top_k_keypoints(keypoints, scores, k: int):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices], scores


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """ Interpolate descriptors at keypoint locations """
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor([(w*s - s/2 - 0.5), (h*s - s/2 - 0.5)],
                              ).to(keypoints)[None]
    keypoints = keypoints*2 - 1  # normalize to (-1, 1)
    args = {'align_corners': True} if torch.__version__ >= '1.3' else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors


class SuperPoint(nn.Module):
    """SuperPoint Convolutional Detector and Descriptor
    SuperPoint: Self-Supervised Interest Point Detection and
    Description. Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. In CVPRW, 2019. https://arxiv.org/abs/1712.07629
    """
    default_config = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(
            c5, self.config['descriptor_dim'],
            kernel_size=1, stride=1, padding=0)

        # path = Path(__file__).parent / 'weights/superpoint_v1.pth'
        # self.load_state_dict(torch.load(str(path)))

        mk = self.config['max_keypoints']
        if mk == 0 or mk < -1:
            raise ValueError('\"max_keypoints\" must be positive or \"-1\"')

        # print('Loaded SuperPoint model')

    def forward(self, data):
        """ Compute keypoints, scores, descriptors for image """
        # Shared Encoder
        x = self.relu(self.conv1a(data['image']))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Compute the dense keypoint scores
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8)
        scores = simple_nms(scores, self.config['nms_radius'])

        # Extract keypoints
        keypoints = [
            torch.nonzero(s > self.config['keypoint_threshold'])
            for s in scores]
        scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

        # Discard keypoints near the image borders
        keypoints, scores = list(zip(*[
            remove_borders(k, s, self.config['remove_borders'], h*8, w*8)
            for k, s in zip(keypoints, scores)]))

        # Keep the k keypoints with highest score
        if self.config['max_keypoints'] >= 0:
            keypoints, scores = list(zip(*[
                top_k_keypoints(k, s, self.config['max_keypoints'])
                for k, s in zip(keypoints, scores)]))

        # Convert (h, w) to (x, y)
        keypoints = [torch.flip(k, [1]).float() for k in keypoints]

        # Compute the dense descriptors
        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)

        # Extract descriptors
        descriptors = [sample_descriptors(k[None], d[None], 8)[0]
                       for k, d in zip(keypoints, descriptors)]

        return {
            'keypoints': keypoints,
            'scores': scores,
            'descriptors': descriptors,
        }

def MLP(channels: List[int], do_bn: bool = True) -> nn.Module:
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def normalize_keypoints(kpts, image_shape):
    """ Normalize keypoints locations based on image image_shape"""
    _, _, height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one*width, one*height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]


class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, feature_dim: int, layers: List[int]) -> None:
        super().__init__()
        self.encoder = MLP([3] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        return self.encoder(torch.cat(inputs, dim=1))


def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: List[str]) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        for layer, name in zip(self.layers, self.names):
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1


def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


class SuperGlue(nn.Module):
    """SuperGlue feature matching middle-end
    Given two sets of keypoints and locations, we determine the
    correspondences by:
      1. Keypoint Encoding (normalization + visual feature and location fusion)
      2. Graph Neural Network with multiple self and cross-attention layers
      3. Final projection layer
      4. Optimal Transport Layer (a differentiable Hungarian matching algorithm)
      5. Thresholding matrix based on mutual exclusivity and a match_threshold
    The correspondence ids use -1 to indicate non-matching points.
    Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. SuperGlue: Learning Feature Matching with Graph Neural
    Networks. In CVPR, 2020. https://arxiv.org/abs/1911.11763
    """
    default_config = {
        'descriptor_dim': 256,
        'weights': 'indoor',
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.kenc = KeypointEncoder(
            self.config['descriptor_dim'], self.config['keypoint_encoder'])

        self.gnn = AttentionalGNN(
            feature_dim=self.config['descriptor_dim'], layer_names=self.config['GNN_layers'])

        self.final_proj = nn.Conv1d(
            self.config['descriptor_dim'], self.config['descriptor_dim'],
            kernel_size=1, bias=True)

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)
        
    def forward(self, data):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        desc0, desc1 = data['descriptors0'], data['descriptors1']  # (1, 256, N)
        kpts0, kpts1 = data['keypoints0'], data['keypoints1']      # (1, N, 2) and (1, M, 2)

        if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:
            shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
            return {
                'matches0': kpts0.new_full(shape0, -1, dtype=torch.int),
                'matches1': kpts1.new_full(shape1, -1, dtype=torch.int),
                'matching_scores0': kpts0.new_zeros(shape0),
                'matching_scores1': kpts1.new_zeros(shape1),
            }

        # Normalize keypoints
        kpts0_normalized = normalize_keypoints(kpts0, data['image0'].shape)
        kpts1_normalized = normalize_keypoints(kpts1, data['image1'].shape)
        
        # Keypoint encoding
        desc0 = desc0 + self.kenc(kpts0_normalized, data['scores0'])
        desc1 = desc1 + self.kenc(kpts1_normalized, data['scores1'])

        # GNN
        desc0, desc1 = self.gnn(desc0, desc1)

        # Final projection
        mdesc0, mdesc1 = desc0, self.final_proj(desc1)  # (1, 256, N) and (1, 256, M)
        # z1, z2 = mdesc0[0], mdesc1[0]
        # c = torch.mm(z1.T, z2)
        # c1 = torch.mm(z1.T, z1)
        # c2 = torch.mm(z2.T, z2)
        # N = z1.shape[1] + z2.shape[1]
        # c = c / N
        # c1 = c1 / N
        # c2 = c2 / N

        # loss_inv = - torch.diagonal(c).sum()
        # iden = torch.eye(c.size(0), device=desc0.device)
        # loss_dec1 = (iden - c1).pow(2).sum()
        # loss_dec2 = (iden - c2).pow(2).sum()
        # loss = loss_inv + 1e-3 * (loss_dec1 + loss_dec2)
        
        # # Compute matching scores via dot product
        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)  # (1, N, M)
        scores = scores / self.config['descriptor_dim']**0.5

        # Run the optimal transport.
        scores = log_optimal_transport(
            scores, self.bin_score,
            iters=self.config['sinkhorn_iterations'])
        
        # Convert log scores to probabilities and exclude dustbin
        probabilities = scores.exp()[:, :-1, :-1]  # shape: (1, N, M)
        
        # Compute expected positions of kpts1 for each kpt0
        expected_kpts1 = torch.einsum('bnm,bmd->bnd', probabilities, kpts1)  # (1, N, 2)
        
        # Prepare matrix A
        N = kpts0.size(1)
        ones = torch.ones((N, 1), device=kpts0.device)
        A = torch.cat([kpts0.squeeze(0), ones], dim=1)  # (N, 3)
        
        # B is expected_kpts1
        B = expected_kpts1.squeeze(0)  # (N, 2)
        
        # Compute AtA and AtB

        # Compute AtA and AtB
        AtA = A.t() @ A  # Shape: (3, 3)
        AtB = A.t() @ B  # Shape: (3, 2)

        # Solve for w in AtA * w = AtB
        w = torch.linalg.solve(AtA, AtB)  # w has shape (3, 2)
        
        # w is of shape (3, 2), we need to transpose it to get (2, 3)
        w = w.t()  # (2, 3)
        
        # Apply the affine transformation
        kpts0_augmented = torch.cat([kpts0, torch.ones_like(kpts0[:, :, :1])], dim=2)  # (1, N, 3)
        transformed_kpts0 = torch.bmm(kpts0_augmented, w.unsqueeze(0).transpose(1, 2))  # (1, N, 2)
        
        # Compute proximity loss between transformed_kpts0 and expected_kpts1
        loss_proximity = torch.nn.functional.mse_loss(transformed_kpts0, expected_kpts1)
        
        # Define positive pairs where transformed_kpts0 and kpts1 are close
        dist_matrix = torch.cdist(transformed_kpts0, kpts1, p=2)  # (1, N, M)
        
        # Use a threshold to define positive pairs
        distance_threshold = self.config.get('distance_threshold', 0.1)
        pos_pair_mask = dist_matrix[0] < distance_threshold  # (N, M)
        pos_indices = torch.nonzero(pos_pair_mask)  # (num_pos_pairs, 2)
        
        num_pos_pairs = pos_indices.size(0)
        
        if num_pos_pairs == 0:
            loss_contrastive = torch.tensor(0.0, device=kpts0.device)
        else:
            # Get descriptors of positive pairs
            pos_desc0 = mdesc0[0, :, pos_indices[:, 0]]  # (256, num_pos_pairs)
            pos_desc1 = mdesc1[0, :, pos_indices[:, 1]]  # (256, num_pos_pairs)
            
            # Compute positive similarities
            pos_similarities = torch.mean(pos_desc0 * pos_desc1, dim=0)  # (num_pos_pairs)
            
            # Negative pairs can be sampled from non-positive pairs
            # For simplicity, let's sample random negatives
            num_neg_samples = num_pos_pairs * self.config.get('neg_to_pos_ratio', 5)
            all_indices0 = torch.arange(kpts0.size(1), device=kpts0.device)
            all_indices1 = torch.arange(kpts1.size(1), device=kpts1.device)
            neg_indices0 = torch.randint(0, kpts0.size(1), (num_neg_samples,), device=kpts0.device)
            neg_indices1 = torch.randint(0, kpts1.size(1), (num_neg_samples,), device=kpts1.device)
            
            # Exclude positive pairs from negatives
            pos_pairs_set = set((i.item(), j.item()) for i, j in pos_indices)
            neg_pairs = [(i.item(), j.item()) for i, j in zip(neg_indices0, neg_indices1)
                        if (i.item(), j.item()) not in pos_pairs_set]
            if len(neg_pairs) == 0:
                loss_contrastive = torch.tensor(0.0, device=kpts0.device)
            else:
                neg_indices0 = torch.tensor([i for i, _ in neg_pairs], device=kpts0.device)
                neg_indices1 = torch.tensor([j for _, j in neg_pairs], device=kpts1.device)
                
                neg_desc0 = mdesc0[0, :, neg_indices0]  # (256, num_neg_samples)
                neg_desc1 = mdesc1[0, :, neg_indices1]  # (256, num_neg_samples)
                
                # Compute negative similarities
                neg_similarities = torch.mean(neg_desc0 * neg_desc1, dim=0)  # (num_neg_samples)
                
                # Compute Contrastive Loss
                margin = self.config.get('contrastive_margin', 1.0)
                pos_loss = (1 - pos_similarities).clamp(min=0).mean()
                neg_loss = (neg_similarities - margin).clamp(min=0).mean()
                loss_contrastive = pos_loss + neg_loss
        
        # Total loss
        loss = loss_proximity + loss_contrastive
        
        # Extract matches from OT probabilities
        max_prob0, indices0 = probabilities.max(dim=2)  # For each keypoint in image0, find the best match in image1
        max_prob1, indices1 = probabilities.max(dim=1)  # For each keypoint in image1, find the best match in image0

        # Mutual nearest neighbors
        mutual0 = torch.arange(indices0.size(1), device=kpts0.device) == indices1[0, indices0[0]]
        mutual1 = torch.arange(indices1.size(1), device=kpts1.device) == indices0[0, indices1[0]]

        # Apply mutual nearest neighbor constraint and a match threshold
        match_threshold = self.config.get('match_threshold', 0.2)
        valid0 = (mutual0 & (max_prob0[0] > match_threshold))
        valid1 = (mutual1 & valid0[indices1[0]])

        # Prepare match indices
        matches0 = torch.where(valid0, indices0[0], indices0.new_full(indices0[0].size(), -1))
        matches1 = torch.where(valid1, indices1[0], indices1.new_full(indices1[0].size(), -1))

        # Prepare matching scores
        matching_scores0 = torch.where(valid0, max_prob0[0], max_prob0.new_zeros(max_prob0[0].size()))
        matching_scores1 = torch.where(valid1, matching_scores0[matches1], max_prob1.new_zeros(max_prob1[0].size()))

        return {
            'matches0': matches0,  # Matches for keypoints in image0
            'matches1': matches1,  # Matches for keypoints in image1
            'matching_scores0': matching_scores0,
            'matching_scores1': matching_scores1,
            "w": w,
            'loss': loss
        }

    def forward_bak(self, data):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        kpts0, kpts1 = data['keypoints0'], data['keypoints1']

        if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:
            shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
            return {
                'matches0': kpts0.new_full(shape0, -1, dtype=torch.int),
                'matches1': kpts1.new_full(shape1, -1, dtype=torch.int),
                'matching_scores0': kpts0.new_zeros(shape0),
                'matching_scores1': kpts1.new_zeros(shape1),
            }

        kpts0 = normalize_keypoints(kpts0, data['image0'].shape)
        kpts1 = normalize_keypoints(kpts1, data['image1'].shape)

        desc0 = desc0 + self.kenc(kpts0, data['scores0'])
        desc1 = desc1 + self.kenc(kpts1, data['scores1'])

        desc0, desc1 = self.gnn(desc0, desc1)

        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        scores = scores / self.config['descriptor_dim']**.5

        # Run the optimal transport.
        scores = log_optimal_transport(
            scores, self.bin_score,
            iters=self.config['sinkhorn_iterations'])

        # Get the matches with score above "match_threshold".
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        valid0 = mutual0 & (mscores0 > self.config['match_threshold'])
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        return {
            'matches0': indices0,
            'matches1': indices1,
            'matching_scores0': mscores0,
            'matching_scores1': mscores1,
        }

class Matching(torch.nn.Module):
    """ Image Matching Frontend (SuperPoint + SuperGlue) """
    def __init__(self, config={}):
        super().__init__()
        self.superpoint = SuperPoint(config.get('superpoint', {}))
        self.superglue = SuperGlue(config.get('superglue', {}))

    def forward(self, data):
        """ Run SuperPoint (optionally) and SuperGlue
        SuperPoint is skipped if ['keypoints0', 'keypoints1'] exist in input
        Args:
          data: dictionary with minimal keys: ['image0', 'image1']
        """
        pred = {}

        if 'keypoints0' not in data:
            pred0 = self.superpoint({'image': data['image0']})
            pred = {**pred, **{k+'0': v for k, v in pred0.items()}}
        if 'keypoints1' not in data:
            pred1 = self.superpoint({'image': data['image1']})
            pred = {**pred, **{k+'1': v for k, v in pred1.items()}}

        # Batch all features
        # We should either have i) one image per batch, or
        # ii) the same number of local features for all images in the batch.
        data = {**data, **pred}

        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])

        # Perform the matching
        pred = {**pred, **self.superglue(data)}

        return pred


class SuperGlueDetector(nn.Module, KeypointDetectorMeta):
    def __init__(self, weight=None):
        super().__init__()
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if weight is None:
            weight = {}
        
        if "superpoint_weights_path" not in weight:
            weight["superpoint_weights_path"] = "/cluster/home/bqhu_jh/projects/SG_0623/vendor/SuperGlue-pytorch/models_superglue/weights/superpoint_v1.pth"

        if "superglue_weights_path" not in weight:
            weight["superglue_weights_path"] = "/cluster/home/bqhu_jh/projects/SG_0623/vendor/SuperGlue-pytorch/models_superglue/weights/superglue_outdoor.pth"

        config = {
            'superpoint': {
                'nms_radius': 4,
                'keypoint_threshold': 0.005,
                'max_keypoints': 1024
            },
            'superglue': {
                'weights': "outdoor",
                'sinkhorn_iterations': 20,
                'match_threshold': 0.2,
            }
        }
        self.matching = Matching(config).eval().to(self.dev)
        self.matching.superpoint.load_state_dict(torch.load(weight["superpoint_weights_path"]))
        self.matching.superglue.load_state_dict(torch.load(weight["superglue_weights_path"]))

    def forward(self, image_F, image_M, method="cv2DMatch", top_k=4096):

        data = {
            'image0': viz.image_viz.im2tensor(image_M)[:, 0:1, :, :].to(self.dev),
            'image1': viz.image_viz.im2tensor(image_F)[:, 0:1, :, :].to(self.dev)
        }

        pred = self.matching(data)
        pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches = pred['matches0']

        valid = matches > -1
        mkpts0 = kpts0[pred["matching_scores0"] > 0, :]
        mkpts1 = kpts1[pred["matching_scores1"] > 0, :]
        if method == "cv2DMatch":
            canvas, matches = warp_corners_and_draw_matches(mkpts1, mkpts0, image_F, image_M)

            l_idxs = [ m.queryIdx for m in matches ]

            return mkpts1, mkpts0, l_idxs, canvas
