# Author: David Harwath, Wei-Ning Hsu
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def compute_perplexity(onehots):
    avg_probs = torch.mean(onehots, dim=0)
    perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
    return perplexity

class TemporalJitter(nn.Module):
    def __init__(self, p_left=0.12, p_right=0.12):
        super(TemporalJitter, self).__init__()
        assert(p_left >= 0 and p_right >= 0)
        assert(p_left + p_right <= 1)
        self.p_left = p_left
        self.p_right = p_right
        self.p_middle = 1.0 - p_left - p_right
        self.sampler = Categorical(torch.tensor([p_left, self.p_middle, p_right])) 

    def forward(self, x):
        if self.training:
            jitters = self.sampler.sample(sample_shape=(x.size(-1),))
            Tinds = torch.arange(x.size(-1)) + jitters
            Tinds = Tinds.to(x.device).clamp(0, x.size(-1)-1)
            return torch.index_select(x, -1, Tinds)
        else:
            return x

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, 
                 epsilon=1e-8, init_std=1, nonneg_init=False, init_ema_mass=1):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(num_embeddings, embedding_dim)
        self._embedding.weight.requires_grad = False
        self._embedding.weight.data.normal_(std=init_std)
        if nonneg_init:
            self._embedding.weight.data.clamp_(min=0)
        self._commitment_cost = commitment_cost

        self.register_buffer(
                '_ema_cluster_size', init_ema_mass * torch.ones(num_embeddings))
        self._ema_w = nn.Parameter(
                init_ema_mass * torch.from_numpy(self._embedding.weight.cpu().numpy()),
                requires_grad=False)
        if nonneg_init:
            self._ema_w.data.clamp_(min=0)
        
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        """
        Args:
            inputs (torch.Tensor): input of shape (B, D)
        Returns:
            loss (torch.Tensor): (B, D)
            quantized (torch.Tensor): (B, D)
            onehots (torch.Tensor): (B, num_embeddings)
        """
        # Calculate distances
        distances = (torch.sum(inputs**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(inputs, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        onehots = torch.zeros(encoding_indices.shape[0],
                              self._num_embeddings)
        onehots = onehots.to(inputs.device)
        onehots.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(onehots, self._embedding.weight)
        
        # Loss
        e_latent_loss = (quantized.detach() - inputs)**2
        loss = self._commitment_cost * e_latent_loss
        
        # Straight-through estimator for gradient
        if self.training:
            quantized = inputs + (quantized - inputs).detach()
        
        return loss, quantized, onehots

    def ema_update(self, inputs, onehots):
        """
        Args:
            inputs (torch.Tensor): input of shape (B, D)
            onehots (torch.Tensor): one-hot encodings of input of shape (B, K)
        """
        assert(self.training)

        # Use EMA to update the embedding vectors
        self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                 (1 - self._decay) * torch.sum(onehots, 0)

        # Redistribute cluster size by interpolating with a uniform assignment
        n = torch.sum(self._ema_cluster_size.data)
        self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
        
        dw = torch.matmul(onehots.t(), inputs)
        self._ema_w = nn.Parameter(
                self._ema_w * self._decay + (1 - self._decay) * dw, 
                requires_grad=False)
        
        self._embedding.weight = nn.Parameter(
                self._ema_w / self._ema_cluster_size.unsqueeze(1), 
                requires_grad=False)

    def get_embedding(self):
        return self._embedding.weight.detach()
