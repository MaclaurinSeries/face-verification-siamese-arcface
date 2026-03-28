import torch
import torch.nn.functional as F
import torch.nn as nn
import math


class ArcFaceLoss(nn.Module):
    def __init__(self, embed_dim, num_classes, s=64.0, m=0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.empty(num_classes, embed_dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, embeddings, labels):
        # embeddings must be L2 normalized
        # weight must be L2 normalized
        cosine = F.linear(embeddings, F.normalize(self.weight))
        cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)

        theta = torch.acos(cosine)
        target_mask = F.one_hot(labels, cosine.size(1)).bool()

        # add margin only to the target class angle
        theta[target_mask] += self.m
        logits = torch.cos(theta) * self.s

        return F.cross_entropy(logits, labels)
