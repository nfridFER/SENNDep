import torch
import torch.nn as nn


class ConceptEncoder(nn.Module):
    def __init__(self, input_dim, concept_dim=50):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, concept_dim),
        )

    def forward(self, x):
        return self.net(x)


class RelevanceParametrizer(nn.Module):
    def __init__(self, concept_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(concept_dim, 64),
            nn.ReLU(),
            nn.Linear(64, concept_dim),
        )

    def forward(self, x):
        return self.net(x)


class SENN(nn.Module):
    """
    Self-Explaining Neural Network
    Returns logit + (theta, concepts) when require_concepts=True.
    """

    def __init__(self, input_dim, concept_dim=50):
        super().__init__()
        self.encoder = ConceptEncoder(input_dim, concept_dim)
        self.parametrizer = RelevanceParametrizer(concept_dim)

    def aggregator(self, theta, concepts):
        return torch.sum(theta * concepts, dim=1, keepdim=True)

    def forward(self, x, require_concepts=True):
        z = self.encoder(x)
        th = self.parametrizer(z)
        logit = self.aggregator(th, z)
        return (logit, th, z) if require_concepts else logit



def gradient_alignment_loss(logits, theta, concepts):
    grads = torch.autograd.grad(outputs=logits.sum(), inputs=concepts, create_graph=True)[0]

    return nn.functional.mse_loss(grads, theta)
