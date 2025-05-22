import torch
import numpy as np
import matplotlib.pyplot as plt

class vec3:
    def __init__(self, data, device='cuda'):
        if isinstance(data, torch.Tensor):
            self.vect = data
        elif isinstance(data, np.ndarray):
            self.vect = torch.tensor(data, dtype=torch.float32,device=device)
        else:
            try:
                self.vect = torch.tensor(data, dtype=torch.float32,device=device)
            except Exception as e:
                raise TypeError(f"Expected numpy array, torch tensor, or list-like object. Got {type(data)}") from e

        if self.vect.shape[-1] != 3:
            raise ValueError("vec3 expects a 3D vector.")

        if not self.vect.is_floating_point():
            self.vect = self.vect.float()

    def __add__(self, other):
        return vec3(self.vect + other.vect)

    def __sub__(self, other):
        return vec3(self.vect - other.vect)

    def __mul__(self, other):  # Element-wise multiply
        return vec3(self.vect * other.vect)

    def dot(self, other):
        return torch.dot(self.vect, other.vect)

    def cross(self, other):
        return vec3(torch.cross(self.vect, other.vect))

    def magnitude(self):
        return torch.norm(self.vect)

    def normalize(self, eps=1e-8):
        norm = self.magnitude()
        return vec3(self.vect / (norm + eps))

    def __repr__(self):
        return f"vec3({self.vect})"

    def __eq__(self, other):
        return torch.allclose(self.vect, other.vect, atol=1e-6)

