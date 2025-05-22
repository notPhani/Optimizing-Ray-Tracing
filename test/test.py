import torch
import numpy as np
import matplotlib.pyplot as plt
import math
class vec3:
    def __init__(self, data, device='cuda'):
        if isinstance(data, vec3): 
            self.vect = data.vect.to(device)
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
    
    def to_array(self):
        return self.vect.detach().cpu().numpy()

#initializing vec3class Material:
    def __init__(self, color, roughness, specularity, em_color, em_strength, ir, reflectiveness):
        self.color = color
        self.roughness = roughness
        self.specularity = specularity
        self.em_color = em_color
        self.em_strength = em_strength  
        self.reflectiveness = reflectiveness
        self.ir = ir

class Ray:
    def __init__(self, origin, direction):
        self.origin = vec3(origin)
        self.direction = vec3(direction).normalize()  # Always normalize ray direction!

    def at(self, t):
        return self.origin + self.direction * t  # Scalar multiplication, not dot product

class Objects:
    class Sphere:
        def __init__(self, radius, center, material):
            self.radius = radius
            self.center = vec3(center)
            self.material = material

        def interact(self, ray: Ray):
            oc = ray.origin - self.center
            a = ray.direction.dot(ray.direction)  # should be 1 if normalized, but safe to keep
            b = 2.0 * oc.dot(ray.direction)
            c = oc.dot(oc) - self.radius * self.radius

            discriminant = b ** 2 - 4 * a * c  # Use power operator ** not ^

            if discriminant < 0:
                return None  # No intersection

            sqrt_disc = torch.sqrt(discriminant)

            # Numerically stable quadratic roots
            q = -0.5 * (b + torch.sign(b) * sqrt_disc)
            t1 = q / a
            t2 = c / q

            # Sort and choose smallest positive t
            t_near = min(t1, t2)
            t_far = max(t1, t2)

            if t_near > 1e-6:
                return t_near
            elif t_far > 1e-6:
                return t_far
            else:
                return None  # Intersection behind the ray origin
        
