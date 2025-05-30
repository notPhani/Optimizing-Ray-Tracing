import torch
import time
import matplotlib.pyplot as plt
import numpy as np

def normalize(v):
    """Normalize a tensor of vectors along the last dimension"""
    return v / (torch.norm(v, dim=-1, keepdim=True) + 1e-8)

class RayBatch:
    def __init__(self, origins, directions, device="cuda"):
        self.origins = torch.as_tensor(origins, device=device, dtype=torch.float32)
        self.directions = torch.as_tensor(directions, device=device, dtype=torch.float32)
        self.directions = self.directions / (torch.norm(self.directions, dim=-1, keepdim=True) + 1e-8)
        if self.origins.shape[-1] != 3 or self.directions.shape[-1] != 3:
            raise Exception("Ray origins/directions must be 3D vectors")
            
    def at(self, t):
        return self.origins + self.directions * t.unsqueeze(-1)
    
    def __repr__(self):
        return f"RayBatch(shape={self.origins.shape})"
    
    def trace(self, scene, num_bounces, max_bounces):
        hit_info = scene.interact(self)
        color = torch.zeros_like(self.origins)
        miss_mask = (hit_info.t == float('inf'))
        #color[miss_mask] = scene.background_color
        hit_mask = ~miss_mask
        return miss_mask
    
    @property
    def shape(self):
        return self.origins.shape
    
    def to(self, device):
        return RayBatch(self.origins.to(device), self.directions.to(device))
    
    
    
class Material:
    def __init__(self, color, roughness, metallic, specularity, em_strength, em_color, ir):
        self.color = torch.tensor(color, dtype=torch.float32)
        self.roughness = float(roughness)
        self.metallic = float(metallic)
        self.specularity = float(specularity)
        self.em_strength = float(em_strength)
        self.em_color = torch.tensor(em_color, dtype=torch.float32)
        self.ir = float(ir) 

    def __repr__(self):
        return (
            f"Material(color={self.color.tolist()},\n roughness={self.roughness},\n "
            f"metallic={self.metallic},\n specularity={self.specularity},\n "
            f"em_strength={self.em_strength},\n em_color={self.em_color.tolist()},\n "
            f"ir={self.ir})"
        )
    
class HitInfo:
    def __init__(self, t, material_idx, normal):
        self.t = t            # (B,) hit distances
        self.material_idx = material_idx  # (B,) material indices
        self.normal = normal  # (B, 3) surface normals

class Object:
    class Sphere:
        def __init__(self, centers, radii, materials):
            """
            Batch-friendly sphere storage
            centers: (N, 3) tensor
            radii: (N,) tensor
            materials: list of Material objects (length N)
            """
            self.centers = centers.to('cuda')  # (N, 3)
            self.radii = radii.to('cuda')      # (N,)
            self.materials = materials

        def intersect(self,ray_batch):
            """
            Batched intersection for ALL spheres vs ALL rays
            Returns: HitInfo with (t, material_idx, hit_normal)
            """
            # ray_batch.origins: (B, 3)
            # ray_batch.directions: (B, 3)
            B = ray_batch.origins.shape[0]
            N = self.centers.shape[0]
            oc = ray_batch.origins[:, None] - self.centers[None]  # (B, N, 3)
            a = torch.einsum('bi,bi->b', ray_batch.directions, ray_batch.directions)  # (B,)
            b = 2 * torch.einsum('bni,bi->bn', oc, ray_batch.directions)  # (B, N)
            c = torch.einsum('bni,bni->bn', oc, oc) - (self.radii**2)[None]  # (B, N)

            discriminant = b**2 - 4 * a[:, None] * c
            hit_mask = discriminant > 1e-6  # (B, N)
            sqrt_disc = torch.sqrt(discriminant[hit_mask])
            q = -0.5 * (b[hit_mask] + torch.sign(b[hit_mask]) * sqrt_disc)
            a_expanded = a.unsqueeze(1).expand(-1, N)  # (B,) → (B, N)
            t1 = q / a_expanded[hit_mask]  # Now works with (B, N) mask

            t2 = c[hit_mask] / q
            
            # Find smallest positive t
            t_vals = torch.where((t1 > 1e-6) & (t1 < t2), t1, t2)
            t_vals = torch.where(t_vals < 1e-6, float('inf'), t_vals)

            # Build hit info
            hit_t = torch.full((B, N), float('inf'), device='cuda')
            hit_t[hit_mask] = t_vals
            
            # Find closest hit per ray
            closest_t, closest_idx = torch.min(hit_t, dim=1)  # (B,)
            
            # Calculate hit normals
            hit_points = ray_batch.at(closest_t)  # (B, 3)
            hit_normals = (hit_points - self.centers[closest_idx]) / self.radii[closest_idx][:, None]
            
            return HitInfo(
                t=closest_t,
                material_idx=closest_idx,
                normal=hit_normals
            )

class Camera:
    def __init__(self, origin, look_at, fov, aspect):
        self.origin = torch.tensor(origin, dtype=torch.float32, device='cuda')
        self.look_at = torch.tensor(look_at, dtype=torch.float32, device='cuda')
        self.fov = torch.tensor(fov, dtype=torch.float32, device='cuda')
        self.aspect = torch.tensor(aspect, dtype=torch.float32, device='cuda')

    def CreateRayBffr(self, width, height, mode):
        # Pixel grid with Y going from 1 (top) to -1 (bottom)
        x = torch.linspace(-1, 1, width, device='cuda') * self.aspect
        y = torch.linspace(1, -1, height, device='cuda')  # FIXED: Top to bottom
        grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')  # (W, H)

        # Camera basis vectors (ensure looking along -Z)
        forward = self.look_at - self.origin
        forward = forward / (torch.norm(forward) + 1e-8)
        up = torch.tensor([0, 1, 0], dtype=torch.float32, device='cuda')
        right = torch.linalg.cross(forward, up)
        right = right / (torch.norm(right) + 1e-8)
        up = torch.linalg.cross(right, forward)
        up = up / (torch.norm(up) + 1e-8)

        if mode == "orth":
            origins = self.origin + grid_x[..., None]*right*width/2 + grid_y[..., None]*up*height/2
            directions = forward.expand_as(origins)
        elif mode == "persp":
            f = 1 / torch.deg2rad(self.fov/2).tan()
            directions = grid_x[..., None]*right + grid_y[..., None]*up + forward*f  # FIXED: +forward
            directions = directions / directions.norm(dim=-1, keepdim=True)
            origins = self.origin.expand(directions.shape)
        
        return RayBatch(origins.reshape(-1, 3), directions.reshape(-1, 3))

class Illumination:
    class Light:
        def __init__(self, center, radius, color, strength):
            self.center = torch.tensor(center, dtype=torch.float32, device='cuda')
            self.radius = torch.tensor(radius, dtype=torch.float32, device='cuda')
            self.color = torch.tensor(color, dtype=torch.float32, device='cuda')
            self.strength = torch.tensor(strength, dtype=torch.float32, device='cuda')
            
            # Create emissive material
            self.material = Material(
                color=[0.0, 0.0, 0.0],        # Black base (pure emissive)
                roughness=0.0,
                metallic=0.0,
                specularity=0.0,
                em_strength=self.strength.item(),
                em_color=self.color.tolist(),
                ir=1.0
            )
            
            # Create sphere representation (batch-compatible)
            self.light = Object.Sphere(
                centers=self.center.unsqueeze(0),  # (1, 3)
                radii=self.radius.unsqueeze(0),    # (1,)
                materials=[self.material]
            )

    class dirLight:
        def __init__(self, position, direction, color, strength):
            self.position = torch.tensor(position, dtype=torch.float32, device='cuda')
            self.direction = torch.tensor(direction, dtype=torch.float32, device='cuda')
            self.direction = self.direction / (torch.norm(self.direction) + 1e-8)  # Normalize
            self.color = torch.tensor(color, dtype=torch.float32, device='cuda')
            self.strength = torch.tensor(strength, dtype=torch.float32, device='cuda')

class Scene:
    def __init__(self, objects, lights, background_color = torch.tensor([0,0,0])):
        self.background_color = background_color
        self.objects = objects
        self.lights = lights
    def interact(self, rayBatch):
        return self.objects.intersect(rayBatch)
    

# Set up a simple scene with a sphere at the origin
sphere_material = Material(
    color=[1, 0, 0], roughness=0.1, metallic=0.0, specularity=0.0,
    em_strength=0.0, em_color=[0, 0, 0], ir=1.0
)
sphere = Object.Sphere(
    centers=torch.tensor([[0.0, 0.0, 0.0]], device='cuda'),
    radii=torch.tensor([1.0], device='cuda'),
    materials=[sphere_material]
)
scene = Scene(objects=sphere, lights=[], background_color=torch.tensor([0, 0, 0], device='cuda'))

# Set your desired image size
width, height = 1080, 1080


camera = Camera([0,0,5], [0,0,0], 90, 1)

ray_batch = camera.CreateRayBffr(width, height, mode="persp")  

# Now you can use ray_batch directly for intersection, shading, etc.
start =time.time()
hit_mask = ray_batch.trace(scene, num_bounces=1, max_bounces=1)
print(time.time()-start)
hit_mask_img = hit_mask.reshape(height, width).cpu().numpy()
plt.imshow(hit_mask_img, cmap='gray')
plt.title('Ray Hit Mask (White = Hit, Black = Miss)')
plt.show()
