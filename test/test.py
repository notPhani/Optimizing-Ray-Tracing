import torch
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def normalize(v):
    return v / (torch.norm(v, dim=-1, keepdim=True) + 1e-8)

class CMJSampler:
    def __init__(self):
        pass

    def _permute(self, x, n, seed):
        # Kensler's reversible hash-based permutation
        x = torch.as_tensor(x, device='cuda', dtype=torch.int64)
        n = int(n)
        mask = n - 1
        if n & mask == 0:  # power of two
            x = (x ^ seed) * 0xe170893d
            x = (x ^ (x >> 16)) ^ (seed * 0x0929eb3f)
            x = x & mask
        else:
            x = (x ^ seed) * 0xe170893d
            x = (x ^ (x >> 16)) ^ (seed * 0x0929eb3f)
            x = x % n
        return x

    def _randfloat(self, x, seed):
        # Kensler's hash-to-float
        x = torch.as_tensor(x, device='cuda', dtype=torch.int64)
        x = x ^ seed
        x = x ^ (x >> 17)
        x = x ^ (x >> 10)
        x = x * 0xb36534e5
        x = x ^ (x >> 12)
        x = x ^ (x >> 21)
        x = x * 0x93fc4795
        x = x ^ 0xdf6e307f
        x = x ^ (x >> 17)
        x = x * (1 | seed >> 18)
        return (x.float() % 4294967808.0) / 4294967808.0

    def generate(self, num_samples, sampler_number=0):
        # CMJ grid size
        m = int(torch.sqrt(torch.tensor(num_samples, dtype=torch.float32)).item())
        n = (num_samples + m - 1) // m  # ceil division

        # Sample indices
        s = torch.arange(num_samples, device='cuda')

        # Pattern seed
        p = int(sampler_number) + 1  # never zero

        # Permutations
        sx = self._permute(s % m, m, p * 0xa511e9b3)
        sy = self._permute(s // m, n, p * 0x63d83595)

        # Jitter
        jx = self._randfloat(s, p * 0xa399d265)
        jy = self._randfloat(s, p * 0x711ad6a5)

        # CMJ sample positions in [0,1)
        x = (s % m + (sy + jx) / n) / m
        y = (s // m + (sx + jy) / m) / n

        # Center and scale to [-0.5, 0.5], then to [-0.4, 0.4] (optional)
        pts = torch.stack([x, y], dim=-1) - 0.5
        pts = pts * 0.8
        return pts


class HaltonSampler:
    def __init__(self, bases=(2, 3), permute=True, skip=20):
        self.bases = bases
        self.permutations = {
            2: torch.tensor([0, 1], device='cuda'),
            3: torch.tensor([0, 2, 1], device='cuda'),
        } if permute else None
        self.skip = skip

    def generate(self, num_samples, sample_number):
        samples = []
        for i in range(self.skip + sample_number, num_samples + self.skip + sample_number):
            sample = []
            for base in self.bases:
                f = 1.0
                n = i
                val = 0.0
                while n > 0:
                    f /= base
                    rem = n % base
                    if self.permutations:
                        rem = self.permutations[base][rem]
                    val += rem * f
                    n = n // base
                sample.append(val - 0.5)
            samples.append(sample)
        pts = torch.tensor(samples, device='cuda') * 0.8  # Scale to [-0.4, 0.4]

        # --- Add reproducible random jitter based on sample_number ---
        rng = torch.Generator(device='cuda')
        rng.manual_seed(sample_number)
        # Critical jitter radius for Halton: 0.5 * delta0 / sqrt(N), delta0 ~ 0.45 [5]
        jitter_radius = 0.5 * 0.45 / (num_samples ** 0.5)
        jitter = (torch.rand(pts.shape, device=pts.device, generator=rng) - 0.5) * 2 * jitter_radius
        pts = pts + jitter
        return pts
    
class Lambertian:
    @staticmethod
    def offset(hit_normals, material_roughness, incoming_dirs):
        reflected_dirs = incoming_dirs - 2*(incoming_dirs*hit_normals).sum(-1, keepdim=True)*hit_normals
        # Create diffusion rays
        N = hit_normals.shape[0]
        device = hit_normals.device

        u1 = torch.rand(N, device=device)
        u2 = torch.rand(N, device=device)
        r = torch.sqrt(u1)
        phi = 2 * torch.pi * u2
        x = r * torch.cos(phi)
        y = r * torch.sin(phi)
        z = torch.sqrt(1 - x**2 - y**2)
        local_dirs = torch.stack([x, y, z], dim=-1)

        up = torch.tensor([0.0, 1.0, 0.0], device=device).expand_as(hit_normals)
        alt_up = torch.tensor([1.0, 0.0, 0.0], device=device).expand_as(hit_normals)
        mask = (torch.abs(hit_normals @ torch.tensor([0.0, 1.0, 0.0], device=device)) > 0.999)
        ref = torch.where(mask.unsqueeze(-1), alt_up, up)  # (N, 3)
        t = torch.cross(hit_normals, ref, dim=-1)
        t = t / (torch.norm(t, dim=-1, keepdim=True) + 1e-8)
        b = torch.cross(hit_normals, t, dim=-1)


        world_dirs = (local_dirs[..., 0:1] * t +
                      local_dirs[..., 1:2] * b +
                      local_dirs[..., 2:3] * hit_normals)
        diffusion_dirs = world_dirs / (torch.norm(world_dirs, dim=-1, keepdim=True) + 1e-8)

        blended = (1-material_roughness).unsqueeze(-1) * reflected_dirs + (material_roughness).unsqueeze(-1) * diffusion_dirs
        return blended / (torch.norm(blended, dim=-1, keepdim=True) + 1e-8)

        
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
        color = torch.zeros_like(self.origins, dtype=torch.float32, device='cuda')
        miss_mask = (hit_info.t == float('inf'))
        hit_mask = ~miss_mask

        # Set missed rays to background color
        color[miss_mask] = scene.background_color.to(color.dtype)

        if hit_mask.any():
            # Gather hit info for rays that hit something
            hit_points = self.origins[hit_mask] + self.directions[hit_mask] * hit_info.t[hit_mask].unsqueeze(-1)
            hit_normals = hit_info.normal[hit_mask]
            hit_materials_idx = hit_info.material_idx[hit_mask]

            # Gather material properties
            all_roughness = torch.tensor([m.roughness for m in scene.objects.materials], device='cuda')
            all_colors = torch.stack([m.color for m in scene.objects.materials], dim=0).to('cuda')
            all_em_strength = torch.tensor([m.em_strength for m in scene.objects.materials], device='cuda')
            all_em_color = torch.stack([m.em_color for m in scene.objects.materials], dim=0).to('cuda')

            hit_materials_roughness = all_roughness[hit_materials_idx]
            hit_materials_color = all_colors[hit_materials_idx]
            hit_materials_em_strength = all_em_strength[hit_materials_idx]
            hit_materials_em_color = all_em_color[hit_materials_idx]
            color[hit_mask] += hit_materials_em_strength.unsqueeze(-1) * hit_materials_em_color

            # If we haven't reached max bounces, spawn new rays for indirect lighting
            if num_bounces <= max_bounces:
                new_origins = hit_points + 1e-4 * hit_normals
                incoming_dirs = self.directions[hit_mask]
                new_dirs = Lambertian.offset(
                    hit_normals=hit_normals,
                    material_roughness=hit_materials_roughness,
                    incoming_dirs=incoming_dirs
                )
                new_rays = RayBatch(new_origins, new_dirs, device='cuda')
                # Recursive call for next bounce
                bounce_color = new_rays.trace(scene, num_bounces + 1, max_bounces)
                # Lambertian: multiply by albedo (material color)
                color[hit_mask] += hit_materials_color * bounce_color

        return color
    
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
            self.centers = centers.to('cuda')  # (N, 3)
            self.radii = radii.to('cuda')      # (N,)
            self.materials = materials

        def intersect(self,ray_batch):
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

    def CreateRayBffr(self, width, height, mode, sampler, sample, AA=True):
        samples = sampler.generate(width*height, sample)
        jitter_grid = samples.reshape(height,width,2)
        # Pixel grid with Y going from 1 (top) to -1 (bottom)
        x = torch.linspace(-1, 1, width, device='cuda') * self.aspect
        y = torch.linspace(1, -1, height, device='cuda')  # FIXED: Top to bottom
        grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')  # (W, H)
        if AA:
            grid_x = grid_x + jitter_grid[..., 0]
            grid_y = grid_y + jitter_grid[..., 1]

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
        
        return grid_x.cpu(),grid_y.cpu(),RayBatch(origins.reshape(-1, 3), directions.reshape(-1, 3))

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
    def __init__(self, objects, lights, background_color = torch.tensor([0.0,0.0,0.0],device='cuda')):
        self.background_color = background_color
        self.objects = objects
        self.lights = lights
    def interact(self, rayBatch):
        return self.objects.intersect(rayBatch)


