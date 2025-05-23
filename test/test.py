import torch

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

        @staticmethod
        def intersect(self,ray_batch):
            """
            Batched intersection for ALL spheres vs ALL rays
            Returns: HitInfo with (t, material_idx, hit_normal)
            """
            # ray_batch.origins: (B, 3)
            # ray_batch.directions: (B, 3)
            B = ray_batch.origins.shape[0]
            N = self.centers.shape[0]

            # Vectorized intersection math 🔥
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

    def generate_rays(self, width, height):
        # Pixel grid in normalized screen space [-1, 1]
        x = torch.linspace(-1, 1, width, device='cuda')
        y = torch.linspace(-1/self.aspect, 1/self.aspect, height, device='cuda')
        grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')  # (W, H)

        # Compute ray directions based on FOV
        focal = 1.0 / torch.tan(torch.deg2rad(self.fov / 2))
        dirs = torch.stack([
            grid_x,
            grid_y,
            -torch.ones_like(grid_x) * focal  # Negative Z for forward direction
        ], dim=-1)  # (W, H, 3)

        # Rotate directions to point towards look_at
        forward = (self.look_at - self.origin).normalize()
        # ... (compute right/up vectors and rotation matrix here) ...

        # Flatten to (W*H, 3) for RayBatch
        dirs = dirs.reshape(-1, 3)
        return RayBatch(origins=self.origin.expand(dirs.shape), directions=dirs)

        
