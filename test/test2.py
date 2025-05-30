import torch
import pygame
import numpy as np
from collections import defaultdict

class RayTracer:
    def __init__(self, width=800, height=600, fov=90, device='cuda'):
        self.width = width
        self.height = height
        self.device = device
        self.fov = np.radians(fov)
        self.scene = []
        self.ml_data = defaultdict(list)
        
        # Camera setup
        self.cam_pos = torch.tensor([0, 0, -5], dtype=torch.float32, device=device)
        self.look_at = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
        
        # PyGame initialization
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        
        # Generate initial grid
        self.create_coordinate_grid()
        
    class Sphere:
        def __init__(self, center, radius, material):
            self.center = torch.tensor(center, dtype=torch.float32)
            self.radius = radius
            self.material = material

    def create_coordinate_grid(self):
        # Create grid lines
        for i in range(-10, 11):
            self.scene.append(self.Sphere([i, 0, 0], 0.05, 'grid'))
            self.scene.append(self.Sphere([0, i, 0], 0.05, 'grid'))
            
    def trace_rays(self):
        # Generate ray directions
        aspect = self.width / self.height
        x = torch.linspace(-1, 1, self.width, device=self.device) * aspect
        y = torch.linspace(1, -1, self.height, device=self.device)
        xx, yy = torch.meshgrid(x, y, indexing='xy')
        
        forward = (self.look_at - self.cam_pos).normalize()
        right = torch.cross(forward, torch.tensor([0, 1, 0], device=self.device)).normalize()
        up = torch.cross(right, forward).normalize()
        
        dirs = xx[..., None] * right + yy[..., None] * up + forward
        dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
        
        # Ray tracing logic
        color_buffer = torch.zeros((self.height, self.width, 3), device=self.device)
        
        for obj in self.scene:
            if isinstance(obj, self.Sphere):
                # Vectorized sphere intersection
                oc = self.cam_pos - obj.center.to(self.device)
                a = torch.sum(dirs * dirs, dim=-1)
                b = 2 * torch.sum(oc * dirs, dim=-1)
                c = torch.sum(oc * oc, dim=-1) - obj.radius**2
                disc = b**2 - 4 * a * c
                
                hit_mask = disc > 0
                hit_pos = self.cam_pos + dirs * (-b - torch.sqrt(disc)) / (2 * a)
                normal = (hit_pos - obj.center.to(self.device)) / obj.radius
                
                # Store ML data
                if 'sphere' in obj.material:
                    self.ml_data['ray_origins'].append(self.cam_pos.cpu())
                    self.ml_data['ray_directions'].append(dirs.cpu())
                    self.ml_data['hit_positions'].append(hit_pos.cpu())
                    self.ml_data['normals'].append(normal.cpu())
                
                # Simple shading
                color = torch.tensor([1, 0, 0] if 'red' in obj.material else [0.5, 0.5, 0.5], 
                                   device=self.device)
                color_buffer[hit_mask] = color
        
        return color_buffer.cpu().numpy()

    def run(self):
        selected_obj = None
        while True:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                
                # Object selection
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = pygame.mouse.get_pos()
                    ray_dir = self.generate_mouse_ray(mx, my)
                    
                    # Find intersected object
                    closest = None
                    for obj in self.scene:
                        if isinstance(obj, self.Sphere):
                            t = self.intersect_sphere(self.cam_pos, ray_dir, obj)
                            if t and (closest is None or t < closest[1]):
                                closest = (obj, t)
                    
                    selected_obj = closest[0] if closest else None
                
                # Object creation
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                    mx, my = pygame.mouse.get_pos()
                    ray_dir = self.generate_mouse_ray(mx, my)
                    t = 10  # Place new sphere 10 units away
                    pos = (self.cam_pos + ray_dir * t).cpu().numpy()
                    self.scene.append(self.Sphere(pos, 0.5, 'red'))
                    

