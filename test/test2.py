# Temporary CPU version for visualization
import numpy as np
import matplotlib.pyplot as
class HaltonSamplerCPU:
    def __init__(self, bases=(2, 3), permute=True, skip=20):
        self.bases = bases
        self.permutations = {
            2: [0, 1],      # Base 2 permutation
            3: [0, 2, 1],   # Base 3 permutation
        } if permute else None
        self.skip = skip

    def generate(self, num_samples):
        samples = []
        for i in range(self.skip, num_samples + self.skip):
            sample = []
            for base in self.bases:
                f = 1.0
                n = i
                val = 0.0
                while n > 0:
                    f /= base
                    rem = n % base
                    if self.permutations: rem = self.permutations[base][rem]
                    val += rem * f
                    n = n // base
                sample.append(val - 0.5)  # Center around [-0.5, 0.5]
            samples.append(sample)
        return np.array(samples) * 0.8  # Scale to [-0.4, 0.4]

# Generate and plot samples
sampler = HaltonSamplerCPU(permute=True, skip=20)
samples = sampler.generate(256)

plt.figure(figsize=(8,8))
plt.scatter(samples[:,0], samples[:,1], s=15, alpha=0.7)
plt.title('Permuted Halton Jitter Pattern\n(Bases 2 & 3, 256 samples)')
plt.xlim(-0.5, 0.5)
plt.ylim(-0.5, 0.5)
plt.grid(True, alpha=0.3)
plt.gca().set_aspect('equal')
plt.show()
