from turbencrypt.make_dataset import Dataset
import numpy as np
from torchvision.datasets import MNIST


class SyntheticDataset:
    def __init__(self, num_samples: int, freq_range=(-20, 20), amplitude_range=(1, 4), single_coord_prob: float = 0.5):
        self.freq_range = freq_range
        self.amplitude_range = amplitude_range
        self.num_samples = num_samples
        self.single_coord_prob = single_coord_prob

    def generate_data(self, grid_size=256, noise_std=0.0):
        coords = np.meshgrid(np.linspace(-1, 1, grid_size), np.linspace(-1, 1, grid_size))
        coords = np.stack(coords, axis=-1)  # (grid_size, grid_size, 2)
        freqs = np.random.uniform(self.freq_range[0], self.freq_range[1], size=(2, self.num_samples))
        amp = np.random.uniform(self.amplitude_range[0], self.amplitude_range[1], size=(self.num_samples, 1, 1))

        # randomly make some field vertical/horizontal
        if np.random.rand() < self.single_coord_prob:
            dim = np.random.choice(2)
            freqs[dim] = 0.
            
        
        field = amp * np.sin(coords @ freqs).transpose(2, 0, 1)  # (N, grid_size, grid_size)
        return field + noise_std * np.random.normal(size=field.shape)


def main():
    sim_config = {
        'viscosity': 1e-2,
        'max_velocity': 7.0,
        'final_time': 25,
        'outer_steps': 10,
        'gridsize': 64,
        'max_courant_num': 0.1
    }

    save_path = f"data/mnist_dataset2.npz"
    dataset = Dataset()
    # dataset.run_sim_from_images(
    #     image_dir="raw_images",
    #     save_path=save_path,
    #     config=sim_config
    # )
    # synthetic_dataset = SyntheticDataset(num_samples=100)
    # forcing_fns = synthetic_dataset.generate_data(grid_size=256, noise_std=0.0)
    # dataset.run_sim_from_tensors(
    #     tensors=forcing_fns,
    #     save_path=save_path,
    #     config=sim_config,
    #     length=synthetic_dataset.num_samples
    # )
    num_samples = 100
    mnist_data = MNIST("data/", train=True).data
    samples = np.random.choice(len(mnist_data), num_samples, replace=False)
    mnist_samples = mnist_data[samples].numpy()/255.0
    dataset.run_sim_from_tensors(
        tensors=mnist_samples,
        save_path=save_path,
        config=sim_config,
        length=num_samples
    )
    

if __name__ == '__main__':
    main()
