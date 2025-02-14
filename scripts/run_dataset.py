from turbencrypt.make_dataset import Dataset
from turbencrypt.data_utils import dict2str

def main():
    sim_config = {
        'viscosity': 1,
        'max_velocity': 7.0,
        'final_time': 25,
        'outer_steps': 10,
        'gridsize': 64,
        'max_courant_num': 0.1
    }

    save_path = f"data/forreal2.npz"
    dataset = Dataset()
    dataset.make_data(
        image_dir="raw_images",
        save_path=save_path,
        config=sim_config
    )
    

if __name__ == '__main__':
    main()
