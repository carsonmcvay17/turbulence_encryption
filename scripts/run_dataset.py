from turbencrypt.make_dataset import Dataset


def main():
    sim_config = {
        'viscosity': 1,
        'max_velocity': 7.0,
        'final_time': 25,
        'outer_steps': 10,
        'gridsize': 64,
        'max_courant_num': 0.1
    }
    dataset = Dataset()
    dataset.make_data(
        image_dir="raw_images",
        save_path="data",
        config=sim_config
    )

if __name__ == '__main__':
    main()
