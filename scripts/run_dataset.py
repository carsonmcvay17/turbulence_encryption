from turbencrypt.make_dataset import Dataset


def main():
    sim_config = {
        'viscosity': 1,
        'max_velocity': 2.0,
        'final_time': 1,
        'outer_steps': 3,
        'gridsize': 64,
        'max_courant_num': 1e-3
    }
    dataset = Dataset()
    dataset.make_data(config=sim_config)

if __name__ == '__main__':
    main()