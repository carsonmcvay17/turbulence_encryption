from turbencrypt.make_movie import movie

def main():
    sim_config = {
        'viscosity': 1e-1,
        'max_velocity': 7.0,
        'final_time': 25,
        'outer_steps': 10,
        'gridsize': 64,
        'max_courant_num': 0.1
    }

    save_path = f"data"
    model = movie()
    model.make_movie(
        image_dir="raw_images/image_3.jpg",
        save_path=save_path,
        config=sim_config
    )
    

if __name__ == '__main__':
    main()
