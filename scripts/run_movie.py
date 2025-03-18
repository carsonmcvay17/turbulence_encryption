from turbencrypt.make_movie import movie

def main():
    sim_config = {
        'viscosity': 1e-2,
        'max_velocity': 7.0,
        'final_time': 30,
        'outer_steps': 50,
        'gridsize': 128,
        'max_courant_num': 0.001
    }

    save_path = f"movie_files/kolmogorov"
    model = movie()
    model.make_movie(
        image_dir="raw_images/image_3.jpg",
        save_path=save_path,
        sim_config=sim_config, 
    )
    

if __name__ == '__main__':
    main()
