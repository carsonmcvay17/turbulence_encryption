import jax.numpy as jnp
import matplotlib.pyplot as plt

from turbencrypt.img_analysis import img_analysis

# load data
# load the data
# data_paths = [
#     "/Users/gilpinlab/turbulence_encryption/data/img_re20_g64_eval.npz",
#     "/Users/gilpinlab/turbulence_encryption/data/img_re70_g64_eval.npz",
#     "/Users/gilpinlab/turbulence_encryption/data/img_re200_g64_eval.npz",
#     "/Users/gilpinlab/turbulence_encryption/data/img_re700_g64_eval.npz",
#     "/Users/gilpinlab/turbulence_encryption/data/img_re2000_g64_eval.npz",
#     "/Users/gilpinlab/turbulence_encryption/data/img_re7000_g64_eval.npz" , 
# ]
data_paths = [
    "/Users/gilpinlab/turbulence_encryption/data/mnist_re20_g64_eval.npz",
    "/Users/gilpinlab/turbulence_encryption/data/mnist_re70_g64_eval.npz",
    "/Users/gilpinlab/turbulence_encryption/data/mnist_re200_g64_eval.npz",
    "/Users/gilpinlab/turbulence_encryption/data/mnist_re700_g64_eval.npz",
    "/Users/gilpinlab/turbulence_encryption/data/mnist_re2000_g64_eval.npz",
    "/Users/gilpinlab/turbulence_encryption/data/mnist_re7000_g64_eval.npz" , 
]
data_dict = {}



for i in range(6):
    data_dict[i] = jnp.load(data_paths[i])


mse_dict = {}

model = img_analysis()
for j in range(6):
    mse_values = []
    for i in range(0, len(data_dict[j]['outputs'])):
        outputs = data_dict[j]['outputs'][i]
        targets = data_dict[j]['targets'][i]
        mse = model.mse(outputs, targets)
        mse_values.append(mse)
        #breakpoint()
    mse_dict[j] = mse_values
    
# average mse for each dataset
avg_mse = [jnp.mean(jnp.array(mse_values)) for mse_values in mse_dict.values()]
re_num = jnp.array([20, 70, 200, 700, 2000, 7000])
plt.plot(re_num, avg_mse)
plt.title("Average MSE for MNIST Dataset Outputs vs Targets")
plt.xlabel("Reynolds Number")
plt.ylabel("Average MSE")
plt.show()
# plotting
# Plotting the MSE values for each dataset
plt.figure(figsize=(10, 6))

# For each dataset, plot the MSE values
for j in range(6):
    plt.plot(mse_dict[j], label=f'Dataset {j+1}', marker='o', linestyle='-', markersize=4)

# Adding labels and title
plt.xlabel('Image Index')
plt.ylabel('MSE')
plt.title('MSE Outputs vs Targets')

# Adding legend
plt.legend()

# Show the plot
plt.show()