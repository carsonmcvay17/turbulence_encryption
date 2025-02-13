import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns

def dict2str(dict_obj: dict) -> str:
    """
    Convert a dictionary to a string representation
    """
    config_str = '-'.join([f"{key}_{value}" for key, value in dict_obj.items()])
    return config_str.replace('.', '_')

def safe_standardize(array: jnp.ndarray, axis: int = 1, epsilon: float = 1e-8) -> jnp.ndarray:
    """
    Standardize an array along an axis safely (ignoring NaNs)


    """
    mean = jnp.nanmean(array, axis=axis, keepdims=True)
    std = jnp.nanstd(array, axis=axis, keepdims=True)
    std = jnp.where(std < epsilon, epsilon, std)
    return (array - mean) / std

def nan_check(data):
    data = jnp.load(data)
    for key in data.files:
        array = data[key]
        if jnp.isnan(array).any():
            print(f"NaNs found in {key} at indices:", jnp.argwhere(jnp.isnan(array)))
        else:
            print(f"No NaNs in {key}")

def visualize_data(data, forcing_idx, component):
    data = jnp.load(data)
    inputs= data['inputs']
    outputs = data['outputs']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(jnp.fft.irfftn(inputs[forcing_idx]),sns.cm.icefire, origin='lower')
    axes[0].set_title(f"trajectory for foricng function {forcing_idx}")

    axes[1].imshow(outputs[forcing_idx, :, :, component],cmap=sns.cm.icefire, origin='lower')
    axes[1].set_title(f"forcing function")
    plt.show()
