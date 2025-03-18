import jax.numpy as jnp


class img_analysis:
    """
    This class holds functions used to analyze the images
    """
    
    def mse(self, img1, img2):
        """
        Calculates mean squared error. Takes in two images, returns the MSE
        """
        return jnp.mean((img1-img2))**2
    
