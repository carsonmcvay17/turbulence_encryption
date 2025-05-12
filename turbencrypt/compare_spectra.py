# imports
import jax.numpy as jnp
from turbencrypt import run_turbulence
import matplotlib.pyplot as plt

class CompSpectra:
    """
    Class to compare the spectra of the flows and the forcing functions
    """

    def compare_spectra(self, traj, forcing):
        """
        Compare the spectra of the flow and the forcing function
        inputs: traj-the evolved trajectory
        forcing - the forcing in the x direction
        """
        # ft the trajectory
        ft_traj = jnp.fft.rfftn(traj, axes=(0,1))
        ft_forcing = jnp.fft.rfftn(forcing, axes=(0,1))

        ft_traj = jnp.fft.fftshift(jnp.abs(ft_traj))
        ft_forcing = jnp.fft.fftshift(jnp.abs(ft_forcing))

        # compare the spectra
        correlation = jnp.corrcoef(ft_traj.ravel(), ft_forcing.ravel())[0, 1]
        similarity = jnp.abs(ft_traj - ft_forcing)
        # plt.subplot(1,2,1)
        # plt.imshow(ft_traj)
        # plt.colorbar()
        # plt.title('Spectra of the Trajectory')

        # plt.subplot(1,2,2)
        # plt.imshow(ft_forcing)
        # plt.colorbar()
        # plt.title('Spectra of the Forcing')
        # # plt.imshow(similarity)
        # # plt.colorbar()
        # # plt.imshow(ft_traj, label='Trajectory Spectra')
        # # plt.imshow(ft_forcing, label='Forcing Spectra')
        # # plt.title('Spectra Comparison of Trajectory and Forcing')
        # plt.suptitle(f'Correlation Between Spectra: {correlation}')
        # plt.legend()
        # plt.show()
        return correlation