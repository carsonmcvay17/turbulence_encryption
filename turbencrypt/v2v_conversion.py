# imports
import jax.numpy as jnp

class vort2vel:
    """
    Class that takes a vorticity field and computes the velocity field
    """
    epsilon: float = 1e-8

    def solvef(self, vort1, vort2):
        """
        takes in two vorticity fields and computes the forcing function
        Returns fourier transformed forcing
        """
        dxdt = vort2-vort1
        vort_field = # unforced Navier stokes eq at time t
        forcing = dxdt-vort_field
        forcing = jnp.fft.fftn(forcing)
        return forcing
    
    def comp_forcings(self, true_forcing, solved_forcing):
        """"
        compares the true forcing and the fourier transformed solved forcing
        """
    



    def convertv2v(self, vort):
        """
        Takes in a vorticity field
        Converts of velocity field using Biot Savart 
        Returns u(x,y), v(x,y)
        """
        u = jnp.empty((vort.shape[0], vort.shape[1]))
        v = jnp.empty((vort.shape[0], vort.shape[1]))

        x = jnp.arange(vort.shape[0])
        y = jnp.arange(vort.shape[1])

        for i in range(vort.shape[0]):
            for j in range(vort.shape[1]):
                u[i,j] = 0
                v[i,j] = 0
                for i_prime in range(vort.shape[0]):
                    for j_prime in range(vort.shape[1]):
                        dx = x[i]-x[i_prime]
                        dy = y[j]-y[j_prime]
                        r2=dx**2+dy**2+self.epsilon
                        factor = vort[i_prime, j_prime]/r2
                        u[i,j]+= -dy*factor*dx*dy
                        v[i,j]+= dx*factor*dx*dy

        return u,v