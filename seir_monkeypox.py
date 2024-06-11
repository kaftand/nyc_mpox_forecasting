
import numpy as np

class SEIR(object):
    """ Discrete time SEIR model."""

    def __init__(self, S0, D_e, D_i, z_t, add_z_to_inf = False):
        # Store the known model parameters
        self.z_t = z_t
        self.S0 = S0
        self.D_e = D_e
        self.D_i = D_i
        self.add_z_to_inf = add_z_to_inf
        # Create a time axis
        self.T = len(z_t)
        self.time = np.arange(self.T)

        # Mark the model as un-fit, which means parameters are missing.
        self._fit = False

    def sample_scenario(self, beta, immune_num, num_samples=1):
        # Allocate storage for the output, and set up the
        # initial condition.
        X = np.zeros((num_samples, 6, self.T))
        X[:, 0, 0] = self.S0

        # Loop over time and collect samples
        for t in range(1, self.T):

            # Update all the deterministic components (S and I)
            X[:, 0, t] = X[:, 0, t - 1] - beta[t] * X[:, 0, t - 1] * (X[:, 2, t - 1]  + self.z_t[t - 1]) - X[:, 4, t - 1]

            add_z = self.z_t[t] if self.add_z_to_inf else 0

           # Update the exposed compartment 
            X[:, 1, t] = beta[t] *  X[:, 0, t - 1] * (X[:, 2, t - 1] + self.z_t[t - 1]) \
                         + X[:, 1, t - 1] * (1. - (1. / self.D_e))  + add_z
             
            #Infectious comparment
            X[:, 2, t] = X[:, 1, t - 1] / self.D_e + X[:, 2, t - 1] * (1. - (1. / self.D_i)) 
            
            #R compartment
            X[:, 3, t] = X[:, 2, t - 1] * (1 / self.D_i) +  X[:, 3, t-1] \
                          + immune_num[t - 1] * X[:, 0, t-1]/ self.S0  
                          
            #Number of people receiving vaccines and moved to R compartment 
            X[:, 4, t] = immune_num[t - 1] * X[:, 0, t - 1]/ self.S0            

            # This includes those who have new infections per day
            X[:, 5, t] =beta[t] * X[:, 0, t - 1] * (X[:, 2, t - 1] + self.z_t[t - 1]) 
                         
            where_are_NaNs = np.isnan(X)
            X[where_are_NaNs]= 0 
            
            # High sig-eps models require by-hand enforcement of positivity (i.e. truncated gaussians).
            X[X[:, 0, t] < 0, 0, t] = 0
            X[X[:, 1, t] < 0, 1, t] = 0
            X[X[:, 2, t] < 0, 2, t] = 0
            X[X[:, 3, t] < 0, 3, t] = 0
            X[X[:, 4, t] < 0, 4, t] = 0
            X[X[:, 5, t] < 0, 5, t] = 0
            
        return X
    
