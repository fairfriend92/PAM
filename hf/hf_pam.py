import numpy as np
import matplotlib.pyplot as plt

e_p = 0         # energy of p-orbital electron
n_loops = 100   # number of loops   
n_d = 0.5       # d-orbital electron number
mix = 0.5       # ratio of old and new n_d 
band_up = []    # up-spin energy band
band_dn = []    # down-spin energy band
energy = []     # range of e_k values

def solve_model(e_p, n_d):
    D = 1       # half-bandwidth
    e_d = 0     # energy of d-orbital electron
    t_pd = 0.5  # orbital hopping
    U = 2.0     # interaction term
    de = 1e-3   # energy differential
    mu = 0      # chemical potential
    n_p_up = 0  # p-orbital up-spin electron number
    n_d_up = 0  # d-orbital up-spin electron number
    n_p_dn = 0.5# p-orbital down-spin electron number
    n_d_dn = 0.5# d-orbital down-spin electron number    
    global band_up 
    global band_dn 
    global energy
        
    for e_k in np.arange(-D + 0.001*de, D - 0.001*de, de):
        is_zero_freq = e_k > - de/2 and e_k < de/2  
        # discard 0 energy to avoid division by 0
        if (not is_zero_freq):
            # compute orbital energy
            E_p = e_p - mu + e_k
            E_d = e_d - mu + U * (n_d - 1/2)
            # compute Bethe lattice DOS
            rho = 2 * np.sqrt(D**2 - e_k**2)/np.pi
            # compute eigenvalues
            a = 1
            b = E_p + E_d
            c = E_p * E_d - t_pd**2        
            lambda_up = (b + np.sqrt(b**2 - 4*a*c))/(2*a)
            lambda_dn = (b - np.sqrt(b**2 - 4*a*c))/(2*a)
            # compute eigenvectors
            v_up = [1, (lambda_up - E_d)/t_pd]
            v_dn = [1, (lambda_dn - E_d)/t_pd]
            # normalize eigenvectors            
            v_up /= np.sqrt(v_up[0]**2 + v_up[1]**2)
            v_dn /= np.sqrt(v_dn[0]**2 + v_dn[1]**2)
            # compute electron number
            if lambda_up < 0:
                n_p_up += v_up[0]**2 * rho * de
                n_d_up += v_up[1]**2 * rho * de
            if lambda_dn < 0:
                n_p_dn += v_dn[0]**2 * rho * de
                n_d_dn += v_dn[1]**2 * rho * de
            
            band_up.append(lambda_up)
            band_dn.append(lambda_dn)
            energy.append(e_k)
            
    return n_d_up + n_d_dn

fig, ax = plt.subplots()
ax.set(xlabel = r'$\epsilon_k$')

for e_p in np.arange(0.0, 1.5, 0.25):
    n_d = 0
    print("solving for e_p = " + str(e_p))
    for i in range(n_loops):
        n_d_old = n_d
        band_up = []
        band_dn = []
        energy = []
        n_d = solve_model(e_p, n_d)
        # mix old and new n_d to speed up convergence
        n_d = mix * n_d + (1 - mix) * n_d_old
        
    plt.plot(energy, band_up, 
             label = r'$\lambda_{\uparrow} \epsilon_p=$' + str(e_p))
    
    plt.plot(energy, band_dn, 
             label = r'$\lambda_{\downarrow} \epsilon_p=$' + str(e_p))

plt.legend()
plt.savefig('bands.png')
plt.close()
