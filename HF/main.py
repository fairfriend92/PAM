import numpy as np
import matplotlib.pyplot as plt

'''
The Coulomb interaction term is treated using the Hartree-Fock approximation.
p-orbital electrons are conductions electrons.
d-orbital electrons are localized electrons.
'''

def fermi_dirac(e, mu, T):
    return 1.0 / (1 + np.exp((e-mu)/T))

def solve_model(n_d, e_p, mu, U, T, e_k_range=None):
    D = 2*t_pp      # half-bandwidth
       
    n_p_up = 0.0    # p-orbital upper band electron number
    n_d_up = 0.0    # d-orbital upper band electron number
    n_p_dn = 0.0    # p-orbital lower band electron number
    n_d_dn = 0.0    # d-orbital lower band electron number  
    
    global band_up 
    global band_dn 
    band_up = []
    band_dn = [] 
    
    e_k_range = np.arange(-2*t_pp, 2*t_pp, de) if e_k_range is None \
                else e_k_range
    
    for e_k in e_k_range:
        # compute orbital energy
        E_p = e_p - mu + e_k
        E_d = e_d - mu + U * (1.0 - n_d)    # HF approximation
        # compute Bethe lattice DOS
        rho = 2 * np.sqrt(D**2 - e_k**2)/np.pi
        # compute eigenvalues
        a = 1
        b = -(E_p + E_d)
        c = E_p * E_d - V**2        
        lambda_up = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
        lambda_dn = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)
        # compute eigenvectors
        v_up = [(E_d - lambda_up)/(E_p - lambda_up), 
                (lambda_up - E_d)/V]
        v_dn = [(E_d - lambda_dn)/(E_p - lambda_dn), 
                (lambda_dn - E_d)/V]
        # normalize eigenvectors            
        v_up /= np.sqrt(v_up[0]**2 + v_up[1]**2)
        v_dn /= np.sqrt(v_dn[0]**2 + v_dn[1]**2)            
        # compute electron number
        n_p_up += 2 * v_up[0]**2 * rho * de * \
                  fermi_dirac(lambda_up, mu, T)
        n_d_up += 2 * v_up[1]**2 * rho * de * \
                  fermi_dirac(lambda_up, mu, T)
        n_p_dn += 2 * v_dn[0]**2 * rho * de * \
                  fermi_dirac(lambda_dn, mu, T)
        n_d_dn += 2 * v_dn[1]**2 * rho * de * \
                  fermi_dirac(lambda_dn, mu, T)               
        
        band_up.append(lambda_up)
        band_dn.append(lambda_dn)
    
    return [n_p_up, n_p_dn, n_d_up, n_d_dn]

# loop until model converges
def converge(e_p, mu, U, T, e_k_range=None):  
    n_loops = 100   # number of loops 
    mix = 0.5       # ratio of old and new n_d 
    n_p = 0.0       # p-orbital electron number
    n_d = 0.0       # d-orbital electron number
    d_n_d = 1e-3    # n_d differential
    for i in range(n_loops):
        n_d_old = n_d
        n = solve_model(n_d, e_p, mu, U, T, e_k_range)
        n_p = n[0] + n[1]
        n_d = n[2] + n[3]
        # mix old and new n_d to speed up convergence
        n_d = mix * n_d + (1 - mix) * n_d_old
        if (abs(n_d_old - n_d) < d_n_d):
            break
    return n

# compute band structures for different e_p values
def bands(e_p_range, mu = 0.529, U = 0.0, T = 1.0/64, k_range=None):
    print('\ncomputing band structures...\n')
    fig, ax = plt.subplots()
    
    e_k_range = []
    if (k_range is None):
        e_k_range = np.arange(-2*t_pp, 2*t_pp, de)
    else:
        e_k_range = [-2*t_pp*np.cos(k) for k in k_range]
    
    for e_p in e_p_range:
        print("solving for e_p = " + str(e_p))
        
        converge(e_p, mu, U, T, e_k_range)
        
        x = e_k_range if k_range is None else k_range
        plt.scatter(x, band_up, s=1,
                    label = r'$\lambda_{\uparrow} \epsilon_p=$' + str(e_p))
        
        plt.scatter(x, band_dn, s=1,
                    label = r'$\lambda_{\downarrow} \epsilon_p=$' + str(e_p))
    ax.legend()
    x_label = r'$\epsilon_k$' if k_range is None else 'k'
    ax.set(xlabel = x_label, ylabel = 'E')
    plt.savefig('figures/bands.pdf')
    plt.close()


# compute n_p, n_d for different mu
def occ_num(mu_range, ep = -1.0, U = 0.0, T = 1.0/64):
    print('\ncomputing occupation numbers...\n')
    fig, ax = plt.subplots()
    n_p_mu = []
    n_d_mu = []
    for mu in mu_range: 
        print("solving for mu = " + str(mu))
        n = converge(e_p, mu, U, T)
        n_p = n[0] + n[1]
        n_d = n[2] + n[3]
        n_p_mu.append(n_p)
        n_d_mu.append(n_d)
    n_tot = [n_p_mu[i] + n_d_mu[i] for i in range(len(n_p_mu))]
    ax.plot(mu_range, n_p_mu, marker='.', label=r'$n_p$')
    ax.plot(mu_range, n_d_mu, marker='.', label=r'$n_d$')
    ax.plot(mu_range, n_tot, marker='.', label=r'$n_{tot}$')
    ax.legend()
    ax.set(xlabel=r'$\mu$', ylabel='occupation number')
    plt.savefig('figures/occ_num.pdf')
    plt.clf()


# plot phase diagram
def phase_diag(mu_range, U_range, e_p = -1.0, T = 1.0/64):
    print('\nplotting phase diagram...\n')
    fig, ax = plt.subplots()
    n_mu_U = []
    for U in U_range:
        print("solving for U = " + str(U))
        n_mu = []
        for mu in mu_range:
            print("solving for mu = " + str(mu))
            n = converge(e_p, mu, U, T)
            n_p = n[0] + n[1]
            n_d = n[2] + n[3]
            n_mu.append(n_p + n_d)
        n_mu_U.append(n_mu)
    n_mu_U.reverse() # order matrix rows from biggest U to smallest
    im_edges = [mu_range[0], mu_range[len(mu_range)-1],
                U_range[0], U_range[len(U_range)-1]]
    plt.imshow(n_mu_U, interpolation='none', 
               extent=im_edges, aspect='auto')
    ax.set_xticks(mu_range)
    ax.set_yticks(U_range)
    ax.set(xlabel=r'$\mu$', ylabel=r'$U$')
    clb = plt.colorbar()
    clb.set_label(r'$n_{tot}$')
    plt.savefig('figures/phase_diag.png')
    plt.clf()
    
def lorentzian(x, x_0, e=0.01):
    return 1/np.pi * e / ((x-x_0)**2 + e**2)

# plot density of states
def dos(mu_range, U_range, e_p = -1.0, T = 1.0/64, k_range=None):
    print('\nplotting density of states...\n')
    domega = 1e-2                          # omega differential
    omega_range = np.arange(-3, 1, domega) # DOS domain 
    
    e_k_range = []
    if (k_range is None):
        e_k_range = np.arange(-2*t_pp, 2*t_pp, de)
    else:
        e_k_range = [-2*t_pp*np.cos(k) for k in k_range]
    
    for U in U_range:
        print("solving for U = " + str(U))
        for mu in mu_range:
            print("solving for mu = " + str(mu))
            fig, ax = plt.subplots()
            n = converge(e_p, mu, U, T, e_k_range)
            n_p_up = n[0]
            n_p_dn = n[1]
            n_d_up = n[2]
            n_d_dn = n[3]
            
            dos_p_up = [] # DOS of p-electrons in upper band
            dos_d_up = [] # DOS of d-electrons in upper band
            dos_p_dn = [] # DOS of p-electrons in lower band
            dos_d_dn = [] # DOS of d-electrons in lower band

            for omega in omega_range:
                norm = de /np.sum(n)
               
                dos_omega_up = np.sum(lorentzian(omega, band_up))
                dos_p_up.append(dos_omega_up * norm * n_p_up)
                dos_d_up.append(dos_omega_up * norm * n_d_up)
                
                dos_omega_dn = np.sum(lorentzian(omega, band_dn))
                dos_p_dn.append(dos_omega_dn * norm * n_p_dn)
                dos_d_dn.append(dos_omega_dn * norm * n_d_dn)
            
            T = np.round(T, 3)
            plt.plot(omega_range, dos_p_up, label=r'$\lambda_{\uparrow} p$')
            plt.plot(omega_range, dos_d_up, label=r'$\lambda_{\downarrow} d$')
            plt.plot(omega_range, dos_p_dn, label=r'$\lambda_{\uparrow} p$')
            plt.plot(omega_range, dos_d_dn, label=r'$\lambda_{\downarrow} d$')
            ax.set_title('U='+str(U)+r' $\mu=$'+str(mu)+' T='+str(T))
            ax.legend()
            ax.set(xlabel=r'$\omega$', ylabel=r'$\rho$')
            plt.savefig('figures/dos/dos_U='+str(U)+'_mu='+str(mu)+ \
                        '_T='+str(T)+'.pdf')
            plt.close()

# constants
e_p = -1.0      # energy of p-orbital electron
t_pp = 0.5      # p-p orbital hopping
e_d = 0.0       # energy of d-orbital electron
V = 0.9         # hybridization 
mu = 0.5        # chemical potential
U = 0.0         # interaction term
T = 1.0/64      # temperature in units of Boltzmann's constant
de = 1e-2       # energy spacing
dk = 1e-1       # momentum spacing

# values range
k_range = np.arange(-np.pi, np.pi, dk)
e_p_range = [e_p]
mu_range = np.arange(-2.5, 2.5, 0.25)
U_range = [U] #np.arange(0, 4, 0.25)

# global variables
band_up = []    # upper energy band
band_dn = []    # lower energy band  

# main code
#bands(e_p_range, mu, U, T, k_range) 
#occ_num(mu_range, e_p, U, T)
#phase_diag(mu_range, U_range, e_p, T)
dos(mu_range, U_range, e_p, T)