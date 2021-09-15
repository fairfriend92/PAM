import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

e_p = 0.0       # energy of p-orbital electron
t_pp = 0.5      # p-p orbital hopping
n_p = 0.0       # p-orbital electron number
U = 0.5         # interaction term
mu = 0.0        # chemical potential
mix = 0.5       # ratio of old and new n_d 
T = 1.0/64      # temperature in units of Boltzmann's constant
de = 1e-3       # energy differential
band_up = []    # upper energy band
band_dn = []    # lower energy band  
energy = []     # range of e_k values

def plane_wave(k):
    return - 2 * t_pp * np.cos(k)

def fermi_dirac(e, mu, T):
    return 1.0 / (1 + np.exp((e-mu)/T))

def solve_model(e_p, n_d, mu, U, T):
    D = 1.0     # half-bandwidth
    e_d = 0.0   # energy of d-orbital electron
    t_pd = 0.9  # p-d orbital hopping     
    n_p_up = 0.0# p-orbital upper band electron number
    n_d_up = 0.0# d-orbital upper band electron number
    n_p_dn = 0.0# p-orbital lower band electron number
    n_d_dn = 0.0# d-orbital lower band electron number    
    global band_up 
    global band_dn 
    global energy
    
    band_up = []
    band_dn = [] 
    energy = [] 
    
    for e_k in np.arange(-2*t_pp, 2*t_pp, de):
        # compute orbital energy
        E_p = e_p - mu + e_k
        E_d = e_d - mu + U * (n_d - 0.5)
        # compute Bethe lattice DOS
        rho = 2 * np.sqrt(D**2 - e_k**2)/np.pi
        # compute eigenvalues
        a = 1
        b = -(E_p + E_d)
        c = E_p * E_d - t_pd**2        
        lambda_up = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
        lambda_dn = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)
        # compute eigenvectors
        v_up = [(E_d - lambda_up)/(E_p - lambda_up), 
                (lambda_up - E_d)/t_pd]
        v_dn = [(E_d - lambda_dn)/(E_p - lambda_dn), 
                (lambda_dn - E_d)/t_pd]
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
        energy.append(e_k)
    
    n_p = n_p_up + n_p_dn
    n_d = n_d_up + n_d_dn
    return n_p, n_d

# loop until model converges
def converge(e_p, n_p, mu, U, T):  
    n_loops = 100   # number of loops 
    n_d = 0.0       # d-orbital electron number
    d_n_d = 1e-3    # n_d differential
    for i in range(n_loops):
        n_d_old = n_d
        n_p, n_d = solve_model(e_p, n_d, mu, U, T)
        # mix old and new n_d to speed up convergence
        n_d = mix * n_d + (1 - mix) * n_d_old
        if (abs(n_d_old - n_d) < d_n_d):
            break
    return n_p, n_d

# compute band structures for different e_p values
def bands():
    mu = 0.529
    T = 1.0/64
    U = 0.0
    print('\ncomputing band structures...\n')
    fig, ax = plt.subplots()
    for e_p in [-1.0]:
        print("solving for e_p = " + str(e_p))
        converge(e_p, n_p, mu, U, T)
            
        plt.plot(energy, band_up, 
                 label = r'$\lambda_{\uparrow} \epsilon_p=$' + str(e_p))
        
        plt.plot(energy, band_dn, 
                 label = r'$\lambda_{\downarrow} \epsilon_p=$' + str(e_p))
    ax.legend()
    ax.set(xlabel = r'$\epsilon_k$')
    plt.savefig('bands.png')
    plt.clf()


# compute n_p, n_d for different mu
def occ_num():
    print('\ncomputing occupation numbers...\n')
    fig, ax = plt.subplots()
    e_p = -1.0
    n_p_mu = []
    n_d_mu = []
    mu_range = np.arange(-2, 2.5, 0.5)
    for mu in mu_range: 
        print("solving for mu = " + str(mu))
        n_p, n_d = converge(e_p, n_p, mu, U, T)
        n_p_mu.append(n_p)
        n_d_mu.append(n_d)
    n_tot = [n_p_mu[i] + n_d_mu[i] for i in range(len(n_p_mu))]
    ax.plot(mu_range, n_p_mu, marker='.', label=r'$n_p$')
    ax.plot(mu_range, n_d_mu, marker='.', label=r'$n_d$')
    ax.plot(mu_range, n_tot, marker='.', label=r'$n_{tot}$')
    ax.legend()
    ax.set(xlabel=r'$\mu$', ylabel='occupation number')
    plt.savefig('occ_num.png')
    plt.clf()


# plot phase diagram
def phase_diag():
    print('\nplotting phase diagram...\n')
    fig, ax = plt.subplots()
    e_p = -1.0
    mu_range = np.arange(0, 2.5, 0.25)
    U_range = np.arange(0, 4, 0.25)
    n_mu_U = []
    for U in U_range:
        print("solving for U = " + str(U))
        n_mu = []
        for mu in mu_range:
            print("solving for mu = " + str(mu))
            n_p, n_d = converge(e_p, n_p, mu, U, T)
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
    plt.savefig('phase_diag.png')
    plt.clf()


# plot density of states
def dos():
    print('\nplotting density of states...\n')
    e_p = -1.0
    mu_range = [0.529]
    U_range = [0]
    T = np.round(1.0 / 64, 3)
    domega = 1e-2                          # omega differential
    omega_range = np.arange(-3, 1, domega) # DOS domain 
    for U in U_range:
        print("solving for U = " + str(U))
        for mu in mu_range:
            print("solving for mu = " + str(mu))
            fig, ax = plt.subplots()
            converge(e_p, n_p, mu, U, T)
            
            dos_up = [] # density of states upper band
            dos_dn = [] # density of states lower band

            for omega in omega_range:
                dos_omega_up = 0
                dos_omega_dn = 0
                for e_up in band_up:
                    if e_up > omega-domega/2 and e_up < omega+domega/2:
                        dos_omega_up += 1
                dos_up.append(dos_omega_up * de/domega)
                for e_dn in band_dn:
                    if e_dn > omega-domega/2 and e_dn < omega+domega/2:
                        dos_omega_dn += 1
                dos_dn.append(dos_omega_dn * de/domega)
            
            plt.plot(omega_range, dos_up, label=r'$\lambda_{\uparrow}$')
            plt.plot(omega_range, dos_dn, label=r'$\lambda_{\downarrow}$')
            ax.set_title('U='+str(U)+r' $\mu=$'+str(mu)+' T='+str(T))
            ax.legend()
            ax.set(xlabel=r'$\omega$', ylabel=r'$\rho$')
            plt.savefig('dos/dos_U='+str(U)+'_mu='+str(mu)+ \
                        '_T='+str(T)+'.png')
            plt.clf()

# main code

#bands()       
dos()