import numpy as np
import matplotlib.pylab as plt
import dmft
from constants import *
from green_func import *   
import print_func as print_f

''' Main loop '''

# TODO: Maybe arrays would be better than lists
tau_U = []
dos_U = []
n_U = [[],[]]
d_U = []
ekin_U = []
Z_U = []
phase_U = []
g_wn = []
sig_wn = []
g_wn_U = [[],[]]
g_tau_U = [[],[]]

for beta in beta_list:  
    # Generate Matsubara freq 
    wn = np.pi * (1 + 2 * np.arange(-N, N, dtype=np.double)) / beta

    # Generate imaginary time 
    dtau = beta/(2*N)   # tau has to be twice as dense as wn...
                        # ...when considering negative freq
    tau = np.arange(dtau/2., beta, dtau, dtype=np.double)
     
    # Seed green function
    g_wn.append(-2.j / (wn + np.sign(wn) * np.sqrt(wn**2 + D**2)))
               #1/(1.j*wn + 1.j*D*np.sign(wn))                
    g_wn.append(g_wn[0])
    
    print_f.generic(wn, g_wn[0], g_wn[1], 
                    r'$\omega_n$', r'$g(\omega_n)$', 
                    "./figures/g_seed.pdf")
    
    # Index of zero frequency
    w0_idx = int(len(w)/2)
    
    dos_beta = [[],[]]
    n_beta = [[],[]]
    d_beta = []
    e_kin_beta = []
    Z_beta = []
    phase_beta = []    
    g_wn_beta = [[],[]]
    g_tau_beta = [[],[]]

    for mu, U in zip(mu_list, U_list):        
        g_wn, sig_wn =  dmft.loop(U, t, mu, g_wn, wn, tau, beta, 
                                  mix=1., conv=1e-3, max_loops=50, 
                                  m_start=0.0)
        
        # Analytic continuation using Pade
        g_w = []
        sig_w = []
        g_w.append(pade(g_wn[0], w, wn))
        g_w.append(pade(g_wn[1], w, wn))
        #sig_w.append(pade(sig_wn[0], w, wn))
        #sig_w.append(pade(sig_wn[1], w, wn))
                
        if U in U_print and beta in beta_print:          
            # Save Green functions
            g_wn_beta[0].append(g_wn[0])
            g_wn_beta[1].append(g_wn[1])
            g_tau_beta[0].append(ift(wn, g_wn[0], tau, beta))
            g_tau_beta[1].append(ift(wn, g_wn[1], tau, beta))
            
            print("T="+f'{1/beta:.3f}'+"\tU="+f'{U:.3}'+"\tmu="+f'{mu:.3}')
                        
            # DOS
            dos_beta[0].append(-g_w[0].imag/np.pi)
            dos_beta[1].append(-g_w[1].imag/np.pi)
            
            # Electron concentration for temp 1/beta and energy w
            n_beta[0].append(2/beta*np.sum(g_wn[0].real) + 0.5)
            n_beta[1].append(2/beta*np.sum(g_wn[1].real) + 0.5)
            
            # Double occupancy
            #d = n**2 + 1/(U*beta)*np.sum(g_wn[0]*sig_wn[0])
            #d_beta.append(d.real)
            
            # Kinetic energy
            e_kin = 0.
            # Sum over Matsubara freq
            for w_n, sig_n in zip(wn[N:], sig_wn[0][N:]):
                # Integral in epsilon
                mu = 0. # ?
                g_k_wn = 1./(1.j*w_n + mu - e - sig_n)
                e_kin += 2./beta * np.sum(de*e*dos_e*g_k_wn)
            #print("E_kin.real="+f'{e_kin.real:.5f}')
            e_kin_beta.append(e_kin.real)
            
            # Quasi-particle weight
            #dSig = (sig_w[0][w0_idx+1].real-sig_w[0][w0_idx].real)/dw
            #Z_beta.append(1/(1-dSig))
    
    if beta in beta_print:
        tau_U.append(tau)
        dos_U.append(dos_beta)
        n_U[0].append(n_beta[0])
        n_U[1].append(n_beta[1])
        d_U.append(d_beta)
        ekin_U.append(e_kin_beta)
        Z_U.append(Z_beta)
        phase_U.append(phase_beta)
        g_wn_U[0].append(g_wn_beta[0])
        g_wn_U[1].append(g_wn_beta[1])
        g_tau_U[0].append(g_tau_beta[0])
        g_tau_U[1].append(g_tau_beta[1])
        
''' Printing functions '''

if model == 'PAM':
    y_labels = ['p electron', 'd electron']
elif model == 'HM':
    y_labels = [r'$\sigma_\uparrow$', r'$\sigma_\downarrow$'] 
      
'''
print_f.green_func(beta_print, tau_U, \
                    g_wn_U_up, g_wn_U_dn, g_tau_U_up, g_tau_U_dn, \
                    U_print, hyst, wn)
'''
#print_f.gf_iw0(beta_print, g_wn_U[0], U_print)
print_f.n(beta_print, n_U, U_print, mu_list, y_labels, 'mu')
#print_f.d(beta_print, d_U, U_print)

print_f.e_kin(beta_print, ekin_U, U_print)
#print_f.phase(beta_list, U_print, g_wn_U[0])

print_f.dos(beta_print, w, dos_U, U_print, y_labels, hyst)
#print_f.Z(beta_print, Z_U, U_print)
