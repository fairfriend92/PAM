import numpy as np
import matplotlib.pylab as plt
from constants import *
from green_func import ft  
from green_func import ift   
import print_func as print_f

def ipt_para_mag(beta, U, g_0_wn, wn, tau, loop):     
    g_0_tau = np.round(ift(wn, g_0_wn, tau, beta), 8)
    
    # IPT self-energy using G0 of quantum impurity
    sigma_tau = U**2 * g_0_tau**3 
    sigma_wn = ft(wn, sigma_tau, tau, beta, a=0.)
    
    # Dyson eq.
    g_wn = g_0_wn / (1.0 - sigma_wn * g_0_wn)
    
    # Print G and self-energy
    '''
    print_f.not_converged(wn, tau, 
                          g_0_wn, g_0_wn, g_0_tau, g_0_tau,
                          sigma_wn, sigma_wn,
                          sigma_tau, sigma_tau, 
                          g_wn, g_wn, loop, U
    '''

    return g_wn, sigma_wn

# TODO: Fix anti-ferromagnetic impurity solver
def ipt_anti_ferr(beta, U, g_0_wn_up, g_0_wn_dn, wn, tau, 
                         n_up, n_dn, loop):    
    g_0_tau_up = np.round(ift(wn, g_0_wn_up, tau, beta), 8)
    g_0_tb_up = np.array([g_0_tau_up[-t] for t in range(len(tau))])
    g_0_tau_dn = np.round(ift(wn, g_0_wn_dn, tau, beta), 8)
    g_0_tb_dn = np.array([g_0_tau_dn[-t] for t in range(len(tau))])   
    
    # IPT self-energy using G0 of quantum impurity
    sigma_tau_up = U**2 * g_0_tau_up* g_0_tb_dn * g_0_tau_dn
    sigma_tau_dn = U**2 * g_0_tau_dn* g_0_tb_up * g_0_tau_up
    sigma_wn_up = ft(wn, sigma_tau_up, tau, beta, a=0.)
    sigma_wn_dn = ft(wn, sigma_tau_dn, tau, beta, a=0.)
    
    '''
    mu = U / 2   
    n_0_up = 2./beta*np.sum(g_0_wn_up) + 0.5
    n_0_dn = 2./beta*np.sum(g_0_wn_dn) + 0.5
    A_up = n_dn * (1. - n_dn) / (n_0_dn * (1. - n_0_dn))
    A_dn = n_up * (1. - n_up) / (n_0_up * (1. - n_0_up))
    B_up = (U * (1. - n_dn) - mu) / (U**2 * n_0_dn * (1. - n_0_dn))
    B_dn = (U * (1. - n_up) - mu) / (U**2 * n_0_up * (1. - n_0_up))
    sigma_wn_up = A_up * sigma_wn_up/(1. - B_up * sigma_wn_up)
    sigma_wn_dn = A_dn * sigma_wn_dn/(1. - B_dn * sigma_wn_dn)
    
    zeta_up = wn*1.j + mu - U*n_dn - sigma_wn_up
    zeta_dn = wn*1.j + mu - U*n_up - sigma_wn_dn
    g_e_up = zeta_dn/np.add.outer(-e**2, (zeta_dn*zeta_up))
    g_e_dn = zeta_up/np.add.outer(-e**2, (zeta_dn*zeta_up))
    dos_de = (dos_e * de).reshape(-1, 1)
    g_wn_up = (dos_de * g_e_up).sum(axis=0)
    g_wn_dn = (dos_de * g_e_dn).sum(axis=0)
    '''
    
    # Dyson eq.
    g_wn_up = g_0_wn_up / (1.0 - sigma_wn_up * g_0_wn_up)
    g_wn_dn = g_0_wn_dn / (1.0 - sigma_wn_dn * g_0_wn_dn)
  
    # Print G and self-energy
    '''
    print_f.not_converged(wn, tau, 
                          g_0_wn_up, g_0_wn_dn, g_0_tau_up, g_0_tau_dn,
                          sigma_wn_up, sigma_wn_dn, 
                          sigma_tau_up, sigma_tau_dn, 
                          g_wn_up, g_wn_dn, loop, U)
    '''

    return [g_wn_up, g_wn_dn], [sigma_wn_up, sigma_wn_dn]

''' Non-interacting quantum impurity Green functions '''

# Hubbard model
def HM_g_0_wn(wn, m, t, mu, U, g_wn, n):
    return 1. / (1.j*wn + m - t**2 * g_wn + mu - U*np.round(n, 8))

# Periodic Anderson model
def PAM_g_0_wn(wn, t, mu, U, V, e_d, e_p, g_p_wn, n_p, n_d):
    # TODO: Check if we should add - U*n after mu 
    return 1. / (1.j*wn + mu - e_d - 
           V**2/(1.j*wn + mu - e_p - t**2*g_p_wn))

def loop(U, t, mu, g_wn, wn, tau, beta, 
         mix=1, conv=1e-3, max_loops=50, 
         m_start=0.):
    converged = False
    loops = 0
        
    file = open("./data/not_converged/dmft_loop_beta="+f'{beta:.3}'+"_U="+f'{U:.3}'+".txt", "w") 
    file.write("n_up\tn_dn\tg_diff_up\tg_diff_dn\n")
    
    while not converged:
        g_wn_old = []
        g_0_wn = []
        n = []
    
        # Initial condition for the magnetization
        m = m_start if loops == 0 else 0.0
        
        # Backup old g
        g_wn_old.append(g_wn[0].copy())
        g_wn_old.append(g_wn[1].copy())
        
        # Occupation numbers
        n.append(2/beta*np.sum(g_wn_old[0].real) + 0.5)
        n.append(2/beta*np.sum(g_wn_old[1].real) + 0.5)
              
        # Non-interacting GF of quantum impurity  
        if (model == 'HM'):
            g_0_wn.append(HM_g_0_wn(wn, m, t, mu, U, g_wn_old[0], n[1]))
            g_0_wn.append(HM_g_0_wn(wn, -m, t, mu, U, g_wn_old[1], n[0]))
        elif (model == 'PAM'):
            g_0_wn.append(PAM_g_0_wn(wn, t, mu, U, V, e_d, e_p, g_wn[0], n[0], n[1]))
        
        # Impurity solver
        if (m_start != 0.):
            g_wn, sigma_wn = ipt_anti_ferr(beta, U, g_0_wn[0], g_0_wn[1], wn, 
                                           tau, n[0], n[1], loops)  
        else:            
            g_wn, sigma_wn = ipt_para_mag(beta, U, g_0_wn[0], wn, tau, loops) 
            g_wn = [g_wn] * 2
            sigma_wn = [sigma_wn] * 2
                    
        # Compute local GF of p-electrons 
        if (model == 'PAM'):
            # TODO: Find better way of doing this
            g_wn[0] = np.array([np.sum(de*dos_e / (1.j*w + mu - e_p - 
                                           V**2 / (1.j*w + mu - e_d - sig_w) - e))
                                for w, sig_w in zip(wn, sigma_wn[0])])
       
        # Check convergence
        converged = np.allclose(g_wn_old[0], g_wn[0], conv) 
        loops += 1
        if loops > max_loops:
            converged = True
        g_wn[0] = mix * g_wn[0] + (1 - mix) * g_wn_old[0]
        g_wn[1] = mix * g_wn[1] + (1 - mix) * g_wn_old[1]
        
        # Write datafile 
        g_diff = [np.abs(np.sum(g_wn[0] - g_wn_old[0])),
                  np.abs(np.sum(g_wn[1] - g_wn_old[1]))]
        file.write(str(n[0])+"\t"+str(n[1])+"\t"+str(g_diff[0])+"\t"+str(g_diff[1])+"\n")
        
    file.close()            
    print("Loops=" + str(loops))
    
    return g_wn, sigma_wn