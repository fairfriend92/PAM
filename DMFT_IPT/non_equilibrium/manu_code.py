#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 15:49:38 2020

@author: diaz
"""

import numpy as np
import matplotlib.pyplot as plt
# plt.rcParams['backend'] = "Qt4Agg"
from time import time
from scipy.optimize import curve_fit
from scipy.signal import find_peaks as fpeaks
import datetime
import os
ppio = time()
ya   = datetime.datetime.now().strftime("%d-%m-%y_%H:%M")

def KK_w(Im_G_w): # input ImGR, output GR (via Kramers-Kronig)
    G_w = -(dw/np.pi) * np.convolve(wPV_,Im_G_w,'same') + (1/np.pi)* np.gradient(Im_G_w) + 1j*Im_G_w
    return G_w

def fdd(x,Tp):
   return 1/(np.exp(x/Tp)+1)

#### Physical parameters

t_perp = 1
t_para = 0
U_min  = 0   ; U_max = 16
T_min  = 0.01  ; T_max = 0.11
E_min  = 0. ; E_max = 0.519
Gamma  = 0.008                  # RODO: ? 


#### Precision parameters

logN_w = 13     # ~log of number of w points
w_max  = 15
logN_k = 7      # ~log of number of k points
N_U    = 9     # number of U points in the parameter trajectory
N_T    = 6
dE     = 0.048 # step of electric field when ramping up or down

first_kernel        = 1
clear_output        = False

save                = False
rundate             = '27-04-21_16:37'
ind                 = 361

alpha               = 0.5   # Relative weight with which the previous solution is mixed with the current one
n_dmft_mixing       = 0      
tolerance_F_iter    = 1e-5
tolerance_dmft_iter = 1e-5
n_iter_F_max        = 800
n_dmft_max          = 800
force_E_code        = False  # force to use the non-equilibrium code even if E = 0
cache               = True   # keep self-energies in cache

#### Axis meshes

N_w   = 2**logN_w + 1
w_min = -w_max
Dw    = w_max - w_min
w_    = np.linspace(w_min,w_max,N_w)
dw    = w_[1]-w_[0]
w0    = np.int32(np.ceil(N_w/2)) - 1

w_aux = np.copy(w_)      ; w_aux[w0] = 1
wPV_  = 1/w_aux          ; wPV_[w0]  = 0             # Principal value of 1/w

N_t   = N_w
#t_max = np.pi/dw
#t_min = -t_max
#t_    = np.linspace(t_min,t_max,N_t) 
t_    = np.fft.fftshift(np.fft.fftfreq(N_w, dw/(2*np.pi)))
t_max = max(t_)
t_min = min(t_)
Dt    = t_max - t_min
dt    = t_[1]-t_[0]
t0    = np.int32(np.ceil(N_t/2)) - 1

zero = np.zeros(N_w)

Nk      = 2**logN_k + 1
k_      = np.linspace(-np.pi,np.pi,Nk)
dk      = k_[1]-k_[0]
k0      = np.int32(np.ceil(Nk/2)) - 1
k_      = np.delete(k_,-1)
N_k     = len(k_)

#### Lattice geometry

epsilon_k = -2*t_para*np.cos(k_) # dispersion in directions perpendicular to E
# epsilon_k = 0

#### Parameter trajectories

U_   = np.linspace(U_min,U_max,N_U)
# U_   = np.flip(np.linspace(U_min,U_max,N_U))

# U_   = np.append(U_, np.flip(np.linspace(0,10.4,53)))

# borr = np.linspace(np.int(12),np.int(12),1).astype(int)
# U_   = np.insert(np.delete(U_,borr),12,np.linspace(11.25,15,16))

# N_E   = int(max(np.floor((E_max - E_min)/dE)+1,1))
# E_    = np.linspace(E_min,E_max,N_E); E_ = np.double(np.floor(E_/dw)*dw); 
# E_min = E_[0]; E_max = E_[-1]
# E_    = np.delete(E_, np.array([0]))
# E_    = np.flip(E_)

# UTE_ = np.array([U_min*np.ones(len(E_)), T_min*np.ones(len(E_)), E_, np.append(first_kernel,np.zeros(len(E_)-1))])
# UTE_ = np.append(UTE_, np.array([U_min*np.ones(len(E_)), T_min*np.ones(len(E_)), np.flip(E_), np.zeros(len(E_))]), axis=1)
# UTE_ = np.delete(UTE_, len(E_), axis = 1) 

# UTE_     = []
# n_trj    = 0
# for U in U_:
#     ute_ = []
#     ute_ = np.array([U*np.ones(len(E_)), T_min*np.ones(len(E_)), E_, np.append(1,np.zeros(len(E_)-1))])
#     ute_ = np.append(ute_, np.array([U*np.ones(len(E_)), T_min*np.ones(len(E_)), np.flip(E_), np.zeros(len(E_))]), axis=1)
#     ute_ = np.delete(ute_, len(E_), axis = 1) 
#     if n_trj == 0:
#         UTE_ = ute_
#     else:        
#         UTE_ = np.append(UTE_, ute_, axis = 1)
#     n_trj = n_trj + 1



# Ramping up U + clear kernel only at first step + Ramping down U


UTE_ = np.array([U_, T_min*np.ones(len(U_)), np.double(np.floor(0.5*E_min/dw)*2*dw)*np.ones(len(U_)), np.append(first_kernel,np.zeros(len(U_)-1))])  
# UTE_ = np.append(UTE_, np.array([np.flip(U_), T_min*np.ones(len(U_)), np.double(np.floor(E_min/dw)*dw)*np.ones(len(U_)), np.append(2,np.zeros(len(U_)-1))]), axis=1)
# UTE_ = np.delete(UTE_, len(U_), axis = 1)

# U_ = np.array([10.5, 11, 11.5, 12, 12.5, 13, 14, 16])
# E_ = np.array([0.1953125 , 0.2734375 , 0.32714844, 0.37597656, 0.41503906, 0.45410156, 0.51757812, 0.6201171875])
# E_ = np.array([0.1953 , 0.274 , 0.328, 0.376, 0.416, 0.455, 0.518, 0.621])

# UTE_ = np.array([U_, T_min*np.ones(len(U_)), np.double(np.floor(E_/dw)*dw)*np.ones(len(U_)), 2*np.ones(len(U_))])  


# E_   = np.array([0.        , 0.00366211, 0.00915527, 0.01281738, 0.01831055, 0.02197266, 0.02746582])

# UTE_ = []
# n_tr = 0
# for E in E_:
#     ute_ = [] 
#     ute_ = np.array([U_, T_min*np.ones(len(U_)), np.double(np.floor(E/dw)*dw)*np.ones(len(U_)), np.append(first_kernel,np.zeros(len(U_)-1))])  
#     ute_ = np.append(ute_, np.array([np.flip(U_), T_min*np.ones(len(U_)), np.double(np.floor(E/dw)*dw)*np.ones(len(U_)), np.append(0,np.zeros(len(U_)-1))]), axis=1)
#     ute_ = np.delete(ute_, len(U_), axis = 1)
#     if n_tr == 0:
#           UTE_ = ute_
#     else:        
#           UTE_ = np.append(UTE_, ute_, axis = 1)
#     n_tr = n_tr + 1

# T_  = np.linspace(T_min, T_max, N_T)
# # T_  = np.array([0.01005781, 0.01027352, 0.01063332, 0.01123596, 0.01214603,
# #        0.01573882, 0.02089866, 0.47416483])
# UTE_  = np.array([U_min*np.ones(len(T_)), T_, np.double(np.floor(E_min/dw)*dw)*np.ones(len(T_)),np.append(first_kernel,np.zeros(len(T_)-1))])

# T_   = np.array([0.01612528, 0.01816771, 0.02088863, 0.02433244, 0.02819432,
#        0.03157184, 0.03307252, 0.66538508])
# U_   = np.array([18.        , 17.16666667, 16.33333333, 15.5       , 14.66666667,
#        13.83333333, 13.        , 12.16666667])
# UTE_ = np.array([U_, T_, np.double(np.floor(E_min/dw)*dw)*np.ones(len(U_)), np.append(first_kernel,np.zeros(len(U_)-1))])  



# n_tr = 0
# for T in T_:
#     ute_ = [] 
#     ute_ = np.array([U_, T*np.ones(len(U_)), np.double(np.floor(E_min/dw)*dw)*np.ones(len(U_)), np.append(first_kernel,np.zeros(len(U_)-1))])
#     ute_ = np.append(ute_, np.flip(ute_,axis=1), axis=1)
#     ute_ = np.delete(ute_, len(U_), axis = 1)                
#     if n_tr == 0:
#           UTE_ = ute_
#     else:        
#           UTE_ = np.append(UTE_, ute_, axis = 1)
#     n_tr = n_tr + 1
# if first_kernel == 3:
#     UTE_      = np.delete(UTE_, np.linspace(0, ind-1, ind).astype(int), axis = 1)
#     UTE_[3,0] = 3

#### Initialize output and auxiliary variables
t_perp_M        = t_perp*np.ones((N_w,N_k))
epsilon_k_M     = epsilon_k*np.ones((N_w,1))

if first_kernel != 0 or clear_output:
    rho0  = []
    op    = []
    cond  = []
    Teff  = []
    Tefi  = []
    J     = []
    JJ    = []
    JH    = []
    n_tot = []
    gap   = []
    garp  = []
    Z     = []
    tpef  = []
    Zp    = []
    tpeff = []
    Zpp   = []
    tpefp = []
    tsc   = []
    
if cache:
    c_Sigma_U_p_R_w = np.zeros((len(UTE_[0,:]), N_w),dtype = 'complex_')
    c_Sigma_U_m_R_w = np.zeros((len(UTE_[0,:]), N_w),dtype = 'complex_')
    c_Sigma_U_p_K_w = np.zeros((len(UTE_[0,:]), N_w),dtype = 'complex_')
    c_Sigma_U_m_K_w = np.zeros((len(UTE_[0,:]), N_w),dtype = 'complex_')
    DOS             = np.zeros((len(UTE_[0,:]), N_w),dtype = 'complex_')
    distro          = np.zeros((len(UTE_[0,:]), N_w),dtype = 'complex_')

    #### Saving parameters
if save == True:
    os.makedirs('Saved/'+ya)
    np.savez_compressed('Saved/'+ya+'/param', t_perp=t_perp, t_para=t_para, Gamma=Gamma,
        w_=w_, k_=k_, UTE_=UTE_)


#### U,T,E loop
n_UTE = 1

for U, T, E, kernel in UTE_.T:
    
    beta = 1/T
    wwE  = np.int32(np.floor(E/dw))
    if wwE > N_w-1: wwE = N_w - 1
    
    if E > 0 and E < 2*dw:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! dw is too large for E !! Increase N_w !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    
    #### Initialization
    # RODO: (1) begin with a guess for Sigma
    if kernel == 1:                         # RODO: Initial guess for Sigma is zero?
        Sigma_U_p_R_w = zero
        Sigma_U_m_R_w = zero 
        Sigma_U_p_K_w = zero 
        Sigma_U_m_K_w = zero 
        F_rp_R_wk     = np.zeros((N_w,N_k))
        F_rm_R_wk     = np.zeros((N_w,N_k))
        F_lp_R_wk     = np.zeros((N_w,N_k))
        F_lm_R_wk     = np.zeros((N_w,N_k))      
        F_rp_K_wk     = np.zeros((N_w,N_k))
        F_rm_K_wk     = np.zeros((N_w,N_k))
        F_lp_K_wk     = np.zeros((N_w,N_k))
        F_lm_K_wk     = np.zeros((N_w,N_k))

    if kernel == 2:
        if t_perp == 0:
            Sigma_U_p_R_w = (U**2/4)*wPV_
            Sigma_U_m_R_w = (U**2/4)*wPV_
            # Sigma_U_p_R_w = (U**2/4)/(w_-0.0001)
            # Sigma_U_m_R_w = (U**2/4)/(w_+0.0001)    
        else:
            Sigma_U_p_R_w = (U**2/4)/(w_-3*t_perp)
            Sigma_U_m_R_w = (U**2/4)/(w_+3*t_perp)
        Sigma_U_p_K_w = 0j * Gamma * np.tanh(0.5*beta*w_) 
        Sigma_U_m_K_w = 0j * Gamma * np.tanh(0.5*beta*w_)
        
        # Sigma_U_p_R_w = (U**2/4)/(w_-3*t_perp - 1j*Gamma)
        # Sigma_U_m_R_w = (U**2/4)/(w_+3*t_perp - 1j*Gamma)
        # Sigma_U_p_K_w = 2j * Sigma_U_p_R_w.imag * np.tanh(0.5*beta*w_) 
        # Sigma_U_m_K_w = 2j * Sigma_U_m_R_w.imag * np.tanh(0.5*beta*w_) 
        
        F_rp_R_wk     = np.zeros((N_w,N_k))
        F_rm_R_wk     = np.zeros((N_w,N_k))
        F_lp_R_wk     = np.zeros((N_w,N_k))
        F_lm_R_wk     = np.zeros((N_w,N_k))      
        F_rp_K_wk     = np.zeros((N_w,N_k))
        F_rm_K_wk     = np.zeros((N_w,N_k))
        F_lp_K_wk     = np.zeros((N_w,N_k))
        F_lm_K_wk     = np.zeros((N_w,N_k))
    
    if kernel == 3:
        point         = '/'+str(ind)
        seed          = np.load('Saved/' + rundate + point + '/fcts.npz') # RODO: Seed function in external file?
        param         = np.load('Saved/'+ rundate + '/param.npz')
        
        if not np.array_equal(w_,param['w_']) or not np.array_equal(k_,param['k_']):
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("!!! Seed meshes incompatible with current settings !!!")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                break
        
        Sigma_U_p_R_w = seed['Sigma_U_p_R_w']
        Sigma_U_m_R_w = seed['Sigma_U_m_R_w'] 
        Sigma_U_p_K_w = seed['Sigma_U_p_K_w'] 
        Sigma_U_m_K_w = seed['Sigma_U_m_K_w'] 
        F_rp_R_wk     = seed['F_rp_R_wk']
        F_rm_R_wk     = seed['F_rm_R_wk']
        F_lp_R_wk     = seed['F_lp_R_wk']
        F_lm_R_wk     = seed['F_lm_R_wk']    
        F_rp_K_wk     = seed['F_rp_K_wk']
        F_rm_K_wk     = seed['F_rm_K_wk']
        F_lp_K_wk     = seed['F_lp_K_wk']
        F_lm_K_wk     = seed['F_lm_K_wk']

    #### Dissipation
    Sigma_th_R_w = zero - 1j*Gamma                      # RODO: Retarded Sigma bath?
    Sigma_th_K_w = 2j * (-Gamma) * np.tanh(0.5*beta*w_) # RODO: Keldysh Sigma bath?
    
    #### DMFT loop  
    vv       = 0
    dmft_err = 1
    n_dmft   = 1
    while (dmft_err > tolerance_dmft_iter or n_dmft < 2) and n_dmft < n_dmft_max:
    # while n_dmft < n_dmft_max/2:
        
        #### Update total self-energies
        Sigma_p_R_w = Sigma_U_p_R_w + Sigma_th_R_w  # RODO: Impurity Sigma + bath Sigma?
        Sigma_m_R_w = Sigma_U_m_R_w + Sigma_th_R_w
        Sigma_p_K_w = Sigma_U_p_K_w + Sigma_th_K_w
        Sigma_m_K_w = Sigma_U_m_K_w + Sigma_th_K_w
        
        w_S_p_R         = np.transpose((w_-Sigma_p_R_w)*np.ones((N_k,N_w))) #Used to compute F faster later
        w_S_m_R         = np.transpose((w_-Sigma_m_R_w)*np.ones((N_k,N_w)))
        Sigma_p_K_w_M   = np.transpose(Sigma_p_K_w*np.ones((N_k,N_w)))
        Sigma_m_K_w_M   = np.transpose(Sigma_m_K_w*np.ones((N_k,N_w)))
        
        #### Compute F via recurrence
        v        = 0
        F_err    = 1
        n_iter_F = 1
        
        while (F_err > tolerance_F_iter or n_iter_F < 2) and n_iter_F < n_iter_F_max:
            if wwE == 0 and not force_E_code:
                F_rp_R_wk = 1./(w_S_p_R - epsilon_k_M + t_perp_M - t_para**2*F_rp_R_wk)
                F_rm_R_wk = 1./(w_S_m_R - epsilon_k_M - t_perp_M - t_para**2*F_rm_R_wk)
                F_lp_R_wk = 1./(w_S_p_R - epsilon_k_M + t_perp_M - t_para**2*F_lp_R_wk)
                F_lm_R_wk = 1./(w_S_m_R - epsilon_k_M - t_perp_M - t_para**2*F_lm_R_wk)
            else:
                F_rp_R_p = np.append(F_rp_R_wk[wwE:N_w],1*F_rp_R_wk[-1]*np.ones((wwE,N_k)),0) # auxiliary "F(w+E,k)"
                F_rm_R_p = np.append(F_rm_R_wk[wwE:N_w],1*F_rm_R_wk[-1]*np.ones((wwE,N_k)),0) # (w,k) matrix form to avoid nesting loops
                F_lp_R_m = np.append(1*F_lp_R_wk[0]*np.ones((wwE,N_k)),F_lp_R_wk[0:N_w-wwE],0)
                F_lm_R_m = np.append(1*F_lm_R_wk[0]*np.ones((wwE,N_k)),F_lm_R_wk[0:N_w-wwE],0)
                
                F_rp_K_p = np.append(F_rp_K_wk[wwE:N_w],1*F_rp_K_wk[-1]*np.ones((wwE,N_k)),0) # auxiliary
                F_rm_K_p = np.append(F_rm_K_wk[wwE:N_w],1*F_rm_K_wk[-1]*np.ones((wwE,N_k)),0)
                F_lp_K_m = np.append(1*F_lp_K_wk[0]*np.ones((wwE,N_k)),F_lp_K_wk[0:N_w-wwE],0)
                F_lm_K_m = np.append(1*F_lm_K_wk[0]*np.ones((wwE,N_k)),F_lm_K_wk[0:N_w-wwE],0)                
                
                F_rp_R_wk = 1./(w_S_p_R - epsilon_k_M + t_perp_M - t_para**2*F_rp_R_p)
                F_rm_R_wk = 1./(w_S_m_R - epsilon_k_M - t_perp_M - t_para**2*F_rm_R_p)
                F_lp_R_wk = 1./(w_S_p_R - epsilon_k_M + t_perp_M - t_para**2*F_lp_R_m)
                F_lm_R_wk = 1./(w_S_m_R - epsilon_k_M - t_perp_M - t_para**2*F_lm_R_m)
                
                F_rp_K_wk = abs(F_rp_R_wk)**2 * (Sigma_p_K_w_M + t_para**2*F_rp_K_p)
                F_rm_K_wk = abs(F_rm_R_wk)**2 * (Sigma_m_K_w_M + t_para**2*F_rm_K_p)
                F_lp_K_wk = abs(F_lp_R_wk)**2 * (Sigma_p_K_w_M + t_para**2*F_lp_K_m)
                F_lm_K_wk = abs(F_lm_R_wk)**2 * (Sigma_m_K_w_M + t_para**2*F_lm_K_m)

            #### Measure of 'convergence' of the algo (F)
            v        = np.append(v,(np.linalg.norm(np.imag(F_rp_R_wk),'fro') + np.linalg.norm(np.imag(F_rp_K_wk),'fro')) * np.sqrt(dw*dk))
            F_err    = abs(v[n_iter_F] - v[n_iter_F-1])
            n_iter_F = n_iter_F + 1
            
        #### Total F functions   
        F_rp_R_p = np.append(F_rp_R_wk[wwE:N_w],1*F_rp_R_wk[-1]*np.ones((wwE,N_k)),0) # auxiliary
        F_rm_R_p = np.append(F_rm_R_wk[wwE:N_w],1*F_rm_R_wk[-1]*np.ones((wwE,N_k)),0) 
        F_lp_R_m = np.append(1*F_lp_R_wk[0]*np.ones((wwE,N_k)),F_lp_R_wk[0:N_w-wwE],0)
        F_lm_R_m = np.append(1*F_lm_R_wk[0]*np.ones((wwE,N_k)),F_lm_R_wk[0:N_w-wwE],0)
        F_rp_K_p = np.append(F_rp_K_wk[wwE:N_w],1*F_rp_K_wk[-1]*np.ones((wwE,N_k)),0) # auxiliary
        F_rm_K_p = np.append(F_rm_K_wk[wwE:N_w],1*F_rm_K_wk[-1]*np.ones((wwE,N_k)),0)
        F_lp_K_m = np.append(1*F_lp_K_wk[0]*np.ones((wwE,N_k)),F_lp_K_wk[0:N_w-wwE],0)
        F_lm_K_m = np.append(1*F_lm_K_wk[0]*np.ones((wwE,N_k)),F_lm_K_wk[0:N_w-wwE],0)            
        
        F_p_R_wk = F_rp_R_p + F_lp_R_m
        F_m_R_wk = F_rm_R_p + F_lm_R_m
        
        if wwE != 0 or force_E_code:
            F_rp_K_p = np.append(F_rp_K_wk[wwE:N_w],1*F_rp_K_wk[-1]*np.ones((wwE,N_k)),0) # auxiliary
            F_rm_K_p = np.append(F_rm_K_wk[wwE:N_w],1*F_rm_K_wk[-1]*np.ones((wwE,N_k)),0)
            F_lp_K_m = np.append(1*F_lp_K_wk[0]*np.ones((wwE,N_k)),F_lp_K_wk[0:N_w-wwE],0)
            F_lm_K_m = np.append(1*F_lm_K_wk[0]*np.ones((wwE,N_k)),F_lm_K_wk[0:N_w-wwE],0)
            
            F_p_K_wk = F_rp_K_p + F_lp_K_m
            F_m_K_wk = F_rm_K_p + F_lm_K_m
        else:    
            F_rp_K_wk     = np.transpose(2j * np.tanh(0.5*beta*w_)*np.ones((N_k,N_w))) * np.imag(F_rp_R_wk)
            F_rm_K_wk     = np.transpose(2j * np.tanh(0.5*beta*w_)*np.ones((N_k,N_w))) * np.imag(F_rm_R_wk)
            F_lp_K_wk     = np.transpose(2j * np.tanh(0.5*beta*w_)*np.ones((N_k,N_w))) * np.imag(F_lp_R_wk)
            F_lm_K_wk     = np.transpose(2j * np.tanh(0.5*beta*w_)*np.ones((N_k,N_w))) * np.imag(F_lm_R_wk)
            
        
        #### Lattice GF via Dyson
        G_p_R_wk = 1./(w_S_p_R - epsilon_k_M + t_perp_M - t_para**2*F_p_R_wk) # RODO: Initially F_R_wk is zero?
        G_m_R_wk = 1./(w_S_m_R - epsilon_k_M - t_perp_M - t_para**2*F_m_R_wk)
        
        if wwE == 0 and not force_E_code:
            G_p_K_wk = np.transpose(2j * np.tanh(0.5*beta*w_)*np.ones((N_k,N_w))) * np.imag(G_p_R_wk)
            G_m_K_wk = np.transpose(2j * np.tanh(0.5*beta*w_)*np.ones((N_k,N_w))) * np.imag(G_m_R_wk)
        else:
            G_p_K_wk = abs(G_p_R_wk)**2 * (Sigma_p_K_w_M + t_para**2*F_p_K_wk)       
            G_m_K_wk = abs(G_m_R_wk)**2 * (Sigma_m_K_w_M + t_para**2*F_m_K_wk)
        
        #### Local GF
        # RODO: (2) Sum the momentum-dependent GF over all momenta to determine local GF
        G_loc_p_R_w = np.sum(G_p_R_wk,1)/N_k # RODO: Summing over momenta?
        G_loc_m_R_w = np.sum(G_m_R_wk,1)/N_k
        G_loc_p_K_w = np.sum(G_p_K_wk,1)/N_k
        G_loc_m_K_w = np.sum(G_m_K_wk,1)/N_k
        
        #### Impurity non-interacting GF
        # RODO: (3) Use Dyson equation to determine the effective medium G0
        G_imp_p_R_w = 1./(1./G_loc_p_R_w + Sigma_U_p_R_w)
        G_imp_m_R_w = 1./(1./G_loc_m_R_w + Sigma_U_m_R_w)

        G_imp_p_A_w = np.conj(G_imp_p_R_w)
        G_imp_m_A_w = np.conj(G_imp_m_R_w)
        
        if wwE == 0 and not force_E_code:
            G_imp_p_K_w = 2j * np.tanh(0.5*beta*w_) * np.imag(G_imp_p_R_w)
            G_imp_m_K_w = 2j * np.tanh(0.5*beta*w_) * np.imag(G_imp_m_R_w)
        else:
            G_imp_p_K_w = abs(G_imp_p_R_w)**2 * (G_loc_p_K_w/(abs(G_loc_p_R_w)**2) - Sigma_U_p_K_w)
            G_imp_m_K_w = abs(G_imp_m_R_w)**2 * (G_loc_m_K_w/(abs(G_loc_m_R_w)**2) - Sigma_U_m_K_w) 
        
        # nanp        = np.isnan(G_imp_p_K_w)
        # nanm        = np.isnan(G_imp_m_K_w)
        
        # G_imp_p_K_w[nanp] = G_loc_p_K_w[nanp] - Sigma_U_p_K_w[nanm]
        # G_imp_m_K_w[nanm] = G_loc_m_K_w[nanm] - Sigma_U_m_K_w[nanm]

        #### Mixing with the previous DMFT solution        
        G_imp_p_R_w_new = np.copy(G_imp_p_R_w) 
        G_imp_m_R_w_new = np.copy(G_imp_m_R_w)
        G_imp_p_K_w_new = np.copy(G_imp_p_K_w)
        G_imp_m_K_w_new = np.copy(G_imp_m_K_w)
        
        # if n_UTE > 1:
        #     ratio_ = np.linspace(0, 1, n_dmft_mixing + 2); ratio_ = np.delete(ratio_, 0)
        #     ra     = ratio_[min(n_dmft, len(ratio_)) - 1]
        #     G_imp_p_R_w = ra*G_imp_p_R_w + (1-ra)*G_imp_p_R_w_old
        #     G_imp_m_R_w = ra*G_imp_m_R_w + (1-ra)*G_imp_m_R_w_old
        #     G_imp_p_K_w = ra*G_imp_p_K_w + (1-ra)*G_imp_p_K_w_old
        #     G_imp_m_K_w = ra*G_imp_m_K_w + (1-ra)*G_imp_m_K_w_old

        if n_dmft > 1:
            G_imp_p_R_w = alpha*prev_G_imp_p_R_w + (1-alpha)*G_imp_p_R_w
            G_imp_m_R_w = alpha*prev_G_imp_m_R_w + (1-alpha)*G_imp_m_R_w
            G_imp_p_K_w = alpha*prev_G_imp_p_K_w + (1-alpha)*G_imp_p_K_w
            G_imp_m_K_w = alpha*prev_G_imp_m_K_w + (1-alpha)*G_imp_m_K_w
        
        
        #### IPT impurity solver
            #### Bonding to orbital basis
        G_imp_11_R_w = 0.5*(G_imp_p_R_w + G_imp_m_R_w)    
        G_imp_12_R_w = 0.5*(G_imp_p_R_w - G_imp_m_R_w)    
        G_imp_11_K_w = 0.5*(G_imp_p_K_w + G_imp_m_K_w)    
        G_imp_12_K_w = 0.5*(G_imp_p_K_w - G_imp_m_K_w)
            
            #### First order terms
        # RODO: (4) Impurity solver
        Sigma_U_11_R_w = -0.5j*U*dw/(2*np.pi)*np.sum(G_imp_11_K_w)
        # Sigma_U_11_R_w = -U*dw/(2*np.pi)*np.sum(G_imp_11_K_w)
        Sigma_U_12_R_w = 0.
        
        if wwE != 0 or force_E_code:
            # Sigma_U_11_K_w = -0.5j*U*dw/(2*np.pi)*np.sum(G_imp_11_R_w + G_imp_11_A_w)
            Sigma_U_11_K_w = 0.
            Sigma_U_12_K_w = 0.
        
            #### Second order
        RR11  = (np.convolve(G_imp_11_R_w.imag, G_imp_11_R_w.imag,'same'))*dw/(2*np.pi)
        RR12  = (np.convolve(G_imp_12_R_w.imag, G_imp_12_R_w.imag,'same'))*dw/(2*np.pi)  
        KK11  = (np.convolve(G_imp_11_K_w.imag, G_imp_11_K_w.imag,'same'))*dw/(2*np.pi)
        KK12  = (np.convolve(G_imp_12_K_w.imag, G_imp_12_K_w.imag,'same'))*dw/(2*np.pi)  
        KR11  = (np.convolve(G_imp_11_K_w.imag, G_imp_11_R_w.imag,'same'))*dw/(2*np.pi)
        KR12  = (np.convolve(G_imp_12_K_w.imag, G_imp_12_R_w.imag,'same'))*dw/(2*np.pi)     
        Im11R  = 0.25*dw/(2*np.pi)*U**2*(-2*np.convolve(KR11, np.flip(G_imp_11_K_w.imag), 'same') + np.convolve(KK11 + 4*RR11, np.flip(G_imp_11_R_w.imag), 'same'))
        Im12R  = 0.25*dw/(2*np.pi)*U**2*(-2*np.convolve(KR12, np.flip(G_imp_12_K_w.imag), 'same') + np.convolve(KK12 + 4*RR12, np.flip(G_imp_12_R_w.imag), 'same'))
        
        Im11R = 0.5*(Im11R + np.flip(Im11R))  # Enforces half-filling symmetry
        # Im12R = 0.5*(Im12R + np.flip(Im12R))
        
        # Im11R  = dw/(2*np.pi)*U**2*(np.convolve(-2*KR11+KK11, np.flipud(G_imp_11_K_w.imag), 'same') + np.convolve(RR11, np.flipud(G_imp_11_R_w.imag), 'same'))
        # Im12R  = dw/(2*np.pi)*U**2*(np.convolve(-2*KR12+KK12, np.flipud(G_imp_12_K_w.imag), 'same') + np.convolve(RR12, np.flipud(G_imp_12_R_w.imag), 'same'))

        Sigma_U_11_R_w = Sigma_U_11_R_w + KK_w(Im11R) 
        Sigma_U_12_R_w = Sigma_U_12_R_w + KK_w(Im12R) 
        
        if wwE == 0 and not force_E_code:
            Sigma_U_11_K_w = 2j * np.tanh(0.5*beta*w_) * np.imag(Sigma_U_11_R_w)
            Sigma_U_12_K_w = 2j * np.tanh(0.5*beta*w_) * np.imag(Sigma_U_12_R_w)
        else:        
            Sigma_U_11_K_w += -0.25j*dw/(2*np.pi)*U**2*(np.convolve(KK11+4*RR11, np.flip(G_imp_11_K_w.imag), 'same') - 8*np.convolve(KR11, np.flip(G_imp_11_R_w.imag), 'same'))
            Sigma_U_12_K_w += -0.25j*dw/(2*np.pi)*U**2*(np.convolve(KK12+4*RR12, np.flip(G_imp_12_K_w.imag), 'same') - 8*np.convolve(KR12, np.flip(G_imp_12_R_w.imag), 'same'))   
            # Sigma_U_11_K_w += 1j*dw/(2*np.pi)*U**2*(np.convolve(-KK11-RR11, np.flipud(G_imp_11_K_w.imag), 'same') + 2*np.convolve(KR11, np.flipud(G_imp_11_R_w.imag), 'same'))
            # Sigma_U_12_K_w += 1j*dw/(2*np.pi)*U**2*(np.convolve(-KK12-RR12, np.flipud(G_imp_12_K_w.imag), 'same') + 2*np.convolve(KR12, np.flipud(G_imp_12_R_w.imag), 'same'))
               
        Sigma_U_11_K_w = 0.5*(Sigma_U_11_K_w - np.flip(Sigma_U_11_K_w))      # Half-filling
        # Sigma_U_12_K_w = 0.5*(Sigma_U_12_K_w - np.flip(Sigma_U_12_K_w))
        
            #### Orbital to bonding basis
        Sigma_U_p_R_w = Sigma_U_11_R_w + Sigma_U_12_R_w
        Sigma_U_m_R_w = Sigma_U_11_R_w - Sigma_U_12_R_w
        Sigma_U_p_K_w = Sigma_U_11_K_w + Sigma_U_12_K_w
        Sigma_U_m_K_w = Sigma_U_11_K_w - Sigma_U_12_K_w     
        
        #### Measure of 'convergence' of the algo (DMFT)
        G_loc_R_w        = 0.5*(G_loc_p_R_w + G_loc_m_R_w)
        G_loc_K_w        = 0.5*(G_loc_p_K_w + G_loc_m_K_w)
        vv               = np.append(vv, (np.linalg.norm(np.imag(G_loc_R_w)) + np.linalg.norm(np.imag(G_loc_K_w))) * np.sqrt(dw) + abs(np.imag(G_loc_R_w[w0])))
        # vv               = np.append(vv, (np.linalg.norm(0.5*np.imag(G_loc_p_R_w)) + np.linalg.norm(np.imag(0.5*G_loc_m_R_w)) + np.linalg.norm(0.25*np.imag(G_loc_p_K_w)) + np.linalg.norm(0.25*np.imag(G_loc_m_K_w))) * np.sqrt(dw) + abs(np.imag(G_loc_R_w[w0])))
        
        # if n_dmft > 1:
        #     dmft_err     = np.sqrt(dw*(np.sum((G_loc_R_w.imag - prev_G_loc_R_w.imag)**2) + np.sum((G_loc_K_w.imag - prev_G_loc_K_w.imag)**2)))
        
        err_half_filling = U*abs(np.sum(G_loc_K_w)*dw/(2*np.pi)) 
        dmft_err         = abs(vv[n_dmft] - vv[n_dmft-1]) + err_half_filling

        prev_G_loc_R_w     = np.copy(G_loc_R_w)
        prev_G_loc_K_w     = np.copy(G_loc_K_w)
        prev_G_imp_p_R_w   = np.copy(G_imp_p_R_w)
        prev_G_imp_p_K_w   = np.copy(G_imp_p_K_w)
        prev_G_imp_m_R_w   = np.copy(G_imp_m_R_w)
        prev_G_imp_m_K_w   = np.copy(G_imp_m_K_w)        

        # plt.figure(404)
        # # plt.clf()
        # plt.plot(w_,-np.pi**(-1)*G_loc_R_w.imag)
        # plt.xlim(-10,10)
        # plt.pause(.005)        
        
        # print(f"n_dmft = {n_dmft}")
        # print(f" rho_0_p_loc = {G_loc_p_R_w[w0]}")
        # print(f" rho_0_p_imp = {G_imp_p_R_w[w0]}")
        
        n_dmft           = n_dmft + 1
        #### End of DMFT loop
        
    # if n_dmft == n_dmft_max: break
    
    #### Saving converged functions
    if save == True:
        os.makedirs('Saved/'+ya+'/'+str(n_UTE))
        np.savez_compressed('Saved/'+ya+'/'+str(n_UTE)+'/fcts', G_loc_p_R_w=G_loc_p_R_w, G_loc_m_R_w=G_loc_m_R_w,
             G_loc_p_K_w=G_loc_p_K_w, G_loc_m_K_w=G_loc_m_K_w, Sigma_U_p_R_w=Sigma_U_p_R_w, 
             Sigma_U_m_R_w=Sigma_U_m_R_w, Sigma_U_p_K_w=Sigma_U_p_K_w, Sigma_U_m_K_w=Sigma_U_m_K_w,
             F_rp_R_wk     = F_rp_R_wk,
             F_rm_R_wk     = F_rm_R_wk,        
             F_lp_R_wk     = F_lp_R_wk,
             F_lm_R_wk     = F_lm_R_wk, 
             F_rp_K_wk     = F_rp_K_wk,
             F_rm_K_wk     = F_rm_K_wk,
             F_lp_K_wk     = F_lp_K_wk,
             F_lm_K_wk     = F_lm_K_wk)

    #### Final quantities
    n_tot     = np.append(n_tot, n_dmft)
    
    df        = 0.5-0.25*G_loc_K_w.imag/G_loc_R_w.imag
    rho       = -(1/np.pi)*np.imag(G_loc_R_w)
    
    rho0      = np.append(rho0,rho[w0])
    op        = np.append(op,np.sum(-(1/np.pi)*np.imag(G_loc_R_w)*wPV_*wPV_)*dw)
    
    td        = (6/np.pi**2)*w_*(df-np.heaviside(-w_,0.5))
    Tefi      = np.append(Tefi,np.sqrt(np.sum(td)*dw))

    w_c       = 0.3
    wwC       = np.int32(np.floor(w_c/dw))
    Ti, par   = curve_fit(fdd,w_[w0-wwC:w0+wwC],df[w0-wwC:w0+wwC])     
    Teff      = np.append(Teff,Ti)
    
    # gra       = np.gradient(rho)
    # garp      = np.append(garp,w_[w0+fpeaks(-gra[w0:-1].real)[0][0].astype(int)])
    # gap       = np.append(gap,w_[w0+fpeaks(rho[w0:-1].real)[0][0].astype(int)])
    
    # Sigma_11_R_w = 0.5*(Sigma_p_R_w + Sigma_m_R_w)
    # Sigma_12_R_w = 0.5*(Sigma_p_R_w - Sigma_m_R_w)
    # dSg          = np.gradient(Sigma_11_R_w.real,dw)
    
    # zi           = 1/(1-dSg[w0])
    # Z            = np.append(Z,zi)
    # tpef         = np.append(tpef,(t_perp-Sigma_12_R_w[w0].real)*zi)
    
    # wp           = w0 + np.ceil(t_perp/dw).astype(int)
    # zp           = 1/(1-dSg[wp])
    # Zp           = np.append(Zp,zp)
    # tpeff        = np.append(tpeff,(t_perp-Sigma_12_R_w[wp].real)*zp)

    # wpp          = w0 + np.ceil(gap[-1]/dw).astype(int)
    # zpp          = 1/(1-dSg[wpp])
    # Zpp          = np.append(Zpp,zpp)
    # tpefp        = np.append(tpefp,(t_perp-Sigma_12_R_w[wpp].real)*zpp)

    # tt = np.array([tpeff[-1],])
    # ee = 1
    # wt = []
    # zt = []
    # while ee > 1e-8 and np.size(tt) < 1e5:
    #     wt     = w0 + np.ceil(tt[-1]/dw).astype(int)
    #     zt     = 1/(1-dSg[wt])
    #     tt     = np.append(tt,(t_perp-Sigma_12_R_w[wt].real)*zt)
    #     ee     = np.abs(tt[-1] - tt[-2])
    #     tt[-1] = 0.5*tt[-1] + 0.5*tt[-2]
    # if np.size(tt) != 1e5: tsc = np.append(tsc,tt[-1])
    # else: tsc = np.append(tsc,0)
        #### Current
    G_loc_pm_w = 0.5 * (G_loc_K_w - G_loc_R_w + np.conj(G_loc_R_w))                                               
                                                     
    F_r_R_loc  = 0.5 * np.sum(F_rp_R_p + F_rm_R_p, 1)/N_k
    F_l_R_loc  = 0.5 * np.sum(F_lp_R_m + F_lm_R_m, 1)/N_k
    F_r_K_loc  = 0.5 * np.sum(F_rp_K_p + F_rm_K_p, 1)/N_k
    F_l_K_loc  = 0.5 * np.sum(F_lp_K_m + F_lm_K_m, 1)/N_k

    F_r_pm_loc = 0.5 * (F_r_K_loc - F_r_R_loc + np.conj(F_r_R_loc))       
    F_l_pm_loc = 0.5 * (F_l_K_loc - F_l_R_loc + np.conj(F_l_R_loc))  

    jx         = G_loc_pm_w * np.conj(F_r_R_loc - F_l_R_loc) + G_loc_R_w * (F_r_pm_loc - F_l_pm_loc)
    Re_jx      = jx.real

    Ji         = -t_para**2 * (dw/(2*np.pi)) * np.sum(Re_jx)
    J          = np.append(J,Ji)
    
    G_pm_wk    = 0.25 * (G_p_K_wk + G_m_K_wk - G_p_R_wk - G_m_R_wk + np.conj(G_p_R_wk) + np.conj(G_m_R_wk))
    
    F_r_R_wk   = 0.5 * (F_rp_R_p + F_rm_R_p)
    F_l_R_wk   = 0.5 * (F_lp_R_m + F_lm_R_m)
    F_r_K_wk   = 0.5 * (F_rp_K_p + F_rm_K_p)
    F_l_K_wk   = 0.5 * (F_lp_K_m + F_lm_K_m)
    
    F_r_pm_wk = 0.5 * (F_r_K_wk - F_r_R_wk + np.conj(F_r_R_wk))       
    F_l_pm_wk = 0.5 * (F_l_K_wk - F_l_R_wk + np.conj(F_l_R_wk))  
    
    jjx       = np.sum(G_pm_wk * np.conj(F_r_R_wk - F_l_R_wk) + 0.5*(G_p_R_wk + G_m_R_wk) * (F_r_pm_wk - F_l_pm_wk), 1)/N_k
    Re_jjx    = jjx.real

    JJi       = -t_para**2 * (dw/(2*np.pi)) * np.sum(Re_jjx)
    JJ        = np.append(JJ,JJi)    
    
    if E == 0:
        JH = np.append(JH,0)
    else:
        jh = 2*Gamma*dw/E * np.sum(w_*rho*(df-fdd(w_,T)))
        JH = np.append(JH,jh)
    
    if op[-1] > 10:
        cond = np.append(cond,1)
    else:
        cond = np.append(cond,0)
    
    G_imp_p_R_w_old = np.copy(G_imp_p_R_w_new) 
    G_imp_m_R_w_old = np.copy(G_imp_m_R_w_new)
    G_imp_p_K_w_old = np.copy(G_imp_p_K_w_new)
    G_imp_m_K_w_old = np.copy(G_imp_m_K_w_new)
    
    if cache:
        c_Sigma_U_p_R_w[n_UTE-1,:] = np.copy(Sigma_U_p_R_w)
        c_Sigma_U_m_R_w[n_UTE-1,:] = np.copy(Sigma_U_m_R_w)
        c_Sigma_U_p_K_w[n_UTE-1,:] = np.copy(Sigma_U_p_K_w)
        c_Sigma_U_m_K_w[n_UTE-1,:] = np.copy(Sigma_U_m_K_w)
        DOS[n_UTE-1,:]             = np.copy(rho)
        distro[n_UTE-1,:]          = np.copy(df)
    
    #### Figures
    
    plt.figure(11)
    plt.plot(w_,rho)
    plt.xlim(w_min*0-5,5+0*w_max)
    # plt.xlim(-2,2)
    # plt.ylim(-0.0001,0.005)
    plt.pause(.005)
    
    # plt.figure(2)
    # plt.clf()
    # plt.plot(UTE_[0,0:n_UTE],rho0,"-o")
    # plt.xlim(min(U_),max(U_))
    # plt.ylim(0,max(rho0)+0.05)
    # plt.pause(.005)
    
    # plt.figure(3)
    # plt.clf()
    # plt.plot(UTE_[0,0:n_UTE],op,"-o")
    # plt.xlim(U_min,max(U_))
    # plt.ylim(0,max(op)+10)
    # plt.pause(.005)
    
    # plt.figure(4)
    # plt.plot(w_,df)
    # plt.xlim(-0.9,0.9)
    # plt.pause(.005)
    
    # plt.figure(5)
    # # plt.clf()
    # plt.plot(UTE_[0,0:n_UTE-1],J,"-o",UTE_[0,0:n_UTE-1],JJ,"-o",UTE_[0,0:n_UTE-1],JH,"-o")
    # plt.xlim(U_min,U_max)
    # plt.pause(.005)
    
    # for ii in np.linspace(0,len(op)-1,len(op)).astype(int):
    #     plt.figure(6)
    #     if op[ii]>10:
    #         plt.plot(UTE_[0,ii],UTE_[2,ii],"bo")
    #     if op[ii]<10:
    #         plt.plot(UTE_[0,ii],UTE_[2,ii],"rx")
    #     plt.pause(.005)

    # plt.figure(7)
    # plt.clf()
    # plt.plot(E_,np.sqrt(3)*E_/(4*Gamma),UTE_[2,0:n_UTE],Teff,"o",UTE_[2,0:n_UTE],Tefi,"o")
    # #plt.ylim(-0.075,0.5)
    # plt.pause(.005)
    
    print(f"U ={U:6.3f}, T ={T:6.3f}, E ={E:6.3f}, n_dmft = {n_dmft:3.0f}, n_iter_F ={n_iter_F-1:3.0f}, rho_0 ={-(1/np.pi)*np.imag(G_loc_R_w[w0]):6.3f}, t_para ={t_para:6.3f}, t_perp ={t_perp:6.3f}")
    # print("---> Runtime: %s seconds" % round((time()-ppio),2))

    n_UTE += 1

#### Saving overall quantities
if save == True:
    np.savez_compressed('Saved/'+ya+'/qtts', op=op, rho0=rho0, Teff=Teff, Tefi=Tefi, n_tot=n_tot)

print("---> Runtime: %s seconds" % round((time()-ppio),2))

# jj=5
# for jj in np.linspace(0,8,9):
#     plt.figure(49)
#     plt.plot(UTE_[0,np.int(jj*65):np.int((jj+1)*65)],J[np.int(jj*65):np.int((jj+1)*65)],"-o")
#     plt.pause(.005)

# Uu=UTE_[0,:]
# for ii in np.linspace(0,len(op)-1,len(op)).astype(int):
#     plt.figure(61)
#     if n_tot[ii]==n_dmft_max:
#         if Uu[ii]>Uu[ii-1]:
#             plt.plot(UTE_[0,ii],UTE_[2,ii],"bo")
#         if Uu[ii-1]>Uu[ii]:
#             plt.plot(UTE_[0,ii],UTE_[2,ii],"rx")
            
# Tci = []; Uc = []; Ec = []; Tc = []; im=[]
# for o in np.linspace(0,len(op)-1,len(op)).astype(int):
#     if op[o-1]>10 and op[o]<10:
#         Tc  = np.append(Tc,Teff[o-1])
#         Tci = np.append(Tci,Tefi[o-1])
#         Uc  = np.append(Uc,UTE_[0,o-1])
#         Ec  = np.append(Ec,UTE_[2,o-1])
#         im  = np.append(im,o-1)

# ITci = []; IUc = []; IEc = []; ITc = []; it=[]; irh = []
# for o in np.linspace(0,len(op)-1,len(op)).astype(int):
#     if op[o]<50 and op[o+1]>50:
#         ITc  = np.append(ITc,Teff[o])
#         ITci = np.append(ITci,Tefi[o])
#         IUc  = np.append(IUc,UTE_[0,o])
#         IEc  = np.append(IEc,UTE_[2,o])
#         it   = np.append(it,o)
#         irh  = np.append(irh,rho0[o])
        
# for ii in np.linspace(0,len(op)-1,len(op)).astype(int):
#     plt.figure(42)
#     if op[ii]>50:
#         plt.plot(UTE_[0,ii],Tefi[ii],"bo")
#     if op[ii]<50:
#         plt.plot(UTE_[0,ii],Tefi[ii],"rx")

# runfile('/home/diaz/Dropbox/Códigos/Python/moyal_nx_grad.py', wdir='/home/diaz/Dropbox/Códigos/Python', current_namespace=True)