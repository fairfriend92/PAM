from parameters import *
from threading import Thread
import time

# Compute the retarded component of the RHS or the LHS semi-infinite chain F(w)_+-
def getF_R(G_00_R_invMtrx):
    # Create threads 
    F_RList = 2*[None]
    thread_1 = Thread(target = trgtF_R, args = (G_00_R_invMtrx, +1, F_RList, 0))
    thread_2 = Thread(target = trgtF_R, args = (G_00_R_invMtrx, -1, F_RList, 1))

    # Start threads
    thread_1.start()
    thread_2.start()

    # Wait for job completion 
    thread_1.join()
    thread_2.join()
    
    return F_RList[0], F_RList[1]

# Thread for computing the retarded component of F(w)_+-
def trgtF_R(G_00_R_invMtrx, # Inverse of the retarded component G(w, k)_00 
            sign,           # Direction of the chain
            F_RList,        # List where threads store return values
            threadIdx):     # Index of the current thread
               
    # Last element of the inverse of the chain F(w, k)_+-^-1
    F_R_invMtrx = G_00_R_invMtrx + sign*L*E
    
    # Continued fraction
    for x in np.arange(sign*(L-1.), 0., -sign):
        F_R_invMtrx = G_00_R_invMtrx + x*E - t**2/F_R_invMtrx
     
    # Take the inverse 
    F_RMtrx = np.reciprocal(F_R_invMtrx)
    
    # Sum over momenta
    F_RArr = np.sum(F_RMtrx, 1)/N_k
     
    F_RList[threadIdx] = F_RArr
    
def getSig_U(g_0_RArr, g_0_KArr, beta, U):
    # Thread for doing convolutions
    convList = 3*[None] # Stores the results
    def trgtMyConv(a, b, convList, threadIdx):
        convList[threadIdx] = np.convolve(a, b, 'same')*dw/(2.*np.pi)
        
    # Create the threads to do the 2 terms convolutions
    thrg_0_KK = Thread(target = trgtMyConv, args = (g_0_KArr, g_0_KArr, convList, 0)) # g_0_K X g_0_K
    thrg_0_RK = Thread(target = trgtMyConv, args = (g_0_RArr, g_0_KArr, convList, 1)) # g_0_R X g_0_K
    thrg_0_RR = Thread(target = trgtMyConv, args = (g_0_RArr, g_0_RArr, convList, 2)) # g_0_R X g_0_R
    
    # Start the threads wait for completion and store results
    thrg_0_KK.start()
    thrg_0_RK.start()
    thrg_0_RR.start()
    
    thrg_0_KK.join()
    thrg_0_RK.join()
    thrg_0_RR.join()
    
    g_0_KKArr = convList[0]
    g_0_RKArr = convList[1]
    g_0_RRArr = convList[2]
    
    # Create the threads to do the 3 terms convolutions 
    thrg_0_RKK = Thread(target = trgtMyConv, args = (np.conj(np.flip(g_0_RArr)), g_0_KKArr, convList, 0))   # g_0_R* X g_0_K X g_0_K
    thrg_0_KRK = Thread(target = trgtMyConv, args = (np.flip(g_0_KArr), g_0_RKArr, convList, 1))            # g_0_K  X g_0_R X g_0_k
    thrg_0_RRR = Thread(target = trgtMyConv, args = (np.conj(np.flip(g_0_RArr)), g_0_RRArr, convList, 2))   # g_0_R* X g_0_R X g_0_R
    
    # Start the threads wait for completion and store results
    thrg_0_RKK.start()    
    thrg_0_KRK.start()
    thrg_0_RRR.start()
    
    thrg_0_RKK.join()
    thrg_0_KRK.join()
    thrg_0_RRR.join()
    
    g_0_RKKArr = convList[0]
    g_0_KRKArr = convList[1]
    g_0_RRRArr = convList[2]
    
    # Compute the retarded component 
    Sig_U_RArr = - (U/2.)**2*(g_0_RKKArr + g_0_RRRArr + 2.*g_0_KRKArr)
    
    # Compute the Keldysh component, assuming that the real part is zero
    Sig_U_KArr = getKeldyshDFT(Sig_U_RArr.imag, beta)

    return Sig_U_RArr, Sig_U_KArr

# Get Keldysh component using the Dissipation Fluctuation Theorem    
def getKeldyshDFT(imag, beta):
    return 1.j*np.tanh(beta*wArr/2.)*imag # TODO: Should there be a factor 2?

def main(beta, U, mu,
         Sig_U_RArr, Sig_U_KArr,
         Sig_B_RArr, Sig_B_KArr,
         G_pp_RArr=np.zeros(N_w), G_dd_RArr=np.zeros(N_w)):
    
    converged   = False             # Flag to check convergence of the dmft loop
    iter        = 0                 # dmft loop iterator
    
    tic = time.process_time()

    # Update the Keldysh component of the bath self-energy 
    Sig_B_KArr  = getKeldyshDFT(Sig_B_RArr.imag, beta)  

    while not converged and iter < maxIter:
        print("dmft loop iteration=" + str(iter)) 
        
        # Broadcast self-energy to the appropriate matrix shapes
        Sig_U_RMtrx  = np.transpose(Sig_U_RArr*np.ones([N_k, N_w]))
        Sig_B_RMtrx  = np.transpose(Sig_B_RArr*np.ones([N_k, N_w]))
              
        # Inverse of the retarded component of the central site p Green function G(w, k)_00
        G_00_R_invMtrx = wMtrx + mu - e_p - e_kMtrx - Sig_B_RMtrx - V**2/(wMtrx + mu - e_d - Sig_U_RMtrx - Sig_B_RMtrx)

        # Thread for summing G(w, k)_00 in the momenta
        G_00List = [None]       # Stores the result of thread  
        def trgtG_00(G_00List):
            G_00List[0] = np.sum(G_00_R_invMtrx, 1)/N_k

        # Start thread    
        thrG_00 = Thread(target = trgtG_00, args = (G_00List,))
        thrG_00.start()

        # Retarded component of the semi-infinite chains F(w)_+-
        F_rhs_RArr, F_lhs_RArr = getF_R(G_00_R_invMtrx)

        # Wait for thread completion and save result
        thrG_00.join()
        G_00_R_invArr = G_00List[0]

        # Store old value
        oldG_pp_RArr = G_pp_RArr.copy()

        # Retarded component of the local p Green function G(w)_pp 
        G_pp_RArr = np.reciprocal(G_00_R_invArr - t**2*(F_rhs_RArr + F_lhs_RArr))
        
        # Retarded component of the impurity Green function g(w)_0
        g_0_RArr = np.reciprocal(wArr + mu - e_d - Sig_B_RArr - V**2/(wArr + mu - e_p - Sig_B_RArr - t**2*G_pp_RArr)) 
        
        # Store old value
        oldG_dd_RArr = G_dd_RArr.copy()

        # Retarded component of the local d Green function G(w)_dd
        G_dd_RArr = np.reciprocal(np.reciprocal(g_0_RArr) - Sig_U_RArr)
        
        print(oldG_dd_RArr[:10])
        print(G_dd_RArr[:10])
                
        # Keldysh component of the impurity Green function g(w)_0
        g_0_KArr = np.power(g_0_RArr, 2)*Sig_B_KArr

        # Coulomb self-energy according to IPT
        Sig_U_RArr, Sig_U_KArr = getSig_U(g_0_RArr, g_0_KArr, beta, U)
        
        # Mix old and new solutions
        G_pp_RArr = mix*oldG_pp_RArr + (1. - mix)*G_pp_RArr
        G_dd_RArr = mix*oldG_dd_RArr + (1. - mix)*G_dd_RArr
                
        # Check convergence
        converged  = np.allclose(oldG_pp_RArr, G_pp_RArr, error)
        converged &= np.allclose(oldG_dd_RArr, G_dd_RArr, error)
        
        iter = iter+1

    toc = time.process_time()
    print(toc-tic)
    
    return G_pp_RArr, G_dd_RArr
