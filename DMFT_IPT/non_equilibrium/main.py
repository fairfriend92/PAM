from parameters import *
from threading import Thread
import time

# Compute the inverse of the retarded component 
# of the RHS or the LHS semi-infinite chain
def getF_R_inv(G_00_R_inv, sign, 
               resultList,      # List where threads store return values
               threadIdx):      # Index of the current thread
               
    # Last element of the chain
    F_R_inv = G_00_R_inv + sign*L*E
    
    # Continued fraction
    for x in np.arange(sign*(L-1.), 0.):
        F_R_inv = G_00_R_inv + x*E - t**2/F_R_inv
        
    resultList[threadIdx] = F_R_inv

tic = time.process_time()

# Broadcast arrays to the appropriate shapes
eArr        = eArr*np.ones([N_w, N_e])
wArr        = np.transpose(wArr*np.ones([N_e, N_w]))
Sig_U_RArr  = np.transpose(Sig_U_RArr*np.ones([N_e, N_w]))
    
# Inverse of the retarded component of the central site dd Green function: G(w, e)_00^-1
G_00_R_inv = 1.j*wArr + mu - eArr + 1.j*Gamma + - V**2/(1.j*wArr + mu - e_d - Sig_U_RArr)

# Create threads 
resultList = 2*[None]
thread_1 = Thread(target=getF_R_inv,args=(G_00_R_inv, +1, resultList, 0))
thread_2 = Thread(target=getF_R_inv,args=(G_00_R_inv, -1, resultList, 1))

# Start threads
thread_1.start()
thread_2.start()

# Wait for job completion 
thread_1.join()
thread_2.join()

# Inverse of retarded component of the semi-infinite chains: F(w, e)_+-^-1
F_rhs_R_inv = resultList[0]
F_lhs_R_inv = resultList[1]

# Inverse of the retarded component of the dd Green function: G(w, e)_dd^-1
G_dd_R_inv = G_00_R_inv - t**2*(1./F_rhs_R_inv + 1./F_lhs_R_inv)

toc = time.process_time()

print(toc-tic)