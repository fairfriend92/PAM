from parameters import *
from threading import Thread
import matplotlib.pylab as plt
import dmft 

''' Functions '''

# Make figure
def myPlot(x, yList, 
           fileName,
           xLabel='', yLabel='', labelList=None,
           xLimLeft=None, xLimRight=None,
           yLimBott=None, yLimTop=None):
            
    fig, ax = plt.subplots(figsize=(12, 8))
    size = 24 
    ax.set(xlabel=xLabel, ylabel=yLabel)
    ax.xaxis.label.set_size(size)
    ax.yaxis.label.set_size(size)
    if xLimLeft is not None:
        ax.set_xlim(left=xLimLeft)
    if xLimRight is not None:
        ax.set_xlim(right=xLimRight) 
    if yLimBott is not None:
        ax.set_ylim(bottom=yLimBott)
    if yLimTop is not None:
        ax.set_ylim(top=yLimTop)
    if labelList is None:
        labelList = [yLabel for i in yList]
    for y, label in zip(yList, labelList):
        ax.plot(x, y, label=label, marker='s')
        ax.tick_params(axis='both', which='both', labelsize=size, length=int(size/6))
    fig.tight_layout()
    plt.legend(prop={'size': size})
    plt.savefig('./figures/'+fileName+'.pdf') 
    plt.close()
    
  
# Thread for computing and plotting 
# the density of states and the electron concentration
def trgtDos_N(G_pp_R, G_dd_R, beta, U, mu):  
    global n_pList, n_dList
    
    dos_p   = -G_pp_R.imag/np.pi
    dos_d   = -G_dd_R.imag/np.pi    
    
    inputStr = '_beta='+str(beta)+'_U='+str(U)+'_mu='+str(mu)

    # TODO: Save the electron concentration for all betas 
    if beta == minBeta:
        n_pList.append(np.sum(dos_p/(1. + np.exp(beta*wArr))*dw))
        n_dList.append(np.sum(dos_d/(1. + np.exp(beta*wArr))*dw))
        if mu == maxMu:
            myPlot(muArr, [n_pList, n_dList],                
                   'n'+inputStr, 
                    r'$\omega$', 'electron concentration', [r'$n_p$', r'$n_d$'])  
            n_pList = []
            n_dList = []
    
    '''
    myPlot(wArr, [dos_p],               
           'DOS_p'+inputStr, 
           r'$\omega$', r'$\rho(\omega)$')
            
    myPlot(wArr, [dos_d],                
           'DOS_d'+inputStr, 
           r'$\omega$', r'$\rho(\omega)$')
    '''
           
    myPlot(wArr, [dos_p, dos_d],                
           'DOS'+inputStr, 
           r'$\omega$', r'$\rho(\omega)$', ['p electrons', 'd electrons'])  
           
''' Variables '''

thrDos_N    = None      # Thread used for computing and plotting the DOS and electron concentration
n_pList     = []        # List of p electron concentrations for different mu
n_dList     = []        # List of d electron concentrations for different mu
    
''' Main '''

for U in UArr:
    for mu in muArr:
        for beta in betaArr:
            print('beta='+str(beta)+' mu='+str(mu)+' U='+str(U))
        
            G_pp_R, G_dd_R = dmft.main(beta, U, mu,     
                                       Sig_U_RArr, Sig_U_KArr,
                                       Sig_B_RArr, Sig_B_KArr)
            
            # Create and execute thread for computing and plotting 
            # the DOS and electron concentration        
            if thrDos_N is not None:
                thrDos_N.join()                                     # Wait for old thread to finish
            thrDos_N = Thread(target=trgtDos_N,                     
                              args=(G_pp_R, G_dd_R, beta, U, mu))   # Create new thread
            thrDos_N.start()                                        # Start the thread

# Wait for the last thread        
thrDos_N.join()   
