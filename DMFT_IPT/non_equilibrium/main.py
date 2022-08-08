from parameters import *
from threading import Thread
import matplotlib.pylab as plt
import dmft 

''' Variables '''

thrDos = None       # Thread used for computing and plotting the DOS

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
  
# Thread for computing and plotting the density of states  
def trgtDos(G_pp_R, G_dd_R, beta, U):                                         
    myPlot(wArr, [-G_pp_R.imag/np.pi],               
           'DOS_p_beta='+str(beta)+'_U='+str(U), 
           r'$\omega$', r'$\rho(\omega)$')
            
    myPlot(wArr, [-G_dd_R.imag/np.pi],                
           'DOS_d_beta='+str(beta)+'_U='+str(U), 
           r'$\omega$', r'$\rho(\omega)$')    
           
    myPlot(wArr, [-G_pp_R.imag/np.pi, -G_dd_R.imag/np.pi],                
           'DOS_beta='+str(beta)+'_U='+str(U), 
           r'$\omega$', r'$\rho(\omega)$', ['p electrons', 'd electrons'])  
    
''' Main '''

for U in UArr:
    for beta in betaArr:
        G_pp_R, G_dd_R = dmft.main(beta, U,     
                                   Sig_U_RArr, Sig_U_KArr,
                                   Sig_B_RArr, Sig_B_KArr)
        
        # Create and execute thread for computing and plotting the DOS          
        if thrDos is not None:
            thrDos.join()                                                    # Wait for old thread to finish
        thrDos = Thread(target=trgtDos, args=(G_pp_R, G_dd_R, beta, U))      # Create new thread
        thrDos.start()                                                       # Start the thread
        
thrDos.join()   
