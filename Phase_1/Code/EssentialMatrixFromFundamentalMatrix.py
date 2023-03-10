import numpy as np

def getEssentialMatrix(K, F):
    E = K.T.dot(F).dot(K)
    U,s,V = np.linalg.svd(E)
    s = [1,1,0]
    E_corrected = np.dot(U,np.dot(np.diag(s),V))
    return E_corrected