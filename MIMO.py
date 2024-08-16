import numpy as np
import numpy.linalg as nl
import pdb
from scipy.stats import norm

def DFTmat(K):
    kx, lx = np.meshgrid(np.arange(K), np.arange(K))
    omega = np.exp(-2*np.pi*1j/K)
    dftmtx = np.power(omega,kx*lx)
    return dftmtx

def Q(x):
    return 1-norm.cdf(x);


def H(G):
    return np.conj(np.transpose(G));


def AHA(A):
    return np.matmul(H(A),A)

def AAH(A):
    return np.matmul(A,H(A))

def mimo_capacity(Hmat, TXcov, Ncov):
    r, c = np.shape(Hmat);
    inLD = np.identity(r) + nl.multi_dot([nl.inv(Ncov),Hmat,TXcov,H(Hmat)]);
    C = np.log2(nl.det(inLD));
    return C

def QPSK(m,n):
    return ((2*nr.randint(2,size=(m,n))-1)+1j*(2*nr.randint(2,size=(m,n))-1))/np.sqrt(2);

def OPT_CAP_MIMO(Heff,SNR):
    U, S, V = nl.svd(Heff, full_matrices=False)
    t = len(S);
    CAP = 0;
    while not CAP:
        onebylam = (SNR + sum(1/S[0:t]**2))/t;
        if  onebylam - 1/S[t-1]**2 >= 0:
            optP = onebylam - 1/S[0:t]**2;
            CAP = sum(np.log2(1+ S[0:t]**2 * optP));
        elif onebylam - 1/S[t-1]**2 < 0:
            t = t-1;            
    return CAP

def EQ_CAP_MIMO(Heff,SNR):
    U, S, V = nl.svd(Heff, full_matrices=False)
    t = len(S);
    CAP = sum(np.log2(1+ S[0:t]**2 * SNR/t));
    return CAP


def MPAM_DECODER(EqSym,M):
    DecSym = np.round((EqSym+M-1)/2);
    DecSym[np.where(DecSym<0)] = 0;
    DecSym[np.where(DecSym>(M-1))] = M-1      
    return DecSym

def MQAM_DECODER(EqSym,M):
    sqM = np.int(np.sqrt(M));
    DecSym = np.round((EqSym+sqM-1)/2);
    DecSym[np.where(DecSym<0)]=0;
    DecSym[np.where(DecSym>(sqM-1))]=sqM-1      
    return DecSym