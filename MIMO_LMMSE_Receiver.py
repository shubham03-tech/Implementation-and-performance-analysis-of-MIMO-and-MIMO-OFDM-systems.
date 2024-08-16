# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 12:41:59 2022

@author: ramak
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.random as nr
import numpy.linalg as nl
from scipy.special import comb
import MIMO

blockLength = 1000;
nBlocks = 10000;
r = 2;
t = 2;
EbdB = np.arange(1.0,33.1,4.0);
Eb = 10**(EbdB/10);
No = 1;
Es = 2*Eb;
SNR = Es/No;
SNRdB = 10*np.log10(SNR);
BER_LMMSE = np.zeros(len(EbdB));
BERt = np.zeros(len(EbdB));


for blk in range(nBlocks):   
    H = (nr.normal(0.0, 1.0,(r,t))+1j*nr.normal(0.0, 1.0,(r,t)))/np.sqrt(2);
    noise = nr.normal(0.0, np.sqrt(No/2), (r,blockLength))+1j*nr.normal(0.0, np.sqrt(No/2), (r,blockLength));
    BitsI = nr.randint(2,size=(t,blockLength));
    BitsQ = nr.randint(2,size=(t,blockLength));
    Sym = (2*BitsI-1)+1j*(2*BitsQ-1);
    
    for K in range(len(SNRdB)):
        
        TxSym = np.sqrt(Eb[K])*Sym;
        RxSym = np.matmul(H,TxSym);
     # LMMSE Receiver   
        LMMSERx = np.matmul(nl.inv(MIMO.AHA(H)+No*np.identity(t)/Es[K]),MIMO.H(H));
        LMMSEout = np.matmul(LMMSERx,RxSym);
        DecBitsI_LMMSE = (np.real(LMMSEout)>0);
        DecBitsQ_LMMSE = (np.imag(LMMSEout)>0);
        # Bit error for LMMSE Receiver
        BER_LMMSE[K] = BER_LMMSE[K] + np.sum(DecBitsI_LMMSE!=BitsI) + np.sum(DecBitsQ_LMMSE!=BitsQ);
        

BER_LMMSE = BER_LMMSE/blockLength/nBlocks/2/t;
L=r-t+1;    BERt = comb(2*L-1, L)/2**L/SNR**L;
plt.yscale('log')

plt.plot(SNRdB, BER_LMMSE,'b-.s');
plt.plot(SNRdB, BERt,'r-.o');
plt.grid(1,which='both')
plt.suptitle('BER for MIMO_LMMSE Channel')
plt.legend(["LMMSE", "Theory"], loc ="lower left");
plt.xlabel('SNR (dB)')
plt.ylabel('BER')