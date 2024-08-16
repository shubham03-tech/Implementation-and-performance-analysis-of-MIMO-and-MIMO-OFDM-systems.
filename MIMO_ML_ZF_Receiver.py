
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as nr
import numpy.linalg as nl

blockLength = 100;
nBlocks = 10000;
r = 2;
t = 2;
EbdB = np.arange(1.0,19.1,2.0);
Eb = 10**(EbdB/10);
No = 1;
SNR = 2*Eb/No;
SNRdB = 10*np.log10(SNR);
BER_ZF = np.zeros(len(EbdB));
BER_ML = np.zeros(len(EbdB));
S = np.array([[1,1,-1,-1],[1,-1,1,-1]]);
MLout = np.zeros((t,blockLength));

for blk in range(nBlocks):
    H = (nr.normal(0.0, 1.0,(r,t))+1j*nr.normal(0.0, 1.0,(r,t)))/np.sqrt(2);
    noise = nr.normal(0.0, np.sqrt(No/2), (r,blockLength))+ \
    1j*nr.normal(0.0, np.sqrt(No/2), (r,blockLength));
    BitsI = nr.randint(2,size=(t,blockLength));
    Sym = (2*BitsI-1);

    for K in range(len(SNRdB)):
        TxSym = np.sqrt(Eb[K])*Sym;
        RxSym = np.matmul(H,TxSym) + noise;
        # ZF Receiver
        ZFRx = nl.pinv(H);
        ZFout = np.matmul(ZFRx,RxSym);
        DecBitsI_ZF = (np.real(ZFout)>0);
        BER_ZF[K] = BER_ZF[K] + np.sum(DecBitsI_ZF != BitsI) ;
        # ML Receiver
        TxS = np.sqrt(Eb[K])*Sym
        for vx in range (blockLength):
            decIdx = np.argmin(np.sum(np.abs(RxSym[:,vx:vx+1]-np.matmul(H,TxS))**2,axis=0));
            MLout[:,vx:vx+1]=TxS[:,decIdx:decIdx+1]
        DecBitsI_ML = (np.real(MLout)>0)
        BER_ML[K] = BER_ML[K] + np.sum(DecBitsI_ML!=BitsI)# ML Receiver
        


BER_ZF = BER_ZF/blockLength/nBlocks/t;
BER_ML = BER_ML/blockLength/nBlocks/t;


plt.yscale('log')
plt.plot(SNRdB, BER_ZF,'g-s');
plt.plot(SNRdB, BER_ML,'b-.s');
plt.grid(1,which='both')
plt.suptitle('BER for MIMO Channel ML and ZF')
plt.legend(["ZF","ML"], loc ="lower left");
plt.xlabel('SNR (dB)')
plt.ylabel('BER') 

