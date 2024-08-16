import numpy as np
import matplotlib.pyplot as plt
import numpy.random as nr
import numpy.linalg as nl
import numpy.fft as nf
from scipy.special import comb


Nsub = 256;
Ncp = int(Nsub/4);
nBlocks = 100;
L = 2;
r = 2;
t = 2;
EbdB = np.arange(1.0,45.1,4.0);
Eb = 10**(EbdB/10);
No = 1;
SNR = 2*Eb/No;
SNRdB = 10*np.log10(Eb/No);
BER = np.zeros(len(EbdB));
BERt = np.zeros(len(EbdB));
RxSamCP = np.zeros((r,L+Nsub+Ncp-1))+1j*np.zeros((r,L+Nsub+Ncp-1));
RxSamples = np.zeros((r,Nsub))+1j*np.zeros((r,Nsub));
RxSym = np.zeros((r,Nsub))+1j*np.zeros((r,Nsub));
Hfft = np.zeros((r,t,Nsub))+1j*np.zeros((r,t,Nsub));
TxSamples = np.zeros((t,Nsub))+1j*np.zeros((t,Nsub));
TxSamCP = np.zeros((t,Nsub+Ncp))+1j*np.zeros((t,Nsub+Ncp));

for blk in range(nBlocks):    
    noise = nr.normal(0.0, np.sqrt(No/2),(r,L+Nsub+Ncp-1))+1j*nr.normal(0.0, np.sqrt(No/2),(r,L+Nsub+Ncp-1));
    H = (nr.normal(0.0, 1.0,(r,t,L))+1j*nr.normal(0.0, 1.0,(r,t,L)))/np.sqrt(2);
    BitsI = nr.randint(2,size=(t,Nsub));
    BitsQ = nr.randint(2,size=(t,Nsub));
    Sym = (2*BitsI-1)+1j*(2*BitsQ-1);

    for K in range(len(SNRdB)):
        LoadedSym = np.sqrt(Eb[K])*Sym;
        for tx in range(t):
            TxSamples[tx,:] = nf.ifft(LoadedSym[tx,:]);
            TxSamCP[tx,:] = np.concatenate((TxSamples[tx,Nsub-Ncp:Nsub],TxSamples[tx,:])); 

        RxSamCP = np.zeros((r,L+Nsub+Ncp-1))+1j*np.zeros((r,L+Nsub+Ncp-1));
        for rx in range(r):
            for tx in range(t):
                RxSamCP[rx,:]+= np.convolve(H[rx,tx,:],TxSamCP[tx,:])
                Hfft[rx,tx,:] = nf.fft(np.concatenate((H[rx,tx,:],np.zeros(Nsub-L))));
            RxSamCP[rx,:]+= noise[rx,:];
            RxSamples[rx,:] = RxSamCP[rx,Ncp:Ncp+Nsub];
            RxSym[rx,:] = nf.fft(RxSamples[rx,:]);
        for nx in range(Nsub):
            ZFout = np.matmul(nl.pinv(Hfft[:,:,nx]),RxSym[:,nx]);
            DecBitsI = (np.real(ZFout)>0);
            DecBitsQ = (np.imag(ZFout)>0);
            BER[K] = BER[K]+np.sum(DecBitsI!=BitsI[:,nx]) + np.sum(DecBitsQ!=BitsQ[:,nx]);
                
BER = BER/nBlocks/Nsub/t/2;
SNReff = SNR*L/Nsub;
L=r-t+1;    
BERt = comb(2*L-1, L)/2**L/SNReff**L; # BER for ZF from formula

plt.yscale('log')
plt.plot(SNRdB, BER,'g-');
plt.plot(SNRdB, BERt,'ro');
plt.grid(1,which='both')
plt.suptitle('BER for MIMO OFDM Channel')
plt.legend(["Simulation", "Theory"], loc ="lower left");
plt.xlabel('SNR (dB)')
plt.ylabel('BER') 