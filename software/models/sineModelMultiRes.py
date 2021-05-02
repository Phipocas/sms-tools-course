import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import blackmanharris, triang, get_window
from scipy.fftpack import ifft, fftshift
import math
import dftModel as DFT
import utilFunctions as UF


def sineModelMultiRes(x, fs, t, W, N, B, h):
    """
    Analysis/synthesis of a sound using the sinusoidal model, without sine tracking
    x: input array sound, w: analysis window, N: size of complex spectrum, t: threshold in negative dB 
    returns y: output array sound
    """
        
    hM1 = [int(math.floor((w.size+1)/2)) for w in W]                     # half analysis window size by rounding
    hM2 = [int(math.floor(w.size/2)) for w in W]                        # half analysis window size by floor
    Ns = 512                                                # FFT size for synthesis (even)
    H = h                                               # Hop size used for analysis and synthesis
    hNs = Ns//2                                             # half of synthesis FFT size
    pin = [max(hNs, hm1) for hm1 in hM1]                                    # init sound pointer in middle of anal window       
    pend = [(x.size - max(hNs, hm1)) for hm1 in hM1]                          # last sample to start a frame
    yw = np.zeros(Ns)                                       # initialize output sound frame
    y = np.zeros(x.size)                                   # initialize output array
    w = [(w / sum(w)) for w in W]                                          # normalize analysis window
    sw = np.zeros(Ns)                                       # initialize synthesis window
    ow = triang(2*H)                                        # triangular window
    sw[hNs-H:hNs+H] = ow                                    # add triangular window
    bh = blackmanharris(Ns)                                 # blackmanharris window
    bh = bh / sum(bh)                                       # normalized blackmanharris window
    sw[hNs-H:hNs+H] = sw[hNs-H:hNs+H] / bh[hNs-H:hNs+H]     # normalized synthesis window
    ipfreq_matrix = np.array([[]])
    ipmag_matrix = np.array([[]])
    ipphase_matrix = np.array([[]])

    for i in range(len(pin)):
        while pin[i]<pend[i]:                                         # while input sound pointer is within sound 
        #-----analysis-----             
            x1 = x[pin[i]-hM1[i]:pin[i]+hM2[i]]                               # select frame
            mX, pX = DFT.dftAnal(x1, w[i], N[i])                        # compute dft
            ploc = UF.peakDetection(mX, t)                        # detect locations of peaks
            iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc)   # refine peak values by interpolation
            ipfreq = fs*iploc/float(N[i])                            # convert peak locations to Hertz
            ipfreq_band = np.array([peak for peak in ipfreq if B[i]<peak<=B[i+1]])
            ipmag_band = np.array([ipmag[j] for j in range(len(ipfreq)) if B[i]<ipfreq[j]<=B[i+1]])
            ipphase_band = np.array([ipphase[j] for j in range(len(ipfreq)) if B[i]<ipfreq[j]<=B[i+1]])
        #-----synthesis-----
            Y = UF.genSpecSines(ipfreq_band, ipmag_band, ipphase_band, Ns, fs)   # generate sines in the spectrum         
            fftbuffer = np.real(ifft(Y))                          # compute inverse FFT
            yw[:hNs-1] = fftbuffer[hNs+1:]                        # undo zero-phase window
            yw[hNs-1:] = fftbuffer[:hNs+1] 
            y[pin[i]-hNs:pin[i]+hNs] += sw*yw                           # overlap-add and apply a synthesis window
            pin[i] += H                                              # advance sound pointer
    return y

if __name__ == '__main__':

    window = 'blackmanharris'
    w1 = get_window(window, 8191)
    w2 = get_window(window, 2047)
    w3 = get_window(window, 255)
    w = [w1, w2, w3]

    N = [8192, 4096, 512]
    B = [0, 1000, 5000, 22050]


    fs, x = UF.wavread(filename = '../../sounds/orchestra.wav')
    y = sineModelMultiRes(x, fs, t=-80, W=w, N=N, B=B, h=64)

    UF.wavwrite(y, fs, "../../sounds/A10_b1.wav")

    # create figure to show plots
    plt.figure(figsize=(9, 6))

    # frequency range to plot
    maxplotfreq = 5000.0

    # plot the input sound
    plt.subplot(3,1,1)
    plt.plot(np.arange(x.size)/float(fs), x)
    plt.axis([0, x.size/float(fs), min(x), max(x)])
    plt.ylabel('amplitude')
    plt.xlabel('time (sec)')
    plt.title('input sound: x')

    # plot the output sound
    plt.subplot(3,1,2)
    plt.plot(np.arange(y.size)/float(fs), y)
    plt.axis([0, y.size/float(fs), min(y), max(y)])
    plt.ylabel('amplitude')
    plt.xlabel('time (sec)')
    plt.title('output sound: y')

    plt.show()