#!/usr/bin/python3

# DT2119, Lab 1 Feature Extraction

import numpy as np
from scipy import signal, fftpack
from toolslab1 import trfbank, lifter
import matplotlib.pyplot as plt

# Function given by the exercise ----------------------------------

def mfcc(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512,
         nceps=13, samplingrate=20000, liftercoeff=22):
    """Computes Mel Frequency Cepstrum Coefficients.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        nceps: number of cepstrum coefficients to compute
        samplingrate: sampling rate of the original signal
        liftercoeff: liftering coefficient used to equalise scale of MFCCs

    Returns:
        N x nceps array with lifetered MFCC coefficients
    """
    frames = enframe(samples, winlen, winshift)
    preemph = preemp(frames, preempcoeff)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    mspec = logMelSpectrum(spec, samplingrate)
    ceps = cepstrum(mspec, nceps)
    return lifter(ceps, liftercoeff)#logMelSpectrum(spec, samplingrate)


def mfcc_both(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512,
         nceps=13, samplingrate=20000, liftercoeff=22):
    """Computes Mel Frequency Cepstrum Coefficients.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        nceps: number of cepstrum coefficients to compute
        samplingrate: sampling rate of the original signal
        liftercoeff: liftering coefficient used to equalise scale of MFCCs

    Returns:
        N x nceps array with lifetered MFCC coefficients
    """
    frames = enframe(samples, winlen, winshift)
    preemph = preemp(frames, preempcoeff)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    mspec = logMelSpectrum(spec, samplingrate)
    ceps = cepstrum(mspec, nceps)
    lmfcc = lifter(ceps, liftercoeff)
    return lmfcc, mspec

def mfccNOLIFT(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512,
         nceps=13, samplingrate=20000, liftercoeff=22):
    """Computes Mel Frequency Cepstrum Coefficients.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        nceps: number of cepstrum coefficients to compute
        samplingrate: sampling rate of the original signal
        liftercoeff: liftering coefficient used to equalise scale of MFCCs

    Returns:
        N x nceps array with lifetered MFCC coefficients
    """
    frames = enframe(samples, winlen, winshift)
    preemph = preemp(frames, preempcoeff)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    mspec = logMelSpectrum(spec, samplingrate)
    ceps = cepstrum(mspec, nceps)
    lift = lifter(ceps, liftercoeff)
    return lift


# Functions to be implemented ----------------------------------

def enframe(samples, winlen, winshift):
    """
    Slices the input samples into overlapping windows.

    Args:
        winlen: window length in samples.
        winshift: shift of consecutive windows in samples
    Returns:
        numpy array [N x winlen], where N is the number of windows that fit
        in the input signal
    """
    N = int(np.floor((len(samples) - winlen) / winshift + 1))
    frames = np.zeros([N, winlen])

    for i in range(N):
        frames[i, :] = samples[i*winshift : i*winshift + winlen]

    return frames


def preemp(input, p=0.97):
    """
    Pre-emphasis filter.

    Args:
        input: array of speech frames [N x M] where N is the number of frames and
               M the samples per frame
        p: preemhasis factor (defaults to the value specified in the exercise)

    Output:
        output: array of pre-emphasised speech samples
    Note (you can use the function lfilter from scipy.signal)
    """

    return signal.lfilter([1, -p], 1, input, axis=1)


def windowing(input):
    """
    Applies hamming window to the input frames.

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
    Output:
        array of windowed speech samples [N x M]
    Note (you can use the function hamming from scipy.signal, include the sym=0 option
    if you want to get the same results as in the example)
    """

    window = signal.windows.hamming(np.shape(input)[1], sym=0)
    #plt.plot(window)
    #plt.show()
    return input * window

def powerSpectrum(input, nfft):
    """
    Calculates the power spectrum of the input signal, that is the square of
    the modulus of the FFT

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
        nfft: length of the FFT
    Output:
        array of power spectra [N x nfft]
    Note: you can use the function fft from scipy.fftpack
    """
    return abs(fftpack.fft(input, nfft))**2

def logMelSpectrum(input, samplingrate):
    """
    Calculates the log output of a Mel filterbank when the input is the power
    spectrum

    Args:
        input: array of power spectrum coefficients [N x nfft] where N is the
         number of frames and nfft the length of each spectrum
        samplingrate: sampling rate of the original signal (used to calculate
        the filterbank shapes)
    Output:
        array of Mel filterbank log outputs [N x nmelfilters] where nmelfilters
        is the number of filters in the filterbank
    Note: use the trfbank function provided in tools.py to calculate the
       filterbank shapes and nmelfilters
    """
    nfft = np.shape(input)[1]
    trfBank = trfbank(samplingrate, nfft)

    return np.log(np.dot(input, trfBank.T))


def cepstrum(input, nceps):
    """
    Calulates Cepstral coefficients from mel spectrum applying Discrete Cosine
    Transform

    Args:
        input: array of log outputs of Mel scale filterbank [N x nmelfilters]
            where N is the number of frames and nmelfilters the length of the
            filterbank
        nceps: number of output cepstral coefficients
    Output:
        array of Cepstral coefficients [N x nceps]
    Note: you can use the function dct from scipy.fftpack.realtransforms
    """
    return fftpack.dct(input, norm="ortho")[:, 0:nceps]

def dtw(x, y, dist):
    """Dynamic Time Warping.

    Args:
        x, y: arrays of size NxD and MxD respectively, where D is the dimensionality
              and N, M are the respective lenghts of the sequences
        dist: distance function (can be used in the code as dist(x[i], y[j]))

    Outputs:
        d: global distance between the sequences (scalar) normalized to len(x)+len(y)
        LD: local distance between frames from x and y (NxM matrix)
        AD: accumulated distance between frames of x and y (NxM matrix)
        path: best path thtough AD

    Note that you only need to define the first output for this exercise.
    """

    H = x.shape[0]
    K = y.shape[0]
    AD = np.zeros([H+1, K+1])
    AD[:, 0] = np.inf
    AD[0, :] = np.inf
    AD[0, 0] = 0

    LD = np.zeros([H, K])

    for h in range(H):
        for k in range(K):
            LD[h,k] = dist(x[h,:], y[k,:])

    for h in range(H):
        for k in range(K):
            AD[h+1,k+1] = LD[h,k] + np.min([AD[h, k+1], AD[h, k], AD[h+1, k]])

    h = H
    k = K
    path = [(h,k)]

    while h != 1 or k != 1:
        ind = np.argmin([AD[h, k-1], AD[h-1, k-1], AD[h-1, k]])
        if ind == 0:
            k = k-1
        elif ind == 1:
            k = k-1
            h = h-1
        elif ind == 2:
            h = h-1

        path.append((h,k))
    # append when they are both one
    path.append((h,k))

    AD = AD[1:, 1:]

    d = AD[H-1, K-1]

    dt = np.dtype('int', 'int')

    path = np.array(path, dtype=dt)

    return d, LD, AD, path
