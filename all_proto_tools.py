#!/usr/bin/python3

# DT2119, Lab 1 Feature Extraction

import numpy as np
from scipy import signal, fftpack
import matplotlib.pyplot as plt


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
    return fftpack.dct(input)[:, 0:nceps]


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

    # Append when they are both one
    path.append((h,k))
    AD = AD[1:, 1:]
    d = AD[H-1, K-1]
    dt = np.dtype('int', 'int')
    path = np.array(path, dtype=dt)

    return d, LD, AD, path


def tidigit2labels(tidigitsarray):
    """
    Return a list of labels including gender, speaker, digit and repetition
    information for each utterance in tidigitsarray. Useful for plots.
    """
    labels = []
    nex = len(tidigitsarray)
    for ex in range(nex):
        labels.append(tidigitsarray[ex]['gender'] + '_' +
                      tidigitsarray[ex]['speaker'] + '_' +
                      tidigitsarray[ex]['digit'] + '_' +
                      tidigitsarray[ex]['repetition'])
    return labels


def dither(samples, level=1.0):
    """
    Applies dithering to the samples. Adds Gaussian noise to the samples to
    avoid numerical errors in the subsequent FFT calculations.

        samples: array of speech samples
        level: decides the amount of dithering (see code for details)

    Returns:
        array of dithered samples (same shape as samples)
    """
    return samples + level*np.random.normal(0,1, samples.shape)


def lifter(mfcc, lifter=22):
    """
    Applies liftering to improve the relative range of MFCC coefficients.

       mfcc: NxM matrix where N is the number of frames and M the number of MFCC
             coefficients
       lifter: lifering coefficient

    Returns:
       NxM array with liftered coefficients
    """
    nframes, nceps = mfcc.shape
    cepwin = 1.0 + lifter/2.0 * np.sin(np.pi * np.arange(nceps) / lifter)
    return np.multiply(mfcc, np.tile(cepwin, nframes).reshape((nframes,nceps)))


def hz2mel(f):
    """Convert an array of frequency in Hz into mel."""
    return 1127.01048 * np.log(f/700 +1)


def trfbank(fs, nfft, lowfreq=133.33, linsc=200/3., logsc=1.0711703,
            nlinfilt=13, nlogfilt=27, equalareas=False):
    """Compute triangular filterbank for MFCC computation.

    Inputs:
    fs:         sampling frequency (rate)
    nfft:       length of the fft
    lowfreq:    frequency of the lowest filter
    linsc:      scale for the linear filters
    logsc:      scale for the logaritmic filters
    nlinfilt:   number of linear filters
    nlogfilt:   number of log filters

    Outputs:
    res:  array with shape [N, nfft], with filter amplitudes for each column.
            (N=nlinfilt+nlogfilt)
    From scikits.talkbox"""
    # Total number of filters
    nfilt = nlinfilt + nlogfilt
    freqs = np.zeros(nfilt+2)
    freqs[:nlinfilt] = lowfreq + np.arange(nlinfilt) * linsc
    freqs[nlinfilt:] = freqs[nlinfilt-1] * logsc ** np.arange(1, nlogfilt + 3)

    if equalareas:
        heights = np.ones(nfilt)
    else:
        heights = 2./(freqs[2:] - freqs[0:-2])

    # Compute filterbank coeff (in fft domain, in bins)
    fbank = np.zeros((nfilt, nfft))
    # FFT bins (in Hz)
    nfreqs = np.arange(nfft) / (1. * nfft) * fs
    for i in range(nfilt):
        low = freqs[i]
        cen = freqs[i+1]
        hi = freqs[i+2]

        lid = np.arange(np.floor(low * nfft / fs) + 1,
                        np.floor(cen * nfft / fs) + 1, dtype=np.int)
        lslope = heights[i] / (cen - low)
        rid = np.arange(np.floor(cen * nfft / fs) + 1,
                        np.floor(hi * nfft / fs) + 1, dtype=np.int)
        rslope = heights[i] / (hi - cen)
        fbank[i][lid] = lslope * (nfreqs[lid] - low)
        fbank[i][rid] = rslope * (hi - nfreqs[rid])

    return fbank

def concatTwoHMMs(hmm1, hmm2):
    """ Concatenates 2 HMM models

    Args:
       hmm1, hmm2: two dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be different for each)

    Output
       dictionary with the same keys as the input but concatenated models:
          startprob: K+1 array with priori probability of state
          transmat: (K+1)x(K+1) transition matrix
             means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models

    Example:
       twoHMMs = concatHMMs(phoneHMMs['sil'], phoneHMMs['ow'])

    See also: the concatenating_hmms.pdf document in the lab package
    """
    result_hmm = dict()
    M = hmm1["transmat"].shape[0]

    # parametric to allow sp
    K = M + (hmm2["transmat"].shape[0] - 1)

    # arrays to concatenate
    sp_1 = hmm1["startprob"][:(M-1)]
    sp_2 = hmm1["startprob"][(M-1)]*hmm2["startprob"]

    result_hmm["startprob"] = np.concatenate((sp_1,sp_2))

    # compute transition prob
    result_hmm["transmat"] = np.zeros([K,K])
    result_hmm["transmat"][:M,:M] = hmm1["transmat"]
    result_hmm["transmat"][M-1:,M-1:] = hmm2["transmat"]

    result_hmm["transmat"][:M-1,M-1:] = np.outer(hmm1["transmat"][:M-1,-1],hmm2["startprob"])
    result_hmm["transmat"][-1,-1] = 1

    # concatenate means
    result_hmm["means"] = np.vstack((hmm1["means"], hmm2["means"]))

    # concatenate variances
    result_hmm["covars"] = np.vstack((hmm1["covars"], hmm2["covars"]))

    # concatenate names
    result_hmm["name"] = hmm1["name"] + ", " + hmm2["name"]

    return result_hmm


def concatHMMs(hmmmodels, namelist):
    """ Concatenates HMM models in a left to right manner

    Args:
       hmmmodels: dictionary of models indexed by model name.
       hmmmodels[name] is a dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances
       namelist: list of model names that we want to concatenate

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be
      different in each model)

    Output
       combinedhmm: dictionary with the same keys as the input but
                    combined models:
         startprob: K+1 array with priori probability of state
          transmat: (K+1)x(K+1) transition matrix
             means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models

    Example:
       wordHMMs['o'] = concatHMMs(phoneHMMs, ['sil', 'ow', 'sil'])
    """
    concat = hmmmodels[namelist[0]]
    for idx in range(1,len(namelist)):
        concat = concatTwoHMMs(concat, hmmmodels[namelist[idx]])

    return concat


def gmmloglik(log_emlik, weights):
    """Log Likelihood for a GMM model based on Multivariate Normal Distribution.

    Args:
        log_emlik: array like, shape (N, K).
            contains the log likelihoods for each of N observations and
            each of K distributions
        weights:   weight vector for the K components in the mixture

    Output:
        gmmloglik: scalar, log likelihood of data given the GMM model.
    """

def forward(log_emlik, log_startprob, log_transmat):
    """Forward (alpha) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: log transition probability from state i to j

    Output:
        forward_prob: NxM array of forward log probabilities for each of the
        M states in the model
    """
    N = log_emlik.shape[0]
    M = log_emlik.shape[1]
    forward_prob = np.zeros(log_emlik.shape)

    forward_prob[0, :] = log_startprob[:-1] + log_emlik[0, :]

    for n in range(1, N):
        for j in range(M):
            forward_prob[n, j] = logsumexp(forward_prob[n-1, :] +
                    log_transmat[:-1, j]) + log_emlik[n, j]

    return forward_prob


def backward(log_emlik, log_startprob, log_transmat):
    """Backward (beta) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        backward_prob: NxM array of backward log probabilities for each of the
        M states in the model
    """

    N = log_emlik.shape[0]
    M = log_emlik.shape[1]

    backward_prob = np.zeros([N,M])

    for i in reversed(range(N-1)):
        for j in range(M):
            backward_prob[i,j] = logsumexp(log_transmat[j,:-1] +
                    log_emlik[i+1,:] + backward_prob[i+1,:])

    return backward_prob

def viterbi(log_emlik, log_startprob, log_transmat, forceFinalState=True):
    """Viterbi path.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j
        forceFinalState: if True, start backtracking from the final state in
                  the model, instead of the best state at the last time step

    Output:##### Section 5.3
        viterbi_loglik: log likelihood of the best path
        viterbi_path: best path
    """
    N = log_emlik.shape[0]
    M = log_emlik.shape[1]

    V = np.zeros(log_emlik.shape)
    B = np.zeros(log_emlik.shape)

    # Initialization
    V[0, :] = log_startprob[:-1] + log_emlik[0, :]
    B[0, :] = 0

    # Induction
    for i in range(1, N):
        for j in range(M):
            V[i, j] = np.max(V[i-1, :] + log_transmat[:-1, j]) + log_emlik[i, j]
            B[i, j] = np.argmax(V[i-1, :] + log_transmat[:-1, j])

    # Termination
    best = np.max(V[-1, :])
    sN = np.argmax(B[-1, :])

    # Backtracking
    st = np.zeros(N, dtype="int32")
    st[-1] = sN
    for i in reversed(range(N-1)):
        st[i] = B[i+1, int(st[i+1])]


    viterbi_loglik = best
    viterbi_path = st

    return viterbi_loglik, viterbi_path


def statePosteriors(log_alpha, log_beta):
    """State posterior (gamma) probabilities in log domain.

    Args:
        log_alpha: NxM array of log forward (alpha) probabilities
        log_beta: NxM array of log backward (beta) probabilities
    where N is the number of frames, and M the number of states

    Output:
        log_gamma: NxM array of gamma probabilities for each of the
        M states in the model
    """
    return log_alpha + log_beta - logsumexp(log_alpha[-1,:])



def updateMeanAndVar(X, log_gamma, varianceFloor=5.0):
    """ Update Gaussian parameters with diagonal covariance

    Args:
         X: NxD array of feature vectors
         log_gamma: NxM state posterior probabilities in log domain
         varianceFloor: minimum allowed variance scalar
    were N is the lenght of the observation sequence, D is the
    dimensionality of the feature vectors and M is the number of
    states in the model

    Outputs:
         means: MxD mean vectors for each state
         covars: MxD covariance (variance) vectors for each state
    """
    N = X.shape[0]
    M = log_gamma.shape[1]
    D = X.shape[1]
    means  = np.zeros((M, D))
    covars = np.zeros((M, D))

    for i in range(M):
        means[i, :] = np.dot(X.T, np.exp(log_gamma[:, i])) / \
                np.sum(np.exp(log_gamma[:, i]))

        C = X.T - means[i,:].reshape((D, 1))

        res = 0
        for j in range(N):
            res = res + np.exp(log_gamma[j, i]) * np.outer(C[:, j], C[:, j])

        covars[i, :] = np.diag(res) / np.sum(np.exp(log_gamma[:, i]))

    covars[covars < varianceFloor] = varianceFloor

    return means, covars


def logsumexp(arr, axis=0):
    """Computes the sum of arr assuming arr is in the log domain.
    Returns log(sum(exp(arr))) while minimizing the possibility of
    over/underflow.
    """
    arr = np.rollaxis(arr, axis)
    vmax = arr.max(axis=0)
    if vmax.ndim > 0:
        vmax[~np.isfinite(vmax)] = 0
    elif not np.isfinite(vmax):
        vmax = 0
    with np.errstate(divide="ignore"):
        out = np.log(np.sum(np.exp(arr - vmax), axis=0))
        out += vmax
        return out


def log_multivariate_normal_density_diag(X, means, covars):
    """Compute Gaussian log-density at X for a diagonal model

    Args:
        X: array like, shape (n_observations, n_features)
        means: array like, shape (n_components, n_features)
        covars: array like, shape (n_components, n_features)

    Output:
        lpr: array like, shape (n_observations, n_components)
    From scikit-learn/sklearn/mixture/gmm.py
    """
    n_samples, n_dim = X.shape
    lpr = -0.5 * (n_dim * np.log(2 * np.pi) + np.sum(np.log(covars), 1)
                  + np.sum((means ** 2) / covars, 1)
                  - 2 * np.dot(X, (means / covars).T)
                  + np.dot(X ** 2, (1.0 / covars).T))
    return lpr


def words2phones(wordList, pronDict, addSilence=True, addShortPause=True):
    """ word2phones: converts word level to phone level transcription adding
    silence

    Args:
       wordList: list of word symbols
       pronDict: pronunciation dictionary. The keys correspond to words in wordList
       addSilence: if True, add initial and final silence
       addShortPause: if True, add short pause model "sp" at end of each word
    Output:
       list of phone symbols
    """
    phone = []
    for digit in wordList:
        phone = phone + pronDict[digit] + ["sp"]

    phone = ["sil"] + phone + ["sil"]

    return phone


def forcedAlignment(lmfcc, phoneHMMs, phoneTrans):
    """ forcedAlignmen: aligns a phonetic transcription at the state level

    Args:
       lmfcc: NxD array of MFCC feature vectors (N vectors of dimension D)
              computed the same way as for the training of phoneHMMs
       phoneHMMs: set of phonetic Gaussian HMM models
       phoneTrans: list of phonetic symbols to be aligned including initial and
                   final silence

    Returns:
       list of strings in the form phoneme_index specifying, for each time step
       the state from phoneHMMs corresponding to the viterbi path.
    """

    HMM = concatHMMs(phoneHMMs, phoneTrans)
    result = []

    stateTrans = [phone + '_' + str(stateid) for phone in phoneTrans \
                            for stateid in range(phoneHMMs[phone]['means'].shape[0])]

    data_log_lik = log_multivariate_normal_density_diag(
                    lmfcc, HMM["means"], HMM["covars"])
    viterbi_loglik, viterbi_path = viterbi(data_log_lik,
                    np.log(HMM["startprob"]),
                    np.log(HMM["transmat"]))

    for i in viterbi_path:
        result = result + [stateTrans[i]]

    return result


def path2info(path):
    """
    path2info: parses paths in the TIDIGIT format and extracts information
               about the speaker and the utterance

    Example:
    path2info('tidigits/disc_4.1.1/tidigits/train/man/ae/z9z6531a.wav')
    """
    rest, filename = os.path.split(path)
    rest, speakerID = os.path.split(rest)
    rest, gender = os.path.split(rest)
    digits = filename[:-5]
    repetition = filename[-5]

    return gender, speakerID, digits, repetition


def loadAudio(filename):
    """
    loadAudio: loads audio data from file using pysndfile

    Note that, by default pysndfile converts the samples into floating point
    numbers and rescales them in the range [-1, 1]. This is avoided by specifying
    the option dtype=np.int16 which keeps both the original data type and range
    of values.
    """
    if False:
        from pysndfile import sndio
        sndobj = sndio.read(filename)
        samplingrate = sndobj[1]
        samples = np.array(sndobj[0])*np.iinfo(np.int16).max

    if False:
        import wave
        f = wave.open(filename, mode="rb")
        samplingrate = f.getframerate()
        samples = f.readframes(f.getnframes())
        f.close()

    if True:
        import soundfile as sf
        data, fs = sf.read(filename,dtype="int16")
        samplingrate = fs #sndobj[1]
        samples = np.array(data, dtype="float64")
    return samples, samplingrate


def frames2trans(sequence, outfilename=None, timestep=0.01):
    """
    Outputs a standard transcription given a frame-by-frame
    list of strings.

    Example (using functions from Lab 1 and Lab 2):
    phones = ['sil', 'sil', 'sil', 'ow', 'ow', 'ow', 'ow', 'ow', 'sil', 'sil']
    trans = frames2trans(phones, 'oa.lab')

    Then you can use, for example wavesurfer to open the wav file and the transcription
    """
    sym = sequence[0]
    start = 0
    end = 0
    trans = ''
    for t in range(len(sequence)):
        if sequence[t] != sym:
            trans = trans + str(start) + ' ' + str(end) + ' ' + sym + '\n'
            sym = sequence[t]
            start = end
        end = end + timestep
    trans = trans + str(start) + ' ' + str(end) + ' ' + sym + '\n'
    if outfilename != None:
        with open(outfilename, 'w') as f:
            f.write(trans)

    return trans

