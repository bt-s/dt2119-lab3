#!/usr/bin/python3

import numpy as np
import unittest
import matplotlib.pyplot as plt

from lab1_proto import enframe, preemp, windowing, powerSpectrum, logMelSpectrum,\
    cepstrum, mfcc
from lab1_tools import lifter


### 3 Data
print(np.__version__)
# Example utterance
example = np.load('lab1_example.npz', allow_pickle=True)['example'].item()
samples = example["samples"]

# Data set
data = 0
#data = np.load('lab1_data.npz', allow_pickle=True)['data']
### 4 Mel Frequency Cepstrum Coefficients step-by-step

# 4.1 Enframe
enframeRes = enframe(example["samples"], 400, 200)
print(np.testing.assert_almost_equal(enframeRes, example["frames"]))

# 4.2 Pre-emphasis
preEmpRes = preemp(enframeRes)
print(np.testing.assert_almost_equal(preEmpRes, example["preemph"]))

# 4.3 Hamming Window
hammingRes = windowing(preEmpRes)
print(np.testing.assert_almost_equal(hammingRes, example["windowed"], decimal=10))

# 4.4 Fast Fourier Transform
FFTRes = powerSpectrum(hammingRes, 512)
print(np.testing.assert_almost_equal(FFTRes, example["spec"], decimal=7))

# 4.5 Mel filterbank log spectrum
melFBankRes = logMelSpectrum(FFTRes, 20000)
print(np.testing.assert_almost_equal(melFBankRes, example["mspec"], decimal=7))

# 4.6 Cosine Transform and Liftering
MFCCRes = cepstrum(melFBankRes, 13)
LMFCCRes = lifter(MFCCRes)
print(np.testing.assert_almost_equal(MFCCRes, example["mfcc"], decimal=7))
print(np.testing.assert_almost_equal(LMFCCRes, example["lmfcc"], decimal=7))
print(LMFCCRes-example["lmfcc"])
### 5 Feature Correlation
featureMatrix = np.zeros([1, 40])
featureMatrix = np.zeros([1, 13])
for d in data:
    A = mfcc(d["samples"])
    featureMatrix = np.vstack([featureMatrix, mfcc(d["samples"])])

featureMatrix = featureMatrix[1:, :]
featureMatrix = featureMatrix - np.mean(featureMatrix, axis=0)
featureMatrix = 1.0/float(featureMatrix.shape[0]) * featureMatrix.T@featureMatrix

### 6 Comparing Utterances

### 7 Explore Speech Segments with Clustering

### Plots
#plt.pcolormesh(enframeRes.T)
#plt.pcolormesh(preEmpRes.T)
#plt.pcolormesh(hammingRes.T)
#plt.pcolormesh(FFTRes.T)
#plt.pcolormesh(melFBankRes.T)
#plt.pcolormesh(MFCCRes.T)
#plt.pcolormesh(LMFCCRes.T)
#plt.pcolormesh(featureMatrix)
#plt.show()

