import numpy as np
from lab3_tools import *
from lab1_proto import mfcc
from prondict import prondict
from lab2_proto import concatHMMs
from lab3_proto import *

###########################################################
######## INITIALIZATION ###################################
###########################################################

example = np.load('lab3_example.npz')["example"]
example.shape=(1,)
example = example[0]
one_speaker_model = np.load('lab2_models_onespkr.npz')
all_model = np.load('lab2_models_all.npz')

isolated = {}
for digit in prondict.keys():
    isolated[digit] = ['sil'] + prondict[digit] + ['sil']


phoneHMMs = one_speaker_model['phoneHMMs'].item()

wordHMMs = {}
for key in isolated.keys():
    wordHMMs[key] = concatHMMs(phoneHMMs, isolated[key])

phoneHMMs_all = all_model['phoneHMMs'].item()

wordHMMs_all = {}
for key in isolated.keys():
    wordHMMs[key] = concatHMMs(phoneHMMs, isolated[key])


phones = sorted(phoneHMMs.keys())
nstates = {phone: phoneHMMs[phone]['means'].shape[0] for phone in phones}
stateList = [ph + '_' + str(id) for ph in phones for id in range(nstates[ph])]

print(stateList)

###########################################################
######## INITIALIZATION END ###############################
###########################################################


filename = './tidigits/disc_4.1.1/tidigits/train/man/nw/z43a.wav'
#filename = 'tidigits/disc_4.1.1/tidigits/train/man/ae/z9z6531a.wav'

samples, samplingrate = loadAudio(filename)
print(samples)
lmfcc = mfcc(samples)

wordTrans = list(path2info(filename)[2])
print(wordTrans)
phoneTrans = words2phones(wordTrans, prondict)

# phone transcription

print(phoneTrans)
print(example["phoneTrans"])

utteranceHMM = concatHMMs(phoneHMMs, phoneTrans)

#lmfcc = mfcc(samples)
