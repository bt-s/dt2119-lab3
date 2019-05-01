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

## 4.1 Target Class Definition

phones = sorted(phoneHMMs.keys())
nstates = {phone: phoneHMMs[phone]['means'].shape[0] for phone in phones}
stateList = [ph + '_' + str(id) for ph in phones for id in range(nstates[ph])]

#print(stateList)
print(example.keys())
np.seterr(divide='ignore')
print("#"*50 + "\n" + "#"*50 + "\n" + "#"*50)
###########################################################
######## INITIALIZATION END ###############################
###########################################################


###########################################################
######## 4. Preparing the Data for DNN Training ###########


filename = './tidigits/disc_4.1.1/tidigits/train/man/nw/z43a.wav'
#filename = 'tidigits/disc_4.1.1/tidigits/train/man/ae/z9z6531a.wav'

# 4.2 Forced Alignment

samples, samplingrate = loadAudio(filename)

print("Samples")
print(samples)

print(len(samples))

lmfcc = mfcc(samples)
print(lmfcc.shape)
print(example["lmfcc"].shape)
# check lmfcc

error_lmfcc = np.max(np.abs(lmfcc - example["lmfcc"]))
print("LMFCC error is : ", error_lmfcc)


wordTrans = list(path2info(filename)[2])
#print(wordTrans)


# phone transcription
phoneTrans = words2phones(wordTrans, prondict)


# Check the result is the same

check = True

for i, e in enumerate(phoneTrans):
    check = check and (e == example["phoneTrans"][i])

print("PhoneTrans is : ", check)

utteranceHMM = concatHMMs(phoneHMMs, phoneTrans)

stateTrans = [phone + '_' + str(stateid) for phone in phoneTrans \
                        for stateid in range(nstates[phone])]

# Check if it is correct

check = True

for i, e in enumerate(stateTrans):
    check = check and (e == example["stateTrans"][i])

print("StateTrans is : ", check)


# aligning states


aligned_states = forcedAlignment(lmfcc, phoneHMMs, phoneTrans)


check = True

for i, e in enumerate(aligned_states):
    check = check and (e == example["viterbiStateTrans"][i])

print("viterbiStateTrans is : ", check)
