import os
import numpy as np
from lab3_tools import *
from lab1_proto import *
from prondict import prondict
from lab2_proto import concatHMMs
from lab3_proto import *

traindata = []
all_model = np.load('lab2_models_all.npz')

isolated = {}
for digit in prondict.keys():
    isolated[digit] = ['sil'] + prondict[digit] + ['sil']


phoneHMMs_all = all_model['phoneHMMs'].item()
phones = sorted(phoneHMMs_all.keys())
nstates = {phone: phoneHMMs_all[phone]['means'].shape[0] for phone in phones}
stateList = [ph + '_' + str(id) for ph in phones for id in range(nstates[ph])]

count = 0

for root, dirs, files in os.walk('tidigits/disc_4.1.1/tidigits/train'):
    for file in files:
        if file.endswith('.wav'):
            filename = os.path.join(root, file)
            print(float(count)/86.23, "%%")
            count = count + 1
            samples, samplingrate = loadAudio(filename)

            ###...your code for feature extraction and forced alignment
            lmfcc, mspec = mfcc_both(samples)
            wordTrans = list(path2info(filename)[2])
            phoneTrans = words2phones(wordTrans, prondict)
            targets = forcedAlignment(lmfcc, phoneHMMs_all, phoneTrans)
            targets = [stateList.index(t) for t in targets]
            #print(targets)
            traindata.append({'filename': filename, 'lmfcc': lmfcc,
            'mspec': 'mspec', 'targets': targets})


print(float(count)/86.23, "%%")