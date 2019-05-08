import os
import numpy as np
from lab3_tools import *
from lab1_proto import *
from prondict import prondict
from lab2_proto import concatHMMs
from lab3_proto import *

ratio = np.load("ratio.npy")

testdata = []
all_model = np.load('lab2_models_all.npz', allow_pickle=True)

isolated = {}
for digit in prondict.keys():
    isolated[digit] = ['sil'] + prondict[digit] + ['sil']


phoneHMMs_all = all_model['phoneHMMs'].item()
phones = sorted(phoneHMMs_all.keys())
nstates = {phone: phoneHMMs_all[phone]['means'].shape[0] for phone in phones}
stateList = [ph + '_' + str(id) for ph in phones for id in range(nstates[ph])]

count = 0

for root, dirs, files in os.walk('tidigits/disc_4.2.1/tidigits/test'):
    for file in files:
        if file.endswith('.wav'):
            filename = os.path.join(root, file)
            print(float(count)/87.00, "%%")
            count = count + 1
            samples, samplingrate = loadAudio(filename)

            lmfcc, mspec = mfcc_both(samples)
            if False:
                lmfcc = np.multiply(lmfcc, np.dot(np.ones([lmfcc.shape[0], 1]),
                    ratio.reshape(len(ratio),1).T))

            wordTrans = list(path2info(filename)[2])
            phoneTrans = words2phones(wordTrans, prondict)
            targets = forcedAlignment(lmfcc, phoneHMMs_all, phoneTrans)
            targets = [stateList.index(t) for t in targets]
            #print(targets)
            testdata.append({'filename': filename, 'lmfcc': lmfcc,
            'mspec': mspec, 'targets': targets})


print(float(count)/87.00, "%%")
np.savez("data/testdata.npz", testdata=testdata)
