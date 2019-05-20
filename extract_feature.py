import os
import numpy as np
from all_proto_tools import *
from prondict import prondict

ratio = np.load("ratio.npy")

all_model = np.load('lab2_models_all.npz', allow_pickle=True)

isolated = {}
for digit in prondict.keys():
    isolated[digit] = ['sil'] + prondict[digit] + ['sil']


phoneHMMs_all = all_model['phoneHMMs'].item()
phones = sorted(phoneHMMs_all.keys())
nstates = {phone: phoneHMMs_all[phone]['means'].shape[0] for phone in phones}
stateList = [ph + '_' + str(id) for ph in phones for id in range(nstates[ph])]


def get_data(filename):
    count = 0
    dataset = []
    for root, dirs, files in os.walk(filename):
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
                dataset.append({'filename': filename, 'lmfcc': lmfcc, 'mspec': mspec,
                    'targets': targets})

    print(float(count)/87.00, "%%")

    return dataset


if __name__ == '__main__':
    train_data = get_data('tidigits/disc_4.1.1/tidigits/train')
    test_data  = get_data('tidigits/disc_4.2.1/tidigits/test')

    val   = np.concatenate((train_data[:461], train_data[-462:]))
    train = train_data[461:-462]

    np.save("data/val.npy", val)
    np.save("data/train.npy", train)
    np.save("data/test.npy", test)
    np.savez("data/traindata.npz", train_data=train_data)
    np.savez("data/testdata.npz", test_data=test_data)

