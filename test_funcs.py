import numpy as np
from all_tools_proto import mfcc
from prondict import prondict

###########################################################
######## INITIALIZATION ###################################
###########################################################
example = np.load('lab3_example.npz', allow_pickle=True)["example"]
example.shape=(1,)
example = example[0]
one_speaker_model = np.load('lab2_models_onespkr.npz', allow_pickle=True)
all_model = np.load('lab2_models_all.npz', allow_pickle=True)

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
print(len(stateList))

print(example.keys())
np.seterr(divide='ignore')
print("#"*50 + "\n" + "#"*50 + "\n" + "#"*50)
###########################################################
######## INITIALIZATION END ###############################
###########################################################


###########################################################
######## 4. Preparing the Data for DNN Training ###########


filename = './tidigits/disc_4.1.1/tidigits/train/man/nw/z43a.wav'

# 4.2 Forced Alignment
samples, samplingrate = loadAudio(filename)

print("Error on Samples : ", np.sum(np.abs(samples - example["samples"])))

lmfcc = mfcc(samples)
lmfcc2 = mfcc(example["samples"])

print(lmfcc-lmfcc2, "\n"*3)

print("(size ",lmfcc.shape == example["lmfcc"].shape, ")")

# check lmfcc
error_lmfcc = np.max(np.abs(lmfcc - example["lmfcc"]))
print(lmfcc)
print("\n\n\n\n")
print(example["lmfcc"])

ratio =  example["lmfcc"] / lmfcc

print("LMFCC error is : ", error_lmfcc)

if error_lmfcc > 0.1:
    lmfcc = example["lmfcc"]
    error_lmfcc = np.max(np.abs(lmfcc - example["lmfcc"]))
    print("New LMFCC error is : ", error_lmfcc)

wordTrans = list(path2info(filename)[2])

# phone transcription
phoneTrans = words2phones(wordTrans, prondict)

# Check the result is the same
check = True
for i, e in enumerate(phoneTrans):
    check = check and (e == example["phoneTrans"][i])

print("PhoneTrans is : ", check)
utteranceHMM = concatHMMs(phoneHMMs_all, phoneTrans)

# check the HMM is correct
for k, values in example["utteranceHMM"].items():
    print("Error on ",k, " is : ", \
            np.max(np.abs(utteranceHMM[k] - example["utteranceHMM"][k]))\
            , "  ( size:",(utteranceHMM[k].shape == example["utteranceHMM"][k].shape),")")

    stateTrans = [phone + '_' + str(stateid) for phone in phoneTrans \
            for stateid in range(nstates[phone])]

    # Check if it is correct
check = True

for i, e in enumerate(stateTrans):
    check = check and (e == example["stateTrans"][i])

print("StateTrans is : ", check)


# aligning states
data_log_lik = log_multivariate_normal_density_diag(
        lmfcc, utteranceHMM["means"], utteranceHMM["covars"])

error_data_log_lik = np.max(np.abs(data_log_lik - example["obsloglik"]))
print("data_log_lik error is : ", error_data_log_lik \
        ,"  ( size:",(data_log_lik.shape == example["obsloglik"].shape),")")

viterbi_loglik, viterbi_path = viterbi(data_log_lik,
        np.log(utteranceHMM["startprob"]),
        np.log(utteranceHMM["transmat"]))

print("Viterbi log lik error is : ", np.abs(viterbi_loglik - example["viterbiLoglik"]))

aligned_states = forcedAlignment(lmfcc, phoneHMMs_all, phoneTrans)

check = True
for i, e in enumerate(aligned_states):
    check = check and (e == example["viterbiStateTrans"][i])

print("viterbiStateTrans is : ", check, "  (",len(aligned_states) ==  len(example["viterbiStateTrans"]),")")

