import numpy as np
import keras
import sys
from all_tools_proto import path2info
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils, to_categorical

def dynamic_features(feature):
    N = feature.shape[0]
    M = feature.shape[1] * 7
    dyn = np.zeros((N, M), dtype=np.float32)

    feature = np.vstack((np.flip(feature[1:4, :], 0),
                      feature,
                      np.flip(feature[-4:-1, :], 0)))
    for i in range(N):
        dyn[i, :] = feature[i:i+7].flatten()

    return dyn




def load_data():

    traindata = np.load("./data/train.npy", allow_pickle=True)
    valdata = np.load("./data/val.npy", allow_pickle=True)
    testdata = np.load("./data/testdata.npz", allow_pickle=True)["testdata"]

    return (traindata, valdata, testdata)



def stack(data, feature):
    C = 0
    sizes = np.zeros(len(data), dtype="int32")

    for i in range(len(data)):
        sizes[i] = data[i][feature].shape[0]

    C = np.sum(sizes, dtype="int32")
    N =  data[0][feature].shape[1]
    output_data = np.zeros((C, N), dtype="float32")
    out_labels = np.zeros((C, 1), dtype="float32")

    start_idx = 0

    for i in range(len(data)):
        output_data[start_idx: start_idx + sizes[i], :] = data[i][feature]
        out_labels[start_idx: start_idx + sizes[i]] = np.array(data[i]["targets"]).reshape(sizes[i],1)
        start_idx += sizes[i]

    return output_data, to_categorical(out_labels, num_classes=61)

def normalize(data, feature, dyn_feature):
    from sklearn.preprocessing import StandardScaler

    if dyn_feature:
        for dataset in data:
            for i in range(len(dataset)):
                dataset[i][feature] = dynamic_features(dataset[i][feature])

    traindata, trainy = stack(data[0], feature)
    valdata, valy = stack(data[1], feature)
    testdata, testy = stack(data[2], feature)


    scaler = StandardScaler(copy=False)
    scaler.fit_transform(traindata)
    mean = scaler.mean_
    var  = scaler.var_

    scaler.transform(valdata)
    scaler.transform(testdata)

    data = [traindata, valdata, testdata]
    labels = [trainy, valy, testy]
    return (data, labels)


def normalize_spekers(data, feature, dyn_feature):
    from sklearn.preprocessing import StandardScaler

    if dyn_feature:
        for dataset in data:
            for i in range(len(dataset)):
                dataset[i][feature] = dynamic_features(dataset[i][feature])

    man_idx = []
    wom_idx = []

    for i in range(len(data[0])):
        gender, speakerID, digits, repetition = path2info(data[0][i]["filename"])
        if gender == "man":
            man_idx.append(i)
        else:
            wom_idx.append(i)

    traindataman, trainyman = stack(data[0][man_idx], feature)
    traindatawom, trainywom = stack(data[0][wom_idx], feature)


    scalerman = StandardScaler(copy=False)
    scalerman.fit_transform(traindataman)
    #mean = scalerman.mean_
    #var  = scalerman.var_


    scalerwom = StandardScaler(copy=False)
    scalerwom.fit_transform(traindatawom)
    #mean = scalerwom.mean_
    #var  = scalerwom.var_


    man_idx = []
    wom_idx = []

    for i in range(len(data[1])):
        gender, speakerID, digits, repetition = path2info(data[1][i]["filename"])
        if gender == "man":
            man_idx.append(i)
        else:
            wom_idx.append(i)



    valdataman, valyman = stack(data[1][man_idx], feature)
    valdatawom, valywom = stack(data[1][wom_idx], feature)
    scalerman.transform(valdataman)
    scalerwom.transform(valdatawom)


    print(np.mean(valdataman, axis=0))
    print(np.mean(valdatawom, axis=0))
    print(np.var(valdataman, axis=0))
    print(np.var(valdatawom, axis=0))


    man_idx = []
    wom_idx = []

    for i in range(len(data[2])):
        gender, speakerID, digits, repetition = path2info(data[2][i]["filename"])
        if gender == "man":
            man_idx.append(i)
        else:
            wom_idx.append(i)


    testdataman, testyman = stack(data[2][man_idx], feature)
    testdatawom, testywom = stack(data[2][wom_idx], feature)

    scalerman.transform(testdataman)
    scalerwom.transform(testdatawom)

    traindata = np.vstack((traindataman,traindatawom))
    valdata = np.vstack((valdataman,valdatawom))
    testdata = np.vstack((testdataman,testdatawom))

    trainy = np.vstack((trainyman,trainywom))
    valy = np.vstack((valyman,valywom))
    testy = np.vstack((testyman,testywom))


    data = [traindata, valdata, testdata]
    labels = [trainy, valy, testy]
    return (data, labels)


def run_network(data,labels, epochs = 10, batch_size=256):

    X, Xval, Xtest = data
    Y, Yval, Ytest = labels

    no_features = X.shape[1]
    no_phonemes = 61
    print("input:", no_features)
    model = Sequential()

    ### LAYERS
    # Layer 1
    model.add(Dense(256, input_dim=no_features, kernel_initializer='he_normal'))
    model.add(Activation('relu'))

    # Layer 2
    model.add(Dense(256, input_dim=256, kernel_initializer='he_normal'))
    model.add(Activation('relu'))

    # Layer 3
    model.add(Dense(256, input_dim=256, kernel_initializer='he_normal'))
    model.add(Activation('relu'))

    # Layer 4
    model.add(Dense(no_phonemes, input_dim=256, kernel_initializer='he_normal'))
    model.add(Activation('softmax'))


    ### COMPILER
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy", "categorical_accuracy"])
    ### TRAINING
    #data   = np.random.random((1000, 13))
    #labels = np.random.randint(61, size=(1000, 1))
    #one_hot_labels = to_categorical(labels, num_classes=61)

    checkpoint = keras.callbacks.ModelCheckpoint('best_model.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='min')
    validation = (Xval, Yval)

    # setup tensorboard
    callback = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, \
                write_graph=True, write_images=True)
    model.summary()
    model.fit(X, Y, epochs=epochs, batch_size=batch_size, \
                validation_data=validation, callbacks=[callback, checkpoint])



    ### EVALUATE
    saved_model = keras.models.load_model('best_model.h5')

    score = model.evaluate(Xtest, Ytest)

    print("Final score : ", score)
    for i in range(len(score)):
        print(model.metrics_names[i], " : ", score[i])

    score = saved_model.evaluate(Xtest, Ytest)
    print("Final score on best model: ", score)
    for i in range(len(score)):
        print(model.metrics_names[i], " : ", score[i])


    return model


# Intermediate layers should be ReLU
# Output layer should be softmax

if __name__ == "__main__":


    if len(sys.argv) == 3:
        feature = sys.argv[1]
        dyn_feature = bool(sys.argv[2])
    elif len(sys.argv) == 2:
        feature = sys.argv[1]
        dyn_feature = False
    else:
        feature = "lmfcc"
        dyn_feature = False

    print("load data")

    data = load_data()


    print("normalize")


    data = normalize_spekers(data, feature, dyn_feature)

    print("Shuffle data")

    for i in range(3):

        print(data[0][i].shape, data[1][i].shape)
        indexes = np.random.permutation(data[0][i].shape[0])
        data[0][i] = data[0][i][indexes, : ]
        data[1][i] = data[1][i][indexes, : ]






    print("train")

    run_network(data[0],data[1], epochs = 5, batch_size=256)

    print("end")
