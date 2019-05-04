import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils, to_categorical

# Feature vectors: 
no_features = 13 #91
no_phonemes = 61
# Number of states:

model = Sequential()

### LAYERS
# Layer 1
model.add(Dense(256, input_dim=no_features))
model.add(Activation('relu'))

# Layer 2
model.add(Dense(256, input_dim=256))
model.add(Activation('relu'))

# Layer 3
model.add(Dense(256, input_dim=256))
model.add(Activation('relu'))

# Layer 4
model.add(Dense(no_phonemes, input_dim=256))
model.add(Activation('softmax'))


### COMPILER
model.compile(optimizer="rmsprop",
              loss="categorical_crossentropy",
              metrics=["accuracy", "categorical_accuracy", "mae"])

### TRAINING
data   = np.random.random((1000, 13))
labels = np.random.randint(61, size=(1000, 1))
one_hot_labels = to_categorical(labels, num_classes=61)


validation = (valX, valY)


model.fit(data, one_hot_labels, epochs=10, batch_size=32, validation_data=validation)


### EVALUATE

#score = model.evaluate(x_test, y_test)


# Intermediate layers should be ReLU
# Output layer should be softmax


