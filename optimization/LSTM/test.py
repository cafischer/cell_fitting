from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
import numpy as np

# test data
n = 100000
data = np.random.random(n)
labels = np.zeros(len(data))
labels[data < 0.3] = 0
labels[np.logical_and(data > 0.3, data < 0.6)] = 1
labels[data > 0.6] = 2
labels = to_categorical(labels, 3)

data_test = np.array([0.11, 0.45, 0.88])
labels_test = np.array([0, 1, 2])

# for a single-input model
model = Sequential()
model.add(Dense(3, input_dim=1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['accuracy'])

# train the model, iterating on the data in batches
model.fit(data, labels, nb_epoch=10, batch_size=50)

# test the model
labels_predicted = model.predict_classes(data_test, batch_size=1, verbose=0)

print labels_test
print labels_predicted