from keras.models import Sequential
from keras.utils.visualize_util import plot
from keras.layers import LSTM, Dense, Activation
import numpy as np

# test data
def fun(t, a):
    return 3 * t - a

n = 10000
t = np.arange(0, 10, 1)
a = np.random.random(n)
data = np.zeros((n, len(t), 1), dtype=object)
for i in range(n):
    data[i, :, 0] = fun(t, a[i])
labels = a

n_test = 3
a_test = np.random.random(n_test)
data_test = np.zeros((n_test, len(t), 1), dtype=object)
for i in range(n_test):
    data_test[i, :, 0] = fun(t, a_test[i])
labels_test = a_test

# model
model = Sequential()
model.add(LSTM(output_dim=len(t), input_shape=(len(t), 1)))
model.add(Dense(1, input_dim=len(t), activation='linear'))
model.compile(optimizer='rmsprop', loss='mse')

plot(model)

# train the model, iterating on the data in batches
model.fit(data, labels, nb_epoch=10, batch_size=100)

# test the model
labels_predicted = model.predict(data_test, batch_size=1, verbose=0)

print labels_test
print labels_predicted