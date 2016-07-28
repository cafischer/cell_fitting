from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# test data
n = 1000000
x = np.random.random(n)
a = np.random.randint(0, 10, n)
data = (x - a)
labels = a

x_test = np.array([0.11, 0.45, 0.45, 0.8])
a_test = np.array([2, 1, 9, 9])
data_test = (x_test - a_test)
labels_test = a_test

# for a single-input model
model = Sequential()
model.add(Dense(1, input_dim=1, activation='tanh'))
model.add(Dense(1, input_dim=1, activation='tanh'))
model.add(Dense(1, input_dim=1, activation='tanh'))
model.add(Dense(1, input_dim=1, activation='linear'))
model.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['accuracy'])

# train the model, iterating on the data in batches
model.fit(data, labels, nb_epoch=10, batch_size=50)

# test the model
labels_predicted = model.predict(data_test, batch_size=1, verbose=0)

print labels_test
print labels_predicted