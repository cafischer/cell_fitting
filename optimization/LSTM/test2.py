from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# test data
def fun(x):
    return 3 * (x)

n = 100000
data = np.random.random(n)
labels = fun(data)

data_test = np.array([0.11, 0.45, 0.88])
labels_test = fun(data_test)

# for a single-input model
model = Sequential()
model.add(Dense(1, input_dim=1, activation='linear'))
model.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['accuracy'])

# train the model, iterating on the data in batches
model.fit(data, labels, nb_epoch=10, batch_size=100)

# test the model
labels_predicted = model.predict(data_test, batch_size=1, verbose=0)

print labels_test
print labels_predicted

json_string = model.to_json()
open('model.json', 'w').write(json_string)
model.save_weights('model_weights.h5', overwrite=True)