import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import LSTM, Dense

# parameter
save_dir = './data/toymodel1/na8st/3/'
n_epochs = 5
batch_size = 10
percent_train = 0.9

# load data
with open(save_dir+'data.npy', 'r') as f:
    data = np.load(f)
with open(save_dir+'labels.npy', 'r') as f:
    labels = np.load(f)

# divide in train and test
(n_samples, len_time, n_inputs) = np.shape(data)
idx = int(n_samples * percent_train)
data_train = data[:idx, :, :]
labels_train = labels[:idx]
data_test = data[idx:, :, :]
labels_test = labels[idx:]

n_outputs = 1  # for now na8st gbar

"""
# create LSTM network
model = Sequential()
model.add(LSTM(output_dim=len_time, input_shape=(len_time, n_inputs)))
model.add(Dense(len_time, input_dim=len_time, activation='sigmoid'))
model.add(Dense(n_outputs, input_dim=len_time, activation='linear'))
model.compile(optimizer='rmsprop', loss='mse')

# training
model.fit(data_train, labels_train, nb_epoch=n_epochs, batch_size=batch_size, verbose=1)

# save network and weights
open(save_dir+'model.json', 'w').write(model.to_json())
model.save_weights(save_dir+'model_weights.h5', overwrite=True)
"""
# load network
model = model_from_json(open(save_dir+'model.json').read())
model.load_weights(save_dir+'model_weights.h5')

# testing
labels_predicted = model.predict(data_test)

print labels_test
print labels_predicted.T