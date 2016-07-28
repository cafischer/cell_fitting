from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.layers import Merge
from keras.utils.visualize_util import plot
import numpy as np

# test data
def fun(x, a):
    return 3 * (-x+a)

n = 100000
x = np.random.random(n)
a = np.random.random(n) * 2
data = fun(x, a)
labels = a

x_test = np.array([0.11, 0.45, 0.45, 0.8])
a_test = np.array([0.1, 1, 2, 2])
data_test = fun(x_test, a_test)
labels_test = a_test

# for a single-input model
left_branch = Sequential()
left_branch.add(Dense(1, input_dim=1, activation='linear'))
#left_branch = model_from_json(open('sin_model.json').read())
#left_branch.load_weights('sin_model_weights.h5')

right_branch = Sequential()
right_branch.add(Dense(1, input_dim=1, activation='linear'))
#right_branch.set_weights([np.array([[1]]), np.array([0])])

merged = Merge([left_branch, right_branch], mode='sum')

final_model = Sequential()
final_model.add(merged)
final_model.add(Dense(1, input_dim=1, activation='linear'))
#final_model.set_weights(left_branch.get_weights()+right_branch.get_weights()+[np.array([[1]]), np.array([0])])
final_model.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['accuracy'])

#plot(final_model)

# train the model, iterating on the data in batches
final_model.fit([data, x], labels, nb_epoch=10, batch_size=100)

# test the model
labels_predicted = final_model.predict([data_test, x_test], batch_size=1, verbose=0)

print labels_test
print labels_predicted