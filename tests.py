import numpy as np
import matplotlib.pyplot as plt
import keras
import pickle

from mnist import MNIST

import naive_nn as nnn
import twin_nn as tnn
import dbm
from preprocessing import *
# from oneshot import *

np.random.seed(0) # set seed

# load train data
X, y, alphabet_dict, char_dict = load_imgs('./data/omniglot/images_background')
n_classes, n_examples, w, h = X.shape

# load test data
X_test, y_test, alphabet_dict_test, _ = load_imgs('./data/omniglot/images_evaluation')

# tests
tests = ['preprocessing', 'dbm']
pretrained = True


for test in tests:
  if test =='preprocessing':
    X_train = preprocess_data_dbm(X)
    np.savetxt('data/dbm_train.dat', X_train)

  elif test == 'naive_nn':
    for i in range(2, 11):
      nnn.test_oneshot(i, 500, X_test, y_test, alphabet_dict_test, language=None, verbose=1)

  elif test == 'twin_nn':
    if not pretrained:
      twin_nn = tnn.create_model((w,h,1))
      twin_nn.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['binary_accuracy'])

      twin_nn.summary()

      batch_size = 32
      history = twin_nn.fit_generator(generator=tnn.training_generator(X, batch_size=batch_size),
                                      steps_per_epoch=(X.shape[0] * X.shape[1] / batch_size),
                                      epochs=30)
      twin_nn.save('models/twin_nn')
    else:
      twin_nn = keras.models.load_model('models/twin_nn')

      for i in range(2, 11):
        tnn.test_oneshot(twin_nn, i, 500, X_test, y_test, alphabet_dict_test, language=None, verbose=1)

  elif test == 'dbm':
    # X_train = np.loadtxt('data/dbm_train.dat')
    # dbm_model = dbm.create_model(X_train.shape[0], 100, 200)
    # dbm.train(dbm_model, X_train)

    with open('data/dbm_model.pickle', 'rb') as handle:
      dbm_model = pickle.load(handle)
    for i in range(2, 11):
      dbm.test_oneshot(dbm_model, i, 500, X_test, y_test, alphabet_dict_test, language=None, verbose=1)
