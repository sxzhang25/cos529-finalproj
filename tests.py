import pickle
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras import backend as K
import tensorflow as tf
from sklearn.manifold import SpectralEmbedding

from mnist import MNIST

import naive_nn as nnn
import twin_nn as tnn
import dbm as dbm
from preprocessing import *
from oneshot import *


def plot_oneshot_task(pairs):
  test_img = pairs[0][1].reshape((105,105))
  support_imgs = pairs[1]

  fig = plt.figure(figsize=(15,3))
  fig.add_subplot(1, 11, 1)
  ax = plt.imshow(test_img)
  ax.axes.get_xaxis().set_ticks([])
  ax.axes.get_yaxis().set_ticks([])
  plt.title('input image')
  for i,support_img in enumerate(support_imgs):
    ax = fig.add_subplot(1, 11, i+2)
    plt.imshow(support_img.reshape((105,105)))
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    plt.title(i+1)
  plt.show()

def single_oneshot_task(X, y, N=10):
  pairs, targets = create_oneshot_task(X, y, alphabet_dict, N=10)
  naive_result = nnn.predict(pairs)
  dbm_result = dbm.dbm_predict(dbm_model, pairs)
  tnn_result = np.argmax(twin_nn.predict(pairs))

  plot_oneshot_task(pairs)

  print('Nearest neighbors prediction:       %d' % (naive_result+1))
  print('Deep Boltzmann Machine prediction:  %d' % (dbm_result+1))
  print('Twin neural network prediction:     %d' % (tnn_result+1))

  return pairs

def create_1_data(batch_size, X, category=None):
  '''
  create a batch of n pairs, all from the same class
  '''
  n_classes, n_examples, w, h = X.shape

  # choose a random category if none given
  if category is None:
    category = np.random.choice(n_classes)

  # initialize two empty arrays for input image batch
  pairs = [np.zeros((batch_size, h, w, 1)) for i in range(2)]

  for i in range(batch_size):
    idx1 = np.random.randint(0, n_examples)
    pairs[0][i,:,:,:] = X[category, idx1].reshape(w,h,1)
    idx2 = np.random.randint(0, n_examples)
    pairs[1][i,:,:,:] = X[category, idx2].reshape(w,h,1)

  return category, pairs

def create_0_data(batch_size, X, category=None):
  '''
  create a batch of n pairs, half from the same class and half from different
  classes
  '''
  n_classes, n_examples, w, h = X.shape

  # choose a random category if none given
  if category is None:
    category1 = np.random.choice(n_classes)
  else:
    category1 = category

  # initialize two empty arrays for input image batch
  pairs = [np.zeros((batch_size, h, w, 1)) for i in range(2)]

  for i in range(batch_size):
    idx1 = np.random.randint(0, n_examples)
    pairs[0][i,:,:,:] = X[category1, idx1].reshape(w,h,1)
    idx2 = np.random.randint(0, n_examples)

    # add a random number to the category modulo n_classes
    category2 = (category1 + np.random.randint(1, n_classes)) % n_classes

    pairs[1][i,:,:,:] = X[category2, idx2].reshape(w,h,1)

  return category, pairs

def draw_feature_vecs(X, model, n_samples):
  # create data from same and different classes
  c, data1 = create_1_data(n_samples, X)
  _, data0 = create_0_data(n_samples, X, category=c)

  # isolate last layer of model before dense layer
  layer_output = model.layers[-2].output
  activation_model = keras.models.Model(inputs=model.input, outputs=layer_output)

  features1 = activation_model.predict(data1)
  features0 = activation_model.predict(data0)
  features = np.concatenate((features1, features0), axis=0)

  # create diffusion map
  embedding = SpectralEmbedding(n_components=2)
  features_transformed = embedding.fit_transform(features)

  # plot classes
  fig = plt.figure(figsize=(8,6))
  ax = fig.add_subplot(111)
  ax.scatter(features_transformed[:n_samples,0], features_transformed[:n_samples,1], c='r', s=10, label='same class (1)')
  ax.scatter(features_transformed[n_samples:,0], features_transformed[n_samples:,1], c='b', s=10, label='diff class (0)')
  plt.title('Feature vectors for inputs from same/different classes')
  ax.legend()
  plt.show()


np.random.seed(0) # set seed

# load train data
X, y, alphabet_dict, char_dict = load_imgs('./data/omniglot/images_background')
n_classes, n_examples, w, h = X.shape

# load test data
X_test, y_test, alphabet_dict_test, _ = load_imgs('./data/omniglot/images_evaluation')

# tests
tests = ['all']
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
    # X_test = preprocess_data(X_test)

    with open('models/dbm_model.pickle', 'rb') as handle:
      dbm_model = pickle.load(handle)

    for i in range(2, 11):
      dbm.test_oneshot(dbm_model, i, 500, X_test, y_test, alphabet_dict_test, language=None, verbose=1)

  elif test == 'all':
    # load models
    twin_nn = keras.models.load_model('models/twin_nn')

    with open('models/dbm_model.pickle', 'rb') as handle:
      dbm_model = pickle.load(handle)

    # accuracies over N-way learning
    naive_accs = []
    twin_accs = []
    dbm_accs = []

    print('\n------NAIVE NEAREST NEIGHBORS------')
    for i in range(2, 21):
      naive_acc = nnn.test_oneshot(i, 500, X_test, y_test, alphabet_dict_test,
                                   language=None, verbose=1)
      naive_accs.append(naive_acc)

    print('\n------DEEP BOLTZMANN MACHINE------')
    for i in range(2, 21):
      dbm_acc = dbm.test_oneshot(dbm_model, i, 500, X_test, y_test, alphabet_dict_test,
                                 language=None, verbose=1)
      dbm_accs.append(dbm_acc)

    print('\n------TWIN NEURAL NETWORK------')
    for i in range(2, 21):
      twin_acc = tnn.test_oneshot(twin_nn, i, 500, X_test, y_test, alphabet_dict_test,
                                  language=None, verbose=1)
      twin_accs.append(twin_acc)

    # plot accuracies of different models
    y = np.arange(2, 21)
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    ax.plot(y, naive_accs, label='nearest neighbors')
    ax.plot(y, dbm_accs, label='deep boltzmann machine')
    ax.plot(y, twin_accs, label='twin neural network')
    plt.xlabel('N')
    plt.ylabel('Accuracy')
    plt.title('N-way one-shot recognition task accuracies')
    plt.legend()
    plt.show()

  elif test == 'iso':
    pairs = single_oneshot_task(X, y)

  elif test == 'tnn_conv':
    # isolate first convolutional layer
    layers = twin_nn.layers
    filters1 = layers[2].layers[0].weights[0]

    # plot filters
    fig = plt.figure()
    for i in range(filters1.shape[3]):
      ax = fig.add_subplot(8,8,i+1)
      filter_i = filters1[:,:,:,i].numpy().reshape(10,10)
      plt.imshow(filter_i, cmap='Greys',  interpolation='nearest')
      ax.axes.get_xaxis().set_ticks([])
      ax.axes.get_yaxis().set_ticks([])
    plt.show()

  elif test == 'tnn_features':
    draw_feature_vecs(X, twin_nn, 800)
