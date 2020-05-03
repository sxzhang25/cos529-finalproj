import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import keras

from mnist import MNIST

import naive_nn as nnn
import twin_nn as tnn
from preprocessing import *
from oneshot import *

np.random.seed(0) # set seed

# load train data
X, y, alphabet_dict, char_dict = load_imgs('./data/omniglot/images_background')
n_classes, n_examples, w, h = X.shape

# load test data
X_test, y_test, alphabet_dict_test, _ = load_imgs('./data/omniglot/images_evaluation')

# tests
tests = ['twin_nn']
pretrained = True


for test in tests:
  if test == 'basic':
    anchors = np.array([[1,1], [1,-1], [-1,1], [-1,-1]])
    labels = np.array([0, 0, 1, 2], dtype='int')
    data = 2 * np.random.rand(500, 2) - 1

    naive_nn = nnn.Naive_NN(anchors, labels)
    y_ = naive_nn.classify(data)

    naive_nn.plot()

  elif test == 'naive_nn':
    mndata = MNIST('data/mnist')
    scaler = StandardScaler()
    cpc = 2  # centers per class

    # prepare anchors
    anchors, anchor_labels = mndata.load_training()
    anchor_labels = np.array(anchor_labels)

    anchor_idxs = np.zeros((0,), dtype='int')
    for i in range(10):
      idxs = np.where(anchor_labels==i)[0]
      new_idxs = np.array([idxs[i] for i in np.random.randint(0, len(idxs), cpc)])
      anchor_idxs = np.concatenate((anchor_idxs, new_idxs), axis=0)

    anchors = np.array([anchors[idx] for idx in anchor_idxs]).T
    anchor_labels = np.array([anchor_labels[idx] for idx in anchor_idxs])

    test_data, test_labels = mndata.load_testing()  # load test data
    test_data = np.array(test_data).T
    test_labels = np.array(test_labels)

    index = np.random.randint(0, len(anchor_labels))  # choose an index
    print(mndata.display(anchors.T[index]))

    pca = PCA(n_components=2, svd_solver='arpack')
    pca.fit(np.hstack((anchors, test_data)))
    anchors_, test_data_ = pca.components_.T[:10 * cpc], pca.components_.T[10 * cpc:]

    naive_nn = nnn.Naive_NN(anchors_, anchor_labels)
    y_ = naive_nn.classify(test_data_)

    naive_nn.plot()

    print('\nAccuracy: %2.2f %%' % (100 * len(np.where(y_==test_labels)[0]) / len(y_)))

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

      for i in range(1, 11):
        tnn.test_oneshot(twin_nn, i, 500, X_test, y_test, alphabet_dict_test, language=None, verbose=1)
