import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from mnist import MNIST

import naive_nn as nnn

np.random.seed(0) # set seed

# tests
tests = ['mnist']

for test in tests:
  if test == 'basic':
    anchors = np.array([[1,1], [1,-1], [-1,1], [-1,-1]])
    labels = np.array([0, 0, 1, 2], dtype='int')
    data = 2 * np.random.rand(500, 2) - 1

    naive_nn = nnn.Naive_NN(anchors, labels)
    y_ = naive_nn.classify(data)
    
    naive_nn.plot()
    
  elif test == 'mnist':
    mndata = MNIST('data')
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

