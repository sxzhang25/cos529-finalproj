import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import colorsys

from oneshot import *

def predict(data):
  '''
  given labeled anchor examples, classify each data point using nearest neighbors
  '''
  distances = np.zeros(data[0].shape[0])
  for i in range(distances.shape[0]):
    distances[i] = np.linalg.norm(data[0][i] - data[1][i])
  y = np.argmin(distances)

  return y

def test_oneshot(N, k, data, labels, alphabet_dict, language=None, verbose=0):
  '''
  Test average N-way oneshot learning accuracy of model over k one-shot tasks
  '''
  correct = 0
  if verbose:
    print("Evaluating model on {} random {}-way one-shot learning tasks...".format(k,N))

  for i in range(k):
    inputs, targets = create_oneshot_task(data, labels, alphabet_dict, N=N, language=language)
    y = predict(inputs)
    if y == np.argmax(targets):
      correct += 1

  accuracy = (100 * correct / k)
  if verbose:
    print("Average %d-way one-shot accuracy: %4.2f%%" % (N, accuracy))

  return accuracy
