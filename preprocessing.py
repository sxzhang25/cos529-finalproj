import numpy as np
import os
from mnist import MNIST
import cv2

from sklearn.decomposition import PCA

def load_imgs(path, n=0):
  '''
  load images from path
  '''

  X = []
  y = []

  alphabet_dict = {}
  char_dict = {}

  curr_y = n

  # load every alphabet separately
  for alphabet in os.listdir(path):
    print("Loading alphabet:", alphabet)
    alphabet_dict[alphabet] = [curr_y, None]
    alphabet_path = os.path.join(path, alphabet)

    # load every character into its own column in array
    for char in os.listdir(alphabet_path):
      char_dict[curr_y] = (alphabet, char)
      char_imgs = []
      char_path = os.path.join(alphabet_path, char)

      # read all images of current character
      for filename in os.listdir(char_path):
        img_path = os.path.join(char_path, filename)
        img = cv2.imread(img_path, 0)
        char_imgs.append(img)
        y.append(curr_y)

      try:
        X.append(np.stack(char_imgs))
      except ValueError as e:  # edge case: last one
        print(e)
        print("ERROR: char_imgs\n", char_imgs)

      alphabet_dict[alphabet][1] = curr_y
      curr_y += 1

  y = np.vstack(y)
  X = np.stack(X)
  return X, y, alphabet_dict, char_dict

def load_mnist(path, n=0):
  # load mnist digits
  mndata = MNIST(path)
  images, labels = mndata.load_testing()
  
  # sort mnist digits by label
  idx = np.argsort(labels)
  images_sorted = [images[i] for i in idx]
  labels_sorted = [labels[i] for i in idx]
  
  X = []
  y = labels_sorted

  alphabet_dict = {}
  char_dict = {}
  curr_y = n

  # load every alphabet separately
  alphabet = 'MNIST digits'
  print("Loading alphabet:", alphabet)
  alphabet_dict[alphabet] = [curr_y, None]

  # load every character into its own column in array
  for i in range(10):
    char_dict[curr_y] = (alphabet, i)
    char_imgs = []

    # read all images of current character
    for i in range(1000*i, 1000*(i+1)):
      img = -np.array(images_sorted[i])
      img = np.round(img / 255).reshape(28, 28)
      img = np.round(cv2.resize(img, (105, 105)))
      char_imgs.append(img)

    try:
      X.append(np.stack(char_imgs))
    except ValueError as e:  # edge case: last one
      print(e)
      print("ERROR: char_imgs\n", char_imgs)

    alphabet_dict[alphabet][1] = curr_y
    curr_y += 1

  alphabet_dict[alphabet][1] = 10
  X = np.stack(X)
  
  return X, y, alphabet_dict, char_dict

def preprocess_data(X):
  X = np.reshape(X, (X.shape[0], X.shape[1], -1))
  X = np.reshape(X, (-1, X.shape[2])).T

  return X
