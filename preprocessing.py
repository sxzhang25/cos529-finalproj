import numpy as np
import os
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

def preprocess_data_dbm(X):
  X = np.reshape(X, (964, 20, -1))
  X = np.reshape(X, (-1, 11025))

  return X
