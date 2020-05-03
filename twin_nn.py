import numpy as np

from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from keras.regularizers import l2
from keras import backend as K

from oneshot import *

def get_batch(batch_size, X):
  '''
  create a batch of n pairs, half from the same class and half from different
  classes
  '''
  n_classes, n_examples, w, h = X.shape

  # randomly sample several classes to use in the batch
  categories = np.random.choice(n_classes, size=(batch_size,), replace=False)

  # initialize two empty arrays for input image batch
  pairs = [np.zeros((batch_size, h, w, 1)) for i in range(2)]

  # initialize target vectors
  targets = np.zeros((batch_size,))
  targets[batch_size // 2:] = 1  # half of the targets are '1's (different class)

  for i in range(batch_size):
    category1 = categories[i]
    idx1 = np.random.randint(0, n_examples)
    pairs[0][i,:,:,:] = X[category1, idx1].reshape(w,h,1)
    idx2 = np.random.randint(0, n_examples)

    # pick images of same class for first half, different for second half
    if i >= batch_size // 2:
      category2 = category1
    else:
      # add a random number to the category modulo n_classes
      category2 = (category1 + np.random.randint(1, n_classes)) % n_classes

    pairs[1][i,:,:,:] = X[category2, idx2].reshape(w,h,1)

  return pairs, targets

def training_generator(X, batch_size=32):
  '''
  a generator for model.fit_generator()
  '''
  while True:
    pairs, targets = get_batch(batch_size, X)
    yield(pairs, targets)

def create_model(input_shape):
  '''
  set up twin neural net model architecture
  '''

  # define tensors for two input images
  left_input = Input(input_shape)
  right_input = Input(input_shape)

  # CNN architecture
  model = Sequential()
  model.add(Conv2D(64, (10,10), activation='relu',
            input_shape=input_shape,
            kernel_regularizer=l2(2e-4)))
  model.add(MaxPooling2D())
  model.add(Conv2D(128, (7,7), activation='relu', kernel_regularizer=l2(2e-4)))
  model.add(MaxPooling2D())
  model.add(Conv2D(128, (4,4), activation='relu', kernel_regularizer=l2(2e-4)))
  model.add(MaxPooling2D())
  model.add(Conv2D(256, (4,4), activation='relu', kernel_regularizer=l2(2e-4)))
  model.add(Flatten())
  model.add(Dense(4096, activation='sigmoid', kernel_regularizer=l2(1e-3)))

  # generate encodings (feature vectors) for the two images
  encoded_l = model(left_input)
  encoded_r = model(right_input)

  # add a custom layer to compute L1 difference between the encodings
  L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
  L1_distance = L1_layer([encoded_l, encoded_r])

  # dense layer with sigmoid unit to generate similarity score
  prediction = Dense(1, activation='sigmoid')(L1_distance)

  # connect inputs with outputs
  twin_nn = Model(inputs=[left_input, right_input], outputs=prediction)

  return twin_nn

def test_oneshot(model, N, k, data, labels, alphabet_dict, language=None, verbose=0):
  '''
  Test average N-way oneshot learning accuracy of model over k one-shot tasks
  '''
  correct = 0
  if verbose:
    print("Evaluating model on {} random {}-way one-shot learning tasks...".format(k,N))

  for i in range(k):
    inputs, targets = create_oneshot_task(data, labels, alphabet_dict, N=N, language=language)
    probs = model.predict(inputs)
    if np.argmax(probs) == np.argmax(targets):
      correct += 1

  accuracy = (100 * correct / k)
  if verbose:
    print("Average %d-way one-shot accuracy: %4.2f%%" % (N, accuracy))

  return accuracy
