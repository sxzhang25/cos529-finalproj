import numpy as np

def create_oneshot_task(X, labels, alphabet_dict, N=1, language=None):
  '''
  create pairs of test images, support set for testing N-way one-shot learning
  '''
  n_classes, n_examples, w, h = X.shape

  indices = np.random.randint(0, n_examples, size=(N,))
  if language is not None:  # select characters from specified language
    low, high = alphabet_dict[language]
    if N < high - low:
      raise ValueError("ERROR: this language ({}) has less than {} letters".format(language, N))
    categories = np.random.choice(range(low, high), size=(N,), replace=False)
  else:
    categories = np.random.choice(n_classes, size=(N,), replace=False)

  true_category = categories[0]
  ex1, ex2 = np.random.choice(n_examples, replace=False, size=(2,))
  test_img = np.asarray([X[true_category, ex1,:,:]] * N).reshape(N,w,h,1)

  targets = np.zeros((N,))
  targets[0] = 1  # first support image is from the same class

  support_imgs = X[caetgories, indices,:,:]
  support_imgs[0,:,:,:] = X[true_category, ex2]  # set same class comparison image
  pairs = [test_img, support_imgs]

  return pairs, targets

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
      print("Average %d-way one-shot accuracy: %4.2f%%" % (accuracy, N))

    return accuracy
