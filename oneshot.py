import numpy as np

def create_oneshot_task(X, labels, alphabet_dict, N=1, seed=2, language=None):
  '''
  create pairs of test images, support set for testing N-way one-shot learning
  '''
  np.random.seed(seed)
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
  test_img = np.asarray([X[true_category, ex1,:,:]] * N)
  test_img = np.expand_dims(test_img, axis=3)

  targets = np.zeros((N,))
  targets[0] = 1  # first support image is from the same class

  support_imgs = X[categories, indices,:,:]
  support_imgs[0,:,:] = X[true_category, ex2]  # set same class comparison image
  support_imgs = np.expand_dims(support_imgs, axis=3)
  pairs = [test_img, support_imgs]

  return pairs, targets
