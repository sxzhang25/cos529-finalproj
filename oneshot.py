import numpy as np

def create_oneshot_task(X, labels, alphabet_dict, N=1, language=None, task_type='simple'):
  '''
  create pairs of test images, support set for testing N-way one-shot learning
  '''
  n_classes, n_examples, w, h = X.shape
  M = 1

  indices = np.random.randint(0, n_examples, size=(N,))
  if language is not None:  # select characters from specified language
    low, high = alphabet_dict[language]
    if N < high - low:
      raise ValueError("ERROR: this language ({}) has less than {} letters".format(language, N))
    if task_type == 'simple':
      categories = np.random.choice(range(low, high), size=(N-M+1,), replace=False)
    elif task_type == 'general':
      M = np.random.randint(1, N // 2 + 1)
      categories = np.random.choice(range(low, high), size=(N-M+1,), replace=True)
    else:
      raise ValueError("ERROR: unknown task type ({})".format(task_type))
  else:
    if task_type == 'simple':
      categories = np.random.choice(n_classes, size=(N-M+1,), replace=False)
    elif task_type == 'general':
      M = np.random.randint(1, N // 2 + 1)
      categories = np.random.choice(n_classes, size=(N-M+1,), replace=True)
    else:
      raise ValueError("ERROR: unknown task type ({})".format(task_type))

  true_category = categories[0]
  exs = np.random.choice(n_examples, replace=False, size=(M+1,))
  test_img = np.asarray([X[true_category, exs[0],:,:]] * N)
  test_img = np.expand_dims(test_img, axis=3)

  targets = np.zeros((N,))
  targets[:M] = 1  # first support image is from the same class

  categories = np.concatenate((np.ones(M, dtype=int) * true_category, categories[1:]))
  support_imgs = X[categories, indices,:,:]
  support_imgs[:M,:,:] = X[true_category, exs[1:]]  # set same class comparison image
  support_imgs = np.expand_dims(support_imgs, axis=3)
  pairs = [test_img, support_imgs]

  return pairs, targets, M