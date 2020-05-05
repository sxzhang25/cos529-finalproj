import numpy as np

from oneshot import *

def initialize_W(a, size):
  '''
  initialize random weight matrix according to normalized initialization
  http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
  '''
  W = np.random.uniform(-a, a, size=size)
  return W

def initialize_b(size):
  '''
  initialize bias vector
  '''
  return np.zeros((size,))

def create_model(n_visible, n_hidden1, n_hidden2):
  '''
  creates a two-layer deep boltzmann machine

  matrix notation pulled from:temp
  http://proceedings.mlr.press/v5/tempsalakhutdinov09a/salakhutdinov09a.pdf
  '''
  model = {}

  a = np.sqrt(6 / n_visible + n_hidden1) # initialization factor
  W = initialize_W(a, (n_visible, n_hidden1)) # visible-to-hidden weight matrices
  b_w = initialize_b(n_hidden1) # bias vector
  a = np.sqrt(6 / n_hidden1 + n_hidden2);
  J = initialize_W(a, (n_hidden1, n_hidden2)) # hidden-to-hidden weigh matrices
  b_j = initialize_b(n_hidden2) # bias vector

  # initialize persistent markov chains
  b_v = initialize_b(n_visible)
  b_h1 = initialize_b(n_hidden1)
  b_h2 = initialize_b(n_hidden2)

  # set model
  model['W'] = W
  model['b_w'] = b_w
  model['J'] = J
  model['b_j'] = b_j
  model['b_v'] = b_v
  model['b_h1'] = b_h1
  model['b_h2'] = b_h2

  return model

def sigmoid(X):
  return 1 / (1 + np.exp(-X))

def cross_entropy(X, model):
  W, J, b_v = model['W'], model['J'], model['b_v']
  h2 = np.random.rand(J.shape[1], X.shape[1])
  h1, _, _ = gibbs_sampling(h2, h2, X, model, 1)
  v = sigmoid(W @ h1 + np.array([b_v,] * X.shape[1]).T)
  cross_entropy = -np.mean(np.sum(X * np.log(v) + (1 - X) * np.log(1 - v), axis=0))
  return cross_entropy

def mean_field_update(X, model, delta):
  W, J = model['W'], model['J']
  b_w, b_j = model['b_w'], model['b_j']
  mu1 = np.random.rand(J.shape[0], X.shape[1])
  mu2 = np.random.rand(J.shape[1], X.shape[1])
  for i in range(10):
    mu1_ = sigmoid(W.T @ X + J @ mu2 + np.array([b_w,] * X.shape[1]).T)
    mu2_ = sigmoid(J.T @ mu1 + np.array([b_j,] * X.shape[1]).T)

    delta_mu1 = np.linalg.norm(mu1_ - mu1)
    delta_mu2 = np.linalg.norm(mu2_ - mu2)

    mu1, mu2 = mu1_, mu2_
    if (delta_mu1 < delta and delta_mu2 < delta):
      break

  return mu1, mu2

def gibbs_sampling(h1, h2, v, model, n_steps):
  W, J = model['W'], model['J']
  b_v, b_h1, b_h2 = model['b_v'], model['b_h1'], model['b_h2']
  for i in range(n_steps):
    h1 = sigmoid(W.T @ v + J @ h2 + np.array([b_h1,] * v.shape[1]).T)
    h1 = np.random.binomial(1, h1)
    h2 = sigmoid(J.T @ h1 + np.array([b_h2,] * v.shape[1]).T)
    h2 = np.random.binomial(1, h2)
    v = sigmoid(W @ h1 + np.array([b_v,] * v.shape[1]).T)
    v = np.random.binomial(1, v)
  return h1, h2, v

def train(model, X_train, n_epochs=10, K=100, batch_size=32, mf_delta=1, gibbs_steps=2, lr=0.001):
  np.random.seed(0)
  W, J = model['W'], model['J']
  b_v, b_h1, b_h2 = model['b_v'], model['b_h1'], model['b_h2']

  train_error = []

  n_batches = X_train.shape[1] // batch_size
  for epoch in range(1, n_epochs+1):
    # persistent markov chains
    n_hidden1, n_hidden2 = model['J'].shape
    v = np.random.binomial(1, 0.5, (X_train.shape[0], K))
    h1 = np.random.binomial(1, 0.5, (n_hidden1, K))
    h2 = np.random.binomial(1, 0.5, (n_hidden2, K))

    sys.stdout.write('\n')
    for i in range(1, n_batches+1):
      batch = X_train[:, np.random.permutation(X_train.shape[1])[:batch_size]]

      mu1, mu2 = mean_field_update(batch, model, mf_delta)
      h1, h2, v = gibbs_sampling(h1, h2, v, model, gibbs_steps)

      h1_batch = sigmoid(W.T @ batch + J @ mu2 + np.array([b_h1,] * batch.shape[1]).T)
      h1_v = sigmoid(W.T @ v + J @ h2 + np.array([b_h1,] * K).T)
      h2_batch = sigmoid(J.T @ mu1 + np.array([b_h2,] * batch.shape[1]).T)
      h2_v = sigmoid(J.T @ h1 + np.array([b_h2,] * K).T)

      # update weights
      W += lr * (batch @ mu1.T / batch.shape[1] - v @ h1.T / v.shape[1])
      J += lr * (mu1 @ mu2.T / mu1.shape[1] - h1 @ h2.T / h1.shape[1])
      b_v += lr * (np.sum(batch, axis=1) / batch.shape[1] - np.sum(v, axis=1) / v.shape[1])
      b_h1 += lr * (np.sum(h1_batch, axis=1) / h1_batch.shape[1] - np.sum(h1_v, axis=1) / h1_v.shape[1])
      b_h2 += lr * (np.sum(h2_batch, axis=1) / h2_batch.shape[1] - np.sum(h2_v, axis=1) / h2_v.shape[1])

      sys.stdout.write('\r')
      sys.stdout.write("Epoch %d:   %d/%d  [%-20s] %d%%" % (epoch, i, n_batches, '='*(20*i // n_batches),
                                              (100 / n_batches)*i))
      sys.stdout.flush()

    # update learning rate
    lr *= 0.5

  return train_error, model

def get_dbm_features(model, x):
  W, J = model['W'], model['J']
  b_w, b_j = model['b_w'], model['b_j']
  vh1 = W.T @ x + b_w
  vh2 = J.T @ vh1 + b_j
  return np.concatenate((vh1, vh2))

def dbm_predict(model, inputs):
  test_img, support_imgs = inputs
  test_img_feature = get_dbm_features(model, np.reshape(test_img[0], (-1,)))
  support_img_features = [get_dbm_features(model, np.reshape(support_img, (-1,))) for support_img in support_imgs]

  min_dist = np.inf
  closest = 0
  for i,feature_vec in enumerate(support_img_features):
    dist = np.linalg.norm(test_img_feature - feature_vec)
    if dist < min_dist:
      min_dist = dist
      closest = i
  return closest

def test_oneshot(model, N, k, data, labels, alphabet_dict, language=None, verbose=0):
  '''
  Test average N-way oneshot learning accuracy of model over k one-shot tasks
  '''
  correct = 0
  if verbose:
    print("Evaluating model on {} random {}-way one-shot learning tasks...".format(k,N))

  for i in range(k):
    inputs, targets = create_oneshot_task(data, labels, alphabet_dict, N=N, language=language)
    y = dbm_predict(model, inputs)
    if y == np.argmax(targets):
      correct += 1

  accuracy = (100 * correct / k)
  if verbose:
    print("Average %d-way one-shot accuracy: %4.2f%%" % (N, accuracy))

  return accuracy
