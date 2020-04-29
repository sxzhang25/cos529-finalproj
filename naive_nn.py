import numpy as np
import matplotlib.pyplot as plt

class Naive_NN:
  def __init__(self, anchors, labels):    
    self.anchors = anchors # data points
    self.labels = labels # labels    

  def classify(self, data):
    '''
    given labeled anchor examples, classify each data point using nearest neighbors
    '''
    self.data = data
    
    self.y = np.zeros(data.shape[0], dtype='int')
    for i,x in enumerate(data):
      min_dist = np.inf
      for j,a in enumerate(anchors):
        dist = np.linalg.norm(x - a)
        if dist < min_dist:
          self.y[i] = self.labels[j]
          min_dist = dist
    
    return self.y.copy()
    
  def plot(self):
    N = self.labels.shape[0] # number of labels
    
    # define colormap
    cmap = plt.cm.jet
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    
    # plot data
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(self.anchors[:,0], self.anchors[:,1], c=self.labels, cmap=cmap)
    ax.scatter(self.data[:,0], self.data[:,1], c=self.y, cmap=cmap)
    plt.show()
    
    
anchors = np.array([[1,1], [1,-1], [-1,1], [-1,-1]])
labels = np.array([0, 0, 1, 2], dtype='int')
data = 2 * np.random.rand(500, 2) - 1

naive_nn = Naive_NN(anchors, labels)
y_ = naive_nn.classify(data)
print(y_)
naive_nn.plot()