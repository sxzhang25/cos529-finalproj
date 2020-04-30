import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import colorsys

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
      for j,a in enumerate(self.anchors):
        dist = np.linalg.norm(x - a)
        if dist < min_dist:
          self.y[i] = self.labels[j]
          min_dist = dist

    return self.y.copy()

  def plot(self):
    '''
    plot 2d projection of nearest neighbors
    '''
    def man_cmap(cmap, value=1.0):
      colors = cmap(np.arange(cmap.N))
      hls = np.array([colorsys.rgb_to_hls(*c) for c in colors[:,:3]])
      hls[:,1] *= value
      rgb = np.clip(np.array([colorsys.hls_to_rgb(*c) for c in hls]), 0,1)
      return mcolors.LinearSegmentedColormap.from_list("", rgb)

    N = len(self.labels) # number of labels

    # define colormap
    cmap = plt.cm.jet
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    # plot data
    fig = plt.figure()
    ax = fig.add_subplot(111)
    test = ax.scatter(self.data[:,0],
                      self.data[:,1],
                      s=1,
                      c=self.y,
                      cmap=man_cmap(cmap, 1.25))

    anchors = ax.scatter(self.anchors[:,0],
                         self.anchors[:,1],
                         s=20,
                         c=self.labels,
                         cmap=man_cmap(cmap, 0.75))

    legend = ax.legend(*test.legend_elements(), title="Classes", prop={'size': 5})
    ax.add_artist(legend)
    plt.title('Nearest Neighbors Classifier')

    plt.show()
