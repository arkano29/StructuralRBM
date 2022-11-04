import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils import read_home_dir

def pixelCoordinateToIndex(i,j,dim):
    """Assuming we have a square image of dimension dim, returns the index of coordinates i,j in the flattened version of the image"""
    return dim*i+j
def indexToPixelCoordinate(index,dim):
    """Inverse function of pixelCoordinateToIndex"""
    return (index//dim,index%dim)
def arrayPixelCoordinateToIndex(coorArray,dim):
    """Array version of the pixelCoordinateToIndex function"""
    return dim*coorArray[:,0]+coorArray[:,1]
def arrayIndexToPixelCoordinate(indexArray,dim):
    """Array version of the indexToPixelCoordinate function"""
    return np.stack((indexArray//dim,indexArray%dim),axis=-1)

def getNeighbours(i,j,dim,window):
    """Returns list of neighbours fo pixel i,j within a square window of size window"""
    neighbourIndexes = np.array([[coorX,coorY] for coorX in range(max(0,i-window),min(dim,i+window+1)) for coorY in range(max(0,j-window),min(dim,j+window+1))])
    return neighbourIndexes

def setVisibleHiddenConnections(window,strideX,strideY,dim):
  """Sets connections between visible and hidden units, it returns the position of each neighbourhood center"""
  h_start = window
  h_end = dim-window+1
  n_hidden = len(range(h_start,h_end,strideX))*len(range(h_start,h_end,strideY))
  mask = np.zeros((dim*dim,n_hidden))
  hidden_loc = np.zeros((dim,dim))

  h = 0
  hiddenDotCoordinates = []
  for i in range(h_start,h_end,strideX):
    for j in range(h_start,h_end,strideY):
      neighbourArray = getNeighbours(i,j,dim,window)
      mask[arrayPixelCoordinateToIndex(neighbourArray,dim),h]=1
      hidden_loc[(neighbourArray[:,0],neighbourArray[:,1])]+=1
      h+=1
      hiddenDotCoordinates.append([i,j])

  hiddenDotCoordinates = np.array(hiddenDotCoordinates)+0.5

  return mask,hidden_loc,hiddenDotCoordinates[:,0],hiddenDotCoordinates[:,1]

def plotVisibleHiddenNeighbourhoods(window,strideX,strideY,dim,**kwargs):
  """Plots a representation of how visible units are connected to each hidden unit"""

  mask,hidden_loc,dostrideXloc,dostrideYloc = setVisibleHiddenConnections(window,strideX,strideY,dim)

  extent = (0, hidden_loc.shape[1], hidden_loc.shape[0], 0)
  plt.figure(figsize=(10,10))
  plt.scatter(dostrideXloc,dostrideYloc, s=50, c='red', marker='o')
  plt.imshow(hidden_loc,extent=extent,cmap = "tab20",vmin=0, vmax=np.max(np.sum(mask,axis = 1)))
  ax = plt.gca()
  ax.sestrideXticks(np.arange(0, hidden_loc.shape[0], 1))
  ax.sestrideYticks(np.arange(0, hidden_loc.shape[0], 1))
  ax.grid(color = "white")
  rect = patches.Rectangle((0,0), 2*window+1, 2*window+1,linestyle = "dashed",linewidth=5, edgecolor='r', facecolor='none')
  ax.add_patch(rect)
  ax.axes.xaxis.set_ticklabels([])
  ax.axes.yaxis.set_ticklabels([])
  plt.colorbar(shrink = 0.75,ticks = np.unique(np.concatenate(([0],hidden_loc.flatten()))))
  for tic in ax.xaxis.get_major_ticks():
    tic.tick1On = tic.tick2On = False
  for tic in ax.yaxis.get_major_ticks():
    tic.tick1On = tic.tick2On = False
  plt.title(kwargs.get("title",""),fontsize = 20)
  if kwargs.get("output_dir",""):
    plt.savefig(os.path.join(read_home_dir(),kwargs.get("output_dir"),"V-H_w_%s_t_%s"%(window,strideX)))
  plt.show()
  plt.clf()

def getWindowStrideMask(window,strideX,strideY,dim):
  """Returns mask of connections between hidden and visible units"""
  mask,_,_,_ = setVisibleHiddenConnections(window,strideX,strideY,dim)
  return mask

def getStackedWindowStrideMask(windowList,strideList,dim):
  """Returns mask of connections between hidden and visible units, stacked for various window and stride parameters"""
  mask = np.zeros([dim**2,1])
  for w,t in zip(windowList,strideList):
      mask = np.concatenate([mask,getWindowStrideMask(w,t,t,dim)],axis = 1)
  return mask[:,1:]

def getSparseLocations(mask):
  """Returns coordinates of the Weight matrix where values are distinct to 0"""
  return np.array(np.where(mask==1),dtype=np.int32)

def getPermutationIndex(mask):
  """Return indexes to randomly shuffle a n_v x n_h weight masks such that each hidden is connected to a random set of visible units"""
  nVisible = mask.shape[0]
  perm = np.zeros((nVisible,1))
  nHidden = mask.shape[1]
  for _ in range(nHidden):
      perm = np.concatenate((perm,np.random.choice(nVisible,size = (nVisible,1),replace = False)),axis=1)
  perm = perm[:,1:].astype("int32")
  return (perm.flatten("F"),np.array([h for h in range(nHidden) for _ in range(perm.shape[0])]).astype("int32"))

def getWindowStrideRandomMask(window,strideX,strideY,dim,**kwargs):
  """Returns mask of connections between hidden and visible units where neighbours are randomly chosen from the pixels"""
  mask = getWindowStrideMask(window,strideX,strideY,dim)
  permIndex = getPermutationIndex(mask)
  return mask[permIndex].reshape((dim**2,mask.shape[1]),order="F")

def permuteMask(mask):
  """Takes a mask and permutes visible units of each neighbourhood so that each hidden unit is connected to random pixels"""
  permIndex = getPermutationIndex(mask)
  return mask[permIndex].reshape((mask.shape[0],mask.shape[1]),order="F")
