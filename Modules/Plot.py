import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np
from Save_Load import from_dataset_to_array,load_rbm

dim = 28

def removeTicks(ax):
  for tic in ax.xaxis.get_major_ticks():
    tic.tick1On = tic.tick2On = False
  for tic in ax.yaxis.get_major_ticks():
    tic.tick1On = tic.tick2On = False
def evaluate_corruption_recon(rbm,corruptions,**kwargs):
  """Computes evaluation metric for each corruption and returns them in an array"""
  perm = kwargs.get("original_permutation",[False])
  data,_ = from_dataset_to_array("identity",split = "test",**kwargs)
  eval_size = kwargs.get("eval_size",2**10)
  data = data[:eval_size].reshape((-1,28*28))
  if any(perm):
    data = data[:,perm]
  evals = np.zeros((eval_size,1))
  for corruption in corruptions:
    states,_ = from_dataset_to_array(corruption,split = "test",**kwargs)
    states = states[:eval_size].reshape((-1,28*28))
    if any(perm):
      states = states[:,perm]
    error = rbm.recon_mse(data,states)
    evals = np.concatenate((evals,error.numpy().reshape(-1,1)),axis = 1)
  return evals[:,1:]

def evaluate_models(names,corruptions,**kwargs):
  evals = []
  for name in names:
    rbm = load_rbm(name,**kwargs)
    evals.append(evaluate_corruption_recon(rbm,corruptions,**kwargs))
  return evals
  
def compare_boxplots(data_list,**kwargs):
  """Plots a set of boxplots of the error values"""
  figsize = kwargs.get("figsize",(16,10))
  MAIN = kwargs.get("MAIN")
  fig = plt.figure(1, figsize=figsize)
  ax = fig.add_subplot(111) #Create an axes instance
  cmap = matplotlib.cm.get_cmap(kwargs.get("cmap","hsv"))  #Create a colormap instance
  d = data_list[0].shape[1]
  D = len(data_list)
  b = [None for _ in range(D*d)] #Boxplot plots list
  color_list = ["greenyellow","aquamarine","blue","darkviolet","crimson","darkorange"]
  for j in range(D):
    col = color_list[j] #Define color for each model
    for i in range(d):
      b[j*d+i] = ax.boxplot(data_list[j][:,i],patch_artist=True,boxprops=dict(facecolor=col,color="black"),
                              positions = [i+(j-(D-1)/2)/D],showmeans = kwargs.get("showmeans",False),
                              whiskerprops=dict(color=col,linestyle="--"),capprops=dict(color="black"),
                              medianprops=dict(color="black"),flierprops=dict(marker="h",color=col,alpha=1),
                              manage_ticks = False, showfliers = True,notch = True)
      for flier in b[j*d+i]["fliers"]:
        flier.set_markerfacecolor(col)
  title = kwargs.get("title","")
  if title:
    plt.title(title,fontsize = 30)
  if kwargs.get("ylabel",""):
    ax.set_ylabel(kwargs.get("ylabel",""),fontsize = 30)
  if len(kwargs.get("ylim",[])):
    ylim = kwargs.get("ylim",[])
    plt.ylim(ylim[0],ylim[1])
  if kwargs.get("names",""):
    ax.legend([b[j*d]["boxes"][0] for j in range(D)], kwargs.get("names"),loc='upper right',fontsize = 20)
  if kwargs.get("labels",""):
    ax.set_xticks(range(len(kwargs.get("labels"))))
    ax.set_xticklabels(kwargs.get("labels"),fontsize = 15)
  if kwargs.get("file_name",""):
    fig.savefig(os.path.join(MAIN,"RBM Images",kwargs.get("file_name","")))
  
def plot_digit_grid(states,n = 10,title = "",**kwargs):
  """Plots a grid of digits"""
  figsize = kwargs.get("figsize",(10,10))
  fig, axs = plt.subplots(n, n, figsize=figsize)

  for i in range(n):
    for j in range(n):
      axs[i,j].imshow(states[n*j+i].reshape((dim,dim)),cmap = plt.get_cmap('gray'))
      axs[i,j].axis('off')
  plt.subplots_adjust(wspace=0, hspace=0)
  if title:
    fig.suptitle(title,fontsize = 40)
  return fig,axs

def plot_reconstruction(rbm,corruption,n,title = "",save = False,**kwargs):
  """Plots a grid of original corrupted images+their reconstruction"""
  MAIN = kwargs.get("MAIN")
  states,_ = from_dataset_to_array(data_name = corruption,split = "test",**kwargs)
  states = states[:n**2]
  if any(kwargs.get("original_permutation",[False])):
    aux_states = states.copy()
    states = aux_states[:,kwargs.get("original_permutation")]
  rbm.k_gibbs = kwargs.get("k",rbm.k_gibbs)  
  _,_,recon,_ = rbm.gibbs_sampling(states)
  rbm.k_gibbs = 1
  recon = recon.numpy()
  if any(kwargs.get("original_permutation",[False])):
    aux_recon = recon.copy()
    aux_states = states.copy()
    recon[:,kwargs.get("original_permutation")] = aux_recon
    states[:,kwargs.get("original_permutation")] = aux_states
  figsize = kwargs.get("figsize",(10,5))
  fig, axs = plt.subplots(n, 2*n, figsize=figsize)

  for i in range(n):
    for j in range(n):
      axs[i,2*j].imshow(states[n*j+i].reshape((dim,dim)),cmap = plt.get_cmap('gray'))
      axs[i,2*j].axis('off')
      axs[i,2*j+1].imshow(recon[n*j+i].reshape((dim,dim)),cmap = plt.get_cmap('gray'))
      axs[i,2*j+1].axis('off')

  plt.subplots_adjust(wspace =0,hspace=0)
  if title:
    fig.suptitle(title,fontsize = 40)
  if save:
    extention = set_extention(**kwargs)
    sample = ""
    if kwargs.get("k",0):
      sample = str(kwargs.get("k"))+"th_sample"
    corrup = ""
    if kwargs.get("corruption",""):
      corrup = "_"+corruption+"_"
    fig.savefig(os.path.join(MAIN,"RBM Images","Recons_"+sample+corrup+extention))
  return fig,axs

def boxplots(data,**kwargs):
  """Plots a set of boxplots of the error values"""
  figsize = kwargs.get("figsize",(16,10))
  MAIN = kwargs.get("MAIN")
  fig = plt.figure(1, figsize=figsize)
  ax = fig.add_subplot(111) #Create an axes instance
  cmap = matplotlib.cm.get_cmap(kwargs.get("cmap","tab20"))  #Create a colormap instance
  d = data.shape[1]
  b = [None for _ in range(d)] #Boxplot plots list
  for i in range(d):
    col = cmap(i/d) #Define color for each method
    b[i] = ax.boxplot(data[:,i],patch_artist=True,boxprops=dict(facecolor=col,color="black"),
                      positions = [i],showmeans = kwargs.get("showmeans",False),
                              whiskerprops=dict(color=col,linestyle="--"),capprops=dict(color="black"),medianprops=dict(color="black"),
                              flierprops=dict(marker="o",color=col,alpha=1),manage_ticks = False, showfliers = True)
    for flier in b[i]["fliers"]:
      flier.set_markerfacecolor(col)
  title = kwargs.get("title","")
  if title:
    plt.title(title,fontsize = 30)
  if len(kwargs.get("ylim",[])):
    ylim = kwargs.get("ylim",[])
    plt.ylim(ylim[0],ylim[1])
  if kwargs.get("labels"):
    ax.legend([box["boxes"][0] for box in b], kwargs.get("labels"),loc='center left', bbox_to_anchor=(1, 0.5))
  if kwargs.get("file_name",""):
    fig.savefig(os.path.join(MAIN,"RBM Images",kwargs.get("file_name","")))

def plot_errors(errors,names,ylabel = "Error",title = "Training error"):
  epoch_list = [i+1 for i in range(len(errors[0]))]
  fig = plt.figure(1, figsize=(10,5))
  ax = fig.add_subplot(111) #Create an axes instance
  for i in range(len(errors)):
    ax.plot(epoch_list,errors[i],label = names[i])
  plt.legend(fontsize = 15)
  plt.ylabel(ylabel,fontsize = 20)
  plt.xlabel("$Epoch$",fontsize = 20)
  plt.title(title,fontsize = 30)
  plt.show()


