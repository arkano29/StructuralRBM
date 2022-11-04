import os
import pickle
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import Launch_Experiments
from sklearn.preprocessing import OneHotEncoder
from utils import read_home_dir
import medmnist
from medmnist import INFO
import dataset_without_pytorch
from dataset_without_pytorch import get_loader
  
dtype = Launch_Experiments.dtype

def save_rbm(rbm,**kwargs):
    "Save RBM"
    MAIN = read_home_dir()
    name = rbm.get_name()
    print("Saving : %s"%name)
    if not os.path.exists(os.path.join(MAIN,kwargs.get("output_dir","Trained RBM/MNIST"))):
      os.system("mkdir %s"%os.path.join(MAIN,kwargs.get("output_dir","Trained RBM/MNIST")))
    with open(os.path.join(MAIN,kwargs.get("output_dir","Trained RBM/MNIST"),name), 'wb') as output:
        pickle.dump(rbm, output, protocol=pickle.HIGHEST_PROTOCOL)
        
def load_rbm(file_name,**kwargs):
    "Load RBM"
    MAIN = read_home_dir()
    with open(os.path.join(MAIN,kwargs.get("output_dir","Trained RBM/MNIST"),file_name), 'rb') as input:
        rbm = pickle.load(input)
    return rbm

def check_rbm(file_name,**kwargs):
    "Return whether an RBM has already been trained or not, or if it has been already trained but with enough Epochs"
    MAIN = read_home_dir()
    full_name = os.path.join(MAIN,kwargs.get("output_dir","Trained RBM/MNIST"),file_name)
    exists =  os.path.isfile(full_name)
    return exists

def from_dataset_to_array(data_name = "",split = "train",**kwargs):
  """Loads MNIST corrupted dataset (normalized between 0 and 1) for specific corruption(data_name)"""
  N = 10000
  if data_name == "fashion_mnist":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    images = x_train.reshape((-1,28*28)).astype(dtype)/255
    labels = y_train
    if split == "train":
      images = images[N:]
      labels = labels[N:]
    elif split == "test":
      images = x_test.reshape((-1,28*28)).astype(dtype)/255
      labels = y_test
    elif split == "val":
      images = images[:N]
      labels = labels[:N]
    enc = OneHotEncoder(sparse = False,dtype = "float32")
    enc.fit(np.array([i for i in range(10)]).reshape(-1,1))
    labels = enc.transform(labels.reshape(-1,1))
  elif data_name in Launch_Experiments.CORRUPTIONS:
    if split == "val":
      ds = tfds.load("mnist_corrupted/"+data_name, split="train",shuffle_files = False,batch_size = -1) 
    else:
      ds = tfds.load("mnist_corrupted/"+data_name, split=split,shuffle_files = False,batch_size = -1) 
    images = ds["image"].numpy().reshape((-1,28*28)).astype(dtype)/255
    labels = ds["label"].numpy()
    enc = OneHotEncoder(sparse = False,dtype = "float32")
    enc.fit(np.array([i for i in range(10)]).reshape(-1,1))
    labels = enc.transform(labels.reshape(-1,1))
    if split == "train":
        images = images[N:]
        labels = labels[N:]
    elif split == "val":
        images = images[:N]
        labels = labels[:N]
  
  else:
    data_flag = data_name
    download = True

    info = INFO[data_flag]
    nClasses = max([int(k) for k in info["label"].keys()])+1
  
    SPLIT = split
    BATCH_SIZE = info["n_samples"][SPLIT]

    DataClass = getattr(dataset_without_pytorch, info['python_class'])

    # load the data
    train_dataset = DataClass(split=SPLIT, download=download)

    # encapsulate data into dataloader form
    train_loader = get_loader(dataset=train_dataset, batch_size=BATCH_SIZE)

    for images,labels in train_loader:
      break
    
    images = images.reshape((-1,images.shape[1]*images.shape[2])).astype(dtype)/255
    labels = labels.astype(dtype)
    if labels.shape[1]==1:
      enc = OneHotEncoder(sparse = False,dtype = "float32")
      enc.fit(np.array([i for i in range(nClasses)]).reshape(-1,1))
      labels = enc.transform(labels)
  transform = kwargs.get("transform","")
  perm = kwargs.get("perm",[])
  if transform == "invert":
      images = 1-images
  if transform == "permute":
      images = images[:,perm]
  if kwargs.get("size"):
      if kwargs.get("size")<images.shape[0]:
        subsampling = np.random.choice(images.shape[0],kwargs.get("size"),replace = False)
        return images[subsampling],labels[subsampling]
  return images,labels
