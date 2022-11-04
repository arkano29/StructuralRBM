import tensorflow as tf
import os
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import OneHotEncoder

def ProcessClasses(oneHotClass):
    try:
        nClasses = len(oneHotClass[0])
        enc = OneHotEncoder(sparse = False)
        enc.fit(np.array([i for i in range(nClasses)]).reshape(-1,1))
        return enc.inverse_transform(oneHotClass)[:,0]
    except:
        return oneHotClass

def obtainClassWeights(Classes):
    if Classes is None:
        return None
    return compute_sample_weight("balanced",ProcessClasses(Classes))
  
def sample_bernoulli(ps):
  """Samples binary variables from ps"""
  return tf.nn.relu(tf.sign(ps - tf.random.uniform(tf.shape(ps))))
  
def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

def read_home_dir():
  with open("./MAIN_DIR.txt","r") as file:
      MAIN = file.readline()[:-1]
  file.close()
  return MAIN

def generateDocumentation(file):
    pass

def combineDictionaries(dictionary1,dictionary2):
    """Notice that if repetitions exist the last dictionary will overwrite the previous"""
    return {**dictionary1,**dictionary2}

def remove_all_but(but=[],show = True,rm = False):
    MAIN = read_home_dir()
    i = 0
    if len(but):
        for dir in os.listdir(os.path.join(MAIN,"Trained RBM")):
            if not "." in dir:
                for file in os.listdir(os.path.join(MAIN,"Trained RBM",dir)):
                    file_dir = os.path.join(MAIN,"Trained\ RBM",dir,file)
                    remove = True
                    for b in but:
                        if b in file:
                            remove = False
                            break
                    if remove:
                        i+=1
                        if show:
                            print("Remove %s"%(file_dir))
                        if rm:
                            os.system("rm %s"%(file_dir))
    print("Removed %s files"%i)


def early_stopping(self,metrics):
        """Check if trainig should stop earlier given some metrics, returns index of where the optimal eval metric is to restore params"""
        ind = 0
        stop = False
        reason = ""
        if self.epoch+1==self.epochs:
            opt = metrics[0].get("optimize","min")
            val_error = self.history["val_"+metrics[0]["name"]]
            if opt == "min":
                stop = True
                ind = np.argmin(val_error)
            elif opt == "max":
                stop = True
                ind = np.argmax(val_error)
        return ind,stop,reason
        if self.n_updates>self.min_updates:
            for metric in metrics:
                min_delta = metric.get("min_delta",0)
                max_delta = metric.get("max_delta",0)
                max_value = metric.get("max_value",0)
                patience = metric.get("patience",self.epochs)
                opt = metric.get("optimize","min")
                val_error = self.history["val_"+metric["name"]]

                prev_ind = -2
                if metric["name"] == "LL":
                    prev_ind = -1-self.LL_epochs
                    if len(val_error)<=self.LL_epochs:
                        continue

                if min_delta:
                    if abs(val_error[-1]-val_error[prev_ind])<min_delta:
                        reason = "minimum delta achieved with %s"%metric["name"]
                        stop = True
                        ind = -1
                if max_delta:
                    if abs(val_error[-1]-val_error[prev_ind])>max_delta:
                        reason = "maximum delta achieved with %s"%metric["name"]
                        stop = True
                        ind = -1
                if max_value:
                    if val_error[-1]>max_value:
                        reason = "maximum value achieved with %s"%metric["name"]
                        stop = True
                        ind = -1
                if len(val_error)>patience:
                    if opt == "min":
                        if val_error[prev_ind]<val_error[-1]:
                            reason = "overfitting with %s"%metric["name"]
                            stop = True
                            ind = np.argmin(val_error)
                    elif opt == "max":
                        if val_error[prev_ind]>val_error[-1]:
                            reason = "overfitting with %s"%metric["name"]
                            stop = True
                            ind = np.argmax(val_error)
                if stop:
                    break
            return ind,stop,reason
        else:
            return ind,stop,reason

