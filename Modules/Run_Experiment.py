import os
from Save_Load import from_dataset_to_array,save_rbm,check_rbm
import numpy as np
from utils import read_home_dir
import Launch_Experiments
from optparse import OptionParser
import tensorflow as tf
from Launch_Experiments import PARAMS_DICT

from RBM import RBM,SBM
from ClassRBM import ClassRBM,ClassSBM 


def Experiment(args):

    """Computes a 10-repeated-hold-out experiment for a model with different seeds, model parameters are given by args"""
    
    SEED = args.get("seed",0)
    MAIN = read_home_dir()

    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    
    #######################################################
    
    TASK = args.get("TASK") # "Unsupervised" or "Supervised"
    DATA_NAME= args.get("DATA_NAME") # "MNIST" or "FashionMNIST"
    nHidden = args.get("nHidden",0)
    
    window = args.get("window",0)
    stride = args.get("stride",0)
    
    windowList = args.get("windowList",[])
    strideList = args.get("strideList",[])

    n_fold = args.get("n_fold",0)

    train_size = args.get("train_size",0)
    batch_size = Launch_Experiments.batch_size

    
    #######################################################
    EPOCHS = Launch_Experiments.EPOCHS
    lr = Launch_Experiments.lr
    momentum = Launch_Experiments.momentum
    
    n_step = Launch_Experiments.n_step
    
    unsuper_metrics = Launch_Experiments.unsuper_metrics
               
    super_metrics = Launch_Experiments.super_metrics

    train_corrupted,val_corrupted = None,None

    if DATA_NAME == "MNIST":
        train_data,train_data_y = from_dataset_to_array(data_name = "identity", split = "train",size = train_size)
        val_data,val_data_y = from_dataset_to_array(data_name = "identity", split = "val")
    elif DATA_NAME == "FashionMNIST":
        train_data,train_data_y = from_dataset_to_array(data_name = "fashion_mnist", split = "train",size = train_size)
        val_data,val_data_y = from_dataset_to_array(data_name = "fashion_mnist", split = "val")
    else:
        train_data,train_data_y = from_dataset_to_array(data_name = DATA_NAME, split = "train",size = train_size)
        val_data,val_data_y = from_dataset_to_array(data_name = DATA_NAME, split = "val")
    OUTPUT_DIR = "Trained RBM/"+DATA_NAME
    if not os.path.exists(os.path.join(MAIN,OUTPUT_DIR)):
        os.mkdir(os.path.join(MAIN,OUTPUT_DIR))
        
    if TASK == "Unsupervised":
        method = "LL"
        train_corrupted,_ = from_dataset_to_array(data_name = "shot_noise", split = "train",size = train_size)
        val_corrupted,_ = from_dataset_to_array(data_name = "shot_noise", split = "val")
        metrics = unsuper_metrics
    elif TASK == "Supervised":
        method = "discriminative"
        metrics = super_metrics
    
    # Set model arguments
    kwargs = dict(batch_size = batch_size,nVisible = train_data.shape[1],nHidden = nHidden, nClass = train_data_y.shape[1], train_size = train_size,
                        window = window, stride = stride, windowList = windowList, strideList = strideList, n_fold = n_fold, name = "_"+DATA_NAME,
                        epochs = EPOCHS,k_gibbs = 1, MAIN = MAIN,metrics = metrics,seed = SEED, output_dir = OUTPUT_DIR, method = method,
                        train_data=train_data,val_data=val_data,train_corrupted=train_corrupted,val_corrupted=val_corrupted,n_step=n_step,
                        train_data_y=train_data_y,val_data_y=val_data_y,min_updates = Launch_Experiments.PATIENCE,
                        lr = lr,momentum = momentum, dtype = Launch_Experiments.dtype,LL_epochs = 100)
                        
    if TASK == "Unsupervised":
        if nHidden:
            model = RBM(**kwargs)
        else:
            model = SBM(**kwargs)
    elif TASK == "Supervised":
        if nHidden:
            model = ClassRBM(**kwargs)
        else:
            model = ClassSBM(**kwargs)
        
    model.batch_size = batch_size
    model.epochs = EPOCHS
    
    print("Name of the RBM: %s"%model.get_name())
    name = model.get_name()
    if check_rbm(name,**kwargs):
        print("Already computed and stored in %s"%(os.path.join(MAIN,OUTPUT_DIR,name)))
    else:
        model.fit(**kwargs) # Train model
    
    save_rbm(model,**kwargs) # Save model
        
__version__ = "1.0"
def command_line_arg():
    """Main function that takes the input variable, index of the parameters dict"""
    usage = "usage: %prog [options] arg1 arg2"
    par = OptionParser(usage=usage, version= __version__)

    par.add_option('-i', '--ind', dest = "i",type="int",default = 0)
    par.add_option('-s', '--seed', dest = "seed",type="int",default = 0)
    par.add_option('-d', '--data', dest = "data_name",type="str",default = "MNIST")
    par.add_option('-t', '--task', dest = "task",type="str",default = "Unsupervised")

    return  par.parse_args()
    
if __name__ == '__main__':
    opts, args = command_line_arg()
    DATA_NAME = opts.data_name
    TASK = opts.task
    MAIN = read_home_dir()
    MAIN_DICT = [dict(item,DATA_NAME = DATA_NAME, TASK = TASK) for item in PARAMS_DICT]
    args = MAIN_DICT[opts.i]
    args["seed"] = opts.seed
    Experiment(args)
    