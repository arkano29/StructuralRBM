from BaseRBM import BaseRBM
from Mask import getWindowStrideMask,getStackedWindowStrideMask,permuteMask
import numpy as np
import tensorflow as tf
import time
import sparseupdate

class RBM(BaseRBM):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.modelBaseName = "RBM"

class SBM(BaseRBM):

    def __init__(self,**kwargs):
        self.nVisible = kwargs.get("nVisible",10)
        self.window = kwargs.get("window",0)
        self.windowList = kwargs.get("windowList",0)
        self.strideList = kwargs.get("strideList",0)
        self.stride = kwargs.get("stride",0)
        self.dim = kwargs.get("dim",28)
        if self.window:
            self.mask = getWindowStrideMask(self.window,self.stride,self.stride,self.dim)
        else:
            self.mask = getStackedWindowStrideMask(self.windowList,self.strideList,self.dim)
        if kwargs.get("n_fold"):
            self.mask = np.concatenate([self.mask for _ in range(kwargs.get("n_fold"))],axis = 1)
        if kwargs.get("permute"):
            self.mask = permuteMask(self.mask)
            
        self.nHidden = self.mask.shape[1]
        self.sparse_loc = np.array(np.where(self.mask==1),dtype=np.int32)
        super().__init__(**kwargs)
        self.modelBaseName = "SBM"
        self.W = self.mask*self.W
    
    def LL_gradient(self,batch):
        """Computes gradients for maximizing LogLikelihood for RBM parameters"""
        v,h,v_k,h_k = self.gibbs_sampling(batch)
        dW = np.zeros([self.nVisible,self.nHidden])
        start = time.time()
        dW_array = np.zeros([self.sparse_loc.shape[1]],dtype = np.float32)
        sparseupdate.LL_W_updates_Cython(0, 0, v.numpy().astype(np.float32), h.numpy().astype(np.float32), v_k.numpy().astype(np.float32),
                                             h_k.numpy().astype(np.float32), self.sparse_loc[0], self.sparse_loc[1], dW_array)

        dW[self.sparse_loc[0],self.sparse_loc[1]] = dW_array
        self.dW_compute_time.append(time.time()-start)
        db = tf.reduce_mean(h-h_k,axis = 0)
        da = tf.reduce_mean(v-v_k,axis = 0)

        return dW,db,da





    