from BaseRBM import BaseRBM
import tensorflow as tf
import numpy as np
import time
from Mask import getWindowStrideMask,getStackedWindowStrideMask,permuteMask
import sparseupdate


class ClassRBM(BaseRBM):

    def __init__(self,nClass = 2,**kwargs):
        self.nClass = nClass
        super().__init__(**kwargs)
        self.modelBaseName = "ClassRBM"

        self.evaluation_metric = "val_log_loss"
        self.best_metric = 100
    
    def initParametersAndGradients(self):
        self.W = self.xavier_init(self.nVisible, self.nHidden)
        self.visible_bias = tf.zeros([self.nVisible])
        self.hidden_bias = tf.zeros([self.nHidden])
        self.U = self.xavier_init(self.nClass,self.nHidden)
        self.class_bias = tf.zeros([self.nClass])
    
        self.delta_W = tf.zeros([self.nVisible, self.nHidden])
        self.delta_visible_bias = tf.zeros([self.nVisible])
        self.delta_hidden_bias = tf.zeros([self.nHidden])
        self.delta_U = tf.zeros([self.nClass, self.nHidden])
        self.delta_class_bias = tf.zeros([self.nClass])

    def predict(self,v):
        """Predict probabilities for each class given visible variables, 
        from https://github.com/rangwani-harsh/pytorch-rbm-classification/blob/master/classification_rbm.py"""
        precomputed_factor = self.visibleTimesW(v) + self.hidden_bias
        class_probabilities = np.zeros((v.shape[0], self.nClass))

        for y in range(self.nClass):
            prod = tf.zeros(v.shape[0])
            prod += self.class_bias[y]
            for j in range(self.nHidden):
                prod += tf.math.softplus(precomputed_factor[:,j] + self.U[y, j])
                
            class_probabilities[:, y] = prod  

        copy_probabilities = np.zeros(class_probabilities.shape)

        for c in range(self.nClass):
          for d in range(self.nClass):
            copy_probabilities[:, c] += np.exp(-1*class_probabilities[:, c] + class_probabilities[:, d])
            
        copy_probabilities = 1/copy_probabilities

        class_probabilities = copy_probabilities

        return class_probabilities

    def sample_class_from_probs(self,p):
        return tf.one_hot(tf.math.argmax(p,axis = 1),depth = self.nClass)

    def computeAssociations(self,unfolded_input,sum):
        start = time.time()
        associations = tf.reduce_mean(tf.einsum("nij,nj->nij",unfolded_input, sum), axis = 0)
        return associations,time.time()-start

    def discriminative_training(self, input_data, class_label):
        """https://github.com/rangwani-harsh/pytorch-rbm-classification/blob/master/classification_rbm.py"""
        
        batch_size = input_data.shape[0]
        
        o_y_j = tf.nn.sigmoid(tf.reshape(tf.repeat(self.visibleTimesW(input_data)+self.hidden_bias,self.nClass),shape=(batch_size,self.nHidden,self.nClass))+tf.transpose(self.U))

        class_probabilities = self.predict(input_data) # Much of the computational cost lies here

        positive_sum = np.zeros([batch_size,self.nHidden])
        class_weight_grad = np.zeros([self.nClass,self.nHidden])

        for i,c in enumerate(tf.argmax(class_label,axis = 1)):
            positive_sum[i] += o_y_j[i, : , c]
            class_weight_grad[c ,:] += positive_sum[i]

        unfolded_input = tf.reshape(tf.repeat(input_data,self.nHidden),shape = (batch_size,self.nVisible,self.nHidden))

        positive_associations,elapsed_time = self.computeAssociations(unfolded_input,positive_sum)
        self.dW_compute_time.append(elapsed_time)

        negative_sum  = np.zeros([batch_size,self.nHidden])

        for c in range(self.nClass):
            term = tf.einsum("nj,n->nj",o_y_j[:,:,c],class_probabilities[:,c])
            class_weight_grad[c, :] -= tf.reduce_sum(term, axis = 0)  
            negative_sum += term

        negative_associations,elapsed_time = self.computeAssociations(unfolded_input,negative_sum)
        self.dW_compute_time[-1]+=elapsed_time

        dW = (positive_associations - negative_associations)

        dU = (class_weight_grad)

        db = tf.reduce_mean(positive_sum - negative_sum, axis = 0)

        dc = tf.reduce_mean(class_label- class_probabilities, axis = 0)	
        
        da = 0

        return dW,dU,da,db,dc
        
    def step(self,batch,batch_class):
        """Implements a training step"""
        dW,dU,da,db,dc =self.discriminative_training(batch,batch_class)
        
        self.delta_W = self.apply_momentum(self.delta_W,dW)
        self.delta_U = self.apply_momentum(self.delta_U,dU)
        self.delta_visible_bias = self.apply_momentum(self.delta_visible_bias,da)
        self.delta_hidden_bias = self.apply_momentum(self.delta_hidden_bias,db)
        self.delta_class_bias = self.apply_momentum(self.delta_class_bias,dc)
        
        self.W+=self.delta_W
        self.U+=self.delta_U
        self.hidden_bias+=self.delta_hidden_bias
        self.visible_bias+=self.delta_visible_bias
        self.class_bias+=self.delta_class_bias

    def compareMetrics(self,new_metric):
        return self.best_metric > new_metric

class ClassSBM(ClassRBM):

    def __init__(self,nClass = 2,**kwargs):
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
        kwargs["nHidden"] = self.nHidden
        self.sparse_loc = np.array(np.where(self.mask==1),dtype=np.int32)
        self.nClass = nClass

        super().__init__(**kwargs)
        self.modelBaseName = "ClassSBM"
        self.W = self.mask*self.W

    def computeAssociations(self,unfolded_input,sum):
        start = time.time()
        associations_array = np.zeros([self.sparse_loc.shape[1]],dtype = np.float32)
        associations = np.zeros([self.nVisible, self.nHidden])
        sparseupdate.discriminative_W_updates_Cython(0, 0, unfolded_input.numpy().astype(np.float32), sum.astype(np.float32), self.sparse_loc[0], self.sparse_loc[1], associations_array)
        associations[self.sparse_loc[0],self.sparse_loc[1]] = associations_array
        return associations,time.time()-start 

