import tensorflow as tf
import numpy as np
import time
from tqdm.notebook import tqdm_notebook as tqdm
from utils import sample_bernoulli,isnotebook,obtainClassWeights,ProcessClasses
from sklearn.metrics import roc_auc_score as ROC
from sklearn.metrics import accuracy_score as ACCURACY
from sklearn.metrics import log_loss


class BaseRBM:
    """RBM model for unsupervised task:
    -nVisible: number of visible units
    -nHidden: number of hidden units
    -lr: learning rate
    -momentum: momentum applied as D<- momentum*old+(1-momentum)*new
    -kGibbs: number of Gibbs sampling steps (default 1)
    -seed: random seed
    """
    def __init__(
        self,
        nVisible = 10,
        nHidden = 10,
        window = 0,
        stride = 0,
        windowList = [],
        strideList = [],
        lr = 0.1,
        momentum = 0,
        kGibbs = 1,
        seed = 42,
        **kwargs
    ):

        self.modelBaseName = "BaseRBM"
        self.nVisible = nVisible
        self.nHidden = nHidden
        self.lr = lr
        self.momentum = momentum
        self.kGibbs = kGibbs
        self.method = kwargs.get("method","LL")

        self.window = window
        self.windowList = windowList
        self.strideList = strideList
        self.stride = stride
        
        self.seed = seed
        tf.random.set_seed(seed)
        np.random.seed(seed)
            
        self.initParametersAndGradients()
        
        self.last_epoch = 0
        self.opt_ind = -1
        self.history = dict()
        self.approx_log_Z = 1
        self.M = kwargs.get("M",1000) # AIS number of samples
        self.nAIS = kwargs.get("nAIS",2)
        self.training_time = 0
        self.dW_compute_time = []
        self.baserate_bias = tf.zeros([self.nVisible])

        self.train_size = kwargs.get("train_size",0)
        self.epochs = kwargs.get("epochs",1)
        
        self.name = kwargs.get("name","") # Variable to distinct models with for instance the name of the dataset
        self.permute = kwargs.get("permute",False)
        self.evaluation_metric = "val_LL"
        self.best_metric = -1000

    def initParametersAndGradients(self):
        self.W = self.xavier_init(self.nVisible, self.nHidden)
        self.visible_bias = tf.zeros([self.nVisible])
        self.hidden_bias = tf.zeros([self.nHidden])
    
        self.delta_W = tf.zeros([self.nVisible, self.nHidden])
        self.delta_visible_bias = tf.zeros([self.nVisible])
        self.delta_hidden_bias = tf.zeros([self.nHidden])

    def xavier_init(self,fan_in, fan_out, *, const=1.0, dtype=tf.dtypes.float32):
        """Returns weigths using Xavier initialization"""
        k = const * np.sqrt(6.0 / (fan_in + fan_out))
        return tf.random.uniform((fan_in, fan_out), minval=-k, maxval=k, dtype=dtype)

    def visibleTimesW(self,v):
        return tf.einsum("ni,ij->nj",v,self.W)

    def hiddenTimesW(self,h):
        return tf.einsum("nj,ij->ni",h,self.W)
    
    def visibleWHidden(self,v,h):
        return tf.einsum("ni,ij,nj->n",v,self.W,h)

    def prob_hidden(self,v):
        """Returns probabilities of hidden variables being equal to 1"""
        return tf.nn.sigmoid(self.visibleTimesW(v)+self.hidden_bias)
        
    def sample_hidden(self,v):
        """Returns samples of binary hidden variables"""
        return sample_bernoulli(self.prob_hidden(v))
        
    def prob_visible(self,h):
        """Returns probabilities of visible variables being equal to 1"""
        return tf.nn.sigmoid(self.hiddenTimesW(h)+self.visible_bias)
    
    def sample_visible(self,h):
        """Returns samples of binary visible variables"""
        return sample_bernoulli(self.prob_visible(h))
        
    def reconstruct(self,v):
        """"Reconstructs input data v with a single Gibbs sampling step"""
        return self.prob_visible(self.sample_hidden(v))
        
    def recon_mse(self,data,corrupted_data):
        """Returns reconstructions error"""
        return tf.reduce_mean(tf.math.squared_difference(data,self.reconstruct(corrupted_data)),axis = 1)
        
    def energy(self,v,h):
        """Returns configuration energy"""
        return (-self.visibleWHidden(v,h)-tf.einsum("j,nj->n",self.hidden_bias,h)-tf.einsum("i,ni->n",self.visible_bias,v))
  
    def free_energy(self,v):
        """Returns free energy"""
        return - tf.einsum("ni,i->n",v,self.visible_bias) - tf.reduce_sum(tf.math.softplus(self.hidden_bias + self.visibleTimesW(v)), axis=1)
    
    def ais_free_energy_term(self,v,beta):
        return (1-beta)*tf.einsum("ni,i->n",v,self.baserate_bias)+beta*tf.einsum("ni,i->n",v,self.visible_bias)+tf.reduce_sum(tf.math.softplus(beta*(self.hidden_bias + self.visibleTimesW(v))), axis=1)+tf.reduce_sum(tf.math.softplus((1-beta)*self.hidden_bias))
  
    def sample_visible_beta(self,v,beta):
        h = sample_bernoulli(tf.nn.sigmoid(beta*(self.visibleTimesW(v)+self.hidden_bias)))
        return tf.nn.sigmoid(beta*(self.hiddenTimesW(h)+self.visible_bias)+(1-beta)*self.baserate_bias)

    def AIS_log_Z(self):
        """Execute AIS with M runs of K steps, and returns the approximated log Z"""
        n1 = 50*self.nAIS
        n2 = 400*self.nAIS
        n3 = 1000*self.nAIS
        beta1 = np.linspace(0,0.5,n1,endpoint=False)
        beta2 = np.linspace(0.5,0.9,n2,endpoint=False)
        beta3 = np.linspace(0.9,1,n3+1)
        beta_list = np.concatenate([beta1,beta2,beta3])
        v_sample = tf.repeat(tf.expand_dims(tf.nn.sigmoid(self.baserate_bias),0),self.M,axis = 0)#tf.convert_to_tensor(np.random.binomial(self.M, p=tf.nn.sigmoid(self.baserate_bias),size = (self.M,self.nVisible)),dtype = self.visible_bias.dtype) #Sample initial v_1, with parameters \theta={0,a,b}
        logZ0 = tf.reduce_sum(tf.math.softplus(self.baserate_bias))+tf.reduce_sum(tf.math.softplus(self.hidden_bias))#+self.nHidden*np.log(2)
        w = np.zeros(self.M) #Initialize \hat{w}
        for i in range(len(beta_list)-1): #Run K-1 times
          logp_k = self.ais_free_energy_term(v_sample,beta_list[i]) #Log(p_k)
          logp_k1 = self.ais_free_energy_term(v_sample,beta_list[i+1]) #Log(p_{k+1})
          w+= logp_k1 - logp_k #Next term of \hat{w}_i=log(w_i)=-delta_beta*E(v_i)
          v_sample = self.sample_visible_beta(v_sample,beta_list[i+1]) # Draw visible samples with a Gibbs sampling step
        maximum = np.max(w) #Compute maximum value of \hat{w}_i, for efficient computation of log(\sum w_i)
        w-=maximum #Subtract it
        w = np.exp(w) #Compute w_i=e^{\hat{w}_i}
        return maximum+np.log(np.sum(w))+logZ0-np.log(self.M) #Return maximum(\hat{w}_i)+log(\sum w_i)+log Z_A-log M
        
    def approximated_LL(self,v):
        """Returns approximated LL of v, the approximated logZ must be computed previously"""
        return -tf.reduce_mean(self.free_energy(v))-self.approx_log_Z

    def gibbs_sampling(self,v_init):
        """Performs Gibbs sampling and returns first visible+hidden state as well as the last visible+hidden sample"""
        h = self.sample_hidden(v_init)
        h_k = tf.identity(h)
        v_k = self.prob_visible(h_k)
        for _ in range(self.kGibbs-1):
          h_k = self.sample_hidden(v_k)
          v_k = self.prob_visible(h_k)
        h_k = self.prob_hidden(v_k) # The last hidden units do not need to sampled (Hinton,2012)
        return v_init,h,v_k,h_k

    def clamped_gibbs_sampling_from_vis_state(self,vis_state,index,value):
        """Generates a new sample (with binary values 0,1), where certain variables are set to some values,
        and are forced to keep those values while others are sampled (clamped sampling)"""
        v = vis_state
        for _ in range(self.kGibbs):
            v = self.prob_visible(self.sample_hidden(v))
            v = v.numpy()
            v[:,index] = value
            v = tf.convert_to_tensor(v)
        return v
      
    def apply_momentum(self,old,new):
        """Applies momentum to gradients"""
        return tf.add(self.momentum*old,self.lr*new)
        
    def LL_gradient(self,batch):
        """Computes gradients for maximizing LogLikelihood for RBM parameters"""
        v,h,v_k,h_k = self.gibbs_sampling(batch)
        startTime = time.time()
        dW = tf.reduce_mean(tf.einsum("ni,nj->nij",v,h),axis = 0)-tf.reduce_mean(tf.einsum("ni,nj->nij",v_k,h_k),axis = 0)
        self.dW_compute_time.append(time.time()-startTime)
        db = tf.reduce_mean(h-h_k,axis = 0)
        da = tf.reduce_mean(v-v_k,axis = 0)

        return dW,db,da
        
    def step(self,batch,batch_class):
        """Implements a training step"""
        dW,db,da =self.LL_gradient(batch)
        
        self.delta_W = self.apply_momentum(self.delta_W,dW)
        self.delta_hidden_bias = self.apply_momentum(self.delta_hidden_bias,db)
        self.delta_visible_bias = self.apply_momentum(self.delta_visible_bias,da)
        
        self.W+=self.delta_W
        self.hidden_bias+=self.delta_hidden_bias
        self.visible_bias+=self.delta_visible_bias
    
    def epochIterable(self):
        """Returns iterable depending on being executed in notebook or shell"""
        if not isnotebook():
            return range(self.epochs)
        else:
            return tqdm(range(self.epochs),desc="Epoch",leave = self.leave_bar)


    def fit(self,**kwargs):
        """Main RBM model training function"""
        train_data = kwargs.get("train_data")
        visible_mean = np.mean(train_data,axis = 0)
        visible_mean[visible_mean==0]=0.001
        self.baserate_bias = (np.log(visible_mean) - np.log(1-visible_mean)).astype(kwargs.get("dtype","float32"))
        train_size = train_data.shape[0]
        val_data = kwargs.get("val_data")
        train_data_y = kwargs.get("train_data_y")
        val_data_y = kwargs.get("val_data_y")
        self.trainClassWeights = obtainClassWeights(train_data_y)
        self.valClassWeights = obtainClassWeights(val_data_y)
        train_corrupted = kwargs.get("train_corrupted")
        val_corrupted = kwargs.get("val_corrupted")
        self.epochs = kwargs.get("epochs",1)
        self.LL_epochs = kwargs.get("LL_epochs",100)
        metrics = kwargs.get("metrics",[{"name":"recon_mse","patience":4,"min_delta":0.00005,"optimize":"min"}])
        for metric in metrics:
            for data in ["train_","val_"]:
                self.history[data+metric["name"]] = []
        
        self.batch_size = kwargs.get("batch_size",train_size//8)
        #self.n_steps = kwargs.get("n_step",train_size//self.batch_size)
        self.leave_bar = kwargs.get("leave_bar",True)
        
        start = time.time()
        for epoch in self.epochIterable(): # Training loop
            self.epoch = epoch
            if epoch%100==0 and not isnotebook():
                print(epoch)
            subsampling = np.random.choice(train_data.shape[0],self.batch_size,replace=False)
            batch = tf.gather(train_data,indices = subsampling,axis = 0)
            batch_class = tf.gather(train_data_y,indices = subsampling,axis = 0)
            self.step(batch,batch_class)
            
            self.compute_metrics(metrics,train_data,val_data,train_data_y,val_data_y,train_corrupted,val_corrupted)

            new_metric = self.history[self.evaluation_metric][-1]
            if self.compareMetrics(new_metric):
                self.best_metric = new_metric
                best_params = (self.W,self.visible_bias,self.hidden_bias)
                ind = epoch
                delta = (self.delta_W,self.delta_visible_bias,self.delta_hidden_bias)
            
        self.W,self.visible_bias,self.hidden_bias = best_params
        self.delta_W,self.delta_visible_bias,self.delta_hidden_bias = delta
        self.opt_ind = ind
                
        self.training_time = time.time()-start
        print("Elaped time %s min"%(self.training_time/60))

    def compareMetrics(self,new_metric):
        return self.best_metric < new_metric
        
    def compute_metrics(self,metrics,train_data,val_data,train_data_y,val_data_y,train_corrupted,val_corrupted):
        """Computes metrics for train and val set, and adds them to the history of the model"""
        for metric in metrics:
            if metric["name"] == "roc":
                self.history["train_roc"].append(ROC(train_data_y,self.predict(train_data)))
                self.history["val_roc"].append(ROC(val_data_y,self.predict(val_data)))
            if metric["name"] == "accuracy":
                if self.epoch%self.LL_epochs==0:
                    self.history["train_accuracy"].append(ACCURACY(ProcessClasses(train_data_y),ProcessClasses(self.sample_class_from_probs(self.predict(train_data))),sample_weight = self.trainClassWeights))
                    self.history["val_accuracy"].append(ACCURACY(ProcessClasses(val_data_y),ProcessClasses(self.sample_class_from_probs(self.predict(val_data))),sample_weight = self.valClassWeights))
                else:
                    self.history["train_accuracy"].append(self.history["train_accuracy"][-1])
                    self.history["val_accuracy"].append(self.history["val_accuracy"][-1])
            if metric["name"] == "log_loss":
                if self.epoch%self.LL_epochs==0:
                    self.history["train_log_loss"].append(log_loss(train_data_y,self.predict(train_data),sample_weight=self.trainClassWeights))
                    self.history["val_log_loss"].append(log_loss(val_data_y,self.predict(val_data),sample_weight=self.valClassWeights))
                else:
                    self.history["train_log_loss"].append(self.history["train_log_loss"][-1])
                    self.history["val_log_loss"].append(self.history["val_log_loss"][-1])
            if metric["name"] == "recon_mse":
                self.history["train_recon_mse"].append(float(tf.reduce_mean(self.recon_mse(train_data,train_corrupted))))
                self.history["val_recon_mse"].append(float(tf.reduce_mean(self.recon_mse(val_data,val_corrupted))))
            if metric["name"] == "free_energy":
                delta_free = abs(float(tf.reduce_mean(self.free_energy(train_data[:val_data.shape[0]]))-tf.reduce_mean(self.free_energy(val_data))))
                self.history["train_free_energy"].append(delta_free)
                self.history["val_free_energy"].append(delta_free)
            if metric["name"] == "LL":
                if self.epoch%self.LL_epochs==0:
                    self.approx_log_Z = self.AIS_log_Z() # ALWAYS compute logZ before computing LL
                    self.history["train_LL"].append(self.approximated_LL(train_data))
                    self.history["val_LL"].append(self.approximated_LL(val_data))
                else:
                    self.history["train_LL"].append(self.history["train_LL"][-1])
                    self.history["val_LL"].append(self.history["val_LL"][-1])
            
    def get_name(self):
        """This function returns file name extentions when saving/loading weights"""
        name = self.modelBaseName
        name+="_permuted"*self.permute
        name+= "_nHidden_"+str(self.nHidden)
        name+=self.name
        name+= "_method_"+self.method
        if self.window:
            name+= "_window_"+str(self.window)
            name+= "_stride_"+str(self.stride)
        elif self.windowList:
            name+= "_window_"+"".join([str(i) for i in self.windowList])
            name+= "_stride_"+"".join([str(i) for i in self.strideList])
        name+="_seed_"+str(self.seed)
        name+="_epochs_"+str(self.epochs)
        name+="_lr_"+str(self.lr)
        name+="_momentum_"+str(self.momentum)
        if self.train_size:
            name+="_train_size_"+str(self.train_size)
        name+="_batch_size_"+str(self.batch_size)
        
        return name.replace(".","")
