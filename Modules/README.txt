README file for modules:
A list of the modules and their functions.

·RBM: object for RBM model, every function has a description of what they do. After training, rbm has an attribute .history which contains lists of specified metrics computed for train and validations sets.

·Mask:
    -hidden_visible_mask(window,t_x,t_y,n_visible,**kwargs): returns mask for a RBM weight parameters.
    -plot_vis_hid(hidden_loc,dot_xloc,dot_yloc,window,t_x,t_y,**kwargs): plots number of connections each visible has with hidden units.
    -neighbours(i,j,dim,window,return_self = False): returns neighbour pixel coordinates.

·Plot: 
    -evaluate_corruption_recon(rbm,corruptions,**kwargs): returns corruption errors of rbm from the list corruptions.
    -evaluate_models(names,corruptions,**kwargs): returns corrupted reconstruction error for RBMs in list "names".
    -compare_boxplots(data_list,**kwargs): plots reconstruction error in boxplots for many RBMs.
    -boxplots(data,**kwargs): same as compare_boxplots, but for a single RBM.
    -plot_digit_grid(states,n = 10,title = "",**kwargs): plots states in grid structure of n x n.
    -plot_reconstruction(rbm,corruption,n,title = "",save = False,**kwargs): given a rbm, computes reconstruction in the corrupted images, and plots the original and reconstructed images in a n x 2n grid.
    -plot_errors(errors,names,ylabel = "Error",title = "Training error"): plots given errors with "names" in legend.

·Save_Load:
    -save_rbm(rbm,**kwargs): saves RBM, kwargs has directory specifications.
    -load_rbm(name,**kwargs): loads RBM, kwargs has directory specifications.
    -from_dataset_to_array(data_name = "",split = "train",**kwargs): loads dataset with name=data_name, split can be train or test.
    
·BayesOpt:
    -objective_function(x): objetive function to be optimized by BayesOpt.
    -BayesOpt(variables,**kwargs): takes variables (hyperparameters) to be optimized, as well as kwargs for training a RBM, and uses Bayesian Optimization to optimize said hyperparameters. Returns a GPyOpt.methods.BayesianOptimization object which contains info about the optimization process.
    
·utils:
    -sample_bernoulli(ps): samples binary data with prob=ps.
    -dict(**kwargs): return kwargs as dict.
