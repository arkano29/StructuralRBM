import os
from optparse import OptionParser

"""HERE THE MOST IMPORTANT PARAMETERS FOR TRAINING ARE SET"""

# 1-Models
PARAMS_DICT_1 = [{"nHidden":196},{"window":1,"stride":2},
                {"nHidden":162},{"windowList":[1,2],"strideList":[3,3]},
                {"nHidden":144},{"window":3,"stride":2},
                {"nHidden":121},{"windowList":[2,3,4],"strideList":[4,4,4]},
                {"nHidden":81},{"window":2,"stride":3},
               ]

# New models:

new_PARAMS_DICT_1 = [{"nHidden":562},{"windowList":[4,4],"strideList":[2,1]},
                {"nHidden":585},{"windowList":[3,4],"strideList":[2,1]},
                {"nHidden":265},{"windowList":[3,4],"strideList":[2,2]},
                {"nHidden":441},{"window":4,"stride":1},
                {"nHidden":121},{"window":4,"stride":2},
                {"nHidden":144},{"window":3,"stride":2},
               ]

# 2-Bigger models(n_fold hidden units)
n_fold = 5
PARAMS_DICT_2 = [{"window":1,"stride":2,"n_fold":n_fold},             
                {"windowList":[1,2],"strideList":[3,3],"n_fold":n_fold},
                {"window":3,"stride":2,"n_fold":n_fold}, 
                {"windowList":[2,3,4],"strideList":[4,4,4],"n_fold":n_fold},
                {"window":2,"stride":3,"n_fold":n_fold},
               ]

# 3-Different train_size
train_sizes = [int(60000/(2**i)) for i in range(5)]
PARAMS_DICT_3 = [dict(item,train_size=train_sizes[i]) for item in PARAMS_DICT_1 for i in range(5)]

# 4-Smaller Different train_size
train_sizes = [2**i for i in range(7,12)]
train_sizes.append(0)#This way we add the whole dataset training process to the list of experiments
PARAMS_DICT_4 = [dict(item,train_size=train_sizes[i]) for item in PARAMS_DICT_1 for i in range(len(train_sizes))]
new_PARAMS_DICT_4 = [dict(item,train_size=train_sizes[i]) for item in new_PARAMS_DICT_1 for i in range(len(train_sizes))]

# 5- Trial debugger single experiment
PARAMS_DICT_5 = [{"nHidden":40},{"nHidden":20}]

PARAMS_DICT = new_PARAMS_DICT_4

EPOCHS = 10000 

lr = 1
momentum = 0.95
batch_size = 2**4
n_step = 1
dtype = "float32"

PATIENCE = 100000

unsuper_metrics = [{"name":"LL","patience":5,"min_delta":0.0000,"optimize":"max"}]
               
super_metrics = [{"name":"log_loss","patience":PATIENCE,"min_delta":0.0000,"optimize":"min"},
                {"name":"accuracy","patience":5,"min_delta":0.0000,"optimize":""},
                ]

CORRUPTIONS = [
                "identity",
                'shot_noise',
                'impulse_noise',
                'glass_blur',
                'motion_blur',
                'spatter',
                'dotted_line',
                'zigzag',
                ]

METRICS_DICT = {'recon_mse_identity': 'Uncorrupted', 'recon_mse_shot_noise': 'Shot noise',
                 'recon_mse_impulse_noise': 'Impulse noise', 'recon_mse_glass_blur': 'Glass blur', 
                 'recon_mse_motion_blur': 'Motion blur', 'recon_mse_spatter': 'Spatter', 
                 'recon_mse_dotted_line': 'Dotted line', 'recon_mse_zigzag': 'Zigzag',
                 'free_energy':'Free energy', 'LL':'Loglikelihood',
                 'roc':'ROC','accuracy':'Accuracy',
                 'accuracy_identity': 'Uncorrupted', 'accuracy_shot_noise': 'Shot noise',
                 'accuracy_impulse_noise': 'Impulse noise', 'accuracy_glass_blur': 'Glass blur', 
                 'accuracy_motion_blur': 'Motion blur', 'accuracy_spatter': 'Spatter', 
                 'accuracy_dotted_line': 'Dotted line', 'accuracy_zigzag': 'Zigzag',
                 'roc_identity': 'Uncorrupted', 'roc_shot_noise': 'Shot noise',
                 'roc_impulse_noise': 'Impulse noise', 'roc_glass_blur': 'Glass blur', 
                 'roc_motion_blur': 'Motion blur', 'roc_spatter': 'Spatter', 
                 'roc_dotted_line': 'Dotted line', 'roc_zigzag': 'Zigzag',
                 "epoch":"Epochs","time":"Training time (min)",
                 }

__version__ = "1.0"
def command_line_arg():
    """Main function that takes the arguments dataset and task from the terminal"""
    usage = "usage: %prog [options] arg1 arg2"
    par = OptionParser(usage=usage, version= __version__)

    par.add_option('-d', '--data_name', dest = "data_name",
            type="str",default = "MNIST")
    par.add_option('-t', '--task',dest = "task",
            type="str",default = "Unsupervised")
    par.add_option('-s', '--seed', dest = "seed",type="int",default = 0)

    return  par.parse_args()
    
def main(DATA_NAME,TASK,SEED):
    
    MAIN_DICT = [dict(item,DATA_NAME = DATA_NAME, TASK = TASK) for item in PARAMS_DICT]
        
    os.system("sbatch --array=0-%s --job-name=%s-%s-%s script_experiments.sl %s %s %s"%(len(MAIN_DICT)-1,DATA_NAME,TASK,SEED,DATA_NAME,TASK,SEED))
    
if __name__ == '__main__':
    opts, args = command_line_arg()
    main(opts.data_name,opts.task,opts.seed)