import os
from optparse import OptionParser
from Launch_Experiments import PARAMS_DICT

__version__ = "1.0"
def command_line_arg():
    """Main function that takes the input variable, index of the parameters dict"""
    usage = "usage: %prog [options] arg1 arg2"
    par = OptionParser(usage=usage, version= __version__)

    return  par.parse_args()
    
if __name__ == '__main__':
    opts, args = command_line_arg()
    Experiments = [("MNIST","Unsupervised"),
                   ("FashionMNIST","Unsupervised"),
                   ("pneumoniamnist","Unsupervised"),
                   ("breastmnist","Unsupervised"),
                   ("MNIST","Supervised"),
                   ("FashionMNIST","Supervised"),
                   ("pneumoniamnist","Supervised"),
                   ("breastmnist","Supervised"),]
    
    for SEED in range(2):
        for exp in Experiments:        
            DATA_NAME = exp[0]
            TASK = exp[1]
            MAIN_DICT = [dict(item,DATA_NAME = DATA_NAME, TASK = TASK) for item in PARAMS_DICT]                  
            os.system("python Launch_Experiments.py -d %s -t %s -s %s"%(DATA_NAME,TASK,SEED))
