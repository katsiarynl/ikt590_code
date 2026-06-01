import numpy as np 
import os 

def load_dataset(config):
  
    X_train_file = "X_train_balanced.npy"
    Y_train_file = "Y_train_balanced.npy"


    X_train = np.load(os.path.join(config.DATA_DIR, X_train_file))
    Y_train = np.load(os.path.join(config.DATA_DIR, Y_train_file))

    X_val = np.load(os.path.join(config.DATA_DIR, "X_val.npy"))
    Y_val = np.load(os.path.join(config.DATA_DIR, "Y_val.npy"))

    X_test = np.load(os.path.join(config.DATA_DIR, "X_test.npy"))
    Y_test = np.load(os.path.join(config.DATA_DIR, "Y_test.npy"))

    return X_train, Y_train, X_val, Y_val, X_test, Y_test