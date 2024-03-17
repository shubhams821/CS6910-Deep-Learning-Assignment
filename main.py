import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import pdb
import math
from tqdm import tqdm
import wandb
wandb.login(key='da5365b4335ad8c7a1df7f3653ec9d0b092e8b09')

import feed_forward_neural_network as FFNN


sweep_config = {
    'method': 'random',
    'name' : 'sweep cross entropy',
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'epochs': {
            'values': [5, 10]
        },
        'num_hidden_layers': {
            'values': [3, 4, 5]
        },
        'hidden_layer_size': {
            'values': [32, 64, 128]
        },
        'weight_decay': {
            'values': [0, 0.0005, 0.5]
        },
        'learning_rate': {
            'values': [1e-3, 1e-4]
        },
        'optimizer': {
            'values': ['sgd', 'mgd', 'nag', 'rmsprop', 'adam', 'nadam']
        },
        'batch_size': {
            'values': [16, 32, 64]
        },
        'weight_initialization': {
            'values': ['random', 'xavier']
        },
        'activation_function': {
            'values': ['sigmoid', 'tanh', 'relu']
        }
    }
}

sweep_id = wandb.sweep(sweep=sweep_config, project='DL_Ass_1')



from tensorflow.keras.datasets import fashion_mnist

# Load Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Preprocess the data (flatten and normalize)
x_train = x_train.reshape((x_train.shape[0], -1)).astype('float32') / 255.0
x_test = x_test.reshape((x_test.shape[0], -1)).astype('float32') / 255.0



from tensorflow.keras.utils import to_categorical

y_train_onehot = to_categorical(y_train)
y_test_onehot = to_categorical(y_test)




def train(x_train, y_train, x_val, y_val,optimizer,Neural_Net):
    if optimizer == 'sgd':
      Neural_Net.stochastic_gradient_descent(x_train, y_train, x_val, y_val)
    elif optimizer == 'mgd':
      Neural_Net.momentum_gradient_descent(x_train, y_train, x_val, y_val)
    elif optimizer == 'nag':
      Neural_Net.nesterov_accelerated_gradient_descent(x_train, y_train, x_val, y_val)
    elif optimizer == 'rmsprop':
      Neural_Net.rmsprop(x_train, y_train, x_val, y_val)
    elif optimizer == 'adam':
      Neural_Net.adam(x_train, y_train, x_val, y_val)
    elif optimizer == 'nadam':
      Neural_Net.nadam(x_train, y_train, x_val, y_val)
    return 0



def train_NN(config):
  eph = config['epochs']
  num_hidden_layers = config['num_hidden_layers']
  hidden_layer_size = config['hidden_layer_size']
  weight_decay = ['weight_decay']
  learning_rate = config['learning_rate']
  optimizer = config['optimizer']
  batchsize = config['batch_size']
  weight_initialization = config['weight_initialization']
  activation_function = config['activation_function']


  l_size =[]
  l_size.append(x_train.shape[1])
  for i in range(num_hidden_layers):
    l_size.append(hidden_layer_size)
  l_size.append(10)
  Neural_Net = FFNN(layer_sizes=l_size,L = num_hidden_layers, epochs=eph, l_rate=learning_rate, batch_size=batchsize, activation_func=activation_function, loss_func='cross_entropy', output_activ_func='softmax', initializer=weight_initialization)
  train(x_train, y_train_onehot, x_test, y_test_onehot, optimizer,Neural_Net)
  test_acc, test_loss, _, _ = Neural_Net.model_performance(x_test, y_test_onehot)
  wandb.log({"val_accuracy": test_acc, "val_loss":test_loss})



def main():
    '''
    WandB calls main function with random combination,

    We log the values obtain from the given hypermeters.

    '''


    with wandb.init() as run:

        run_name="-ac_"+wandb.config.activation_function+"-hs"+str(wandb.config.hidden_layer_size)
        wandb.run.name=run_name
        train_NN(wandb.config)


wandb.agent(sweep_id, function=main, count = 200) # calls main function for count number of times.
wandb.finish() 
