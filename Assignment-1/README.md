# CS6910-Deep-Learning-Assignment - 1

Shubham Singh MA22C043
Instructions to train and evaluate the neural network models:

To train a neural network model for image classification on the Fashion-MNIST dataset using categorical cross-entropy loss, import feed_forward_neural_network.py, and use feed_forward_neural_network.Feed_Forward_Neural_Network with any configuration to create the neural network.
To train a neural network model:

use feed_forward_neural_network.Feed_Forward_Neural_Network.{optimizer}
optimizer: optimization routine 
 (stochastic_gradient_descent, momentum_gradient_descent, nesterov_accelerated_gradient_descent, rmsprop, sdam, nadam)


main.py: creates the sweep and randomly searches for best hyperparameter, it runs for 200 counts maximum.

1. To upload the example images from each class and the confusion matrices given as images Example_Image.png and Confusion_Matrix.png in this repository, run the file titled: confusion_matrix_and_Quest_1.py.

8. To train the model on Squared Error loss: run the file titled: train_for_squared error.py

10. To train the model on MNIST dataset: run the file titled: train_for_mnist.py



Link to the project report: [https://wandb.ai/shubham821/DL_Ass_1_Try_8/reports/Shubham-Singh-CS6910-Assignment-1-Report--Vmlldzo3MTg0MTUz](url)





The NN training framework:

These codes are based on a procedural framework and make no use of classes for NN models like keras does for the simplicity of understanding the code. This code works only for classification tasks and by default assumes that the activation function for the last layer is softmax.

1. Training NN

To train the NN, it takes the training data, the validation data and the hyperparameters and Trains a NN specified by hidden_layer_size and num_hidden_layers. This code provides flexibility in choosing the following hyperparameters:
        
        'epochs': 5, 10
        'num_hidden_layers': 3, 4, 5
        'hidden_layer_size': 32, 64, 128
        'weight_decay': 0, 0.0005, 0.5
        'learning_rate': 1e-3, 1e-4
        'optimizer':'sgd', 'mgd', 'nag', 'rmsprop', 'adam', 'nadam'
        'batch_size': 16, 32, 64
        'weight_initialization':  'random', 'xavier'
        'activation_function': 'sigmoid', 'tanh', 'relu'
       


2. Model Evaluation

The function model_evaluation() takes x_test and y_test, and return test accuracy, test loss, y_test, y_pred.



Note: Wherever you need to log to wandb, please remember to change the name of the entity and project in the corresponding line of code.
