# Machine-Learning-Tensorflow

Contains : 
  - Linear Regression
  - Softmax Regression
  - CNN on MNIST dataset
  - RNN on MNIST dataset
  - RNN models(5) for Protein Family Classification 
  - RNN models() for Protein Secondary Structure Prediction
 
We are going to follow these steps for model creation in tensorflow every time :

    1. Data and all parameters given while class construction.
       Model and data computation graph is also made then itself.
       All the nodes in the dc-graph including predicted value, loss, accuracy
       and optimizer will be designed here itself.

    2. Then there is a function that help in prediction

    3. A function that trains and optimizes which will be called
       number of times we want to iterate (i.e. no of epochs).

    4. A function that gets the cross-validation score.

