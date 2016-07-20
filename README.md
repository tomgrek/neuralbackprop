# neuralbackprop
Sunday afternoon project: make a back-prop learning neural network

This is a barebones neural network implemented in Javascript. It learns via backpropagation. The code isn't optimized,
to keep it easy to read and understand - you can see each step in the learning loop instead of it being a black box.

## How flexible is it?

You can create networks with any number of layers, with any number of neurons per layer, and
an arbitrary number of inputs and outputs. There are three activation functions
provided (sigmoid, tanh, ReLU), but it's easy to see how to add other ones. There's one cost function but again, it's easy
to change it.

It requires a 'labeled' (supervised) training set. Learning is incremental, but can easily be extended to batch.

## How to use it?

Have a look at the code! The two files are almost the same, but one uses convenience functions to build a 
fully interconnected network, and the other builds a more custom network.

```
// With convenience functions:
var trainingData = [[0,0,0],[0,1,1],[1,0,1],[1,1,0]]; 
var net = generateNetwork(trainingData, 4, 6, 1);
fullyInterconnect(net);
```

```
// Custom network:
var trainingData = [[0,0,0],[0,0.1,0],[0.1,0.1,0.1],[0.2,0.2,0.4],[0.2,0.3,0.6],[0.3,0.2,0.6],[0.2,0.4,0.8],[0.3,0.3,0.9]]; 
// split up the training data into individual inputs and their corresponding output
var x1 = new Input(trainingData.map(function(x) { return x[0]}), 'input 1');
var x2 = new Input(trainingData.map(function(x) { return x[1]}), 'input 2');
var y1 = new Output(trainingData.map(function(x) { return x[2]}));

// this function merely sets up the layers variable as a properly dimensioned array.
var layers = neural_network(2);

// build the network. this network is fully connected, 4 input 2 hidden 1 output
// first create the first layer of input neurons
layers[0][0] = new Neuron('tanh', 'input layer neuron 1');
layers[0][1] = new Neuron('sigmoid', 'input layer neuron 2');
layers[0][2] = new Neuron('relu', 'input layer neuron 3');
layers[0][3] = new Neuron('tanh', 'input layer neuron 4');
// then connect these to our inputs, which will feed in the training data
layers[0][0].connect(x1); layers[0][0].connect(x2);
layers[0][1].connect(x1); layers[0][1].connect(x2);
layers[0][2].connect(x1); layers[0][2].connect(x2);
layers[0][3].connect(x1); layers[0][3].connect(x2);

// create the hidden layer and connect each of its two neurons to all four neurons in the input layer
layers[1][0] = new Neuron('tanh', 'hidden layers neuron 1');
layers[1][1] = new Neuron('sigmoid', 'hidden layers neuron 2');
layers[1][0].connect(layers[0][0]); layers[1][0].connect(layers[0][1]); layers[1][0].connect(layers[0][2]); layers[1][0].connect(layers[0][3]);
layers[1][1].connect(layers[0][0]); layers[1][1].connect(layers[0][1]); layers[1][1].connect(layers[0][2]); layers[1][1].connect(layers[0][3]);

// create the output layer, in this case a single neuron
layers[2][0] = new Neuron('relu', 'output neuron');
layers[2][0].connect(layers[1][0]); layers[2][0].connect(layers[1][1]);
```

## How fast is it?

Pretty fast for toy networks (<100 neurons) and perhaps small applications. On my 3 years old ultrabook, the toy sized 
network does 1,000,000 epochs in a couple of seconds.








