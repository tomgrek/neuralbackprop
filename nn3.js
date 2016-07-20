"use strict";

// this is one activation function, anything non-linear is OK
function sigmoid(t) {
    return 1/(1+Math.pow(Math.E, -t));
}
// and this is its derivative
function dsig(t) {
  return sigmoid(t) * (1 - sigmoid(t));
}
// here's a different activation function and its derivative
function relu(t) {
  return Math.max(t,0.01);
}
function drelu(t) {
  return t > 0 ? 1 : 0.01;
}
function tanh(t) {
  return (1 - Math.pow(Math.E, -2*t))/(1 + Math.pow(Math.E, -2*t));
}
function dtanh(t) {
  return 1 - Math.pow(t,2);
}

// the learning rate here is set here, in the learning loop it is gradually decayed
// (as the network converges)
var LEARNING_RATE = 0.1;

// define a fixed input node
var Input = function(samples, name) {
  this.samples = samples;
  name != null ? this.name = name : this.name = 'an input';
}

// and an output node
var Output = function(samples, name) {
  this.samples = samples;
  name != null ? this.name = name : this.name = 'an output expected value';
}

// define a neuron e.g. ('sigmoid', 'input layer neuron 0')
// z is the weighted sum of its inputs (i.e. left hand side)
// a is its output, after the activation function
// delta is the degree of error assigned to this neuron during backprop
// connetions is the neurons it receives inputs from (i.e. on its left hand side)
var Neuron = function(type, name) {
    this.type = type;
    name != null ? this.name = name : this.name = 'a neuron';
    this.connections = [];
    this.z = 0;
    this.a = 0;
    this.delta = 0;
    // activation function can be sigmoid or relu
    switch (this.type) {
      case 'sigmoid' : {
        this.activationFunction = sigmoid;
        this.activationFunctionDeriv = dsig;
        break;
      }
      case 'relu' : {
        this.activationFunction = relu;
        this.activationFunctionDeriv = drelu;
        break;
      }
      case 'tanh' : {
        this.activationFunction = tanh;
        this.activationFunctionDeriv = dtanh;
        break;
      }
      default : {
        this.activationFunction = sigmoid;
        this.activationFunctionDeriv = dsig;
        break;
      }
    }
}

Neuron.prototype.connect = function(connectFrom) {
  // add a connected neuron in layer l-1 to a neuron in layer L, with a random weight
  this.connections.push([connectFrom, Math.random()*2 - 1]);
}

Neuron.prototype.listConnections = function() {
  // for debugging and describing the network
  console.log('this node '+this.name+' receives connections from .......');
  for (var i = 0; i < this.connections.length; i++)
    console.log(this.connections[i][0].name + ' with weight ' + this.connections[i][1]);
}

Neuron.prototype.calc = function(sampleNum) {
  // calculate the neuron's output for one turn
  // samplenum is a real integer (the current sample we're training on)
  // it has no relevance if the neuron isn't an input neuron
  this.z = 0;
  for (var i = 0; i < this.connections.length; i++) {
      var incoming = this.connections[i][0];
      var weighting = this.connections[i][1];

      if (incoming.hasOwnProperty('a')) { // the connection is a previous neuron
        this.z += incoming.a * weighting; // so use that neuron's output
      }
      else {
        this.z += incoming.samples[sampleNum] * weighting; // the connection is a fixed input
      }
  }
  this.a = this.activationFunction(this.z);
}

Neuron.prototype.describeState = function() {
  // for debugging and describing the network
  console.log('i am ' + this.name + ' .......... and ')
  console.log('z, my weighted inputs, is ' + this.z);
  console.log('a, my activation (f(z)), is ' + this.a);
}

function neural_network(layers) {
  // just dimensions an array for a certain number of layers
  var retval = new Array([]);
  for (var i = 0; i < layers; i++)
    retval.push([]);
  return retval;
}

function calcOutput(layers, sampleNum) {
  // calculate the whole network; the forward pass
  for (var i = 0; i < layers.length; i++) {
    for (var j = 0; j < layers[i].length; j++)
      layers[i][j].calc(sampleNum);
  }
}

function showOutput(layers) {
  // just a convenience function to show the output of a network of layers, i.e. the last layer
  var outputlayer = [];
  for (var i = 0; i < layers[layers.length-1].length; i++) {
    outputlayer.push(layers[layers.length-1][i].a);
  }
  console.log(outputlayer);
}

function calcError(layers, output, sampleNum) {
  // calculate difference between desired and actual, for a given sampleNum.
  var error = [];
  calcOutput(sampleNum);
  var temperror = 0;
  for (var i = 0; i < layers[layers.length-1].length; i++) {
    temperror = output[i].samples[sampleNum] - layers[layers.length-1][i].a;
    // it's calculating the absolute (squared) error then also giving it a direction.
    temperror > 0 ? error.push(Math.pow(temperror,2)) : error.push(-Math.pow(temperror,2));
  }
  return error;
}

function calcOutputDelta(layers, errors) {
  // it's simple to calculate the output delta as the scaled error times the derivative of the inputs
  var thisDelta = 0;
  for (var i = 0; i < errors.length; i++) {
    thisDelta = LEARNING_RATE * errors[i] * layers[layers.length-1][i].activationFunctionDeriv(layers[layers.length-1][i].z);
    layers[layers.length-1][i].delta = thisDelta;
  }
}

function calcOtherDeltas(layers) {
  // backpropagation
  // first zero all the deltas for all layers except output
  for (var jj = layers.length-2; jj >= 0; jj--) {
    for (var k = 0; k < layers[jj].length; k++) {
      layers[jj][k].delta = 0;
    }
  }
  // then working backwards, multiply delta at all connected neurons in layer L+1 by weight to this neuron by this neuron's gradient
  for (var jj = layers.length-2; jj >= -1; jj--) {
    for (var k = 0; k < layers[jj+1].length; k++) {
      for (var m = 0; m < layers[jj+1][k].connections.length; m++) {
        if (layers[jj+1][k].connections[m][0].hasOwnProperty('z')) { // check if it's a neuron, i.e. has a z
          layers[jj+1][k].connections[m][0].delta += layers[jj+1][k].delta * layers[jj+1][k].connections[m][1] * layers[jj+1][k].connections[m][0].activationFunctionDeriv(layers[jj+1][k].connections[m][0].z);
        } else { // or if it's a fixed input, do this instead.
          layers[jj+1][k].connections[m][0].delta += layers[jj+1][k].delta * layers[jj+1][k].connections[m][1] * layers[jj+1][k].connections[m][0].samples[j];
        }
      }
    }
  }
}

function adjustOutputLayer(layers) {
  // adjust the weights of each neuron in the output layer by their delta
  for (var jj = 0; jj < layers[layers.length-1].length; jj++) {
    for (var k = 0; k < layers[layers.length-1][jj].connections.length; k++) {
      layers[layers.length-1][jj].connections[k][1] += layers[layers.length-1][jj].delta * layers[layers.length-1][jj].connections[k][0].a;
      }
  }
}

function adjustOtherLayers(layers) {
  // work backwards (ie starting on the right hand side) for the other layers
  for (var jj = layers.length-2; jj >= 0; jj--) {
    for (var k = 0; k < layers[jj].length; k++) {
        for (var m = 0; m < layers[jj][k].connections.length; m++) {
          if (layers[jj][k].connections[m][0].hasOwnProperty('z')) {
            layers[jj][k].connections[m][1] += layers[jj][k].delta * layers[jj][k].connections[m][0].z;
          }
          else {
            layers[jj][k].connections[m][1] += layers[jj][k].delta * layers[jj][k].connections[m][0].samples[j];
          }
        }
    }
  }
}


function generateNetwork(trainingData, numLayers, numPerLayer, numOutputs) {
  // this is a convenience function if you don't want to create a network yourself. Each layer (except the output layer)
  // will have the same number of neurons, numPerLayer.
  numLayers = numLayers-1; // 3 layers - input, hidden, output - at a minimum. -1 because the array will go from 0-2.
  if (numLayers < 2) return false;
  var layers = neural_network(numLayers-1);
  var inputs = [];
  var outputs = [];
  for (var i = 0; i < trainingData[0].length - numOutputs; i++) {
    inputs.push(new Input(trainingData.map(function(x) { return x[i]}), 'fixed input '+i));
  }
  for (var i = trainingData[0].length - numOutputs; i < trainingData[0].length ; i++) {
    outputs.push(new Output(trainingData.map(function(x) { return x[i]}), 'output neuron '+(i - trainingData[0].length+numOutputs)));
  }
  for (var i = 0; i < numLayers-1; i++) {
    for (var j = 0; j < numPerLayer; j++) { // to make a simple, non-customized network, number of neurons in each layer is the same
      layers[i].push(new Neuron('sigmoid', 'layer '+i+' neuron '+j)); // and they're all sigmoid
    }
  }
  for (var i = 0; i < numOutputs; i++) {
    layers[numLayers-1][i] = new Neuron('sigmoid', 'output layer neuron '+i);
  }
  return {inputs: inputs, outputs: outputs, layers: layers};
}

function fullyInterconnect(net) {
  // another convenience function.
  for (var i = 0; i < net.layers[0].length; i++) {
    for (var j = 0; j < net.inputs.length; j++) {
      net.layers[0][i].connect(net.inputs[j]);
    }
  }

  for (var i = 1; i < net.layers.length; i++)
    for (var j = 0; j < net.layers[i].length; j++)
      for (var k = 0; k < net.layers[i-1].length; k++)
        net.layers[i][j].connect(net.layers[i-1][k]);
}

// remember to normalize the training data - inputs and outputs must be between -1 and +1 (if using the sigmoid activation function)

var NUM_RESERVED = 1; // out of the below training data, how many of the last samples do we reserve for testing afterwards?

// remember to normalize the training data - inputs and outputs must be between -1 and +1 (if using the sigmoid activation function)
// let's teach it to multiply inputs x1 and x2 (as if the numbers were above 1)
var trainingData = [[0,0,0],[0,0.1,0],[0.1,0.1,0.1],[0.2,0.2,0.4],[0.2,0.3,0.6],[0.3,0.2,0.6],[0.2,0.4,0.8],[0.3,0.3,0.9]]; // so the final sample here is held back (net doesn't train on it)
// it doesn't matter what you set as the output of the testing samples, just that the array is the same dimension as all the others.
// and the testing sample here is just for illustration; an XOR gate can't have inputs of 0.5.

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

// do 1000000 iterations of training
for (var i = 0; i < 1000000; i++) {
  // for this iteration, pick a random training sample j
  // this is incremental learning, not batch.
  var j = Math.floor(Math.random() * (trainingData.length-NUM_RESERVED));

  // calculate the network for that training sample
  calcOutput(layers,j);

  // calculate the error for that training sample
  var err = calcError(layers,[y1],j);

  // every so often print an update on how the training is going
  if (i%100000==0 ) { console.log('error (for the present sample) is: ' +err); LEARNING_RATE *= 0.91; console.log('decaying learning rate to '+LEARNING_RATE); }

  // calculate the deltas for each layer/neuron, feeding back the output error through each weight
  calcOutputDelta(layers,err);
  calcOtherDeltas(layers);

  // adjust the weights of the output layer
  adjustOutputLayer(layers);
  // then adjust the incoming weights to each neuron by its delta times the 'importance' of that weight
  adjustOtherLayers(layers);

}

// if you want to make predictions based on samples not in the training data, add more training data and
// select only training samples below that position in the array (variable j in the main training loop)
console.log('finished training, here are the outputs for the cases i was trained on');
calcOutput(layers,0); showOutput(layers);
calcOutput(layers,1); showOutput(layers);
calcOutput(layers,2); showOutput(layers);
calcOutput(layers,3); showOutput(layers);
calcOutput(layers,4); showOutput(layers);
calcOutput(layers,5); showOutput(layers);
calcOutput(layers,6); showOutput(layers);
console.log('and here is my output for the cases that were held back from my training. did i generalize well?');
calcOutput(layers,7); showOutput(layers);
