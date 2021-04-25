class NeuralNetwork {
  constructor(inputNodes, hiddenNodes, outputNodes, learningRate) {
    // Setting number of nodes in each layer and the learning rate
    this.inodes = inputNodes;
    this.hnodes = hiddenNodes;
    this.onodes = outputNodes;
    this.lr = learningRate;

    // Initialising random starting weights in matricies
    // Index 'i,j' means from node i to node j in next layer
    this.wIH = randomWeightsMatrix(this.hnodes, this.inodes);
    this.wHO = randomWeightsMatrix(this.onodes, this.hnodes);

  }

  // Setting activation function to the sigmoid function
  activationFunction(x) {
    return sigmoid(x);
  }




  train(inputList, targetList) {
    // Converting target list and input list into a matrix
    let targets = math.matrix([targetList]);
    targets = math.transpose(targets);
    let inputs = math.matrix([inputList]);
    inputs = math.transpose(inputs);

    // Calculating signals into hidden layer
    const hiddenInputs = math.multiply(this.wIH, inputs);
    // Calculating signals leaving hidden layer
    const hiddenOutputs = math.map(hiddenInputs, (value) => {
      return this.activationFunction(value);
    });

    // Calculating signals into output layer
    const finalInputs = math.multiply(this.wHO, hiddenOutputs);
    // Calculating outputs
    const finalOutputs = math.map(finalInputs, (value) => {
      return this.activationFunction(value);
    });




    // Calculating output layer errors
    const outputErrors = math.subtract(targets, finalOutputs);

    // Calculating hidden layer errors
    const hiddenErrors = math.multiply(
      math.transpose(this.wHO), outputErrors
    );

    // Updating weights from hidden layer to output layer
    let temp1 = math.subtract(1, finalOutputs);
    let temp2 = math.dotMultiply(outputErrors, finalOutputs);
    let temp3 = math.dotMultiply(temp2, temp1);
    const delta_wHO = math.multiply(
      this.lr,
      (math.multiply(temp3, math.transpose(hiddenOutputs))));
    this.wHO = math.add(this.wHO, delta_wHO);

    // Updating weights from input layer to hidden layer
    temp1 = math.subtract(1, hiddenOutputs);
    temp2 = math.dotMultiply(hiddenErrors, hiddenOutputs);
    temp3 = math.dotMultiply(temp2, temp1);
    const delta_wIH = math.multiply(
      this.lr,
      (math.multiply(temp3, math.transpose(inputs))));
    this.wIH = math.add(this.wIH, delta_wIH);

  }

  query(inputList) {
    // Converting input list into a matrix
    let inputs = math.matrix([inputList]);
    inputs = math.transpose(inputs);

    // Calculating signals into hidden layer
    const hiddenInputs = math.multiply(this.wIH, inputs);
    // Calculating signals leaving hidden layer
    const hiddenOutputs = math.map(hiddenInputs, (value) => {
      return this.activationFunction(value);
    });

    // Calculating signals into output layer
    const finalInputs = math.multiply(this.wHO, hiddenOutputs);
    // Calculating outputs
    const finalOutputs = math.map(finalInputs, (value) => {
      return this.activationFunction(value);
    });

    return finalOutputs.toArray();

  }


}