// Declaring global variables
let n;
let dataFile;
let testFile;

const inputNodes = 784;
const hiddenNodes = 100;
const outputNodes = 10;
const learningRate = 0.3;

function preload() {
	// Loading in files
	// Each record contains the label as the first entry
	// then 784 pixel values from 0 to 255, all separated
	// by commas
	dataFile = loadStrings("mnist data/mnist_train_100.csv");
	testFile = loadStrings("mnist data/mnist_test_10.csv");
}

function setup() {
	createCanvas(282, 282);
	background(100);

	n = new NeuralNetwork(
		inputNodes, hiddenNodes, outputNodes, learningRate
	);

	console.log("start");
	// Training the network
	let temp = 0;
	for (let record of dataFile) {
		// Splitting the record into an array of numbers and
		// removing the first entry
		const values = split(record, ",");
		const label = values[0];
		values.splice(0, 1);

		// Converting the pixel values from 0-255 to 0.01-1
		const inputList = math.map(values, (value) => {
			return (map(value, 0, 255, 0.01, 1));
		});

		// Creating the target list with all entries at 0.01
		// except for the expected answer, which is at 0.99
		const targetList =
			math.add(math.zeros(10), 0.01).toArray();
		targetList[label] = 0.99;

		n.train(inputList, targetList);
		temp++;

		if ((temp * 100 / dataFile.length) % 10 == 0)
			console.log((temp * 100 / dataFile.length) + "% training completion");
	}

	// Testing the network
	let numberCorrect = 0;
	let total = testFile.length;

	temp = 0;
	for (let record of testFile) {
		// Splitting the record into an array of numbers and
		// removing the first entry
		const values = split(record, ",");
		const correctLabel = values[0];
		values.splice(0, 1);

		// Converting the pixel values from 0-255 to 0.01-1
		const inputList = math.map(values, (value) => {
			return (map(value, 0, 255, 0.01, 1));
		});

		// Getting the output from the network
		const outputs = n.query(inputList);
		const answer = indexOfMax(outputs);

		// Checking the output
		if (answer == correctLabel) numberCorrect++;

		temp++;
		if ((temp * 100 / testFile.length) % 10 == 0)
			console.log((temp * 100 / testFile.length) + "% testing completion");
	}

	console.log(numberCorrect / total * 100 + "% accurate");




}

function draw() {


}



function randomWeightsMatrix(i, j) {
	const x = math.matrix();
	for (let a = 0; a < i; a++) {
		for (let b = 0; b < j; b++) {
			x.set([a, b], randomGaussian(0, pow(j, -0.5)));
		}
	}
	return x;
}

function indexOfMax(arr) {
	if (arr.length === 0) return -1;

	let max = arr[0];
	let maxIndex = 0;

	for (let i = 1; i < arr.length; i++) {
		if (arr[i] > max) {
			maxIndex = i;
			max = arr[i];
		}
	}

	return maxIndex;
}

function sigmoid(x) {
	return 1 / (1 + pow(math.e, -x));
}