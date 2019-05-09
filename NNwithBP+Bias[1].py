import numpy as np
import pandas as pds
from sklearn import preprocessing

#open file of joke threat scores
negThreatScoreFile = open("/Users/UP719920/Documents/Final Year Project/projectFiles/negThreatScoreFile2.txt", "r")
#variable to store negative threat scores
negInputVector = []


#open file of actual threat scores
posThreatScoreFile = open("/Users/UP719920/Documents/Final Year Project/projectFiles/posThreatScoreFile2.txt", "r")
#variable to store positive threat scores
posInputVector = []

#transposes input vector into a column vector
def transpose(vector, ndMin):
    return np.array(vector, ndmin = ndMin).T

def createDataSet(file, threatArray, label):

    dataPoint = []

    for score in file:
        #adds threat score to variable as int
        threatArray.append((int(score[0:-1]), label))

    for i in range(len(threatArray)):
        score = threatArray[i][0]
        dataPoint.append(score)

    #creates input vector from threat score variable
    TSinputVector = transpose(dataPoint, 1)

    return TSinputVector

negThreatScore = createDataSet(negThreatScoreFile, negInputVector, 0.01)
posThreatScore = createDataSet(posThreatScoreFile, posInputVector, 0.99)

dataSet = negInputVector + posInputVector

#import truncnorm form scipy.stats
from scipy.stats import truncnorm as trNorm

#define a sigmoidal function
def sigActFunction(x):
    return 1 / (1 + np.e ** -x)

#function to create random values using normal distribution
def randTruncatedNormal (mean = 0, stdDev = 1, lower = 0, upper = 10):
    return trNorm((lower - mean)/stdDev, (upper - mean)/stdDev, loc = mean, scale = stdDev)


#class to create a neural network with backpropagation
class neuralNetworkBP:

    def __init__(self, numInputLayerNodes, numHiddenLayerNodes, numOutputLayerNodes, learningRate, bias=None):
        self.numInputLayerNodes = numInputLayerNodes
        self.numHiddenLayerNodes = numHiddenLayerNodes
        self.numOutputLayerNodes = numOutputLayerNodes
        self.learningRate = learningRate
        self.bias = bias
        self.generateWeightMatrices()

    def generateWeightMatrices(self):

        #set value of bias node
        if self.bias:
            biasNode = 1
        else:
            biasNode = 0

        #create weights from input layer to hidden layer (ITH)
        ITHweightRange = 1 / np.sqrt(self.numInputLayerNodes + biasNode)
        ITHrandomNormal = randTruncatedNormal(mean = 0, stdDev = 1, lower = -ITHweightRange, upper = ITHweightRange)
        self.ITHweights = ITHrandomNormal.rvs((self.numHiddenLayerNodes, self.numInputLayerNodes + biasNode))

        #create weights from hidden layer to output layer (HTO)
        HTOweightRange = 1 / np.sqrt(self.numHiddenLayerNodes + biasNode)
        HTOrandomNormal = randTruncatedNormal(mean = 0, stdDev = 1, lower = -HTOweightRange, upper = HTOweightRange)
        self.HTOweights = HTOrandomNormal.rvs((self.numOutputLayerNodes, self.numHiddenLayerNodes + biasNode))

    def trainNetwork(self, inputVector, targetVector):

        #set value of bias node
        if self.bias:
            biasNode = 1
        else:
            biasNode = 0

        #update input vector with bias
        if self.bias:
            inputVector = np.concatenate((inputVector, [self.bias]), axis=None)

        inputVector = transpose(inputVector, 2)
        targetVector = transpose(targetVector, 2)

        hiddenLayerOutputVector = sigActFunction(np.dot(self.ITHweights, inputVector))

        if self.bias:
            hiddenLayerOutputVector = np.concatenate((hiddenLayerOutputVector, [self.bias]), axis=None)

        NNoutputVector = sigActFunction(np.dot(self.HTOweights, hiddenLayerOutputVector))

        #claculate how far away the actual output is from the target
        errorFromOutput = targetVector - NNoutputVector

        #updates the values of the output weights - self.HTOweights += outputUpdate
        outputUpdate = errorFromOutput * NNoutputVector * (1.0 - NNoutputVector)

        updateToOutputVector = []
        for outVect in hiddenLayerOutputVector:
            tempOutVect = 0
            tempOutVect += outVect * outputUpdate
            updateToOutputVector.append(outVect * tempOutVect)

        for output in outputUpdate:
            self.HTOweights += output

        #determines the errors from the hidden layer
        errorFromHidden = []
        for weight in self.HTOweights:
            for error in errorFromOutput:
                tempWeight = 0
                tempWeight =+ weight * error
            errorFromHidden.append(weight * tempWeight)

        #updates the values of the hidden weights
        hiddenUpdate = errorFromHidden * hiddenLayerOutputVector * (1.0 - hiddenLayerOutputVector)
        updateToInputVector = []
        for inVect in inputVector:
            tempInVect = 0
            for update in hiddenUpdate:
                tempUpdate = 0
                for updateVal in update:
                    tempUpdate += updateVal
                tempInVect =+ inVect * tempUpdate
            updateToInputVector.append(inVect * tempInVect)

        updateValues = self.learningRate * transpose(updateToInputVector, 2)
        self.ITHweights += updateValues

    #initializes the neural network with a given input vector
    def runNetwork(self, inputVector):

        #update input vector with bias of value 1
        if self.bias:
            inputVector = np.concatenate((inputVector, [1]), axis=None)

        #change input vector for NN into column vector
        NNinputVector = transpose(inputVector, 2)

        #output from hidden layer
        hiddenLayerOutputVector = sigActFunction(np.dot(self.ITHweights, NNinputVector))

        #update output vector with bias of value 1
        if self.bias:
            hiddenLayerOutputVector = np.concatenate((hiddenLayerOutputVector, [[1]]), axis=None)

        #output from output layer
        NNoutputVector = sigActFunction(np.dot(self.HTOweights, hiddenLayerOutputVector))

        return NNoutputVector

    #test the neural network to see how accurate it is
    def testNetwork(self, testVector, testLabel):

        #convert labels into column vectors
        NNtestLabels = transpose(testLabel, 2)

        classificationScore = self.runNetwork(testVector)
        classificationScore = classificationScore
        return classificationScore

#randomize and split the dataset
np.random.shuffle(dataSet)
trainingSetSplit = int(len(dataSet) * 0.66)
trainingDataSet = dataSet[:trainingSetSplit]
testingDataSet = dataSet[trainingSetSplit:]

#create training arrays for scores and labels
dataPoints = []
dataLabels = []
for i in range(trainingSetSplit):
    dataPoints.append(trainingDataSet[i][0])
    dataLabels.append(trainingDataSet[i][1])

#create testing arrays for scores and labels
testDataPoints = []
testDataLabels = []
for i in range(len(testingDataSet)):
    testDataPoints.append(testingDataSet[i][0])
    testDataLabels.append(testingDataSet[i][1])

#normalize training data
npTrainData = np.array(dataPoints)
normTrainDataSet = preprocessing.normalize([npTrainData])
#standardize training data
stdzTrainDataSet = preprocessing.scale(npTrainData)

#normalize testing data
npTestData = np.array(testDataPoints)
normTestDataSet = preprocessing.normalize([npTestData])
#standardize testing data
stdzTestDataSet = preprocessing.scale(npTestData)

oneValNetwork = neuralNetworkBP(1, 5, 1, 0.6, 1)

numOfCorrectLabels = 0
networkPredictions = []
networkActual = []
testResults = []

for stdzScore, scoreLabel in zip(stdzTrainDataSet, trainingDataSet):
    oneValNetwork.trainNetwork(stdzScore, scoreLabel[1])

for stdzScore, testDataEntry in zip(stdzTestDataSet, testingDataSet):
    testEntry = oneValNetwork.testNetwork(stdzScore, testDataEntry[1])
    if testDataEntry[1] == 0.99:
        testResults.append([testEntry, 1])
    else:
        testResults.append([testEntry, 0])

scoreSum = 0
for i in testResults:
    scoreSum += i[0]
avg = float(scoreSum / len(testResults))

for result in testResults:
    if result[0] > avg:
        if result[1] == 1:
            numOfCorrectLabels += 1
            networkPredictions.append(1)
            networkActual.append(1)
        else:
            networkActual.append(0)
            networkPredictions.append(1)
    else:
        if result[1] == 0:
            numOfCorrectLabels += 1
            networkActual.append(0)
            networkPredictions.append(0)
        else:
            networkActual.append(1)
            networkPredictions.append(0)


accuracyPercentage = round(((float(numOfCorrectLabels)/float(len(testingDataSet))) * 100), 2)

print("\n")
print("Accuracy of Model: %d/%s , %%%r \n" %(numOfCorrectLabels, len(testingDataSet), accuracyPercentage))

#-------------------------------------------------------------------------------

#creates and prints confusion matrix
testAct = pds.Series(networkActual, name="Act")
testPre = pds.Series(networkPredictions, name="Pred")
confMatrix = pds.crosstab(testAct, testPre, rownames=["Actual"], colnames=["Predicted"], margins=True)

print("\n")
print(confMatrix)

#to create normalized confusion matrix
#confMatrixNorm = confMatrix / confMatrix.sum(axis=0)

#-------------------------------------------------------------------------------
