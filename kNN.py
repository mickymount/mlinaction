from os import listdir
import numpy as np
import operator

def dataSet():
	group = np.array([[1., 1.1], [1.0,1.0],[0,0],[0,0.1]])
	labels = np.array(['A', 'A', 'B', 'B'])
	return group, labels

def classify0(X, dataS, labels, k):
	dataSize = dataS.shape[0]
	diffMat = np.tile(X, (dataSize, 1)) - dataS
	sqDiff = diffMat**2
	sqDistance = sqDiff.sum(axis=1)
	dist = sqDistance**0.5
	distSort = dist.argsort()
	classCount={}
	for x in range(k):
		voteLabels = labels[distSort[x]]
		classCount[voteLabels] = classCount.get(voteLabels, 0) + 1
	sortClassCount = sorted(classCount.iteritems(),
		key= operator.itemgetter(1), reverse=True)
	return sortClassCount[0][0]

def fileToMatrix(filename):
	fileread = open(filename)
	countLines = len(fileread.readlines())
	returnMat = np.zeros((countLines, 3))
	classLabels = []
	fileread = open(filename)

	index = 0

	for line in fileread.readlines():
		line = line.strip() 
		lFromLine = line.split('\t')
		returnMat[index, :] = lFromLine[0:3]
		labels = {'didntLike':1, 'smallDoses':2, 'largeDoses':3}
		classLabels.append(labels[lFromLine[-1]])
		index += 1
	return returnMat, classLabels
def normalize(dataSet):
	minVal = dataSet.min(0)
	maxVal = dataSet.max(0)
	ranges = maxVal - minVal
	normDataSet = np.zeros(np.shape(dataSet))
	m = dataSet.shape[0]
	normDataSet = dataSet - np.tile(minVal, (m,1))
	normDataSet = normDataSet / np.tile(ranges, (m, 1))
	return normDataSet, ranges, minVal

def datingClassTest():

	hoRatio = 0.10
	datingDataMat, datingLabels = fileToMatrix('datingTestSet.txt')
	normMat, ranges, minVals = normalize(datingDataMat)
	m = normMat.shape[0]
	numTestVecs = int(m*hoRatio)
	errorCount = 0.0

	for i in range(numTestVecs):
		classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m,:], datingLabels[numTestVecs:m],3)
		print('the class came back with: %d, the real answer is %d' % (classifierResult, datingLabels[i]))

		if(classifierResult != datingLabels[i]): errorCount += 1.0
	print('the total error rate is: %f' % (errorCount/float(numTestVecs)))

def classifyPerson():

	resultList = ['not at all', 'in small doses', 'in large doses']
	persentTats = float(raw_input('percentage of time spent playing games? '))
	ffMiles = float(raw_input('frquent flier miles earned /year? '))
	iceCream = float(raw_input('liters of ice cream consumed /year? '))
	datingDataMat, datingLabels = fileToMatrix('datingTestSet.txt')
	normMat, ranges, minVals = normalize(datingDataMat)
	inArr = np.array([ffMiles, persentTats, iceCream])
	classifierResult = classify0((inArr-minVals) / ranges, normMat, datingLabels, 3)
	print('You\'ll probably like this person: ', resultList[classifierResult - 1])

def imgToVec(filename):
	returnVect = np.zeros((1,1024))
	fileread = open(filename)
	for i in range(32):
		lineStr = fileread.readline()
		for j in range(32):
			returnVect[0, 32*i+j] = int(lineStr[j])
	return returnVect
def handwritingClassify0():
	hwLabels = []
	trainingFileList = listdir('trainingDigits')
	m = len(trainingFileList)
	trainingMat = np.zeros((m, 1024))
	for i in range(m):
		fileNameStr = trainingFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		hwLabels.append(classNumStr)
		trainingMat[i, :] = imgToVec('trainingDigits/%s' % fileNameStr)
		testFileList = listdir('testDigits')
		errorCount = 0.0
		mTest = len(testFileList)
		for i in range(mTest):
			fileNameStr = testFileList[i]
			fileStr = fileNameStr.split('.')[0]
			classNumStr = int(fileStr.split('_')[0])
			vectorUnderTest = imgToVec('testDigits/%s' % fileNameStr)
			classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
			print('classifier with: %d, real answer: %d' % (classifierResult, classNumStr))
			if(classifierResult != classNumStr): errorCount += 1.0
		print('total number of errors: %d' % errorCount)
		print('error rate: %f' % (errorCount/float(mTest)))