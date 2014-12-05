import math, os, random, sklearn
import numpy as np
import pandas as pd
import pywt

class FeatureGen:

	def __init__(self):
		self.gaitData = {}
		self.demographics = []
		self.sensorPositions = {}
		self.schema = [
			'Time', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'R1', 'R2', 'R3',
         	'R4', 'R5', 'R6', 'R7', 'R8', 'Total Force Left', 'Total Force Right'
     	]

	def loadGaitData(self):
		for filename in os.listdir('data'):
			#if filename == 'GaCo01_01.txt':
			key, walkNumber = filename.split('_')
			print filename

			# Process data only for non-serial 7 subjects
			if '10' not in walkNumber:
				filename = os.path.join('data', filename)
				df = pd.read_csv(filename, sep='\t', names=self.schema)
				if key not in self.gaitData:
					self.gaitData[key] = [df]
				else:
					self.gaitData[key].append(df)

	def loadDemographics(self):
		infile = open('demographics.txt', 'r')
		demSchema = infile.readline().split('\t')
		df = pd.read_csv('demographics.txt', skiprows=[0], sep = '\t', names=demSchema)
		df = df[df.columns.values[0:20]]  # Delete bs data due to extra tab delimited spaces
		df = df[:166]                     # Delete nan columns
		self.demographics = df

	def loadSensorPositions(self):
		lines = [line.strip('\n').split() for line in open('sensorPositions.txt').readlines()]
		self.sensorPositions['x'] = []
		self.sensorPositions['y'] = []
		for key, x, y in lines:
			if 'L' in key:
				self.sensorPositions['x'].append(int(x))
				self.sensorPositions['y'].append(int(y))

	def normalizeSignals(self):
		for subjectId in self.gaitData:
			weight = float(self.demographics.loc[self.demographics['ID'] == subjectId]['Weight'])
			if not math.isnan(weight):
				for i in range(len(self.gaitData[subjectId])):
					self.gaitData[subjectId][i] = self.gaitData[subjectId][i]/weight

	def getOneMeanFeatures(self, matrix):
		return [1] + [matrix[self.schema[1:]].mean().mean()]

	def getSensorMeanFeatures(self, matrix):
		return [1] + matrix[self.schema[1:]].mean().values.tolist()

	def segmentGaitOld(self, matrix):
		epsilon = 100
		shiftedSignal = np.subtract(matrix['Total Force Left'], epsilon)
		leftStances = np.where(np.diff(np.sign(shiftedSignal)) > 0)
		leftSwings = np.where(np.diff(np.sign(shiftedSignal)) < 0)

		shiftedSignal = np.subtract(matrix['Total Force Right'], epsilon)
		rightStances = np.where(np.diff(np.sign(shiftedSignal)) > 0)
		rightSwings = np.where(np.diff(np.sign(shiftedSignal)) < 0)
		
		return leftStances, leftSwings, rightStances, rightSwings

	# Helper function to segment a signal into swing and stride phases
	def segmentGaitNew(self, matrix):
		epsilon = 100
		shiftedSignal = np.subtract(matrix['Total Force Left'], epsilon)
		leftStances = np.where(np.diff(np.sign(shiftedSignal)) > 0)[0]
		leftStances = [('Stance', time) for time in leftStances]
		leftSwings = np.where(np.diff(np.sign(shiftedSignal)) < 0)[0]
		leftSwings = [('Swing', time) for time in leftSwings]
		leftPhases = sorted(leftStances + leftSwings, key=lambda x:x[1])


		shiftedSignal = np.subtract(matrix['Total Force Right'], epsilon)
		rightStances = np.where(np.diff(np.sign(shiftedSignal)) > 0)[0]
		rightStances = [('Stance', time) for time in rightStances]
		rightSwings = np.where(np.diff(np.sign(shiftedSignal)) < 0)[0]
		rightSwings = [('Swing', time) for time in rightSwings]
		rightPhases = sorted(rightStances + rightSwings, key=lambda x:x[1])

		return leftPhases, rightPhases

	# Calculates mean and variance of phase times based on balance
	def getBalancePhaseTimes(self, matrix):
		diff = matrix['Total Force Left'] - matrix['Total Force Right']
		zeroCrossings = np.where(np.diff(np.sign(diff)))[0]
		phaseTimes = np.diff(zeroCrossings)
		mean = np.mean(phaseTimes)
		variance = np.var(phaseTimes)
		return [mean, variance]
		
	# Calculates mean and variance of phase times based on when signal crosses epsilon
	def getPhaseTimes(self, matrix):
		leftStances, leftSwings, rightStances, rightSwings = self.segmentGaitOld(matrix)
		leftTimes = np.diff(sorted(np.concatenate((leftStances + leftSwings), axis=1)))
		rightTimes = np.diff(sorted(np.concatenate((rightStances + rightSwings), axis=1)))

		return [np.mean(leftTimes), np.var(leftTimes), np.mean(rightTimes), np.var(rightTimes)]

	# Using actual definition of swing and stance time, for both right and left
	def getStrideFeatures(self, matrix):
		stanceTimes, swingTimes = [], []
		leftPhases, rightPhases = self.segmentGaitNew(matrix)
		leftDiffs = np.diff([time for phase, time in leftPhases])
		leftStanceTimes = leftDiffs[::2] if leftPhases[0][0] == 'Stance' else leftDiffs[1::2]
		leftSwingTimes = leftDiffs[1::2] if leftPhases[0][0] == 'Stance' else leftDiffs[::2]
		stanceTimes = np.concatenate((stanceTimes, leftStanceTimes))
		swingTimes = np.concatenate((swingTimes, leftSwingTimes))
	
		rightDiffs = np.diff([time for phase, time in rightPhases])
		rightStanceTimes = rightDiffs[::2] if rightPhases[0][0] == 'Stance' else rightDiffs[1::2]
		rightSwingTimes = rightDiffs[1::2] if rightPhases[0][0] == 'Stance' else rightDiffs[::2]
		stanceTimes = np.concatenate((stanceTimes, rightStanceTimes))
		swingTimes = np.concatenate((swingTimes, rightSwingTimes))
		
		return [np.mean(stanceTimes), np.var(stanceTimes), np.mean(swingTimes), np.var(swingTimes)]

	def getWaveletApproxCoefficients(self, matrix, cropN):  # Currently just for L1
		l1_sensor = matrix[self.schema[1]]
		cA, cD = pywt.dwt(l1_sensor, 'coif1')
		return list(cA[:cropN])

	def calculateCops(self, matrix):
		leftCopsX = np.divide(matrix[self.schema[1:9]].dot(self.sensorPositions['x']), matrix['Total Force Left'])
		leftCopsY = np.divide(matrix[self.schema[1:9]].dot(self.sensorPositions['y']), matrix['Total Force Left'])
		rightCopsX = np.divide(matrix[self.schema[1:9]].dot(self.sensorPositions['x']), matrix['Total Force Left'])
		rightCopsY = np.divide(matrix[self.schema[1:9]].dot(self.sensorPositions['y']), matrix['Total Force Left'])

		leftCopsX[np.logical_not(np.isfinite(leftCopsX))] = 0. 
		leftCopsY[np.logical_not(np.isfinite(leftCopsY))] = 0.
		rightCopsX[np.logical_not(np.isfinite(rightCopsX))] = 0. 
		rightCopsY[np.logical_not(np.isfinite(rightCopsY))] = 0.

		return leftCopsX, leftCopsY, rightCopsX, rightCopsY

	# Calculates center of pressure without using segmentation
	def getCopFeaturesOld(self, matrix):
		leftCopsX, leftCopsY, rightCopsX, rightCopsY = self.calculateCops(matrix)
		return [np.mean(leftCopsX), np.var(leftCopsX), np.mean(leftCopsY), np.var(leftCopsY)]

	# Calculates the heel strike
	def getCopFeaturesNew(self, matrix):
		leftPhases, rightPhases = self.segmentGaitNew(matrix)
		leftCopsX, leftCopsY, rightCopsX, rightCopsY = self.calculateCops(matrix)

		leftHeelStrikeXs = [leftCopsX[time] for phase, time in leftPhases if phase == 'Stance']
		leftHeelStrikeYs = [leftCopsY[time] for phase, time in leftPhases if phase == 'Stance']
		rightHeelStrikeXs = [rightCopsX[time] for phase, time in rightPhases if phase == 'Stance']
		rightHeelStrikeYs = [rightCopsY[time] for phase, time in rightPhases if phase == 'Stance']
		features = [np.mean(leftHeelStrikeXs), np.mean(leftHeelStrikeYs), 
					np.mean(rightHeelStrikeXs), np.mean(rightHeelStrikeYs),
					np.var(leftHeelStrikeXs), np.var(leftHeelStrikeYs), 
					np.var(rightHeelStrikeXs), np.var(rightHeelStrikeYs)]
		return features

	def getFeatures(self, matrix):
		strideFeatures = self.getStrideFeatures(matrix)
		sensorMeanFeatures = self.getSensorMeanFeatures(matrix)
		oldCopFeatures = self.getCopFeaturesOld(matrix)
		newCopFeatures = self.getCopFeaturesNew(matrix)
		return strideFeatures + sensorMeanFeatures + oldCopFeatures + newCopFeatures

 	def getLabel(self, subjectId):
		group = int((self.demographics.loc[self.demographics['ID'] == subjectId]['Group']))
		return int(group == 1)

	def getXY(self):
		self.loadGaitData()
		self.loadDemographics()
		self.loadSensorPositions()
		#self.normalizeSignals()

		X, Y = [], []
		for subjectId in sorted(self.gaitData.keys()):
			for matrix in self.gaitData[subjectId]:
				X.append(self.getFeatures(matrix))
				Y.append(self.getLabel(subjectId))

		# Randomize X and Y to improve accuracy estimations
		XY = zip(X, Y)
		random.shuffle(XY)
		X = [x for x, y in XY]
		Y = [y for x, y in XY]
		return X, Y
