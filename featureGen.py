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

	# Helper function to segment a signal into swing and stride phases
	def segmentGait(self, matrix):
		epsilon = 100
		shiftedSignal = np.subtract(matrix['Total Force Left'], epsilon)
		leftStances = np.where(np.diff(np.sign(shiftedSignal)) > 0)
		leftSwings = np.where(np.diff(np.sign(shiftedSignal)) < 0)

		shiftedSignal = np.subtract(matrix['Total Force Right'], epsilon)
		rightStances = np.where(np.diff(np.sign(shiftedSignal)) > 0)
		rightSwings = np.where(np.diff(np.sign(shiftedSignal)) < 0)
		
		return leftStances, leftSwings, rightStances, rightSwings

	# def getSwingFeaturesOld(self, matrix):
	# 	diff = matrix['Total Force Left'] - matrix['Total Force Right']
	# 	zeroCrossings = np.where(np.diff(np.sign(diff)))[0]
	# 	swingTimes = np.diff(zeroCrossings)
	# 	mean = np.mean(swingTimes)
	# 	variance = np.var(swingTimes)
	# 	return [mean, variance]

	# Averaging stance and swing individually gives 0.765 AUC
	# def getStrideFeatures(self, matrix):
		# stanceTimes, swingTimes = [], []
		# leftPhases, rightPhases = self.segmentGait(matrix)
		# leftDiffs = np.diff([time for phase, time in leftPhases])[0]
		# stanceTimes = np.concatenate((stanceTimes, leftDiffs[::2]))
		# swingTimes = np.concatenate((swingTimes, leftDiffs[1::2]))
		
		# rightDiffs = np.diff([time for phase, time in rightPhases])[0]
		# stanceTimes = np.concatenate((stanceTimes, rightDiffs[::2]), axis=1)
		# swingTimes = np.concatenate((swingTimes, rightDiffs[1::2]), axis=1)
		
		# return [np.mean(stanceTimes), np.var(stanceTimes), np.mean(swingTimes), np.var(swingTimes)]

	# Using actual definition of swing and stance time, for both right and left
	def getSwingFeaturesNew(self, matrix):
		leftStances, leftSwings, rightStances, rightSwings = self.segmentGait(matrix)
		leftTimes = np.diff(sorted(np.concatenate((leftStances + leftSwings), axis=1)))
		rightTimes = np.diff(sorted(np.concatenate((rightStances + rightSwings), axis=1)))
		
		return [np.mean(leftTimes), np.var(leftTimes), np.mean(rightTimes), np.var(rightTimes)]

	def getWaveletApproxCoefficients(self, matrix, cropN):  # Currently just for L1
		l1_sensor = matrix[self.schema[1]]
		cA, cD = pywt.dwt(l1_sensor, 'coif1')
		return list(cA[:cropN])

	def getCopFeaturesOld(self, matrix):
		xCops = []
		yCops = []
		leftXCops = np.divide(matrix[self.schema[1:9]].dot(self.sensorPositions['x']), matrix['Total Force Left'])
		leftYCops = np.divide(matrix[self.schema[1:9]].dot(self.sensorPositions['y']), matrix['Total Force Left'])
		leftXCops = leftXCops[np.isfinite(leftXCops)]
		leftYCops = leftYCops[np.isfinite(leftYCops)]
		return [np.mean(leftXCops), np.var(leftXCops), np.mean(leftYCops), np.var(leftYCops)]

	def getCopFeaturesNew(self, matrix):
		self.segmentGait(matrix)

	def getFeatures(self, matrix):
		swingFeatures = self.getSwingFeaturesNew(matrix)
		sensorMeanFeatures = self.getSensorMeanFeatures(matrix)
		copFeatures = self.getCopFeaturesOld(matrix)
		return swingFeatures + sensorMeanFeatures + copFeatures

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
