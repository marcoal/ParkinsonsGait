import math, os, random, sklearn
import numpy as np
import pandas as pd
import pywt

class FeatureGen:

	def __init__(self):
		self.gaitData = {}
		self.demographics = []
		self.schema = [
			'Time', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'R1', 'R2', 'R3',
         	'R4', 'R5', 'R6', 'R7', 'R8', 'Total Force Left', 'Total Force Right'
     	]

	def loadGaitData(self):
		for filename in os.listdir('data'):
			key, walkNumber = filename.split('_')
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

	def getStrideFeatures(self, matrix):
		diff = matrix['Total Force Left'] - matrix['Total Force Right']
		zeroCrossings = np.where(np.diff(np.sign(diff)))[0]
		strideLengths = np.diff(zeroCrossings)
		mean = np.mean(strideLengths)
		variance = np.var(strideLengths)
		return [mean, variance]

	def getWaveletApproxCoefficients(self, matrix, cropN):  # Currently just for L1
		l1_sensor = matrix[self.schema[1]]
		cA, cD = pywt.dwt(l1_sensor, 'coif1')
		return list(cA[:cropN])

	def getFeatures(self, matrix):
		strideFeatures = self.getStrideFeatures(matrix)
		sensorMeanFeatures = self.getSensorMeanFeatures(matrix)
		return sensorMeanFeatures
		return strideFeatures + sensorMeanFeatures

 	def getLabel(self, subjectId):
		group = int((self.demographics.loc[self.demographics['ID'] == subjectId]['Group']))
		return int(group == 1)

	def getXY(self):
		self.loadGaitData()
		self.loadDemographics()
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
