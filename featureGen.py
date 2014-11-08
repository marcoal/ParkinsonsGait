import numpy as np
import os
import pandas as pd
import sklearn

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
			key = filename.split('_')[0]
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

	def getFeatures(self, matrix):
		return [1] + [matrix[self.schema[1:]].mean().mean()]

	def getLabel(self, subjectId):
		group = int((self.demographics.loc[self.demographics['ID'] == subjectId]['Group']))
		return int(group == 1)

	def getXY(self):
		self.loadGaitData()
		self.loadDemographics()

		X, Y = [], []
		for subjectId in sorted(self.gaitData.keys()):
			for matrix in self.gaitData[subjectId]:
				X.append(self.getFeatures(matrix))
				Y.append(self.getLabel(subjectId))
		return X, Y
