__author__ = 'Marco'

import os
import numpy as np
import pandas as pd

schema = ['Time', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8',
          'Total Force Left', 'Total Force Right']

data = []
keys = []
X = []  # Will store design matrix. Note that for now, it takes the mean over the time for each sensor, to reduce feature space
X_f = []
for filename in os.listdir(os.getcwd() + '\data'):
    df = pd.read_csv("data\\" + filename, sep='\t', names=schema)
    m = np.matrix(df[schema[1:19]])
    X.append(df[schema[1:19]].mean())
    X_f.append(np.array(m).flatten().tolist())
    keys.append([(filename.split('_'))[0], 'auxData'])  # Save filename and auxData for future use, preprocessing
    data.append(df)

infile = open('demographics.txt', 'r')
dem_schema = infile.readline().split('\t')

df = pd.read_csv('demographics.txt', skiprows=[0], sep = '\t', names=dem_schema)
# TODO think of less hacky way to do this
df = df[df.columns.values[0:20]]  #Delete bs data due to extra tab delimited spaces
df = df[:166]                   #Delete nan columns

#Build the target vector for the training set, order implicit in keys[]
#Use the convention that Parkinson's maps to 1, control group maps to 0
y = []
for k in keys:
    identification = k[0]
    group = int((df.loc[df['ID'] == identification]['Group']))
    if group == 1:
        y.append(1)
    else:
        y.append(0)


#At this point, X is the correct design matrix and y the corresponding target vector
#We add the feature x_0 = 1 to allow for intercept term in theta
X = np.matrix(X)
X_f = np.matrix(X_f)
intercept = np.matrix(np.ones(len(X)))
X = np.concatenate((intercept.T, X), axis=1)
X_f = np.concatenate((intercept.T, X_f), axis=1)
y = np.matrix(y)


#Simple Unweighted Linear Regression
from numpy.linalg import inv
theta = inv(X.T * X)*X.T*y.T
theta_f = inv(X_f.T * X_f)*X_f.T*y.T

output = []
nExamples = (y.shape)[1]
for pt in X:
    h = theta.T * pt.T       #hypothesis for training example pt
    if h>0:
        output.append(1);
    else:
        output.append(0);

err = (output != y)
print np.sum(err)/float(nExamples)

output_f = []
nExamples = (y.shape)[1]
for pt in X_f:
    h = theta.T * pt.T       #hypothesis for training example pt
    if h>0:
        output_f.append(1);
    else:
        output_f.append(0);

err = (output_f != y)
print np.sum(err)/float(nExamples)