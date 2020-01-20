# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np 
import sys
import json
import random

# Get file names
#in_file = sys.argv[1]
#json_file = sys.argv[2]

in_file = 'data/2.in'
json_file = 'data/2.json'
file_name = in_file[in_file.index('/')+1:in_file.index('.')]

# Load data
in_data = np.array(np.loadtxt(in_file))

inputs = in_data[:,:-1]
x1 = in_data[:,0]
x2 = in_data[:,0]
y = in_data[:,-1]
pad = np.ones(len(x1))

# Analytic solution
x = np.insert(inputs,[0],1,axis=1)
m = np.linalg.lstsq(x,y)[0]

# Write analytic solution
out_data = open(file_name+'.out','w')
for i in m:
    out_data.write("{0:.4f}\n".format(i))
out_data.write("\n")

# Open and read json file
with open(json_file) as json_file:
    json_data = json.load(json_file)
alpha = json_data['learning rate']
iterations= json_data['num iter']

w = np.ones(len(inputs[0,:])+1)
# h(x) has the form h(x)=w1+w2*x1+w3*x2
def h(x,w):
    return np.dot(w,x)

for i in range(iterations):
    loc = random.randint(0,len(x1)-1)
    temp = x[loc,:]
    y_val = y[loc]
    w += alpha*(y_val-h(temp,w))*temp[:]
    
# Write GD solution
for i in w:
    out_data.write("{0:.4f}\n".format(i))
out_data.close()