# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 10:00:31 2019

@author: NodaroHo
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.patches as mpatches

df = pd.read_csv("C:\Projets\Machine Learning\datasets\FuelConsumptionCo2.csv")

# take a look at the dataset
df.head()

train_x = np.asanyarray(df[['ENGINESIZE']])
train_y = np.asanyarray(df[['CO2EMISSIONS']])

# initialize the variables a and b with random values
a = tf.Variable(20.0)
b = tf.Variable(30.2)
y = a * train_x + b

# compute loss 
loss = tf.reduce_mean(tf.square(y - train_y))
#define optimizer
optimizer = tf.train.GradientDescentOptimizer(0.05)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

loss_values = []
train_data = []
for step in range(500):
    _, loss_val, a_val, b_val = sess.run([train, loss, a, b])
    loss_values.append(loss_val)
    #Print values of step, loss_val, a_val, b_val every 5 steps
    if step % 5 == 0:
        print(step, loss_val, a_val, b_val)
        train_data.append([a_val, b_val])

#Plot data and the linear regression function
plt.plot(train_x, train_y, 'ro')
plt.plot(train_x, a_val*train_x + b_val)
print("\n")
print("Final value for a:", a_val)
print("Final value for b:", b_val)