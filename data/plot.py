import matplotlib.pyplot as plt
import csv
import numpy as np

data = np.genfromtxt('Antmaze_medium_expert_s_l.csv', dtype=float, delimiter=',')

x = data[:, 0]
y = data[:, 1]

plt.scatter(x, y, color='r')
