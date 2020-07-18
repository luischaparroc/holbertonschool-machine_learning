#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

x_bins = np.arange(0, 110, 10)
plt.hist(student_grades, bins=x_bins, edgecolor='black')
plt.xlabel('Grades')
plt.ylabel('Number of Students')
plt.title('Project A')
plt.xticks(x_bins)
plt.xlim(0, 100)
plt.ylim(0, 30)
plt.show()
