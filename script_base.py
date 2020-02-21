import matplotlib.pyplot as plt
plt.rcdefaults()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

objects = ['KNN (best)', 'AlexNet', 'HOG+SVM', 'CNN1 (best)', 'CNN2', 'ResNet50']
y_pos = np.arange(len(objects))

performance1 = [0, 0.7139, 0.6761, 0.7718, 0.7628, 0.7458]
performance2 = [0.5634, 0.6845, 0.6276, 0.7367, 0.6593, 0.6528]
rt = [381.95, 580.93, 20.53, 134.23, 199.64, 977.18]


for i in range(6):
    print(objects[i])
    print(np.square(performance2[i] - performance1[i])/ rt[i])
    print("")

"""
df = pd.DataFrame({'Linear': performance1, 'CFV': performance2}, index=objects)
ax = df.plot.bar(rot=0)

plt.show()

"""