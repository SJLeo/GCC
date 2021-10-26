import numpy as np
import matplotlib.pyplot as plt
# plt.style.use('seaborn-white')

data = np.random.normal(0, 0.8, 1000)
plt.hist(data, 50, density=True, facecolor='g', alpha=0.75)
plt.show()

