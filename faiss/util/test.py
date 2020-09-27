import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-1, 1, 50)
y = np.random.random(50)
print(x.shape, y.shape)
# 第一个是横坐标的值，第二个是纵坐标的值
plt.plot(x, y)
plt.show()