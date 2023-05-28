import numpy as np
from matplotlib import pyplot as plt
import processing_signals as ps

# print(matplotlib.get_backend())

from mpl_toolkits.mplot3d import Axes3D
from scipy import fftpack

first_input_2d = np.genfromtxt("first_input_2d.txt", names=['1st', '2nd', '3rd'], delimiter=',')
second_input_2d = np.genfromtxt("second_input_2d.txt", names=['1st', '2nd', '3rd'], delimiter=',')
impulse_response_2d = np.genfromtxt("impulse_response.txt", names=['1st', '2nd', '3rd'], delimiter=',')

# print(second_input_2d)
# print(type(data[0][0]))
# print(len(list(zip(*data))[2]))

first_input_2d = list(zip(*first_input_2d))
second_input_2d = list(zip(*second_input_2d))
impulse_response_2d = list(zip(*impulse_response_2d))


# plt.plot(impulse_response_2d[0])

rfft_impulse_response = 2 * np.abs(fftpack.rfft(impulse_response_2d[0])) / len(impulse_response_2d[0])
rfft_first_input = 2 * np.abs(fftpack.rfft(first_input_2d[0])) / len(first_input_2d[0])

# plt.plot(rfft_impulse_response)
# plt.plot(rfft_first_input)

# second_input_2d_by_response = fftpack.irfft(rfft_impulse_response * rfft_first_input)

# plt.plot(impulse_response_2d[1])
# plt.plot(impulse_response_2d[2])
plt.show()

plt.plot(first_input_2d[0])
plt.plot(second_input_2d[0])
plt.show()
#
# plt.plot(first_input_2d[1])
# plt.plot(second_input_2d[1])
# plt.show()
#
# plt.plot(first_input_2d[2])
# plt.plot(second_input_2d[2])
# plt.show()