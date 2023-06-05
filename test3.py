import numpy as np
import matplotlib.pyplot as plt
import processing_signals as ps
from scipy import fftpack
import padasip as pa

def signal_1d(a, w, ph, time_counts): # number of oscillations per second
    return a * np.sin(ph + 2 * np.pi * w * time_counts)

right_bound = 10
number_of_point = 10000
# number_of_point = first_input_1d.size
t = np.linspace(0, right_bound, number_of_point, endpoint=False)

initial_weights_1d = np.array([0.1, 0.3])
# initial_weights_1d = np.array([0.1, 0.3, 0.5])
# initial_weights_1d = np.array([0.2, 0.7, 0.9, 0.3])

# first_input_1d = 1 / (1 + t) * signal_1d(0.5, 10, 0, t)
first_input_1d = signal_1d(0.5, 10, 0, t)

ps_noise = ps.processing_of_signal(first_input_1d, initial_weights_1d)
second_input_1d = ps_noise + signal_1d(1, 1, 0, t)

# approx_weights_1d = ps.RLS(first_input_1d, second_input_1d, 3, 0.9)
approx_weights_1d = ps.RLS(first_input_1d, second_input_1d, 2, 0.999)
print(approx_weights_1d)

approx_signal_1d = ps.processing_of_signal(first_input_1d, approx_weights_1d.flatten())

############################
# show results
plt.figure(figsize=(15, 9))

plt.subplot(411);
plt.title("first_input_1d")
plt.plot(first_input_1d, "b", label="noise - target")
plt.subplot(412);
plt.title("second_input_1d")
plt.plot(second_input_1d, "g", label="d - output");
plt.subplot(413);
plt.title("ps_noise")
plt.plot(ps_noise, "r", label="y - output");
plt.subplot(414);
plt.title("approx_signal_1d")
plt.plot(approx_signal_1d, "r", label="y - output");

print(f"MSE = {ps.MSE(ps_noise, approx_signal_1d)}")
plt.show()


############################

plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.plot(t, first_input_1d)
plt.title("first_input_1d")
# plt.show()

freq_bound = 2100

x = fftpack.rfftfreq(first_input_1d.size, right_bound / number_of_point)
y = 2 * np.abs(fftpack.rfft(first_input_1d)) / first_input_1d.size

plt.subplot(1, 2, 2)
plt.plot(x[:freq_bound], y[:freq_bound], '.')
plt.title("AFH of first_input")
# plt.show()




plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.plot(t, second_input_1d)
plt.title("second_input_1d")
# plt.show()

freq_bound = 2100

x = fftpack.rfftfreq(second_input_1d.size, right_bound / number_of_point)
y = 2 * np.abs(fftpack.rfft(second_input_1d)) / second_input_1d.size

plt.subplot(1, 2, 2)
plt.plot(x[:freq_bound], y[:freq_bound], '.')
plt.title("AFH of second_input_1d")
# plt.show()

############################