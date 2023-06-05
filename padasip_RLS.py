import numpy as np
import matplotlib.pylab as plt
import padasip as pa
import processing_signals as ps

def signal_1d(a, w, ph, time_counts): # number of oscillations per second
    return a * np.sin(ph + 2 * np.pi * w * time_counts)

# creation of data

right_bound = 10
number_of_point = 10000
# number_of_point = first_input_1d.size
t = np.linspace(0, right_bound, number_of_point, endpoint=False)
initial_weights_1d = np.array([0.1, 0.3, 0.5])

# x = pa.input_from_history(signal_1d(0.5, 10, 0, t), 3)
x = pa.input_from_history(signal_1d(0.5, 10, 0, t), 2)
first_input_1d = signal_1d(0.5, 10, 0, t)


# noise = initial_weights_1d[0] * x[:,0] + initial_weights_1d[1] * x[:,1] + initial_weights_1d[2] * x[:,2]
ps_noise = initial_weights_1d[0] * x[:,0] + initial_weights_1d[1] * x[:,1]

second_input_1d = ps_noise + signal_1d(1, 1, 0, t)[:-1]

# identification
# f = pa.filters.FilterRLS(n=3, mu=0.9, w="zeros")
f = pa.filters.FilterRLS(n=2, mu=0.999, w="zeros")
y, e, w = f.run(second_input_1d, x)

print(w)
approx_weights_1d = w[-1]
approx_signal_1d = ps.processing_of_signal(first_input_1d, approx_weights_1d.flatten())

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

print(f"MSE = {ps.MSE(ps_noise, approx_signal_1d[:-1])}")
plt.show()