import numpy as np
import matplotlib.pylab as plt
import padasip as pa

def signal_1d(a, w, ph, time_counts): # number of oscillations per second
    return a * np.sin(ph + 2 * np.pi * w * time_counts)

# creation of data

right_bound = 10
number_of_point = 10000
# number_of_point = first_input_1d.size
t = np.linspace(0, right_bound, number_of_point, endpoint=False)
initial_weights_1d = np.array([0.1, 0.3, 0.5])

x = pa.input_from_history(signal_1d(0.5, 10, 0, t), 3)

noise = initial_weights_1d[0] * x[:,0] + initial_weights_1d[1] * x[:,1] + initial_weights_1d[2] * x[:,2]

d = noise + signal_1d(1, 1, 0, t)[:-2]

# identification
f = pa.filters.FilterNLMS(n=3, mu=0.9, w="zeros")
y, e, w = f.run(d, x)

print(w)

# show results
plt.figure(figsize=(15, 9))

plt.subplot(311);
plt.plot(noise, "b", label="noise - target")
plt.subplot(312);
plt.plot(d, "g", label="d - output");
plt.subplot(313);
plt.plot(y, "r", label="y - output");

plt.show()