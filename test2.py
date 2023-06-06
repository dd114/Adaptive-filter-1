import numpy as np
import matplotlib.pyplot as plt
import processing_signals as ps


model = np.load("MODEL.NPY")
model_agc = np.load("MODEL_AGC.NPY")
gather = np.load("GATHER.NPY")

print(model.shape, model_agc.shape, gather.shape)


# shift = 1e+9
# for i in range(96):
#     plt.plot(shift * i + model[i], color='black')
#
# plt.show()
#
# shift_agc = 1e+5 / 5
# for i in range(96):
#     plt.plot(shift_agc * i + model_agc[i], color='black')
#
# plt.show()
#
# for i in range(96):
#     plt.plot(shift * i + gather[i], color='black')

plt.show()

print(f"NMSE = {ps.NMSE(model_agc, gather[0]):.1e}")
print(f"NMSE = {ps.NMSE(model_agc, model[0]):.1e}")

weights_2d = ps.fitting_of_weights(model_agc[0], model[0], 2, 0.9)
# weights_2d = ps.fitting_of_weights(model_agc[0], gather[0], 1, 0.9)
print(weights_2d, weights_2d.shape)

approx_model = ps.processing_of_signal(model_agc[0], weights_2d)

print(f"NMSE = {ps.NMSE(approx_model, gather[0]):.1e}")
print(f"NMSE = {ps.NMSE(approx_model, model[0]):.1e}")


# plt.plot(model_agc[0] * 1e+4)
# plt.plot(model_agc[0] * 23557)

plt.figure(figsize=(15, 6))

max_value1 = max(np.abs(gather[0]).max(), np.abs(approx_model).max())
max_value2 = max(np.abs(model[0]).max(), np.abs(approx_model).max())

plt.subplot(1, 2, 1)
plt.plot(gather[0] / max_value1)
plt.plot(approx_model / max_value1)

plt.subplot(1, 2, 2)
plt.plot(model[0] / max_value2)
plt.plot(approx_model / max_value2)


plt.show()