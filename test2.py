import numpy as np
import matplotlib.pyplot as plt
import processing_signals as ps

def l2(firstSeries, secondSeries):
    return np.sqrt(((firstSeries - secondSeries) ** 2).sum())

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

print(f"l2 = {l2(model_agc, gather[0]):.1e}")
print(f"l2 = {l2(model_agc, model[0]):.1e}")

# weights_2d = ps.fitting_of_weights(model_agc[0], model[0], 2, 0.9)
weights_2d = ps.fitting_of_weights(model_agc[0], gather[0], 1, 0.9)
print(weights_2d, weights_2d.shape)

approx_model = ps.processing_of_signal(model_agc[0], weights_2d)

print(f"l2 = {l2(approx_model, gather[0]):.1e}")
print(f"l2 = {l2(approx_model, model[0]):.1e}")


# plt.plot(model_agc[0] * 1e+4)
# plt.plot(model_agc[0] * 23557)

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.plot(gather[0])
plt.plot(approx_model)

plt.subplot(1, 2, 2)
plt.plot(model[0])
plt.plot(approx_model)


plt.show()