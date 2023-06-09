import numpy as np

def processing_of_signal(input, weights):
    assert len(input.shape) == len(weights.shape), f'The dimensions of the parameters do not match {len(input.shape)} != {len(weights.shape)}'

    # output = np.zeros_like(input)
    output = np.empty(input.shape)
    number_of_weights = weights.shape[0] # bc number of weights to each time layer is located on column (in 2-dim case)

    print(input.shape, weights.shape)

    if len(input.shape) == 1: # signal depends on time
        output = np.convolve(input, weights, mode='same')
    elif len(input.shape) == 2: # signal depends on time and x
        for j in range(input.shape[0]):
            output[j, :] = np.convolve(input[j, :], weights[:, j], mode='same')

    return output

def NLMS_1d(first_input_1d, second_input_1d, number_of_weights_1d, mu_0, epsilon):

    h_k = np.zeros(number_of_weights_1d).reshape(number_of_weights_1d, 1)

    first_input_k = np.zeros(number_of_weights_1d)

    for k in range(1, len(second_input_1d)):
        first_input_k = np.roll(first_input_k, -1)
        first_input_k[-1] = first_input_1d[k]

        first_input_k_vec = first_input_k.reshape(number_of_weights_1d, 1)
        first_input_k_vec_transp = first_input_k_vec.transpose()

        P_k = second_input_1d[k] * first_input_k_vec

        R_k = first_input_k_vec.dot(first_input_k_vec_transp)

        mu_k = mu_0 / ((first_input_k_vec_transp ** 2).sum() + epsilon)
        # mu_k = mu_0 / (3 * np.trace(R_k) + epsilon)
        # mu_k = mu_0 / (3 * np.trace(R_k))

        grad_J = -2 * (P_k - R_k.dot(h_k))
        # print(mu_0 / (3 * np.trace(r)), mu_0 / (3 * np.trace(r) + epsilon), mu_0 / ((first_input_k ** 2).sum() + epsilon), grad_J)
        # print(mu_k, np.trace(r), grad_J)

        h_k -= mu_k * grad_J

    return h_k

def LMS_1d(first_input_1d, second_input_1d, number_of_weights_1d, mu_0, epsilon):

    h_k = np.zeros(number_of_weights_1d).reshape(number_of_weights_1d, 1)

    first_input_k = np.zeros(number_of_weights_1d)

    for k in range(1, len(second_input_1d)):
        first_input_k = np.roll(first_input_k, -1)
        first_input_k[-1] = first_input_1d[k]

        first_input_k_vec = first_input_k.reshape(number_of_weights_1d, 1)
        first_input_k_vec_transp = first_input_k_vec.transpose()

        P_k = second_input_1d[k] * first_input_k_vec

        R_k = first_input_k_vec.dot(first_input_k_vec_transp)

        # mu_k = mu_0 / ((first_input_k_vec_transp ** 2).sum() + epsilon)
        mu_k = mu_0 / (np.trace(R_k) + epsilon)
        # mu_k = mu_0 / (3 * np.trace(R_k) + epsilon)
        # mu_k = mu_0 / (3 * np.trace(R_k))

        grad_J = -2 * (P_k - R_k.dot(h_k))
        # print(mu_0 / (3 * np.trace(r)), mu_0 / (3 * np.trace(r) + epsilon), mu_0 / ((first_input_k ** 2).sum() + epsilon), grad_J)
        # print(mu_k, np.trace(r), grad_J)

        h_k -= mu_k * grad_J

    return h_k

def RLS(first_input_1d, second_input_1d, number_of_weights_1d, lambd=0.9, delta=1):

    h_k = np.zeros(number_of_weights_1d).reshape(number_of_weights_1d, 1)
    first_input_k = np.zeros(number_of_weights_1d)
    iR_k = np.eye(number_of_weights_1d) * (1 / delta)
    R_k = np.eye(number_of_weights_1d) * (delta)

    # print(len(second_input_1d[number_of_weights_1d - 1:]))

    for k in range(1, len(second_input_1d)):
        first_input_k = np.roll(first_input_k, -1)
        first_input_k[-1] = first_input_1d[k]

        first_input_k_vec = first_input_k.reshape(number_of_weights_1d, 1)
        first_input_k_vec_transp = first_input_k_vec.transpose()

        g_k = iR_k.dot(first_input_k_vec) / (lambd + first_input_k_vec_transp.dot(iR_k).dot(first_input_k_vec))
        alpha_k = second_input_1d[k] - h_k.transpose().dot(first_input_k_vec)

        h_k = h_k + g_k * alpha_k
        iR_k = (iR_k - g_k.dot(first_input_k_vec_transp).dot(iR_k)) / lambd
        R_k = lambd * R_k + first_input_k_vec.dot(first_input_k_vec_transp)
        # iR_k = np.linalg.inv(R_k)
        # print(k)
        # print(np.sum(np.linalg.inv(R_k) - iR_k))

    return h_k


def fitting_of_NLMS(first_input, second_input, number_of_weights, mu_0=0.9, epsilon=1):
    assert len(first_input.shape) == len(
        second_input.shape), f'The dimensions of the parameters do not match {len(first_input.shape)} != {len(second_input.shape)}'

    weights = np.empty((number_of_weights, first_input.shape[0]))

    if len(first_input.shape) == 1: # signal depends on time
        weights = NLMS_1d(first_input, second_input, number_of_weights, mu_0, epsilon).flatten()
    elif len(first_input.shape) == 2:  # signal depends on time and x
        # weights = NLMS_1d(first_input[0, :], second_input[0, :], number_of_weights, mu_0,
        #                                  epsilon).reshape(number_of_weights, 1)

        for i in range(0, first_input.shape[0]):
            # weights2 = NLMS_1d(first_input[i, :], second_input[i, :], number_of_weights, mu_0,
            #                               epsilon).reshape(number_of_weights, 1)
            weights[:, i] = NLMS_1d(first_input[i, :], second_input[i, :], number_of_weights, mu_0, epsilon).flatten()

            # weights = np.concatenate((weights, weights2), axis=1)


    return weights

def fitting_of_LMS(first_input, second_input, number_of_weights, mu_0=0.9, epsilon=1):
    assert len(first_input.shape) == len(
        second_input.shape), f'The dimensions of the parameters do not match {len(first_input.shape)} != {len(second_input.shape)}'

    weights = np.empty((number_of_weights, first_input.shape[0]))

    if len(first_input.shape) == 1: # signal depends on time
        weights = LMS_1d(first_input, second_input, number_of_weights, mu_0, epsilon).flatten()
    elif len(first_input.shape) == 2:  # signal depends on time and x
        # weights = NLMS_1d(first_input[0, :], second_input[0, :], number_of_weights, mu_0,
        #                                  epsilon).reshape(number_of_weights, 1)

        for i in range(0, first_input.shape[0]):
            # weights2 = NLMS_1d(first_input[i, :], second_input[i, :], number_of_weights, mu_0,
            #                               epsilon).reshape(number_of_weights, 1)
            weights[:, i] = LMS_1d(first_input[i, :], second_input[i, :], number_of_weights, mu_0, epsilon).flatten()

            # weights = np.concatenate((weights, weights2), axis=1)


    return weights

def fitting_of_RLS(first_input, second_input, number_of_weights, lambd=0.99, delta=1):
    assert len(first_input.shape) == len(
        second_input.shape), f'The dimensions of the parameters do not match {len(first_input.shape)} != {len(second_input.shape)}'

    weights = np.empty((number_of_weights, first_input.shape[0]))

    if len(first_input.shape) == 1: # signal depends on time
        weights = RLS(first_input, second_input, number_of_weights, lambd, delta).flatten()
    elif len(first_input.shape) == 2:  # signal depends on time and x
        # weights = NLMS_1d(first_input[0, :], second_input[0, :], number_of_weights, mu_0,
        #                                  epsilon).reshape(number_of_weights, 1)

        for i in range(0, first_input.shape[0]):
            # weights2 = NLMS_1d(first_input[i, :], second_input[i, :], number_of_weights, mu_0,
            #                               epsilon).reshape(number_of_weights, 1)
            weights[:, i] = RLS(first_input[i, :], second_input[i, :], number_of_weights, lambd, delta).flatten()

            # weights = np.concatenate((weights, weights2), axis=1)


    return weights

def l2(firstSeries, secondSeries):
    return np.sqrt(((firstSeries - secondSeries) ** 2).sum())

def Nl2(firstSeries, secondSeries):
    max_value = max(np.abs(firstSeries).max(), np.abs(secondSeries).max())
    # print(f"max_value = {max_value:.1e}")
    return np.sqrt((((firstSeries - secondSeries) / max_value) ** 2).sum())

def MSE(firstSeries, secondSeries):
    return np.mean(((firstSeries - secondSeries) ** 2), dtype=np.float32)

def NMSE(firstSeries, secondSeries):
    max_value = max(np.abs(firstSeries).max(), np.abs(secondSeries).max())
    # print(f"max_value = {max_value:.1e}")
    return np.mean((((firstSeries - secondSeries) / max_value) ** 2), dtype=np.float32)

def ME(firstSeries, secondSeries):
    return np.mean(np.abs(firstSeries - secondSeries), dtype=np.float32)

def NME(firstSeries, secondSeries):
    max_value = max(np.abs(firstSeries).max(), np.abs(secondSeries).max())
    # print(f"max_value = {max_value:.1e}")
    return np.mean(np.abs(firstSeries - secondSeries), dtype=np.float32) / max_value