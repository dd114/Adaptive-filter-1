import numpy as np

def processing_of_signal(input, weights):
    assert len(input.shape) == len(weights.shape), f'The dimensions of the parameters do not match {len(input.shape)} != {len(weights.shape)}'

    output = np.zeros_like(input)
    number_of_weights = weights.shape[0] # bc number of weights to each time layer is located on column (in 2-dim case)

    print(weights.shape, input.shape)

    match len(input.shape):
        case 1: # signal depends on time
            for i in range(output.size - number_of_weights + 1):
                output[number_of_weights - 1 + i] = (input[i:number_of_weights + i] * weights).sum()

        case 2: # signal depends on time and x

            for j in range(input.shape[0]):
                for i in range(output.shape[1] - number_of_weights + 1):
                    output[j, number_of_weights - 1 + i] = (input[j, i:number_of_weights + i] * weights[:, j]).sum()
                    # print(i, j)

    return output

def getting_weights_1d(first_input_1d, second_input_1d, number_of_weights_1d, mu_0, epsilon):
    # weights = np.random.random(number_of_weights_1d)
    weights_1d = np.ones(number_of_weights_1d)

    # print(len(second_input_1d[number_of_weights_1d - 1:]))

    for k in range(len(second_input_1d[number_of_weights_1d - 1:])):
        first_input_k = first_input_1d[k:number_of_weights_1d + k]
        second_input_k = second_input_1d[k + number_of_weights_1d - 1]

        p = second_input_k * first_input_k

        r = np.empty((number_of_weights_1d, number_of_weights_1d))

        for i in range(number_of_weights_1d):
            for j in range(number_of_weights_1d):
                r[i, j] = first_input_k[i] * first_input_k[j]

        mu_k = mu_0 / ((first_input_k ** 2).sum() + epsilon)
        # mu_k = 0.9 / np.trace(r)

        grad_J = -2 * (p - r.dot(weights_1d))

        # if grad_J == 0:
        #     print("r.dot(weights_1d) = ", r.max())

        weights_1d -= mu_k * grad_J

    return weights_1d


def fitting_of_weights(first_input, second_input, number_of_weights, mu_0=1, epsilon=0.1):
    assert len(first_input.shape) == len(
        second_input.shape), f'The dimensions of the parameters do not match {len(first_input.shape)} != {len(second_input.shape)}'

    weights = np.array([0])

    match len(first_input.shape):
        case 1:  # signal depends on time
            weights = getting_weights_1d(first_input, second_input, number_of_weights, mu_0, epsilon)

        case 2:  # signal depends on time and x
            weights = getting_weights_1d(first_input[0, :], second_input[0, :], number_of_weights, mu_0,
                                         epsilon).reshape(number_of_weights, 1)

            for i in range(1, first_input.shape[0]):
                weights2 = getting_weights_1d(first_input[i, :], second_input[i, :], number_of_weights, mu_0,
                                              epsilon).reshape(number_of_weights, 1)

                # print("1", weights.shape)

                # print(weights, weights2)
                weights = np.concatenate((weights, weights2), axis=1)

                # print("2", weights.shape)

    return weights