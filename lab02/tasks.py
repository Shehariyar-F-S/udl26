"""Lab 2 tasks. Inputs and returned values are torch tensors (returned ones on the CPU, without gradients)."""


def categorical_1d(train_data, test_data, d):
    """
    Categorical distribution over d values, trained by minibatch gradient descent on NLL.

    train_data: (n_train,) integer tensor (uint8), values in {0, ..., d-1}
    test_data: (n_test,) integer tensor (uint8), values in {0, ..., d-1}
    d: number of values

    Returns
    - train_losses: (# iterations,) float tensor, training NLL of every minibatch
    - test_losses: (# epochs + 1,) float tensor, test NLL at initialisation and after every epoch
    - probs: (d,) float tensor, probs[k] = p(x = k), sums to 1

    Losses: mean NLL, nats per example
    """
    # TODO: YOUR CODE HERE
    raise NotImplementedError


def independent_2d(train_data, test_data, d, dset_id):
    """
    Independent variables, p(x0, x1) = p(x0) p(x1), trained by minibatch gradient descent on NLL.

    train_data: (n_train, 2) integer tensor (uint8), values in {0, ..., d-1}
    test_data: (n_test, 2) integer tensor (uint8), values in {0, ..., d-1}
    d: number of values of each variable
    dset_id: dataset index (1 or 2), e.g. for dataset-specific hyperparameters

    Returns
    - train_losses: (# iterations,) float tensor, training NLL of every minibatch
    - test_losses: (# epochs + 1,) float tensor, test NLL at initialisation and after every epoch
    - probs: (d, d) float tensor, probs[i, j] = p(x0 = i, x1 = j), sums to 1

    Losses: mean NLL, nats per example
    """
    # TODO: YOUR CODE HERE
    raise NotImplementedError


def made_2d(train_data, test_data, d, dset_id):
    """
    Autoregressive model, p(x0, x1) = p(x0) p(x1 | x0), MADE (Germain et al., 2015), trained by minibatch gradient descent on NLL.

    train_data: (n_train, 2) integer tensor (uint8), values in {0, ..., d-1}
    test_data: (n_test, 2) integer tensor (uint8), values in {0, ..., d-1}
    d: number of values of each variable
    dset_id: dataset index (1 or 2), e.g. for dataset-specific hyperparameters

    Returns
    - train_losses: (# iterations,) float tensor, training NLL of every minibatch
    - test_losses: (# epochs + 1,) float tensor, test NLL at initialisation and after every epoch
    - probs: (d, d) float tensor, probs[i, j] = p(x0 = i, x1 = j), sums to 1

    Losses: mean NLL, nats per example
    """
    # TODO: YOUR CODE HERE
    raise NotImplementedError


def made_images(train_data, test_data):
    """
    Autoregressive model of binary images, pixels in raster order, MADE (Germain et al., 2015), trained by minibatch gradient descent on NLL.

    train_data: (n_train, H, W) integer tensor (uint8), values in {0, 1}
    test_data: (n_test, H, W) integer tensor (uint8), values in {0, 1}

    Returns
    - train_losses: (# iterations,) float tensor, training NLL of every minibatch
    - test_losses: (# epochs + 1,) float tensor, test NLL at initialisation and after every epoch
    - samples: (100, H, W) integer tensor, new images sampled from trained model

    Losses: mean NLL, bits per pixel
    """
    # TODO: YOUR CODE HERE
    raise NotImplementedError


# TODO: your own helper functions and classes, if you want any
pass
