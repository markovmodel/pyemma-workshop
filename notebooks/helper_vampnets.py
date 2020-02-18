import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def _inv(x, ret_sqrt=False, epsilon=1e-5):
    '''Utility function that returns the inverse of a matrix, with the
    option to return the square root of the inverse matrix.

    Parameters
    ----------
    x: numpy array with shape [m,m]
        matrix to be inverted

    ret_sqrt: bool, optional, default = False
        if True, the square root of the inverse matrix is returned instead

    Returns
    -------
    x_inv: numpy array with shape [m,m]
        inverse of the original matrix
    '''

    # Calculate eigvalues and eigvectors
    eigval_all, eigvec_all = tf.linalg.eigh(x)

    # Filter out eigvalues below threshold and corresponding eigvectors
    eig_th = tf.constant(epsilon)
    index_eig = tf.cast(eigval_all > eig_th, dtype=tf.int32)
    _, eigval = tf.dynamic_partition(eigval_all, index_eig, 2)
    _, eigvec = tf.dynamic_partition(tf.transpose(eigvec_all), index_eig, 2)

    # Build the diagonal matrix with the filtered eigenvalues or square
    # root of the filtered eigenvalues according to the parameter
    eigval_inv = tf.linalg.tensor_diag(1/eigval)
    eigval_inv_sqrt = tf.linalg.tensor_diag(tf.sqrt(1/eigval))

    cond_sqrt = tf.convert_to_tensor(ret_sqrt)

    diag = tf.cond(cond_sqrt, lambda: eigval_inv_sqrt, lambda: eigval_inv)

    # Rebuild the square root of the inverse matrix
    x_inv = tf.matmul(tf.transpose(eigvec), tf.matmul(diag, eigvec))

    return x_inv

def _prep_data(data):
    '''Utility function that transorms the input data from a tensorflow - 
    viable format to a structure used by the following functions in the
    pipeline.

    Parameters
    ----------
    data: tensorflow tensor with shape [b, 2*o]
        original format of the data

    Returns
    -------
    x: tensorflow tensor with shape [o, b]
        transposed, mean-free data corresponding to the left, lag-free lobe
        of the network

    y: tensorflow tensor with shape [o, b]
        transposed, mean-free data corresponding to the right, lagged lobe
        of the network

    b: tensorflow float32
        batch size of the data

    o: int
        output size of each lobe of the network

    '''

    shape = tf.shape(data)
    b = tf.cast(shape[0], tf.float32)


    # Split the data of the two networks and transpose it
    x_biased, y_biased = tf.split(data, 2, axis=1)


    # Subtract the mean
    x = x_biased - tf.reduce_mean(x_biased, axis=0, keepdims=True)
    y = y_biased - tf.reduce_mean(y_biased, axis=0, keepdims=True)

    return x, y, b

def metric_VAMP(y_true, y_pred):
    '''Returns the sum of the top k eigenvalues of the vamp matrix, with k
    determined by the wrapper parameter k_eig, and the vamp matrix defined
    as:
        V = cov_00 ^ -1/2 * cov_01 * cov_11 ^ -1/2
    Can be used as a metric function in model.fit()

    Parameters
    ----------
    y_true: tensorflow tensor.
        parameter not needed for the calculation, added to comply with Keras
        rules for loss fuctions format.

    y_pred: tensorflow tensor with shape [batch_size, 2 * output_size]
        output of the two lobes of the network

    Returns
    -------
    eig_sum: tensorflow float
        sum of the k highest eigenvalues in the vamp matrix
    '''

    x, y, batch_size = _prep_data(y_pred)
    
    matrices = _build_vamp_matrices(x, y, batch_size)
    cov_00_ir, cov_11_ir, cov_01 = matrices

    vamp_matrix = tf.matmul(cov_00_ir, tf.matmul(cov_01, cov_11_ir))

    diag = tf.convert_to_tensor(tf.linalg.svd(vamp_matrix, compute_uv=False))

    sv_sum = tf.reduce_sum(diag)

    return sv_sum


def metric_VAMP2(y_true, y_pred):
    '''Returns the sum of the squared top k eigenvalues of the vamp matrix,
    with k determined by the wrapper parameter k_eig, and the vamp matrix
    defined as:
        V = cov_00 ^ -1/2 * cov_01 * cov_11 ^ -1/2
    Can be used as a metric function in model.fit()

    Parameters
    ----------
    y_true: tensorflow tensor.
        parameter not needed for the calculation, added to comply with Keras
        rules for loss fuctions format.

    y_pred: tensorflow tensor with shape [batch_size, 2 * output_size]
        output of the two lobes of the network

    Returns
    -------
    eig_sum_sq: tensorflow float
        sum of the squared k highest eigenvalues in the vamp matrix
    '''

    x, y, batch_size = _prep_data(y_pred)
    
    matrices = _build_vamp_matrices(x, y, batch_size)
    cov_00_ir, cov_11_ir, cov_01 = matrices

    vamp_matrix = tf.matmul(cov_00_ir, tf.matmul(cov_01, cov_11_ir))

    diag = tf.convert_to_tensor(tf.linalg.svd(vamp_matrix, compute_uv=False))

    pow2_diag = tf.reduce_sum(tf.multiply(diag,diag))

    return pow2_diag


def loss_VAMP2(y_true, y_pred):
    '''Calculates the VAMP-2 score with respect to the network lobes. Same function
    as loss_VAMP2, but the gradient is computed automatically by tensorflow. Added
    after tensorflow 1.5 introduced gradients for eigenvalue decomposition and SVD

    Parameters
    ----------
    y_true: tensorflow tensor.
        parameter not needed for the calculation, added to comply with Keras
        rules for loss fuctions format.

    y_pred: tensorflow tensor with shape [batch_size, 2 * output_size]
        output of the two lobes of the network

    Returns
    -------
    loss_score: tensorflow tensor with shape [batch_size, 2 * output_size].
        gradient of the VAMP-2 score
    '''

    x, y, batch_size = _prep_data(y_pred) 

    cov_00_inv, cov_11_inv, cov_01 = _build_vamp_matrices(x, y, batch_size)

    vamp_matrix = tf.matmul(tf.matmul(cov_00_inv, cov_01), cov_11_inv)

    vamp_score = tf.norm(vamp_matrix)

    return - tf.square(vamp_score)

def _build_vamp_matrices(x, y, b):
    '''Utility function that returns the matrices used to compute the VAMP
    scores and their gradients for non-reversible problems.

    Parameters
    ----------
    x: tensorflow tensor with shape [output_size, b]
        output of the left lobe of the network

    y: tensorflow tensor with shape [output_size, b]
        output of the right lobe of the network

    b: tensorflow float32
        batch size of the data

    Returns
    -------
    cov_00_inv_root: numpy array with shape [output_size, output_size]
        square root of the inverse of the auto-covariance matrix of x

    cov_11_inv_root: numpy array with shape [output_size, output_size]
        square root of the inverse of the auto-covariance matrix of y

    cov_01: numpy array with shape [output_size, output_size]
        cross-covariance matrix of x and y

    '''

    cov_01 = 1/(b-1) * tf.matmul(x, y, transpose_a=True)
    cov_00 = 1/(b-1) * tf.matmul(x, x, transpose_a=True) 
    cov_11 = 1/(b-1) * tf.matmul(y, y, transpose_a=True)

    cov_00_inv_root = _inv(cov_00, ret_sqrt=True)
    cov_11_inv_root = _inv(cov_11, ret_sqrt=True)

    return cov_00_inv_root, cov_11_inv_root, cov_01

def get_its(data, lags, plot = False, calculate_K = True, multiple_runs = False):
    ''' Implied timescales from a trajectory estimated at a series of lag times.

    Parameters
    ----------
    traj: numpy array with size [traj_timesteps, traj_dimensions]
        trajectory data or a list of trajectories

    lags: numpy array with size [lag_times]
        series of lag times at which the implied timescales are estimated

    Returns
    -------
    its: numpy array with size [traj_dimensions - 1, lag_times]
        Implied timescales estimated for the trajectory.

    '''

    def get_single_its(data):

        if type(data) == list:
            outputsize = data[0].shape[1]
        else:
            outputsize = data.shape[1]

        single_its = np.zeros((outputsize-1, len(lags)))

        for t, tau_lag in enumerate(lags):
            if calculate_K:
                koopman_op = estimate_koopman_op(data, tau_lag)
            else:
                koopman_op = data[t]
            k_eigvals, k_eigvec = np.linalg.eig(np.real(koopman_op))
            k_eigvals = np.sort(np.absolute(k_eigvals))
            k_eigvals = k_eigvals[:-1]
            single_its[:,t] = (-tau_lag / np.log(k_eigvals))

        return np.array(single_its)



    if not multiple_runs:

        its = get_single_its(data)

    else:

        its = []
        for data_run in data:
            its.append(get_single_its(data_run))

    if plot:
        plot_its(its, lags, multiple_runs = multiple_runs)
        plt.show()

    return its


def plot_its(its, lag, ylog=False, multiple_runs = False):
    '''Plots the implied timescales calculated by the function
    'get_its'

    Parameters
    ----------
    its: numpy array
        the its array returned by the function get_its
    lag: numpy array
        lag times array used to estimate the implied timescales
    ylog: Boolean, optional, default = False
        if true, the plot will be a logarithmic plot, otherwise it
        will be a semilogy plot

    '''

    func = plt.loglog if ylog else plt.semilogy

    if not multiple_runs:
        func(lag, its[::-1].T)
    else:
        its_mean = np.mean(its, 0)[::-1]
        its_std = np.std(its, 0)[::-1]
        for index_its, m, s in zip(range(len(its)), its_mean, its_std):
            func(lag, m, color = 'C{}'.format(index_its))
            plt.fill_between(lag, m+s, m-s, color = 'C{}'.format(index_its), alpha = 0.2)

    func(lag,lag, 'k')
    plt.fill_between(lag, lag, 0.99, alpha=0.2, color='k');
    
def estimate_koopman_op(trajs, tau, epsilon = 1e-5):
    '''Estimates the koopman operator for a given trajectory at the lag time
        specified. The formula for the estimation is:
            K = C00 ^ -1/2 @ C01 @ C11 ^ -1/2
        if both_corr_mat is True, else:
            K = C00 ^ -1 @ C01
        If force_symmetric is True, the matrices are calculated as:
            C00 = C11 = x.T @ x + y.T @ y
            C01 = x.T @ y + y.T @ x

    Parameters
    ----------
    traj: numpy array with size [traj_timesteps, traj_dimensions]
        Trajectory described by the returned koopman operator

    tau: int
        Time shift at which the koopman operator is estimated

    Returns
    -------
    koopman_op: numpy array with shape [traj_dimensions, traj_dimensions]
        Koopman operator estimated at timeshift tau

    '''

    if type(trajs) == list:
        traj = np.concatenate([t[:-tau] for t in trajs], axis = 0)
        traj_lag = np.concatenate([t[tau:] for t in trajs], axis = 0)
    else:
        traj = trajs[:-tau]
        traj_lag = trajs[tau:]

    koopman_op = np.ones(traj.shape[1])

    c_0 = traj.T @ traj
    c_tau = traj.T @ traj_lag
    c_1 = traj_lag.T @ traj_lag
    
    eigv_all, eigvec_all = np.linalg.eig(c_0)
    include = eigv_all > epsilon
    eigv = eigv_all[include]
    eigvec = eigvec_all[:,include]
    c0_inv = eigvec @ np.diag(1/eigv) @ np.transpose(eigvec)

    koopman_op = c0_inv @ c_tau
    eigv_all, eigvec_all = np.linalg.eig(c_0)
    include = eigv_all > epsilon
    eigv = eigv_all[include]
    eigvec = eigvec_all[:,include]
    c0_inv = eigvec @ np.diag(1/eigv) @ np.transpose(eigvec)

    koopman_op = c0_inv @ c_tau

    return koopman_op    