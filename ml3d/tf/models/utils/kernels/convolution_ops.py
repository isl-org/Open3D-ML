#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Functions defining KPConv as tensorflow ops
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 11/06/2018
#

# ------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#

import numpy as np
import tensorflow as tf
import sys

from .kernel_points import load_kernels as create_kernel_points

# ------------------------------------------------------------------------------------------
#
#          Utility function
#      \**********************/
#


def broadcast_matmul(A, B):
    """
    Compute A @ B, broadcasting over the first `N-2` ranks
    :param A: first matrix [..., d1, d2]
    :param B: second matrix [..., d2, d3]
    :return: result [..., d1, d3]
    """
    """"""
    with tf.variable_scope("broadcast_matmul"):
        return tf.reduce_sum(A[..., tf.newaxis] * B[..., tf.newaxis, :, :],
                             axis=-2)


def radius_gaussian(sq_r, sig, eps=1e-9):
    """
    Compute a radius gaussian (gaussian of distance)
    :param sq_r: input radiuses [dn, ..., d1, d0]
    :param sig: extents of gaussians [d1, d0] or [d0] or float
    :return: gaussian of sq_r [dn, ..., d1, d0]
    """
    return tf.exp(-sq_r / (2 * tf.square(sig) + eps))


def general_gaussian(xyz, L):
    """
    Compute a general gaussian deformable in every direction
    !!! params has to respect b^2 < ac or the gaussian is not defined !!! (Always use Cholesky decomposition)
    :param xyz: input radiuses [dn, ..., d1, d0, 3]
    :param L: Gaussian parameters in the forme of Cholesky decomposition [d1, d0, 3, 3] or [d0, 3, 3] or [3, 3]
    :return: gaussian of sq_xyz [dn, ..., d1, d0]
    """

    if int(xyz.shape[-1]) != 3:
        raise ValueError('general_gaussian only defined for dimension 3')

    # Create symmetric definite-positive matrices
    if len(L.shape) == 3:
        A = tf.matmul(L, tf.transpose(L, [0, 2, 1]))
    elif len(L.shape) == 4:
        A = tf.matmul(L, tf.transpose(L, [0, 1, 3, 2]))
    else:
        raise ValueError(
            'Matrix L in general gaussian have a wrong number of dimension')

    # Multiply by xyz from both sides
    quad = broadcast_matmul(tf.expand_dims(xyz, -2),
                            tf.expand_dims(tf.expand_dims(A, 0), 0))
    quad = broadcast_matmul(quad, tf.expand_dims(xyz, -1))

    return tf.exp(-tf.squeeze(quad))


# ------------------------------------------------------------------------------------------
#
#          Convolutions definitions
#      \******************************/
#


def unary_convolution(features, K_values):
    """
    Simple unary convolution in tensorflow. Equivalent to matrix multiplication (space projection) for each features
    :param features: float32[n_points, in_fdim] - input features
    :param K_values: float32[in_fdim, out_fdim] - weights of the kernel
    :return: output_features float32[n_points, out_fdim]
    """

    return tf.matmul(features, K_values)


def KPConv(query_points,
           support_points,
           neighbors_indices,
           features,
           K_values,
           fixed='center',
           KP_extent=1.0,
           KP_influence='linear',
           aggregation_mode='sum'):
    """
    This function initiates the kernel point disposition before building KPConv graph ops

    :param query_points: float32[n_points, dim] - input query points (center of neighborhoods)
    :param support_points: float32[n0_points, dim] - input support points (from which neighbors are taken)
    :param neighbors_indices: int32[n_points, n_neighbors] - indices of neighbors of each point
    :param features: float32[n_points, in_fdim] - input features
    :param K_values: float32[n_kpoints, in_fdim, out_fdim] - weights of the kernel
    :param fixed: string in ('none', 'center' or 'verticals') - fix position of certain kernel points
    :param KP_extent: float32 - influence radius of each kernel point
    :param KP_influence: string in ('constant', 'linear', 'gaussian') - influence function of the kernel points
    :param aggregation_mode: string in ('closest', 'sum') - whether to sum influences, or only keep the closest

    :return: output_features float32[n_points, out_fdim]
    """

    # Initial kernel extent for this layer
    K_radius = 1.5 * KP_extent

    # Number of kernel points
    num_kpoints = int(K_values.shape[0])

    # Check point dimension (currently only 3D is supported)
    points_dim = int(query_points.shape[1])

    # Create one kernel disposition (as numpy array). Choose the KP distance to center thanks to the KP extent
    K_points_numpy = create_kernel_points(K_radius,
                                          num_kpoints,
                                          num_kernels=1,
                                          dimension=points_dim,
                                          fixed=fixed)
    K_points_numpy = K_points_numpy.reshape((num_kpoints, points_dim))

    # Create the tensorflow variable
    K_points = tf.Variable(K_points_numpy.astype(np.float32),
                           name='kernel_points',
                           trainable=False,
                           dtype=tf.float32)

    return KPConv_ops(query_points, support_points, neighbors_indices,
                      features, K_points, K_values, KP_extent, KP_influence,
                      aggregation_mode)


def KPConv_ops(query_points, support_points, neighbors_indices, features,
               K_points, K_values, KP_extent, KP_influence, aggregation_mode):
    """
    This function creates a graph of operations to define Kernel Point Convolution in tensorflow. See KPConv function
    above for a description of each parameter

    :param query_points:        [n_points, dim]
    :param support_points:      [n0_points, dim]
    :param neighbors_indices:   [n_points, n_neighbors]
    :param features:            [n_points, in_fdim]
    :param K_points:            [n_kpoints, dim]
    :param K_values:            [n_kpoints, in_fdim, out_fdim]
    :param KP_extent:           float32
    :param KP_influence:        string
    :param aggregation_mode:    string
    :return:                    [n_points, out_fdim]
    """

    # Get variables
    n_kp = int(K_points.shape[0])

    # Add a fake point in the last row for shadow neighbors
    shadow_point = tf.ones_like(support_points[:1, :]) * 1e6
    support_points = tf.concat([support_points, shadow_point], axis=0)

    # Get neighbor points [n_points, n_neighbors, dim]
    neighbors = tf.gather(support_points, neighbors_indices, axis=0)

    # Center every neighborhood
    neighbors = neighbors - tf.expand_dims(query_points, 1)

    # Get all difference matrices [n_points, n_neighbors, n_kpoints, dim]
    neighbors = tf.expand_dims(neighbors, 2)
    neighbors = tf.tile(neighbors, [1, 1, n_kp, 1])
    differences = neighbors - K_points

    # Get the square distances [n_points, n_neighbors, n_kpoints]
    sq_distances = tf.reduce_sum(tf.square(differences), axis=3)

    # Get Kernel point influences [n_points, n_kpoints, n_neighbors]
    if KP_influence == 'constant':
        # Every point get an influence of 1.
        all_weights = tf.ones_like(sq_distances)
        all_weights = tf.transpose(all_weights, [0, 2, 1])

    elif KP_influence == 'linear':
        # Influence decrease linearly with the distance, and get to zero when d = KP_extent.
        all_weights = tf.maximum(1 - tf.sqrt(sq_distances) / KP_extent, 0.0)
        all_weights = tf.transpose(all_weights, [0, 2, 1])

    elif KP_influence == 'gaussian':
        # Influence in gaussian of the distance.
        sigma = KP_extent * 0.3
        all_weights = radius_gaussian(sq_distances, sigma)
        all_weights = tf.transpose(all_weights, [0, 2, 1])
    else:
        raise ValueError(
            'Unknown influence function type (config.KP_influence)')

    # In case of closest mode, only the closest KP can influence each point
    if aggregation_mode == 'closest':
        neighbors_1nn = tf.argmin(sq_distances, axis=2, output_type=tf.int32)
        all_weights *= tf.one_hot(neighbors_1nn,
                                  n_kp,
                                  axis=1,
                                  dtype=tf.float32)

    elif aggregation_mode != 'sum':
        raise ValueError(
            "Unknown convolution mode. Should be 'closest' or 'sum'")

    features = tf.concat([features, tf.zeros_like(features[:1, :])], axis=0)

    # Get the features of each neighborhood [n_points, n_neighbors, in_fdim]
    neighborhood_features = tf.gather(features, neighbors_indices, axis=0)

    # Apply distance weights [n_points, n_kpoints, in_fdim]
    weighted_features = tf.matmul(all_weights, neighborhood_features)

    # Apply network weights [n_kpoints, n_points, out_fdim]
    weighted_features = tf.transpose(weighted_features, [1, 0, 2])
    kernel_outputs = tf.matmul(weighted_features, K_values)

    # Convolution sum to get [n_points, out_fdim]
    output_features = tf.reduce_sum(kernel_outputs, axis=0)

    return output_features


def KPConv_deformable(query_points,
                      support_points,
                      neighbors_indices,
                      features,
                      K_values,
                      fixed='center',
                      KP_extent=1.0,
                      KP_influence='linear',
                      aggregation_mode='sum',
                      modulated=False):
    """
    This function initiates the kernel point disposition before building deformable KPConv graph ops

    :param query_points: float32[n_points, dim] - input query points (center of neighborhoods)
    :param support_points: float32[n0_points, dim] - input support points (from which neighbors are taken)
    :param neighbors_indices: int32[n_points, n_neighbors] - indices of neighbors of each point
    :param features: float32[n_points, in_fdim] - input features
    :param K_values: float32[n_kpoints, in_fdim, out_fdim] - weights of the kernel
    :param fixed: string in ('none', 'center' or 'verticals') - fix position of certain kernel points
    :param KP_extent: float32 - influence radius of each kernel point
    :param KP_influence: string in ('constant', 'linear', 'gaussian') - influence function of the kernel points
    :param aggregation_mode: string in ('closest', 'sum') - behavior of the convolution
    :param modulated: bool - If deformable conv should be modulated

    :return: output_features float32[n_points, out_fdim]
    """

    ############
    # Parameters
    ############

    # Radius of the initial positions of the kernel points
    K_radius = 1.5 * KP_extent

    # Number of kernel points
    num_kpoints = int(K_values.shape[0])

    # Check point dimension (currently only 3D is supported)
    points_dim = int(query_points.shape[1])

    #################################
    # Initiate kernel point positions
    #################################

    # Create one kernel disposition (as numpy array). Choose the KP distance to center thanks to the KP extent
    K_points_numpy = create_kernel_points(K_radius,
                                          num_kpoints,
                                          num_kernels=1,
                                          dimension=points_dim,
                                          fixed=fixed)
    K_points_numpy = K_points_numpy.reshape((num_kpoints, points_dim))

    # Create the tensorflow variable
    K_points = tf.Variable(K_points_numpy.astype(np.float32),
                           name='kernel_points',
                           trainable=False,
                           dtype=tf.float32)

    #############################
    # Standard KPConv for offsets
    #############################

    # Create independant weight for the first convolution and a bias term as no batch normalization happen
    if modulated:
        offset_dim = (points_dim + 1) * num_kpoints
    else:
        offset_dim = points_dim * num_kpoints
    shape0 = K_values.shape.as_list()
    shape0[-1] = offset_dim
    K_values0 = tf.Variable(tf.zeros(shape0, dtype=tf.float32),
                            name='offset_conv_weights')
    b0 = tf.Variable(tf.zeros(offset_dim, dtype=tf.float32),
                     name='offset_conv_bias')

    # Get features from standard convolution
    features0 = KPConv_ops(query_points, support_points, neighbors_indices,
                           features, K_points, K_values0, KP_extent,
                           KP_influence, aggregation_mode) + b0

    if modulated:

        # Get offset (in normalized scale) from features
        offsets = features0[:, :points_dim * num_kpoints]
        offsets = tf.reshape(offsets, [-1, num_kpoints, points_dim])

        # Get modulations
        modulations = 2 * tf.sigmoid(features0[:, points_dim * num_kpoints:])

    else:

        # Get offset (in normalized scale) from features
        offsets = tf.reshape(features0, [-1, num_kpoints, points_dim])

        # No modulations
        modulations = None

    # Rescale offset for this layer
    offsets *= KP_extent

    ###############################
    # Build deformable KPConv graph
    ###############################

    # Apply deformed convolution
    return KPConv_deform_ops(query_points, support_points, neighbors_indices,
                             features, K_points, offsets, modulations,
                             K_values, KP_extent, KP_influence,
                             aggregation_mode)


def KPConv_deform_ops(query_points, support_points, neighbors_indices,
                      features, K_points, offsets, modulations, K_values,
                      KP_extent, KP_influence, mode):
    """
    This function creates a graph of operations to define Deformable Kernel Point Convolution in tensorflow. See
    KPConv_deformable function above for a description of each parameter

    :param query_points:        [n_points, dim]
    :param support_points:      [n0_points, dim]
    :param neighbors_indices:   [n_points, n_neighbors]
    :param features:            [n_points, in_fdim]
    :param K_points:            [n_kpoints, dim]
    :param offsets:             [n_points, n_kpoints, dim]
    :param modulations:         [n_points, n_kpoints] or None
    :param K_values:            [n_kpoints, in_fdim, out_fdim]
    :param KP_extent:           float32
    :param KP_influence:        string
    :param mode:                string

    :return:                    [n_points, out_fdim]
    """

    # Get variables
    n_kp = int(K_points.shape[0])
    shadow_ind = tf.shape(support_points)[0]

    # Add a fake point in the last row for shadow neighbors
    shadow_point = tf.ones_like(support_points[:1, :]) * 1000
    support_points = tf.concat([support_points, shadow_point], axis=0)

    # Get neighbor points [n_points, n_neighbors, dim]
    neighbors = tf.gather(support_points, neighbors_indices, axis=0)

    # Center every neighborhood
    neighbors = neighbors - tf.expand_dims(query_points, 1)

    # Apply offsets to kernel points [n_points, n_kpoints, dim]
    deformed_K_points = tf.add(offsets, K_points, name='deformed_KP')

    # Get all difference matrices [n_points, n_neighbors, n_kpoints, dim]
    neighbors = tf.expand_dims(neighbors, 2)
    neighbors = tf.tile(neighbors, [1, 1, n_kp, 1])
    differences = neighbors - tf.expand_dims(deformed_K_points, 1)

    # Get the square distances [n_points, n_neighbors, n_kpoints]
    sq_distances = tf.reduce_sum(tf.square(differences),
                                 axis=3,
                                 name='deformed_d2')

    # Boolean of the neighbors in range of a kernel point [n_points, n_neighbors]
    in_range = tf.cast(
        tf.reduce_any(tf.less(sq_distances, KP_extent**2), axis=2), tf.int32)

    # New value of max neighbors
    new_max_neighb = tf.reduce_max(tf.reduce_sum(in_range, axis=1))

    # For each row of neighbors, indices of the ones that are in range [n_points, new_max_neighb]
    new_neighb_bool, new_neighb_inds = tf.math.top_k(in_range,
                                                     k=new_max_neighb)

    # Gather new neighbor indices [n_points, new_max_neighb]
    new_neighbors_indices = tf.batch_gather(neighbors_indices, new_neighb_inds)

    # Gather new distances to KP [n_points, new_max_neighb, n_kpoints]
    new_sq_distances = tf.batch_gather(sq_distances, new_neighb_inds)

    # New shadow neighbors have to point to the last shadow point
    new_neighbors_indices *= new_neighb_bool
    new_neighbors_indices += (1 - new_neighb_bool) * shadow_ind

    # Get Kernel point influences [n_points, n_kpoints, n_neighbors]
    if KP_influence == 'constant':
        # Every point get an influence of 1.
        all_weights = tf.cast(new_sq_distances < KP_extent**2, tf.float32)
        all_weights = tf.transpose(all_weights, [0, 2, 1])

    elif KP_influence == 'linear':
        # Influence decrease linearly with the distance, and get to zero when d = KP_extent.
        all_weights = tf.maximum(1 - tf.sqrt(new_sq_distances) / KP_extent,
                                 0.0)
        all_weights = tf.transpose(all_weights, [0, 2, 1])

    elif KP_influence == 'gaussian':
        # Influence in gaussian of the distance.
        sigma = KP_extent * 0.3
        all_weights = radius_gaussian(new_sq_distances, sigma)
        all_weights = tf.transpose(all_weights, [0, 2, 1])
    else:
        raise ValueError(
            'Unknown influence function type (config.KP_influence)')

    # In case of closest mode, only the closest KP can influence each point
    if mode == 'closest':
        neighbors_1nn = tf.argmin(new_sq_distances,
                                  axis=2,
                                  output_type=tf.int32)
        all_weights *= tf.one_hot(neighbors_1nn,
                                  n_kp,
                                  axis=1,
                                  dtype=tf.float32)

    elif mode != 'sum':
        raise ValueError(
            "Unknown convolution mode. Should be 'closest' or 'sum'")

    features = tf.concat([features, tf.zeros_like(features[:1, :])], axis=0)

    # Get the features of each neighborhood [n_points, new_max_neighb, in_fdim]
    neighborhood_features = tf.gather(features, new_neighbors_indices, axis=0)

    # Apply distance weights [n_points, n_kpoints, in_fdim]
    weighted_features = tf.matmul(all_weights, neighborhood_features)

    # Apply modulations
    if modulations is not None:
        weighted_features *= tf.expand_dims(modulations, 2)

    # Apply network weights [n_kpoints, n_points, out_fdim]
    weighted_features = tf.transpose(weighted_features, [1, 0, 2])
    kernel_outputs = tf.matmul(weighted_features, K_values)

    # Convolution sum [n_points, out_fdim]
    output_features = tf.reduce_sum(kernel_outputs, axis=0)

    return output_features


# ------------------------------------------------------------------------------------------
#
#          DEV : Alternate deformable KPConv
#      \***************************************/
#


def KPConv_deformable_v2(query_points,
                         support_points,
                         neighbors_indices,
                         features,
                         K_values,
                         fixed='center',
                         KP_extent=1.0,
                         KP_influence='linear',
                         aggregation_mode='sum',
                         modulated=False):
    """
    This alternate version uses a pointwise MLP instead of KPConv to get the offset. It has thus less parameters.
    It also fixes the center point to remain in the center in any case. This definition offers similar performances

    :param query_points: float32[n_points, dim] - input query points (center of neighborhoods)
    :param support_points: float32[n0_points, dim] - input support points (from which neighbors are taken)
    :param neighbors_indices: int32[n_points, n_neighbors] - indices of neighbors of each point
    :param features: float32[n_points, in_fdim] - input features
    :param K_values: float32[n_kpoints, in_fdim, out_fdim] - weights of the kernel
    :param fixed: string in ('none', 'center' or 'verticals') - fix position of certain kernel points
    :param KP_extent: float32 - influence radius of each kernel point
    :param KP_influence: string in ('constant', 'linear', 'gaussian') - influence function of the kernel points
    :param aggregation_mode: string in ('closest', 'sum') - behavior of the convolution
    :param modulated: bool - If deformable conv should be modulated

    :return: output_features float32[n_points, out_fdim]
    """

    ############
    # Parameters
    ############

    # Check point dimension (currently only 3D is supported)
    points_dim = int(query_points.shape[1])

    # Number of kernel points
    num_kpoints = int(K_values.shape[0])

    #################
    # MLP for offsets
    #################

    # Create independant weight for the first convolution and a bias term as no batch normalization happen
    if modulated:
        offset_dim = (points_dim + 1) * (num_kpoints - 1)
    else:
        offset_dim = points_dim * (num_kpoints - 1)
    shape0 = K_values.shape.as_list()

    w0 = tf.Variable(tf.zeros([shape0[1], offset_dim], dtype=tf.float32),
                     name='offset_mlp_weights')
    b0 = tf.Variable(tf.zeros([offset_dim], dtype=tf.float32),
                     name='offset_mlp_bias')

    # Get features from mlp
    features0 = unary_convolution(features, w0) + b0

    if modulated:

        # Get offset (in normalized scale) from features
        offsets = features0[:, :points_dim * (num_kpoints - 1)]
        offsets = tf.reshape(offsets, [-1, (num_kpoints - 1), points_dim])

        # Get modulations
        modulations = 2 * tf.sigmoid(features0[:, points_dim *
                                               (num_kpoints - 1):])

        #  No offset for the first Kernel points
        offsets = tf.concat([tf.zeros_like(offsets[:, :1, :]), offsets],
                            axis=1)
        modulations = tf.concat(
            [tf.zeros_like(modulations[:, :1]), modulations], axis=1)

    else:

        # Get offset (in normalized scale) from features
        offsets = tf.reshape(features0, [-1, (num_kpoints - 1), points_dim])

        # No offset for the first Kernel points
        offsets = tf.concat([tf.zeros_like(offsets[:, :1, :]), offsets],
                            axis=1)

        # No modulations
        modulations = None

    # Rescale offset for this layer
    offsets *= KP_extent

    #################################
    # Initiate kernel point positions
    #################################

    # Radius of the initial positions of the kernel points
    K_radius = 1.5 * KP_extent

    # Create one kernel disposition (as numpy array). Choose the KP distance to center thanks to the KP extent
    K_points_numpy = create_kernel_points(K_radius,
                                          num_kpoints,
                                          num_kernels=1,
                                          dimension=points_dim,
                                          fixed=fixed)
    K_points_numpy = K_points_numpy.reshape((num_kpoints, points_dim))

    # Create the tensorflow variable
    K_points = tf.Variable(K_points_numpy.astype(np.float32),
                           name='kernel_points',
                           trainable=False,
                           dtype=tf.float32)

    ###############################
    # Build deformable KPConv graph
    ###############################

    # Apply deformed convolution
    return KPConv_deform_ops(query_points, support_points, neighbors_indices,
                             features, K_points, offsets, modulations,
                             K_values, KP_extent, KP_influence,
                             aggregation_mode)
