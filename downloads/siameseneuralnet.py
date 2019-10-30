# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 16:33:02 2018

@author: Jackson
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.neighbors.kde import KernelDensity


def create_encoder(hidden_layers, dimension):
    inputs = tf.keras.layers.Input(shape=(2 * dimension,))
    
    current = inputs
    for layer in hidden_layers:
        current = tf.keras.layers.Dense(units=layer, activation=tf.nn.relu)(current)

    x = tf.keras.layers.Dense(units=dimension)(current)
    y = tf.keras.layers.Dense(units=dimension)(current)

    radius = tf.keras.layers.Lambda(lambda q: tf.sqrt(q[0] ** 2 + q[1] ** 2))([x, y])
    theta = tf.keras.layers.Lambda(lambda q: tf.atan2(q[1], q[0]))([x, y])

    return tf.keras.models.Model(inputs, [radius, theta])


def create_decoder(hidden_layers, dimension):
    radius = tf.keras.layers.Input(shape=(dimension,))
    theta = tf.keras.layers.Input(shape=(dimension,))

    x = tf.keras.layers.Lambda(lambda z: z[0] * tf.cos(z[1]))([radius, theta])
    y = tf.keras.layers.Lambda(lambda z: z[0] * tf.sin(z[1]))([radius, theta])

    current = tf.keras.layers.Concatenate(axis=-1)([x, y])

    for layer in hidden_layers:
        current = tf.keras.layers.Dense(units=layer, activation=tf.nn.relu)(current)

    outputs = tf.keras.layers.Dense(units=2 * dimension)(current)

    return tf.keras.models.Model([radius, theta], outputs)


def create_donsker_varadhan_test_function(hidden_layers, dimension):
    inputs = tf.keras.layers.Input(shape=(dimension,))

    cos = tf.keras.layers.Lambda(tf.cos)(inputs)
    sin = tf.keras.layers.Lambda(tf.sin)(inputs)

    current = tf.keras.layers.Concatenate(axis=-1)([cos, sin])

    for layer in hidden_layers:
        current = tf.keras.layers.Dense(units=layer, activation=tf.nn.relu)(current)

    outputs = tf.keras.layers.Dense(units=1)(current)

    return tf.keras.models.Model(inputs, outputs)


def donsker_varadhan_negentropy(test_function, inputs):
    random_inputs = tf.random_uniform(maxval=2. * np.pi, shape=tf.shape(inputs))

    first_term = tf.reduce_mean(test_function(inputs))
    second_term = -tf.log(tf.reduce_mean(tf.exp(test_function(random_inputs))))

    return first_term + second_term


def model_fn(features, labels, mode, params):
    assert labels is None

    phase_space_coords = features['phase_space_coords']
    dimension = params['dimension']
    
    encoder = create_encoder(params['hidden_layers'], dimension)
    decoder = create_decoder(params['hidden_layers'], dimension)

    autoencoder = tf.keras.models.Model(inputs=encoder.inputs, outputs=decoder(encoder.outputs))

    actions, angles = encoder(phase_space_coords)

    recon_phase_space_coords = autoencoder(phase_space_coords)

    if mode == tf.estimator.ModeKeys.TRAIN:

        autoencoder_loss = tf.losses.mean_squared_error(phase_space_coords, recon_phase_space_coords)

        if params['apply_proximity_loss']:

            halo_id = features['halo_id']

            relative_actions = \
                tf.keras.layers.Lambda(lambda a: a[:, tf.newaxis, :] - a[tf.newaxis, :, :])(actions)

            halo_id_equal = \
                tf.keras.layers.Lambda(lambda h: tf.equal(h[:, tf.newaxis], h[tf.newaxis, :]))(halo_id)

            halo_id_equal = tf.keras.backend.cast(halo_id_equal, tf.float32)

            masked_relative_actions = \
                tf.keras.layers.multiply([relative_actions, halo_id_equal[:, :, tf.newaxis]])

            proximity_loss = tf.keras.backend.sum(masked_relative_actions ** 2)

            proximity_loss /= tf.keras.backend.sum(halo_id_equal)

            autoencoder_loss += params['proximity_loss_coeff'] * proximity_loss

        entropy_train_op = None

        if params['apply_donsker_varadhan_loss']:
            test_function = create_donsker_varadhan_test_function(params['hidden_layers'], dimension)

            donsker_varadhan_loss = donsker_varadhan_negentropy(test_function, angles)
            tf.identity(donsker_varadhan_loss, 'donsker_varadhan_loss')

            entropy_optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
            entropy_train_op = entropy_optimizer.minimize(-donsker_varadhan_loss,
                                                          global_step=tf.train.get_or_create_global_step(),
                                                          var_list=test_function.trainable_variables)

            autoencoder_loss += params['donsker_varadhan_coeff'] * donsker_varadhan_loss

        tf.identity(autoencoder_loss, 'autoencoder_loss')

        autoencoder_optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
        autoencoder_train_op = autoencoder_optimizer.minimize(autoencoder_loss,
                                                              global_step=tf.train.get_or_create_global_step(),
                                                              var_list=autoencoder.trainable_variables)

        train_ops = autoencoder_train_op if entropy_train_op is None else \
            tf.group([autoencoder_train_op, entropy_train_op])

        tf.identity(params['learning_rate'], 'learning_rate')

        loss = autoencoder_loss
    else:
        loss = train_ops = None

    if mode == tf.estimator.ModeKeys.EVAL:
        raise NotImplementedError

    predictions = {'actions': actions, 'angles': angles, 'recon_phase_space_coords': autoencoder(phase_space_coords)} \
        if mode == tf.estimator.ModeKeys.PREDICT else None

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_ops, predictions=predictions)


def load_data():
    folder = 'D:'
    phase_space_file_name = folder + 'phase_space.npy'
    # snapshot_file_name = folder + 'snapshot.npy'
    halo_id_file_name = folder + 'halo_id.npy'

    phase_space_coords = np.load(phase_space_file_name)
    # snapshots = np.load(snapshot_file_name)
    halo_id = np.load(halo_id_file_name)

    phase_space_coords -= np.mean(phase_space_coords, axis=0)

    length_scale = np.sqrt(np.mean(phase_space_coords[:3] ** 2))
    phase_space_coords[:3] /= length_scale
    velocity_scale = np.sqrt(np.mean(phase_space_coords[3:] ** 2))
    phase_space_coords[3:] /= velocity_scale

    phase_space_coords = phase_space_coords.astype(np.float32)
    halo_id = halo_id.astype(np.int64)

    return phase_space_coords, halo_id


def contour_plot(points, x_label, y_label):
    kde = KernelDensity(kernel='epanechnikov', bandwidth=0.01).fit(points)
    x_limits = np.min(points[:, 0]), np.max(points[:, 0])
    y_limits = np.min(points[:, 1]), np.max(points[:, 1])
    x, y = np.meshgrid(np.linspace(*x_limits, 300), np.linspace(*y_limits, 300))
    xy = np.stack([x.ravel(), y.ravel()]).T
    z = kde.score_samples(xy).reshape(x.shape)
    levels = np.linspace(z.max() - 10., z.max(), 100)
    plt.contourf(x, y, np.exp(z), levels=np.exp(levels), cmap=plt.cm.gist_rainbow)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def main(_):
    phase_space_coords, halo_id = load_data()

    # points = phase_space_coords[np.isin(halo_id, np.unique(halo_id)[-21:-20]), :2]
    # contour_plot(points, 'x', 'y')

    background = dict(phase_space_coords=phase_space_coords[halo_id == 9999], halo_id=halo_id[halo_id == 9999])

    cluster_ids = np.unique(halo_id)
    cluster_ids = cluster_ids[:-1]  # remove background

    number_of_clusters = len(cluster_ids) // 4
    print('{} distinct clusters, predicting for last {}'.format(len(cluster_ids), number_of_clusters))

    mask = np.isin(halo_id, cluster_ids[-number_of_clusters:])

    clusters = dict(phase_space_coords=phase_space_coords[mask], halo_id=halo_id[mask])

    params = dict(dimension=3, hidden_layers=[64] * 4, learning_rate=1e-4, apply_donsker_varadhan_loss=True,
                  donsker_varadhan_coeff=1., batch_size=128, shuffle_buffer_size=65536, epochs_per_cycle=1,
                  apply_proximity_loss=True, proximity_loss_coeff=1.)

    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir='./tmp', params=params)

    def train_input_fn(features, batch_size, shuffle_buffer_size, epochs_per_cycle):
        dataset = tf.data.Dataset.from_tensor_slices(features)

        return dataset.repeat(epochs_per_cycle).shuffle(buffer_size=shuffle_buffer_size).batch(batch_size)

    estimator.train(input_fn=lambda: train_input_fn(features=background, batch_size=params['batch_size'],
                                                    shuffle_buffer_size=params['shuffle_buffer_size'],
                                                    epochs_per_cycle=params['epochs_per_cycle']))

    def predict_input_fn(features, batch_size):
        return tf.data.Dataset.from_tensor_slices(features).batch(batch_size)

    predictions = estimator.predict(
        input_fn=lambda: predict_input_fn(features=clusters, batch_size=params['batch_size']),
        yield_single_examples=False)

    predictions = list(predictions)

    actions = np.vstack([p['actions'] for p in predictions])
    angles = np.vstack([p['angles'] for p in predictions])
    recon_phase_space_coords = np.vstack([p['recon_phase_space_coords'] for p in predictions])

    contour_plot(actions[:, 0:2], x_label='action 1', y_label='action 2')
    contour_plot(actions[:, 1:3], x_label='action 2', y_label='action 3')
    contour_plot(actions[:, ::2], x_label='action 1', y_label='action 3')

    contour_plot(angles[:, 0:2], x_label='angle 1', y_label='angle 2')
    contour_plot(angles[:, 1:3], x_label='angle 2', y_label='angle 3')
    contour_plot(angles[:, ::2], x_label='angle 1', y_label='angle 3')
    
    contour_plot(recon_phase_space_coords[:, 0:2], x_label='x', y_label='y')
    contour_plot(recon_phase_space_coords[:, 1:3], x_label='y', y_label='z')
    contour_plot(recon_phase_space_coords[:, :3:2], x_label='x', y_label='z')


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()