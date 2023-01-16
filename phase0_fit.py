# -*- coding: utf-8 -*-
# Copyright 2023 Brett D. Roads. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Command line fit script."""

import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # noqa
from pathlib import Path
import shutil
import time

# import tensorflow as tf
from tidy_models import ModelIdentifier
# from tidy_models import multicuda
import numpy as np
import psiz
import tensorflow as tf
import tensorflow_probability as tfp
import tidy_models.databases.pandas.core as db

# Uncomment the following line to force eager execution.
tf.config.run_functions_eagerly(True)

# Uncomment and edit the following to control GPU visibility.
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class StochasticBehaviorModel(psiz.keras.StochasticModel):
    """A behavior model.

    No Gates.

    """

    def __init__(self, behavior=None, **kwargs):
        """Initialize."""
        super(StochasticBehaviorModel, self).__init__(**kwargs)
        self.behavior = behavior

    def call(self, inputs):
        """Call."""
        return self.behavior(inputs)

    def get_config(self):
        """Get configuration."""
        return super(StochasticBehaviorModel, self).get_config()


def infer_model(fp_project, arch_id, input_id, split):
    """Infer model."""
    # Settings.
    n_stimuli = 25
    n_dim = 2
    epochs = 10000
    batch_size = 128
    fp_models = fp_project / Path('assets', 'models')
    fp_datasets = fp_project / Path('assets', 'data', 'datasets')
    fp_db_fit = fp_project / Path('assets', 'dbs', 'db_fit.txt')

    # Load data.
    fp_ds = fp_datasets / Path('ds_{}'.format(input_id))
    tfds_all = tf.data.Dataset.load(str(fp_ds))

    # Count trials.
    n_trial = 0
    for _ in tfds_all:
        n_trial = n_trial + 1

    print('n_trial: {}'.format(n_trial))

    # Partition data into 90% train, 10% validation set.
    if split == -1:
        n_trial_train = n_trial
        tfds_train = tfds_all.shuffle(
            buffer_size=n_trial_train, reshuffle_each_iteration=True
        ).batch(
            batch_size, drop_remainder=False
        )
    elif split == 0:
        n_trial_train = int(np.round(0.9 * n_trial))
        tfds_train = tfds_all.take(n_trial_train).cache().shuffle(
            buffer_size=n_trial_train, reshuffle_each_iteration=True
        ).batch(
            batch_size, drop_remainder=False
        )
        tfds_val = tfds_all.skip(n_trial_train).cache().batch(
            batch_size, drop_remainder=False
        )
    else:
        raise NotImplementedError("Requested split is not implemented.")

    hypers = {
        'n_dim': n_dim
    }
    mid = ModelIdentifier(
        arch_id=arch_id,
        input_id=input_id,
        hypers=hypers,
        n_split=10,
        split=split,
        path=fp_models,
        prefix='emb'
    )
    fp_board = fp_project / Path('assets', 'logs', 'fit', mid.name)

    # Directory preparation.
    fp_board.mkdir(parents=True, exist_ok=True)
    # Remove existing TensorBoard logs.
    if fp_board.exists():
        shutil.rmtree(fp_board)

    # Build VI model.
    if arch_id == 0:
        optimizer = tf.keras.optimizers.Adam(learning_rate=.001)
        model = build_model_0(n_stimuli, n_dim, n_trial_train, optimizer)
    else:
        raise NotImplementedError

    # Define callbacks.
    cb_board = tf.keras.callbacks.TensorBoard(
        log_dir=fp_board,
        histogram_freq=0,
        write_graph=False,
        write_images=False,
        update_freq='epoch',
        profile_batch=0,
        embeddings_freq=0,
        embeddings_metadata=None
    )
    cb_early = tf.keras.callbacks.EarlyStopping(
        'loss',
        patience=100,
        mode='min',
        restore_best_weights=False,
        verbose=1
    )
    callbacks = [cb_board, cb_early]

    # Infer embedding.
    start_time_s = time.time()
    history = model.fit(
        x=tfds_train, epochs=epochs, callbacks=callbacks, verbose=0
    )
    train_time_s = time.time() - start_time_s
    n_epoch_used = len(history.epoch)

    # Save model.
    model.save(mid.pathname, overwrite=True, save_traces=False)

    d_train = model.evaluate(tfds_train, verbose=0, return_dict=True)

    # Update fit database.
    id_data = mid.as_dict()
    assoc_data = {
        'n_epoch': n_epoch_used,
        'train_time_s': train_time_s,
        'loss': d_train['loss'],
        'cce': d_train['cce'],
    }
    if mid.split != -1:
        # Validation set evaluation.
        d_val = model.evaluate(tfds_val, return_dict=True, verbose=0)
        assoc_data.update({
            'val_loss': d_val['loss'],
            'val_cce': d_val['cce']
        })

    # Update fit database.
    if not fp_db_fit.exists():
        db.create_empty_db(
            fp_db_fit,
            columns=['arch_id', 'input_id', 'split_seed', 'n_split', 'split']
        )
    df_fit_log = db.load_db(fp_db_fit)
    df_fit_log = db.update_one(df_fit_log, id_data, assoc_data)
    db.save_db(df_fit_log, fp_db_fit)

    # result = {
    #     'train_loss': d_train['loss'],
    #     'train_cce': d_train['cce'],
    #     'val_loss': d_val['loss'],
    #     'val_cce': d_val['cce'],
    # }

    # print(
    #     'Results'
    #     '    train_loss: {0:.2f} | train_cce: {1:.2f} | '.format(
    #         result['train_loss'], result['train_cce']
    #     )
    # )
    # print(
    #     '    val_loss: {0:.2f} | val_cce: {1:.2f} | '.format(
    #         result['val_loss'], result['val_cce']
    #     )
    # )
    print('    beta: {0:.2f}'.format(
            model.behavior.kernel.similarity.beta.numpy()
        )
    )
    print(
        '    mean scale: {0:.5f}'.format(
            np.sqrt(
                np.mean(
                    model.behavior.percept.embeddings.variance().numpy()
                )
            )
        )
    )


def build_model_0(n_stimuli, n_dim, n_sample_train, optimizer):
    """Build model.

    Arguments:
        n_stimuli: Integer indicating the number of stimuli (not
            including placeholder).
        n_dim: Integer indicating the dimensionality of the embedding.
        n_sample_train: Integer indicating the number of training
            observations. Used to determine KL weight for variational
            inference.

    Returns:
        model: A TensorFlow Keras model.

    """
    kl_weight = 1. / n_sample_train
    prior_scale = .2  # pixel_std

    embedding_posterior = psiz.keras.layers.EmbeddingNormalDiag(
        (n_stimuli + 1),
        n_dim,
        mask_zero=True,
        scale_initializer=tf.keras.initializers.Constant(
            tfp.math.softplus_inverse(prior_scale).numpy()
        ),
        loc_trainable=True,
        scale_trainable=True
    )
    embedding_prior = psiz.keras.layers.EmbeddingShared(
        n_stimuli + 1,
        n_dim,
        mask_zero=True,
        embedding=psiz.keras.layers.EmbeddingNormalDiag(
            1,
            1,
            loc_initializer=tf.keras.initializers.Constant(0.),
            scale_initializer=tf.keras.initializers.Constant(
                tfp.math.softplus_inverse(prior_scale).numpy()
            ),
            loc_trainable=False,
            scale_trainable=True
        )
    )
    percept = psiz.keras.layers.EmbeddingVariational(
        posterior=embedding_posterior,
        prior=embedding_prior,
        kl_weight=kl_weight,
        kl_n_sample=30
    )
    kernel = psiz.keras.layers.DistanceBased(
        distance=psiz.keras.layers.Minkowski(
            rho_initializer=tf.keras.initializers.Constant(2.),
            w_initializer=tf.keras.initializers.Constant(1.),
            trainable=False
        ),
        similarity=psiz.keras.layers.ExponentialSimilarity(
            fit_tau=False,
            fit_gamma=False,
            fit_beta=False,
            beta_initializer=tf.keras.initializers.Constant(10.),
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.),
        )
    )
    behavior = psiz.keras.layers.RankSimilarity(
        n_reference=2, n_select=1, percept=percept, kernel=kernel
    )
    model = StochasticBehaviorModel(behavior=behavior, n_sample=30)
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=optimizer,
        weighted_metrics=[
            tf.keras.metrics.CategoricalCrossentropy(name='cce')
        ]
    )
    return model


if __name__ == "__main__":
    fp_project = Path.home() / Path('projects', 'rob_mok', 'exploration')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--arch_id', type=int, default=0,
        help='Integer indicating the architecture ID. Will default to `0`'
    )
    parser.add_argument(
        '--input_id', type=int, default=0,
        help='Integer indicating the input ID. Will default to `0`'
    )
    parser.add_argument(
        '--split', type=int, default=-1,
        help='Integer indicating the split. Will default to `-1`'
    )
    parser.add_argument(
        '--gpu', type=int, default=-1,
        help='Integer indicating GPU to use. Will default to CPU only.'
    )
    args = parser.parse_args()
    if args.gpu == -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "{0}".format(args.gpu)

    infer_model(fp_project, args.arch_id, args.input_id, args.split)
