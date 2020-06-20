from etn import coordinates, networks, transformers
import tensorflow as tf
import numpy as np
import fire
from tqdm import tqdm
import logging
import warnings
import os
import time
from functools import partial


class MNISTModel:
    # transformer defaults
    tf_default_opts = {
        'input_shape': (64, 64, 1),
        'kernel_size': 3,
        'nf': 32,
        'strides': (2, 1),
    }
    
    # classification network defaults
    net_default_opts = {
        'input_shape': (64, 64, 1),
        'nf': 32,
        'p_dropout': 0.3,
        'pad_mode': (None, 'cyclic'),
        'pool': (True, True, False),
    }
    
    # optimizer defaults
    optimizer_default_opts = {
        'amsgrad': True,
        'lr': 2e-3,
        'weight_decay': 0.,
    }
    
    # learning rate schedule defaults
    lr_default_schedule = {
        'step_size': 1,
        'gamma': 0.99,
    }
    
    # dataset mean and standard deviation
    normalization_mean = tf.constant([16.2884], dtype=tf.float32)
    normalization_std = tf.constant([56.2673], dtype=tf.float32)
    
    def __init__(self,
                 tfs=[transformers.ShearX,
                      transformers.HyperbolicRotation,
                      transformers.PerspectiveX,
                      transformers.PerspectiveY],
                 coords=coordinates.logpolar_grid,
                 net=networks.make_basic_cnn,
                 equivariant=True,
                 downsample=1,
                 tf_opts=tf_default_opts,
                 net_opts=net_default_opts,
                 seed=None,
                 load_path=None,
                 loglevel='INFO'):
        """MNIST model"""
        tf_opts_copy = dict(self.tf_default_opts)
        tf_opts_copy.update(tf_opts)
        
        net_opts_copy = dict(self.net_default_opts)
        net_opts_copy.update(net_opts)

        # configure logging
        numeric_level = getattr(logging, loglevel.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError('Invalid log level: %s' % loglevel)
        logging.basicConfig(level=numeric_level)

        logging.info(str(self))

        if load_path is not None:
            ckpt = tf.train.Checkpoint(model=self.model)
            ckpt.restore(load_path).assert_consumed()
            logging.info('Model loaded at {}'.format(load_path))

        if net is None:
            raise ValueError('net parameter must be specified')

        if seed is not None:
            tf.random.set_seed(seed)
            np.random.seed(seed)
    
            # build transformer sequence
        if len(tfs) > 0:
            pose_module = networks.make_equivariant_pose_predictor if equivariant \
                else networks.make_direct_pose_predictor
            tfs = [getattr(transformers, tfr) if type(tfr) is str else tfr for tfr in tfs]
            seq = transformers.TransformerSequence(*[tfr(pose_module, **tf_opts) for tfr in tfs])
            # seq = transformers.TransformerParallel(*[tfr(pose_module, **tf_opts) for tfr in tfs])
            logging.info('Transformers: %s' % ' -> '.join([tfr.__name__ for tfr in tfs]))
            logging.info('Pose module: %s' % pose_module.__name__)
        else:
            seq = None

        # get coordinate function if given as a string
        if type(coords) is str:
            if hasattr(coordinates, coords):
                coords = getattr(coordinates, coords)
            elif hasattr(coordinates, coords + '_grid'):
                coords = getattr(coordinates, coords + '_grid')
            else:
                raise ValueError('Invalid coordinate system: ' + coords)
        logging.info('Coordinate transformation before classification: %s' % coords.__name__)

        # define network
        if type(net) is str:
            net = getattr(networks, net)
        network = net(**net_opts)
        logging.info('Classifier architecture: %s' % net.__name__)

        self.tfs = tfs
        self.coords = coords
        self.downsample = downsample
        self.net = net
        self.equivariant = equivariant
        self.tf_opts = tf_opts
        self.net_opts = net_opts
        self.seed = seed
        self.model = self._build_model(net=network, transformer=seq, coords=coords, downsample=downsample)

        logging.info('Net opts: %s' % str(net_opts))
        logging.info('Transformer opts: %s' % str(tf_opts))

    def _build_model(self, net, transformer, coords, downsample):
        return networks.TransformerCNN(
            net=net,
            transformer=transformer,
            coords=coords,
            downsample=downsample)

    def _save(self, path):
        ckpt = tf.train.Checkpoint(model=self.model)
        ckpt.save(path)
        

    def _load_dataset(self, path, which='train') -> (tf.data.Dataset, int):
        # override in subclasses to handle custom preprocessing / different data formats
        which = which + '.npy'
        path = os.path.join(path, which)
        ds_dict = np.load(path, allow_pickle=True)
        ds_dict = ds_dict.item()
    
        x = ds_dict['x']
        y = ds_dict['y']
        feature_ds = tf.data.Dataset.from_tensor_slices(x)
        label_ds = tf.data.Dataset.from_tensor_slices(y)
    
        def _normalize(image):
            image = (image - self.normalization_mean) / self.normalization_std
            return image
    
        feature_ds.map(_normalize)
        dataset = tf.data.Dataset.zip((feature_ds, label_ds))
        return dataset, len(x)
    
    def __str__(self):
        return "Projective MNIST classification"

    def train(self,
              num_epochs=300,
              batch_size=128,
              valid_batch_size=100,
              path=None,
              optimizer='Adam',
              optimizer_opts={'amsgrad': True},
              lr_schedule={'initial_learning_rate': 2e-3, 'decay_steps': 1, 'decay_rate': 0.99, 'staircase': True},
              save_path=None):
        """Train the model."""
        optimizer_opts_copy = dict(self.optimizer_default_opts)
        optimizer_opts_copy.update(optimizer_opts)

        lr_schedule_copy = dict(self.lr_default_schedule)
        lr_schedule_copy.update(lr_schedule)
        
        if save_path is not None:
            logging.info('Saving model with lowest validation error to %s' % save_path)
        else:
            warnings.warn('save_path not specified: model will not be saved')
    
        # load training and validation data
        if path is None:
            raise ValueError('path must be specified')
    
        logging.info('Loading training data from %s' % path)
        train_ds, num_train = self._load_dataset(path, which='train')
        train_ds = train_ds.shuffle(500).batch(batch_size, drop_remainder=True)
    
        logging.info('Loading validation data from %s' % path)
        valid_ds, _ = self._load_dataset(path, which='valid')
        valid_ds = valid_ds.batch(valid_batch_size, drop_remainder=False)
        
        scheduler = tf.keras.optimizers.schedules.ExponentialDecay(**lr_schedule)
        lr_schedule['decay_steps'] = lr_schedule['decay_steps'] * (num_train // batch_size)
        optim = getattr(tf.keras.optimizers, optimizer)(learning_rate=scheduler, **optimizer_opts)
    
        best_err = float('inf')
        start_time = time.time()
        for i in range(num_epochs):
            # train for one epoch
            logging.info('Training epoch %d' % (i + 1))
            train_losses = self._train(optim, train_ds)
        
            # evaluate on validation set
            logging.info('Evaluating model on validation set')
            valid_loss, valid_err = self._test(valid_ds)
            logging.info('Validation loss = %.2e, validation error = %.4f' % (valid_loss, valid_err))
        
            # save model with lowest validation error seen so far
            if (save_path is not None) and (valid_err < best_err):
                logging.info(
                    'Saving model with better validation error: %.2e (previously %.2e)' % (valid_err, best_err))
                best_err = valid_err
                self._save(save_path)
    
        logging.info('Finished training in %.1f s' % (time.time() - start_time))
        return self

    def test(self, batch_size=100, path=None):
        """Test the model."""
        if path is None:
            raise ValueError('test_path must be specified')
        logging.info('loading test data from %s' % path)
        ds, _ = self._load_dataset(path, which='test')
        ds = ds.shuffle(10000).batch(batch_size, drop_remainder=False)

        loss, err_rate = self._test(ds)
        logging.info('Test loss = %.2e' % loss)
        logging.info('Test error = %.4f' % err_rate)
        return loss, err_rate

    def predict(self, x, tf_output=False):
        """Predict a distribution over labels for a single example."""
        if tf.rank(x) == 3:
            x = tf.expand_dims(x, 0)
        out = self.model(x, training=False)
        logits = out[0]
        probs = tf.nn.softmax(logits, axis=-1)
        if tf_output:
            return probs, out[1]
        else:
            return probs

    def _train(self, optim, ds):
        losses = []
        for x, y in tqdm(ds):
            y = tf.one_hot(y, 10)
            with tf.GradientTape() as tape:
                logits, _ = self.model(x)
                loss = tf.nn.softmax_cross_entropy_with_logits(y, logits)
                loss = tf.math.reduce_mean(loss)
            grads = tape.gradient(loss, self.model.trainable_weights)
            optim.apply_gradients(zip(grads, self.model.trainable_weights))
            losses.append(loss.numpy())
        return losses

    def _test(self, ds):
        total_loss = 0.
        total_err = 0.
        count = 0
        for x, y in tqdm(ds):
            y = tf.one_hot(y, 10, dtype=tf.int64)
            count += tf.shape(x)[0]
            logits, _ = self.model(x, training=False)
            yhat = tf.argmax(logits, axis=-1)
            total_err += tf.math.reduce_sum(tf.cast(y != yhat, dtype=tf.int64)).numpy()
            total_loss += tf.math.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(y, logits)).numpy()
        loss = total_loss / count
        err_rate = total_err / count
        return loss, err_rate


if __name__ == '__main__':
    fire.Fire(MNISTModel)