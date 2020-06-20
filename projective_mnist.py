import tensorflow as tf
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array
import numpy as np
import PIL
from tqdm import tqdm
import fire
import logging
import os


def projective_mnist(data_dir, seed=1, output_size=64, copies=8, num_train=10000, num_valid=5000):
    logging.info('Projective MNIST dataset')
    logging.info('seed = %d, output_size = %d, copies = %d, num_train = %d, num_valid = %d'
                 % (seed, output_size, copies, num_train, num_valid))
    
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    
    train_path = os.path.join(data_dir, 'train.npy')
    valid_path = os.path.join(data_dir, 'valid.npy')
    test_path = os.path.join(data_dir, 'test.npy')
    
    (mnist_train_x, mnist_train_y), (mnist_test_x, mnist_test_y) = tf.keras.datasets.mnist.load_data()
    mnist_train_x = np.expand_dims(mnist_train_x, -1)
    mnist_test_x = np.expand_dims(mnist_test_x, -1)
    num_test = len(mnist_test_x)
    
    np.random.seed(seed)
    idxs = np.random.choice(len(mnist_train_x), size=(num_train + num_valid), replace=False)
    train_idxs = idxs[:num_train]
    valid_idxs = idxs[num_train:]
    
    logging.info('Generating pose parameters')
    train_params = [random_projective_transform() for _ in range(copies * num_train)]
    valid_params = [random_projective_transform() for _ in range(num_valid)]
    test_params = [random_projective_transform() for _ in range(copies * num_test)]
    
    logging.info('Transforming training examples')
    train_data = np.empty([copies*len(train_idxs), output_size, output_size, 1], dtype=np.float32)
    train_labels = np.empty([copies*len(train_idxs)], dtype=np.int32)
    for i, idx in enumerate(tqdm(train_idxs)):
        label = mnist_train_y[idx]
        img = array_to_img(mnist_train_x[idx])
        img = img.convert(mode='F')
        for j in range(copies):
            params = train_params[j * num_train + i]
            timg = projective(img, canvas=(output_size, output_size), **params)
            timg = np.clip(img_to_array(timg), 0., 255.)
            train_data[j * len(train_idxs) + i] = timg
            train_labels[j * len(train_idxs) + i] = label
    np.save(train_path, {'x': train_data, 'y': train_labels}, allow_pickle=True)
    logging.info('Saved training set to %s' % train_path)

    logging.info('Transforming validation examples')
    valid_data = np.empty([len(valid_idxs), output_size, output_size, 1], dtype=np.float32)
    valid_labels = np.empty([len(valid_idxs)], dtype=np.int32)
    for i, idx in enumerate(tqdm(valid_idxs)):
        label = mnist_train_y[idx]
        img = array_to_img(mnist_train_x[idx])
        img = img.convert(mode='F')
        timg = projective(img, canvas=(output_size, output_size), **valid_params[i])
        timg = np.clip(img_to_array(timg), 0., 255.)
        valid_data[i] = timg
        valid_labels[i] = label
    np.save(valid_path, {'x': valid_data, 'y': valid_labels}, allow_pickle=True)
    logging.info('Saved validation set to %s' % valid_path)

    logging.info('Transforming test examples')
    test_data = np.empty([copies * num_test, output_size, output_size, 1], dtype=np.float32)
    test_labels = np.empty([copies * num_test], dtype=np.int32)
    for idx in tqdm(range(num_test)):
        label = mnist_test_y[idx]
        img = array_to_img(mnist_test_x[idx])
        img = img.convert(mode='F')
        for j in range(copies):
            params = test_params[j * num_test + idx]
            timg = projective(img, canvas=(output_size, output_size), **params)
            timg = np.clip(img_to_array(timg), 0., 255.)
            test_data[j * len(train_idxs) + idx] = timg
            test_labels[j * len(train_idxs) + idx] = label
    np.save(test_path, {'x': test_data, 'y': test_labels}, allow_pickle=True)
    logging.info('Saved test set to %s' % test_path)

    logging.info('Saving pose parameters to %s' % data_dir)
    np.save(os.path.join(data_dir, 'train_params.npy'), train_params)
    np.save(os.path.join(data_dir, 'valid_params.npy'), valid_params)
    np.save(os.path.join(data_dir, 'test_params.npy'), test_params)
    
    logging.info('Done')


def save_params(params, path):
    with tf.io.TFRecordWriter(path) as writer:
        for idx in tqdm(range(len(params))):
            example = convert_params(params[idx])
            writer.write(example.SerializeToString())


def transform_and_save(img, label, params, writer, output_size=64):
    timg = projective(img, canvas=(output_size, output_size), **params)
    timg = np.clip(img_to_array(timg), 0., 255.)
    example = convert_image(timg, label)
    writer.write(example.SerializeToString())


def random_projective_transform():
    pc = 0.8
    pa = np.random.uniform(-1, 1)
    pb = np.random.uniform(-1, 1) * (1 - np.abs(pa))
    perspective = (pc * pa, pc * pb)
    
    s = np.exp(np.random.uniform(0., np.log(2.)))
    aspect = np.exp(np.random.uniform(-np.log(1.5), np.log(1.5)))
    scale = (s * aspect, s / aspect)
    
    angle = np.random.uniform(-np.pi, np.pi)
    
    shear = np.random.uniform(-1.5, 1.5)
    return {
        'translation': (0., 0.),
        'angle': angle,
        'shear': shear,
        'perspective': perspective,
        'scale': scale,
    }


def projective(img, canvas=(64, 64), translation=(0., 0.),
               angle=0., scale=(1., 1.), shear=0.,
               perspective=(0., 0.)):
    t = translation
    s = scale
    p = perspective
    ca, sa = np.cos(angle), np.sin(angle)
    
    f = canvas[0] / img.size[0]  # assume same aspect ratio
    p = (f * p[0], f * p[1])
    
    mat = np.array([
        s[0] * ca + t[0] * p[0],
        s[1] * (shear * ca - sa) + t[0] * p[1],
        s[0] * sa + t[1] * p[0],
        s[1] * (shear * sa + ca) + t[1] * p[1],
    ]).reshape([2, 2])
    
    pa = [
        ((-mat[0, 0] + mat[0, 1] + t[0]) / (-p[0] + p[1] + 1.),
         (-mat[1, 0] + mat[1, 1] + t[1]) / (-p[0] + p[1] + 1.)),  # (-1, +1)
        
        ((mat[0, 0] + mat[0, 1] + t[0]) / (p[0] + p[1] + 1.),
         (mat[1, 0] + mat[1, 1] + t[1]) / (p[0] + p[1] + 1.)),  # (+1, +1)
        
        ((-mat[0, 0] - mat[0, 1] + t[0]) / (-p[0] - p[1] + 1.),
         (-mat[1, 0] - mat[1, 1] + t[1]) / (-p[0] - p[1] + 1.)),  # (-1, -1)
        
        ((mat[0, 0] - mat[0, 1] + t[0]) / (p[0] - p[1] + 1.),
         (mat[1, 0] - mat[1, 1] + t[1]) / (p[0] - p[1] + 1.)),  # (+1, -1)
    ]
    
    w, h = canvas
    img = img.transform(canvas, PIL.Image.PERSPECTIVE,
                        data=(1, 0, -w // 2 + img.size[0] // 2,
                              0, 1, -h // 2 + img.size[1] // 2,
                              0, 0),
                        resample=PIL.Image.NEAREST)
    
    pa = [(w * (x + 1) / 2, h * (1 - y) / 2) for x, y in pa]
    pb = [(0, 0), (w, 0), (0, h), (w, h)]
    params = _find_coeffs(pa, pb)
    img = img.transform(canvas, PIL.Image.PERSPECTIVE, data=params, resample=PIL.Image.BICUBIC)
    return img


def _find_coeffs(pa, pb):
    # pa: target coordinates, pb: source coordinates
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])
    
    A = np.matrix(matrix, dtype=np.float)
    B = np.array(pb).reshape(8)
    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def convert_image(image, label):
    image_shape = image.shape
    
    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'label': _int64_feature(label),
        'image': _float_feature(image.flatten()),
    }
    
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example


def convert_params(params):
    translation = params['translation']
    angle = params['angle']
    shear = params['shear']
    perspective = params['perspective']
    scale = params['scale']

    feature = {
        'translation_x': _float_feature([translation[0]]),
        'translation_y': _float_feature([translation[1]]),
        'angle': _float_feature([angle]),
        'shear': _float_feature([shear]),
        'perspective_x': _float_feature([perspective[0]]),
        'perspective_y': _float_feature([perspective[1]]),
        'scale_x': _float_feature([scale[0]]),
        'scale_y': _float_feature([scale[1]])
    }
    
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    fire.Fire(projective_mnist)
