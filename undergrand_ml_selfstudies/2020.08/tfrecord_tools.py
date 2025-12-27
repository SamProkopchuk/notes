'''
Functions for writing to and reading from TFRecord files.

Also provide examples of using these objects.
'''
import numpy as np
import os.path
import tensorflow as tf

from os import makedirs

'''
Basic Functions for serializing and parsing ndarrays and tensors
'''


def infer_parse_dtype(dtype):
    if isinstance(dtype, np.dtype):
        dtype = dtype.type

    elif dtype in (np.float16, np.float32, np.float64,
                   tf.float16, tf.float32, tf.float64, float):
        return tf.float64
    elif dtype in (np.uint8, np.uint16, np.uint32, np.uint64,
                   np.int8, np.int16, np.int32, np.int64,
                   tf.uint8, tf.uint16, tf.uint32, tf.uint64,
                   tf.int8, tf.int16, tf.int32, tf.int64,
                   tf.bool, int):
        return tf.int64
    elif dtype in (bytes, tf.string):
        return tf.string
    else:
        raise ValueError(f'Cannot infer tensorflow decoding dtype from dtype {dtype}.')


def serialize_feature(X, Y):
    '''
    Serialize (already encoded if desired) feature.
    '''
    def _dtype_feature(vector):
        if isinstance(vector, np.ndarray):
            # Is an ndarray
            dtype = vector.dtype.type
        elif tf.is_tensor(vector):
            # Is a tensor
            dtype = vector.dtype
        elif type(vector) is bytes:
            # Eg: Is a byte string
            dtype = type(vector)

        if dtype in (np.float16, np.float32, np.float64,
                     tf.float16, tf.float32, tf.float64, float):
            return tf.train.Feature(float_list=tf.train.FloatList(value=vector))
        elif dtype in (np.uint8, np.uint16, np.uint32, np.uint64,
                       np.int8, np.int16, np.int32, np.int64,
                       tf.uint8, tf.uint16, tf.uint32, tf.uint64,
                       tf.int8, tf.int16, tf.int32, tf.int64,
                       tf.bool, int):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=vector))
        elif dtype in (bytes, tf.string):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[vector if dtype is bytes else vector.numpy()]))
        else:
            raise ValueError(f'Given vector is of incompadible type: "{dtype}."')

    feature = {
        'X': _dtype_feature(X),
        'Y': _dtype_feature(Y)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()


'''
Encoding/Saving/Decoding Functions
'''


def encode_feature(X, Y, encoding_type=None, quality=95):
    '''
    Perform desired encoding for X and return
    Returns X, Y as tensors.
    '''
    X = tf.convert_to_tensor(X)
    Y = tf.convert_to_tensor(Y)
    if encoding_type is None:
        X = tf.keras.backend.flatten(X)
    elif encoding_type == 'JPEG':
        # Ensure X has the compadible shape;
        # ie: (height, width, channels)
        # Rather than (1, height, width, channels):
        X = tf.reshape(X, X.shape[-3:])
        X = tf.io.encode_jpeg(X, optimize_size=True, quality=quality)
    else:
        raise ValueError(f'Unknown encoding type: {encoding_type}.')
    return X, Y


def iterable_to_tfrecord(
        iterable,
        file_path,
        compression=None,
        encoding=None,
        quality=95,
        verbose=False,
        interval=100):
    '''
    Save all X, Y pairs from iterable to file_path in tfrecords format.
    X, Y can be either tensors or ndarrays (will be converted to tensors)
    '''
    def _set_type_check_cache(X, Y):
        cache = {
            'Xtype': type(X),
            'Ytype': type(Y)
        }
        return cache

    def _check_tcc(cache, X, Y):
        if not (cache['Xtype'] is type(X)):
            raise TypeError(f'Inconsistent X dtypes: {cache["Xtype"]}, {type(X)}')
        elif not (cache['Ytype'] is type(Y)):
            raise TypeError(f'Inconsistent Y dtypes: {cache["Ytype"]}, {type(Y)}')

    if verbose:
        basename=os.path.basename(file_path)
        idx=0

    tcc=None
    with tf.io.TFRecordWriter(file_path, options = compression) as writer:
        for X, Y in iterable:
            if tcc is None:
                tcc=_set_type_check_cache(X, Y)
            else:
                _check_tcc(tcc, X, Y)

            if verbose:
                if idx != 0 and idx % interval == 0:
                    print(f'Wrote {idx} images to {basename}')
                idx += 1
            X_, Y_=encode_feature(X, Y, encoding, quality = quality)
            feature=serialize_feature(X_, Y_)
            writer.write(feature)
    if verbose:
        print(f'Done! (Wrote {idx} images to {basename})')


def tfrecord_to_tfds(file_path, output_shapes, output_dtypes, compression = None, encoding = None):
    '''
    Given file_path to tfrecord file, returns tensorflow Dataset for which
    all Decompressing, parsing, reshaping is done.

    Should only be used with TFRecords saved using tfrecord_tools function.
    '''
    X_shape, Y_shape=output_shapes
    X_dtype, Y_dtype=output_dtypes

    raw_ds=tf.data.TFRecordDataset(file_path, compression_type = compression)
    if encoding is None:
        feature_format={
            'X': tf.io.FixedLenFeature(X_shape, infer_parse_dtype(X_dtype)),
            'Y': tf.io.FixedLenFeature(Y_shape, infer_parse_dtype(Y_dtype))
        }
    elif encoding == 'JPEG':
        feature_format = {
            'X': tf.io.VarLenFeature(tf.string),
            'Y': tf.io.FixedLenFeature(Y_shape, infer_parse_dtype(Y_dtype))
        }
    else:
        raise ValueError(f'Unknown encoding type: {encoding}.')

    def _proto_to_tensor(example_proto):
        parsed = tf.io.parse_single_example(example_proto, feature_format)
        X = parsed['X']
        Y = parsed['Y']
        return X, Y

    ds = raw_ds.map(_proto_to_tensor)
    if encoding == 'JPEG':
        def _decode(X, Y):
            X = tf.reshape(tf.sparse.to_dense(X), [])
            X = tf.io.decode_jpeg(X, channels=X_shape[-1])
            X = tf.reshape(tf.cast(X, X_dtype), X_shape)
            return X, Y
        ds = ds.map(_decode)

    return ds

'''
Run example
'''


def example(overwrite=False):
    '''
    Retrive tensorflow dataset for mnist.
    Save to given path with JPEG encoding.
    Then load into tensorflow dataset and read first X, Y pair.

    If overwrite is True, overwrite TFRecord if it exists, write if it exists.
    If overwrite is False, write if TFRecord does not exist.
    Then read TFRecord etc.
    '''
    import tensorflow_datasets as tfds
    import matplotlib.pyplot as plt

    mnist_test_iter = tfds.load(
        'mnist',
        split = 'test',
        # data_dir='/run/media/sam/7C6C-8EE6/Datasets/tensorflow_datasets/',
        batch_size = 1,
        shuffle_files = False,
        download = True,
        as_supervised = True)
    mnist_test_iter=mnist_test_iter.as_numpy_iterator()

    save_dir='./__temp__/'  # '/run/media/sam/7C6C-8EE6/Datasets/misc/tfds_mnist/'
    makedirs(save_dir, exist_ok = True)
    file_name='mnist.tfrecord'
    file_path=os.path.join(save_dir, file_name)

    if overwrite or not os.path.exists(file_path):
        iterable_to_tfrecord(
            mnist_test_iter,
            file_path,
            compression = 'ZLIB',
            encoding = 'JPEG',
            quality = 100,
            verbose = True)

    ds=tfrecord_to_tfds(
        file_path,
        output_shapes = ([28, 28, 1], []),
        output_dtypes = (tf.uint8, tf.int64),
        compression = 'ZLIB',
        encoding = 'JPEG')
    ds=ds.as_numpy_iterator()

    X, Y=next(ds)
    print(f'Label: {int(np.squeeze(Y))}')
    plt.imshow(X.reshape(28, 28))
    plt.show()


if __name__ == '__main__':
    example(overwrite = True)
