'''
Objects for writing to and reading from TFRecord files.

Also provide examples of using these objects.
'''
import numpy as np
import os.path
import tensorflow as tf


class TFRecordManager(object):

    def __init__(self):
        self._X_dtype_feature = None
        self._Y_dtype_feature = None

    @staticmethod
    def infer_parse_dtype(dtype):
        if isinstance(dtype, np.dtype):
            dtype = dtype.type

        if dtype in (np.float16, np.float32, np.float64,
                     tf.float16, tf.float32, tf.float64, float):
            return tf.float64
        if dtype in (np.uint8, np.uint16, np.uint32, np.uint64,
                     np.int8, np.int16, np.int32, np.int64,
                     tf.uint8, tf.uint16, tf.uint32, tf.uint64,
                     tf.int8, tf.int16, tf.int32, tf.int64,
                     tf.bool, int):
            return tf.int64
        if dtype in (bytes, tf.string):
            return tf.string
        raise ValueError(f'Cannot infer tensorflow decoding dtype from dtype {dtype}.')

    def serialize_feature(self, X, Y):
        '''
        Serialize (already encoded if desired) feature.
        '''
        if self._X_dtype_feature is self._Y_dtype_feature is None:
            def _dtype_feature(vector):
                if isinstance(vector, np.ndarray):
                    # Is an ndarray
                    dtype = vector.dtype.type
                elif tf.is_tensor(vector):
                    # Is a tensor
                    dtype = vector.dtype
                else:
                    # Eg: Is a byte string
                    dtype = type(vector)

                if dtype in (np.float16, np.float32, np.float64,
                             tf.float16, tf.float32, tf.float64, float):
                    return lambda vector_: tf.train.Feature(float_list=tf.train.FloatList(value=vector_))
                if dtype in (np.uint8, np.uint16, np.uint32, np.uint64,
                             np.int8, np.int16, np.int32, np.int64,
                             tf.uint8, tf.uint16, tf.uint32, tf.uint64,
                             tf.int8, tf.int16, tf.int32, tf.int64,
                             tf.bool, int):
                    return lambda vector_: tf.train.Feature(int64_list=tf.train.Int64List(value=vector_))
                if dtype is bytes:
                    return lambda vector_: tf.train.Feature(bytes_list=tf.train.BytesList(value=[vector_]))
                raise ValueError(f'Given vector is of incompadible type: "{dtype}."')

            self._X_dtype_feature = _dtype_feature(X)
            self._Y_dtype_feature = _dtype_feature(Y)

        feature = {
            'X': self._X_dtype_feature(X),
            'Y': self._Y_dtype_feature(Y)
        }
        return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()


class NumpyTFRecordManager(TFRecordManager):
    '''
    A class containing helpful methods for saving ndarray data to tfrecords.
    Please only use it to parse tfrecords that it,
    (or another correctly initialized instance) has created itself,
    as it assumes a specific saved feature format.

    Arguments:
        file_path: String. File path to save and load data from.
        compression_type: None or String. Compression type of TFRecord at path.
            - None: No Compression
            - String: Either "GZIP" or "ZLIB"
        encoding_type: None or String. Must be None if X aren't images.
            - None: No image encoding will be done on X -> X will be flattened.
            - String: "JPEG"
    '''

    def __init__(
            self,
            file_path,
            compression_type=None,
            encoding_type=None):
        super().__init__()
        if encoding_type not in (None, 'JPEG'):
            raise ValueError(('Currently supported image encodings: '
                              '"JPEG", None (default).'))
        self._file_path = file_path
        self._compression_type = compression_type
        self._encoding_type = encoding_type

    def encoded_feature(self, X, Y):
        if self._encoding_type is None:
            return X.flatten(), Y.flatten()
        elif self._encoding_type == 'JPEG':
            X = tf.convert_to_tensor(X, dtype=tf.uint8)
            # Ensure X has shape (height, width, channels)
            # Rather than (1, height, width, channels):
            X = tf.reshape(X, X.shape[-3:])
            X = tf.io.encode_jpeg(X, optimize_size=True).numpy()
            return X, Y
        else:
            raise ValueError(f'Unknown encoding type: {self._encoding_type}.')

    def iterable_to_tfrecord(
            self,
            np_iterable,
            verbose=False,
            interval=100):
        '''
        Save all X, Y pairs from np_iterable to file_path in tfrecords format.
        '''
        if verbose:
            basename = os.path.basename(self._file_path)
            idx = 0

        with tf.io.TFRecordWriter(self._file_path, options=self._compression_type) as writer:
            for X, Y in np_iterable:
                if verbose:
                    if idx != 0 and idx % interval == 0:
                        print(f'Wrote {idx} images to {basename}')
                    idx += 1
                X_, Y_ = self.encoded_feature(X, Y)
                feature = self.serialize_feature(X_, Y_)
                writer.write(feature)

        if verbose:
            print(f'Done! (Wrote {idx} images to {basename})')

    def tfrecord_to_tfds(self, output_shapes, output_dtypes, file_path=None):
        X_shape, Y_shape = output_shapes
        X_dtype, Y_dtype = output_dtypes

        if file_path is None:
            file_path = self._file_path

        raw_ds = tf.data.TFRecordDataset(
            file_path, compression_type=self._compression_type)
        if self._encoding_type is None:
            feature_format = {
                'X': tf.io.FixedLenFeature(
                    X_shape,
                    super().infer_parse_dtype(X_dtype)),
                'Y': tf.io.FixedLenFeature(
                    Y_shape,
                    super().infer_parse_dtype(Y_dtype))
            }
        elif self._encoding_type == 'JPEG':
            feature_format = {
                'X': tf.io.VarLenFeature(tf.string),
                'Y': tf.io.FixedLenFeature(
                    Y_shape,
                    super().infer_parse_dtype(Y_dtype))
            }
        else:
            raise ValueError(f'Unknown encoding type: {self._encoding_type}.')

        def _proto_to_tensor(example_proto):
            parsed = tf.io.parse_single_example(example_proto, feature_format)
            X = parsed['X']
            Y = parsed['Y']
            return X, Y

        ds = raw_ds.map(_proto_to_tensor)
        if self._encoding_type == 'JPEG':
            def _decode(X, Y):
                X = tf.reshape(tf.sparse.to_dense(X), [])
                X = tf.io.decode_jpeg(X, channels=X_shape[-1])
                X = tf.reshape(tf.cast(X, X_dtype), X_shape)
                return X, Y
            ds = ds.map(_decode)

        return ds


def example(overwrite_if_exist=False):
    '''
    Retrive tensorflow dataset for mnist.
    Save to given path with JPEG encoding.
    Then load into tensorflow dataset and read first X, Y pair.
    '''
    import tensorflow_datasets as tfds
    import matplotlib.pyplot as plt

    mnist_test_iter = tfds.load(
        'mnist',
        split='test',
        # data_dir='/run/media/sam/7C6C-8EE6/Datasets/tensorflow_datasets/',
        batch_size=1,
        shuffle_files=False,
        download=False,
        as_supervised=True)
    mnist_test_iter = mnist_test_iter.as_numpy_iterator()

    save_prefix = './__temp__/' # '/run/media/sam/7C6C-8EE6/Datasets/misc/tfds_mnist/'
    file_name = 'mnist.tfrecord'
    file_path = os.path.join(save_prefix, file_name)

    tfr_manager = NumpyTFRecordManager(
        file_path=file_path,
        compression_type='ZLIB',
        encoding_type='JPEG')

    if overwrite_if_exist or not os.path.exists(file_path):
        tfr_manager.iterable_to_tfrecord(mnist_test_iter, verbose=True)

    ds = tfr_manager.tfrecord_to_tfds(
        output_shapes=([28,28,1], []),
        output_dtypes=(tf.uint8, tf.int64))
    ds = ds.as_numpy_iterator()

    X, Y = next(ds)
    print(f'Image: {X}, type: {X.dtype}, shape: {X.shape}')
    print(f'Label: {Y}, type: {Y.dtype}, shape: {Y.shape}')
    plt.imshow(X.reshape(28,28))
    plt.show()


if __name__ == '__main__':
    example()
