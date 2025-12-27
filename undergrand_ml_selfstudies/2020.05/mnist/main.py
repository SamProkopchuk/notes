import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


def main():
    print("Getting training data")
    (X_train_orig, Y_train_orig), (X_test_orig, Y_test_orig) = tfds.as_numpy(tfds.load(
        'mnist',
        split=['train', 'test'],
        batch_size=-1,
        as_supervised=True,
    ))

    X_train = np.reshape(X_train_orig, (60000, 28*28))
    X_test = np.reshape(X_test_orig, (10000, 28*28))
    X_test, X_train = X_test/255.0, X_train/255.0

    lb = LabelBinarizer()
    Y_train = lb.fit_transform(Y_train_orig)
    Y_test = lb.transform(Y_test_orig)

    X_train, Y_train = shuffle(X_train, Y_train)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(784),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(10),
    ])

    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    model.compile(
        optimizer='adam',
        loss=loss_fn,
        metrics=['categorical_accuracy']
    )

    model.fit(X_train, Y_train, epochs=12)
    
    model.evaluate(X_test, Y_test)

    print(sess.run(pred, feed_dict={x: tst_x}))


if __name__ == "__main__":
    main()