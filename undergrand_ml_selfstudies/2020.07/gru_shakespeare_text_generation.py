'''
This GRU example follows a tutorial found here:
https://www.tensorflow.org/tutorials/text/text_generation
The given tutorial format is more notebook-focused.
I have changed the code slightly to make it more-understandable (for me).
'''

import os.path
import numpy as np
import tensorflow as tf
import time

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import GRU
from tensorflow.keras.models import Sequential

BATCH_SIZE = 64
EPOCHS = 10

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000


def get_dataset(with_info=False, sequence_length=100):
    def get_text(path='./data/shakespeare.txt'):
        abspath = os.path.abspath(path)
        path_to_file = tf.keras.utils.get_file(
            abspath, 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
        with open(path_to_file, 'rb') as f:
            return f.read().decode()

    def chunk_to_supervised(chunk):
        X = chunk[:-1]
        Y = chunk[1:]
        return X, Y

    text = get_text()
    vocab = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(vocab)}
    idx2char = np.array(vocab)
    text_as_int = np.array([char2idx[c] for c in text])

    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    sequences = char_dataset.batch(sequence_length + 1, drop_remainder=True)
    dataset = sequences.map(chunk_to_supervised)

    if with_info:
        info = {
            'vocab': vocab,
            'char2idx': char2idx,
            'idx2char': idx2char,
            'text_as_int': text_as_int
        }
        return dataset, info
    return dataset


def build_model(vocab_size, embedding_dim, gru_units, batch_size):
    model = Sequential()
    model.add(Embedding(
        vocab_size, embedding_dim,
        batch_input_shape=[batch_size, None]))
    model.add(GRU(
        gru_units, return_sequences=True, stateful=True,
        recurrent_initializer='glorot_uniform'))
    model.add(Dense(vocab_size))
    return model


def generate_text(model, start_string, char2idx, idx2char, num_generate=1000, temperature=1.0):
    '''
    Given RNN model and start string, generate num_generate further characters.
    Use temperature to waiver redictability of output text.

    "Higher temperatures results in more surprising text.
    Experiment to find the best setting."
     - https://www.tensorflow.org/tutorials/text/text_generation

    '''
    # Converting our start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the character returned by
        # the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(
            predictions, num_samples=1)[-1, 0].numpy()

        # We pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))


def main():
    ds, info = get_dataset(with_info=True)
    vocab_size = len(info['vocab'])
    char2idx = info['char2idx']
    idx2char = info['idx2char']

    ds = ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    # The embedding dimension (dimention input integer char data is encoded to)
    embedding_dim = 256
    # Number of GRU memory units
    gru_units = 1024

    checkpoint_dir = './__temp__/training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_{epoch}')
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

    if (os.path.exists(checkpoint_dir) and
            input(('Checkpoint directory already found, ') +
                  ('try to load an existing model?')) in ('y', 'Y', '')):
        model = build_model(vocab_size, embedding_dim, gru_units, batch_size=1)
        model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
        print(model.summary())

    else:
        model = build_model(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            gru_units=gru_units,
            batch_size=BATCH_SIZE)
        print(model.summary())
        opt = 'adam'
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer=opt, loss=loss_fn)
        history = model.fit(dataset, epochs=EPOCHS,
                            callbacks=[checkpoint_callback])

    print(generate_text(model, start_string=u'Hola signiorita',
                        char2idx=char2idx, idx2char=idx2char))


if __name__ == '__main__':
    main()
