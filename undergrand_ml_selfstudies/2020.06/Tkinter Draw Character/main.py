import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
import pickle
from scipy.signal import convolve2d

# Prevent tensorflow-GPU RAM issues
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


class CharRecognitionCanvas():
    """
    nn_objects -> {"lb": LabelBinarizer, "model": tf.keras.Model}
        - The nn should be set to recognize 28x28x1 hand written characters.
    """

    FILTER = np.ones((2, 2)) / 2.3

    def __init__(self, base_size_length, multiplier, line_width, nn_objects: dict, convolutions=2):
        self.root = Tk()
        self._init_canvas(
            base_size_length * multiplier,
            base_size_length * multiplier
        )
        self.line_width = line_width
        self.brush_up(None)

        self.multiplier = multiplier
        self._init_canvas_matrix(base_size_length)

        self.lb = nn_objects["lb"]
        self.model = nn_objects["model"]

        self.convolutions = convolutions

    def _init_canvas(self, height, width):
        self.canvas = Canvas(self.root, width=width, height=height)
        self.canvas.pack()
        self.canvas.bind("<Motion>", self.motion)
        self.canvas.bind("<ButtonPress-1>", self.brush_down)
        self.canvas.bind("<ButtonRelease-1>", self.brush_up)
        self.canvas.bind("<Double-Button-1>", self.b1_double)

    def _init_canvas_matrix(self, side_length):
        self.canvas_matrix = np.zeros((side_length, side_length))

    def _update_matrix(self):
        assert((self.x_old, self.y_old) != (None, None))

        self.canvas_matrix[
            max(
                0,
                min(
                    self.canvas_matrix.shape[0],
                    self.y_old // self.multiplier
                )
            ),
            max(
                0,
                min(
                    self.canvas_matrix.shape[1],
                    self.x_old // self.multiplier
                )
            )
        ] = 255.0

    def brush_down(self, event):
        self.is_brush_down = True

    def brush_up(self, event):
        self.is_brush_down = False
        self.x_old, self.y_old = None, None

    def b1_double(self, event):
        plt.imshow(self.get_convolved())
        plt.show()
        self.canvas_matrix = np.zeros(self.canvas_matrix.shape)
        self.canvas.delete("all")

    def motion(self, event):
        if self.is_brush_down:
            if (self.x_old, self.y_old) != (None, None):
                event.widget.create_line(
                    self.x_old, self.y_old,
                    event.x, event.y,
                    smooth=True, fill="black", width=self.multiplier
                )
            self.x_old, self.y_old = event.x, event.y
            self._update_matrix()

        else:
            convolved = self.get_convolved() / 255.0
            pred = self.model.predict(convolved.reshape((1, -1)))
            pred = np.where(pred == np.amax(pred), 1, 0)
            print(self.lb.inverse_transform(pred))

    def get_convolved(self):
        convolved_matrix = self.canvas_matrix
        for i in range(self.convolutions):
            convolved_matrix = convolve2d(
                convolved_matrix, CharRecognitionCanvas.FILTER, mode='same')
        convolved_matrix = np.where(
            convolved_matrix > 255, 255, convolved_matrix)
        return convolved_matrix

    def run(self):
        self.root.mainloop()


def load_lb_and_model():
    f = open("./__temp__/pickel/lb", "rb")
    lb = pickle.load(f)

    model = tf.keras.models.load_model("./__temp__/models/mnist_save1")
    return lb, model


def main():
    lb, model = load_lb_and_model()
    nn_objects = {"lb": lb, "model": model}

    canvas = CharRecognitionCanvas(
        28,
        multiplier=20,
        line_width=20,
        nn_objects=nn_objects
    )

    canvas.run()

if __name__ == "__main__":
    main()
