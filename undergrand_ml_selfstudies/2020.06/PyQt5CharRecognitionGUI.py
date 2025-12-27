import sys
import numpy as np
from PyQt5 import QtGui, QtWidgets, QtCore
from scipy.signal import convolve2d


def _greyscale_to_hex(intensity):
    intensity = int(max(0., min(intensity, 255.)))
    return '#' + (f'{intensity:02x}' * 3)


def _rgb_to_hex(r, g, b):
    r = int(max(0., min(r, 255.)))
    g = int(max(0., min(g, 255.)))
    b = int(max(0., min(b, 255.)))
    return f'#{r:02x}{g:02x}{b:02x}'


class NumpyPixelCanvas(QtWidgets.QLabel):
    """
    A QLabel continaing a QPixmap for pixel art which kept track of in a 2darray

    Arguments:
        np_repr: A 2d array that the dimentions of the canvas is based upon.
        multiplier: integer determining how many times larger the canvas will be
            than the rows/cols of the 2darray.
            - height given by np_repr.shape[0] * multiplier
            - width  given by np_repr.shape[1] * multiplier
    """

    def __init__(self, np_repr: np.ndarray, multiplier: int, painted_val=255.):
        super().__init__()

        assert(np_repr.ndim == 2)

        self.np_repr = np_repr
        self.multiplier = multiplier
        self.painted_val = painted_val

        w = np_repr.shape[0] * multiplier
        h = np_repr.shape[1] * multiplier

        pixmap = QtGui.QPixmap(w, h)
        self.setPixmap(pixmap)

        self.pen_color = QtGui.QColor('#ffffff')
        self.history = []
        self.clearCanvas()

    def updateRepr(self, row, col):
        if (0 <= row < self.np_repr.shape[0] and
                0 <= col < self.np_repr.shape[1]):
            self.np_repr[row, col] = self.painted_val

    def setRepr(self, np_repr: np.ndarray):
        assert(self.np_repr.shape == np_repr.shape)
        np_repr[np_repr > self.painted_val] = self.painted_val
        self.np_repr = np_repr

    def getRepr(self):
        return self.np_repr

    def mouseMoveEvent(self, e):
        qPainter = QtGui.QPainter(self.pixmap())
        x = round(e.x() / self.multiplier) * self.multiplier
        y = round(e.y() / self.multiplier) * self.multiplier
        s = self.multiplier
        qPainter.fillRect(x, y, s, s, self.pen_color)
        qPainter.end()
        self.update()

        self.updateRepr(y // self.multiplier, x // self.multiplier)

    def mouseReleaseEvent(self, e):
        self.updateHistory()

    def updateHistory(self):
        self.history.append(self.np_repr.copy())

    def undo(self):
        if self.history:
            self.history.pop()
        if self.history:
            self.setRepr(self.history.pop())
            self.renderRepr()
        else:
            self.clearCanvas()

    def renderRepr(self):
        self.clearCanvas(clear_repr=False)
        qPainter = QtGui.QPainter(self.pixmap())
        for row, col in np.argwhere(self.np_repr != 0):
            color = QtGui.QColor(_greyscale_to_hex(self.np_repr[row, col]))
            x = col * self.multiplier
            y = row * self.multiplier
            qPainter.fillRect(x, y, self.multiplier, self.multiplier, color)
        qPainter.end()
        self.update()
        self.updateHistory()

    def clearCanvas(self, clear_repr=True):
        self.pixmap().fill(QtGui.QColor('#000000'))
        self.update()
        if clear_repr:
            self.np_repr = np.zeros_like(self.np_repr)
            self.updateHistory()


class Button(QtWidgets.QPushButton):
    """A child of QPushButton class that can be initialized in a clean manner"""

    def __init__(self, *args, color=None, size=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.color = color
        if color is not None:
            self.setStyleSheet(f'background-color: {color};')
            self.color = color

        if size is not None:
            if not isinstance(size, QtCore.QSize):
                size = QtCore.QSize(*size)
            self.setFixedSize(size)


class Slider(QtWidgets.QSlider):
    """
    A child of QSlider class that can be initialized in a clean manner
    Supports float intervals
    """

    def __init__(self, min, max, *args, interval=None, horizontal=False, onValueChanged=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._interval = interval
        self._onValueChanged = onValueChanged
        if interval is None:
            self.setMinimum(min)
            self.setMaximum(max)
        else:
            self.setMinimum(int(np.floor(min / interval)))
            self.setMaximum(int(np.ceil(max / interval)))
            self.setTickInterval(1)
        if horizontal:
            self.setOrientation(QtCore.Qt.Horizontal)
        if onValueChanged is not None:
            self.valueChanged.connect(onValueChanged)

    def value(self):
        if self._interval is None:
            return super().value()
        return round(super().value() * self._interval, 2)


class LabeledSlider(QtWidgets.QWidget):
    """
    A Widget containing a QHBoxLayout that includes a label and a slider
    The label displays prefix + the slider's current value.
    """

    def __init__(self, *args, prefix='', **kwargs):
        super().__init__()

        hbox = QtWidgets.QHBoxLayout()
        self.setLayout(hbox)

        self.prefix = prefix

        self.label = QtWidgets.QLabel()
        self.slider = Slider(*args, onValueChanged=self.updateLabel, **kwargs)
        self.updateLabel()

        hbox.addWidget(self.label)
        hbox.addWidget(self.slider)

    def updateLabel(self):
        self.label.setText(f'{self.prefix}{self.slider.value()}')

    def value(self):
        return self.slider.value()


class ListWidget(QtWidgets.QWidget):
    """
    A widget continaing a QListWidget which allows having a minimum size.
    """

    def __init__(self, *args, minw=None, minh=None, **kwargs):
        super().__init__()
        if None not in (minw, minh):
            self.setMinimumSize(minw, minh)

        self.hbox = QtWidgets.QHBoxLayout()
        self.setLayout(self.hbox)

        self.list_widget = QtWidgets.QListWidget()
        self.hbox.addWidget(self.list_widget)

    def addItem(self, item):
        self.list_widget.addItem(item)

    def addItems(self, item):
        self.list_widget.addItems(item)

    def clear(self):
        self.list_widget.clear()


class DrawingRecognitionTestingGUI(QtWidgets.QMainWindow):
    """
    GUI for testing prediction functions.

    Arugments:
        np_repr: 2darray, used as input for NumpyPixelCanvas initialization.
        multiplier: int, used as input for NumpyPixelCanvas initialization.
        prediction_function:
            function or callable used to convert np_repr of canvas to some prediction.
    """

    def __init__(self, np_repr, multiplier, prediction_function=None):
        super().__init__()

        self.f = prediction_function

        self.canvas = NumpyPixelCanvas(np_repr, multiplier)

        central_widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout()
        central_widget.setLayout(main_layout)

        self.addDrawingArea(main_layout)
        self.addEvaluationArea(main_layout)

        self.setCentralWidget(central_widget)

    def addDrawingArea(self, main_layout):
        vbox = QtWidgets.QVBoxLayout()

        clr = Button('Clear', color='Red')
        clr.pressed.connect(self.canvas.clearCanvas)

        vbox.addWidget(self.canvas)
        vbox.addWidget(clr)

        main_layout.addLayout(vbox)

    def addEvaluationArea(self, main_layout):
        vbox = QtWidgets.QVBoxLayout()
        filter_shape_slider = LabeledSlider(
            2, 4, interval=1, horizontal=True, prefix='Filter Shape (nxn): ')
        filter_value_slider = LabeledSlider(
            0, 1, interval=0.01, horizontal=True, prefix='Filter values: ')

        def _convolveCanvasRepr():
            repr = self.canvas.getRepr()
            filter_shape = filter_shape_slider.value()
            filter = np.full((filter_shape, filter_shape),
                             fill_value=filter_value_slider.value())
            convolved_repr = convolve2d(
                repr, filter, mode='same', fillvalue=0.)
            self.canvas.setRepr(convolved_repr)
            self.canvas.renderRepr()

        convolve = Button('Convolve', color='DarkCyan')
        convolve.pressed.connect(_convolveCanvasRepr)

        undo = Button('Undo', color='LightSlateGrey')
        undo.pressed.connect(self.canvas.undo)

        list_widget = ListWidget(minw=100, minh=100)

        def _addPredictions():
            list_widget.clear()
            x = self.canvas.getRepr()
            preds = self.f(x)
            list_widget.addItems(preds)

        predict = Button('Predict', color='Green')
        if self.f is not None:
            predict.pressed.connect(_addPredictions)

        vbox.addWidget(filter_shape_slider)
        vbox.addWidget(filter_value_slider)
        vbox.addWidget(convolve)
        vbox.addWidget(undo)
        vbox.addWidget(list_widget)
        vbox.addWidget(predict)
        main_layout.addLayout(vbox)


class predictionCallable(object):
    """
    A callable that can be treated as a prediction function

    Arguments:
        preprocessing_function: None or function:
            None: no preprocessing
            function: will be applied to input of __call__ before being passed to function.
        function: A function that converts the
            perhaps preprocessed input to some output.
        postprocessing_function: None or function:
            None: no postprocessing
            function: will be tweak ouput if needed.
    """

    def __init__(self, preprocessing_function, function, postprocessing_function=None):
        self.pre_f = preprocessing_function
        self.f = function
        self.post_f = postprocessing_function

    def __call__(self, X):
        if self.pre_f is not None:
            X = self.pre_f(X)
        X = self.f(X)
        if self.post_f is not None:
            X = self.post_f(X)
        return X


def example():
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # Apparently smth is wrong with my CuDNN installation
    # so this is done so that NN is run on CPU.
    import tensorflow as tf

    num = {i: i for i in range(10)}
    lower = {i + 36: chr(97 + i) for i in range(26)}
    upper = {i + 10: chr(65 + i) for i in range(26)}
    d = {**num, **lower, **upper}

    def preprocess(X):
        return X.T.reshape(1, 28, 28, 1) / 255.

    def postprocess(Y):
        Y = np.squeeze(Y)

        def ind_to_str(ind):
            lst = [d[i] for i in ind]
            return map(str, lst)
            # return '\n'.join(lst)
        TOP = 10
        ind = np.argpartition(Y, -TOP)[-TOP:]
        ind = ind[np.argsort(Y[ind])[::-1]]
        return ind_to_str(ind)

    model = tf.keras.models.load_model('emnist_CNN')

    pred_callable = predictionCallable(
        preprocessing_function=preprocess,
        function=model.predict,
        postprocessing_function=postprocess
    )

    np_repr = np.full((28, 28), fill_value=0.)
    app = QtWidgets.QApplication(sys.argv)
    window = DrawingRecognitionTestingGUI(
        np_repr, multiplier=30, prediction_function=pred_callable)
    window.show()
    app.exec_()


def main():
    example()

if __name__ == '__main__':
    main()
