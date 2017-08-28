from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QWidget, QSlider, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QInputDialog
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from cycler import cycler

__author__ = 'caro'


class ViewHandTuner(QWidget):

    def __init__(self, name_variables, precision_slds, lower_bounds, upper_bounds, slider_fun,
                 button_names, button_funs, init_vals=None):
        super(ViewHandTuner, self).__init__()

        # initialize window
        self.setGeometry(350, 40, 1000, 550)
        self.setWindowTitle('HandTuner')

        # create left-hand side (slider and buttons)
        if init_vals is None:
            init_vals = lower_bounds
        layout_left = QVBoxLayout()
        self.slds = list()
        for i in range(len(name_variables)):
            layout_sld, sld = self.create_slider(name_variables[i], i, precision_slds[i],
                                            lower_bounds[i], upper_bounds[i], slider_fun, init_vals[i])
            self.slds.append(sld)
            layout_left.addLayout(layout_sld)
        layout_left.addSpacing(50)
        for i in range(len(button_names)):
            btn = self.create_button(button_names[i], button_funs[i])
            layout_left.addWidget(btn)

        # create right-hand side (images)
        layout_right = QVBoxLayout()
        self.imgs = list()
        for i in range(2):
            self.imgs.append(dict())
            self.imgs[i]['fig'] = pl.figure(tight_layout=True)
            self.imgs[i]['ax'] = self.imgs[i]['fig'].add_subplot(111)
            self.imgs[i]['canvas'] = FigureCanvas(self.imgs[i]['fig'])
            self.toolbar = NavigationToolbar(self.imgs[i]['canvas'], self, coordinates=True)
            layout_right.addWidget(self.imgs[i]['canvas'])
            layout_right.addWidget(self.toolbar)

        # create outer layout
        layout_outer = QHBoxLayout()
        layout_outer.addLayout(layout_left)
        layout_outer.addSpacing(50)
        layout_outer.addLayout(layout_right)
        self.setLayout(layout_outer)

        # start GUI
        self.show()

    def create_slider(self, label, idx, precision_sld, lower_bound, upper_bound, slider_fun, init_val=None):
        # create slider
        sld = QSlider(QtCore.Qt.Horizontal, self)
        sld.id = idx
        sld.precision = precision_sld

        # display label
        layout_disp = QHBoxLayout()
        sld.label = QLabel(label)
        sld.label.setAlignment(QtCore.Qt.AlignBottom | QtCore.Qt.AlignLeft)
        layout_disp.addWidget(sld.label)

        # display of current value
        sld.setRange(int(round(lower_bound/precision_sld)),
                     int(round(upper_bound/precision_sld)))
        if init_val is None:
            init_val = lower_bound
        sld.setValue(int(round(init_val / sld.precision)))
        sld.value = QLabel('Value: ' + str(init_val))
        sld.value.setAlignment(QtCore.Qt.AlignBottom | QtCore.Qt.AlignRight)
        layout_disp.addWidget(sld.value)

        # slider layout
        layout_sld = QVBoxLayout()
        layout_sld.addLayout(layout_disp)
        layout_sld.addWidget(sld)

        # settings
        sld.setFocusPolicy(QtCore.Qt.NoFocus)
        sld.valueChanged.connect(slider_fun)
        return layout_sld, sld

    def create_button(self, label, button_fun):
        btn = QPushButton(label, self)
        btn.clicked.connect(button_fun)
        return btn

    def set_cmap(self, ax, cmap_name='jet'):
        cmap = pl.get_cmap(cmap_name)
        colors = cmap(np.linspace(0.1, 0.9, 10))
        ax.set_prop_cycle(cycler('color', colors))

    def plot_on_ax(self, img_id, x, y, color=None, linewidth=1.0, label=None, xlabel=None, ylabel=None, title=''):
        ax = self.imgs[img_id]['ax']
        ax.plot(x, y, color=color, linewidth=linewidth, label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if label is not None:
            ax.legend(loc='upper right')

    def plot_img(self, img_id):
        self.imgs[img_id]['fig'].canvas.draw()

    def clear_ax(self, img_id):
        self.imgs[img_id]['ax'].cla()
        self.set_cmap(self.imgs[img_id]['ax'])