from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QWidget, QSlider, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QInputDialog
import numpy as np
import matplotlib.pyplot as pl
from cycler import cycler

__author__ = 'caro'


class ViewHandTuner(QWidget):

    def __init__(self, name_variables, precision_slds, lower_bounds, upper_bounds, slider_fun,
                 button_names, button_funs):
        super(ViewHandTuner, self).__init__()

        # initialize window
        self.setGeometry(350, 40, 1000, 550)
        self.setWindowTitle('HandTuner')

        # create left-hand side (slider and buttons)
        layout_left = QVBoxLayout()
        self.slds = list()
        for i in range(len(name_variables)):
            layout_sld, sld = self.create_slider(name_variables[i], i, precision_slds[i],
                                            lower_bounds[i], upper_bounds[i], slider_fun)
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
            self.imgs[i]['fig'] = pl.figure(figsize=(7, 4.2), tight_layout=True)
            self.imgs[i]['ax'] = self.imgs[i]['fig'].add_subplot(111)
            self.imgs[i]['qpixmap'] = QLabel()
            layout_right.addWidget(self.imgs[i]['qpixmap'])

        # create outer layout
        layout_outer = QHBoxLayout()
        layout_outer.addLayout(layout_left)
        layout_outer.addSpacing(50)
        layout_outer.addLayout(layout_right)
        self.setLayout(layout_outer)

        # start GUI
        self.show()

    def create_slider(self, label, idx, precision_sld, lower_bound, upper_bound, slider_fun):
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
        sld.value = QLabel('Value: 0')
        sld.value.setAlignment(QtCore.Qt.AlignBottom | QtCore.Qt.AlignRight)
        layout_disp.addWidget(sld.value)

        # slider layout
        layout_sld = QVBoxLayout()
        layout_sld.addLayout(layout_disp)
        layout_sld.addWidget(sld)

        # settings
        sld.setFocusPolicy(QtCore.Qt.NoFocus)
        sld.valueChanged.connect(slider_fun)
        sld.setRange(int(np.round(lower_bound/precision_sld, 0)),
                     int(np.round(upper_bound/precision_sld, 0)))
        return layout_sld, sld

    def create_button(self, label, button_fun):
        btn = QPushButton(label, self)
        btn.clicked.connect(button_fun)
        return btn

    def set_cmap(self, ax, cmap_name='jet'):
        cmap = pl.get_cmap(cmap_name)
        colors = cmap(np.linspace(0.1, 0.9, 10))
        ax.set_prop_cycle(cycler('color', colors))

    def plot_on_ax(self, img_id, x, y, color=None, linewidth=1.0, label=None, xlim=None, ylim=None,
                xlabel=None, ylabel=None, title=''):
        ax = self.imgs[img_id]['ax']
        ax.plot(x, y, color=color, linewidth=linewidth, label=label)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if label is not None:
            ax.legend()

    def plot_img(self, img_id):
        canvas = self.imgs[img_id]['fig'].canvas
        canvas.draw()
        size = canvas.size()
        im = QtGui.QImage(canvas.buffer_rgba(), size.width(), size.height(), QtGui.QImage.Format_ARGB32)
        self.imgs[img_id]['qpixmap'].setPixmap(QtGui.QPixmap(im))

    def clear_ax(self, img_id):
        self.imgs[img_id]['ax'].cla()
        self.set_cmap(self.imgs[img_id]['ax'])