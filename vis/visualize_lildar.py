import pyqtgraph
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import numpy as np
from argparse import ArgumentParser
from functools import partial

parser = ArgumentParser()
parser.add_argument('filename', type=str, nargs='+')
parser.add_argument('--no_intensity', action='store_true')
args = parser.parse_args()

app = QtGui.QApplication([])

central = QtGui.QWidget()
central.resize(640, 480)
central.setWindowTitle('visualize LiDAR')
layout = QtGui.QVBoxLayout()
central.setLayout(layout)

w = gl.GLViewWidget()
w.opts['distance'] = 20
layout.addWidget(w)

g = gl.GLGridItem()
w.addItem(g)

a = gl.GLAxisItem()
w.addItem(a)

def onStateChange(sp, check, i):
    sp.setVisible(check.isChecked())

def onColorChange(sp, wc):
    sp.setData(color=wc.color('float'))

cs = [(1,0,0,1), (0,1,0,1), (0,0,1,1)]
for i, fn in enumerate(args.filename):
    colorf = cs[i%len(cs)]
    color255 = [int(c*255) for c in colorf]
    if args.no_intensity:
        data = np.fromfile(fn, dtype=np.float32).reshape(-1, 3)
    else:
        data = np.fromfile(fn, dtype=np.float32).reshape(-1, 4)
    sp = gl.GLScatterPlotItem(pos=data[:,:3], size=0.1, color=colorf)
    w.addItem(sp)
    layout2 = QtGui.QHBoxLayout()
    layout.addLayout(layout2)
    check = QtGui.QCheckBox(fn)
    check.toggle()
    check.stateChanged.connect(partial(onStateChange, sp, check))
    wc = pyqtgraph.ColorButton(color=color255)
    wc.sigColorChanged.connect(partial(onColorChange, sp))
    wc.setMaximumWidth(40)
    layout2.addWidget(check)
    layout2.addWidget(wc)

central.show()

if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

