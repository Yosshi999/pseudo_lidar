from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import numpy as np
from argparse import ArgumentParser
from functools import partial

parser = ArgumentParser()
parser.add_argument('filename', type=str, nargs='+')
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

cs = [(1,0,0,1), (0,1,0,1), (0,0,1,1)]
for i, fn in enumerate(args.filename):
    data = np.fromfile(fn, dtype=np.float32).reshape(-1, 4)
    sp = gl.GLScatterPlotItem(pos=data[:,:3], size=0.1, color=cs[i%len(cs)])
    w.addItem(sp)
    check = QtGui.QCheckBox(fn)
    check.toggle()
    check.stateChanged.connect(partial(onStateChange, sp, check))
    layout.addWidget(check)

central.show()

if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

