from PyQt5 import QtWidgets
import PyQt5.QtCore as QtCore
from PyQt5.QtCore import pyqtSignal,QThread
from threading import Thread

#-------------
# thread handling class

class ScanCore(QThread):

    siginc  = pyqtSignal(int,int)  # signal for progress (gives two int arguments)
    sigterm = pyqtSignal(bool)       # signal for termination (gives bool argument)

    def __init__(self):
        QThread.__init__(self)
        self.abort=False

#----------------------------------
# all thread related functions

    def doAbort(self):    # this routine is called by other class to stop thread
        self.abort=True

    def run(self):
        self.abort=False
        Thread(target=self.runner).start()


    def runner(self):

        # define the loop
        nsteps=10
        isteps=0
        self.siginc.emit(0,nsteps)   # give signal that thread has started


        while(isteps < nsteps and not self.abort):
            # step 1 - set actuator
            # the core measurement - put your core routine in

            isteps += 1
            self.siginc.emit(isteps,nsteps) # signal for increment

        self.sigterm.emit(not self.abort)  # signal for end of measurement

#
#   put here the normal pyqt5 declaration etc.
#
#------------------------
# parent class for thread

class Scantest(QtWidgets.QMainWindow, Ui_ScantestGUI):
    def __init__(self):
        super(Scantest, self).__init__()
        self.setupUi(self)

        self.scan=ScanCore()
        # connect the GUI button to stop the progress
        self.UIStop.clicked.connect(self.scan.doAbort)
        # connect the GUI botton to start thread
        self.UIStart.clicked.connect(self.start)
        # connect the signals from the thread class to the handling function
        self.scan.siginc.connect(self.progress)
        self.scan.sigterm.connect(self.done)


    def start(self):
        self.scan.run()

    @QtCore.pyqtSlot(int)
    def done(self,val):
        if val:
            print('Scan ended')
        else:
            print('Scan aborted')

    #progress handler, incrementing progress bar
    @QtCore.pyqtSlot(int)
    def progress(self,val1,val2):
      print('Step', val1,'of',val2,'Steps done...')


