import sys
import re
import numpy as np
from scipy.io import loadmat
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import pyqtRemoveInputHook

import tracking
import elegant_matrix
from h5_storage import loadH5Recursive

pyqtRemoveInputHook() # for pdb to work
re_time = re.compile('(\\d{4})-(\\d{2})-(\\d{2}):(\\d{2})-(\\d{2})-(\\d{2})')

# Of Tracker
# def __init__(self, magnet_file, timestamp, struct_lengths, n_particles, n_emittances, screen_bins, screen_cutoff, smoothen, profile_cutoff, len_screen, energy_eV='file', forward_method='matrix', compensate_negative_screen=True, optics0='default', quad_wake=True, bp_smoothen=0, override_quad_beamsize=False, quad_x_beamsize=(0., 0.)):

class StartMain(QtWidgets.QMainWindow):

    def __init__(self):
        super(StartMain, self).__init__()
        uic.loadUi('GUI.ui', self)
        self.InitializeTracker.clicked.connect(self.init_tracker)
        self.CalibrateScreen.clicked.connect(self.screen_calibration)
        self.CalibrateStreaker.clicked.connect(self.streaker_calibration)
        self.ObtainReconstructionData.clicked.connect(self.obtain_reconstruction_data)

        self.tracker_initialized = False
        self.screen_calibrated = False
        self.streaker_calibrated = False

    def init_tracker(self):
        sum_checks = self.ImportFileCheck.isChecked()
        assert sum_checks == 1
        if self.ImportFileCheck.isChecked():
            magnet_file = self.ImportFile.text()
        else:
            raise ValueError('Not implemented')

        time_match = re_time.match(self.ImportFileTime.text().strip())
        if time_match is None:
            raise ValueError('Wrong format of ImportFileTime: %s\nMust be: yyyy-MM-dd:hh-mm-ss' % self.ImportFileTime.text())
        args = list(time_match.groups())
        timestamp = elegant_matrix.get_timestamp(*args)
        struct_lengths = [float(self.StructLength1.text()), float(self.StructLength1.text())]
        n_particles = int(float(self.N_Particles.text()))
        n_emittances = [float(self.TransEmittanceX.text())*1e-9, float(self.TransEmittanceY.text())*1e-9]
        screen_bins = int(self.ScreenBins.text())
        screen_cutoff = float(self.ScreenCutoff.text())
        smoothen = float(self.ScreenSmoothen.text())*1e-6
        profile_cutoff = float(self.ProfileCutoff.text())
        len_screen = int(self.ScreenLength.text())
        quad_wake = self.UseQuadCheck.isChecked()
        bp_smoothen = float(self.ProfileSmoothen.text())*1e-15
        override_quad_beamsize = self.OverrideQuadCheck.isChecked()
        quad_x_beamsize = [float(self.QuadBeamsize1.text())*1e-6, float(self.QuadBeamsize2.text())*1e-6]

        self.tracker_kwargs = {
                'magnet_file': magnet_file,
                'timestamp': timestamp,
                'struct_lengths': struct_lengths,
                'n_particles': n_particles,
                'n_emittances': n_emittances,
                'screen_bins': screen_bins,
                'screen_cutoff': screen_cutoff,
                'smoothen': smoothen,
                'profile_cutoff': profile_cutoff,
                'len_screen': len_screen,
                'quad_wake': quad_wake,
                'bp_smoothen': bp_smoothen,
                'override_quad_beamsize': override_quad_beamsize,
                'quad_x_beamsize': quad_x_beamsize,
                }
        self.tracker = tracking.Tracker(**self.tracker_kwargs)
        print('Tracker successfully initialized')
        self.tracker_initialized = True
        return self.tracker

    @staticmethod
    def _check_check(widgets, errormessage):
        checks = [x.isChecked() for x in widgets]
        if sum(checks) != 1:
            raise ValueError(errormessage)

    def screen_calibration(self):
        widgets = (self.ScreenImportCalibrationCheck, self.ScreenDirectCalibrationCheck, self.ScreenLiveCalibrationCheck)
        self._check_check(widgets, 'Check screen calibration checkmarks')

        if self.ScreenImportCalibrationCheck.isChecked():
            filename = self.ImportCalibration.text().strip()
            if filename.endswith('.h5'):
                dict_ = loadH5Recursive(filename)
            elif filename.endswith('.mat'):
                dict_ = loadmat(filename)
            else:
                raise ValueError('Must be h5 or mat file. Is: %s' % filename)

            x_axis = dict_['x_axis']*1e-6
            key = self.ImportCalibrationKey.text()
            data = dict_[key]
            index = self.ImportCalibrationIndex.text()
            if index != 'None':
                index = int(index)
                data = data[index]

            assert len(data.shape) == 2
            n_projections = data.shape[0]

            all_mean_x = np.zeros(n_projections)
            for n_proj in range(n_projections):
                projx = data[n_proj]
                mean = np.sum(projx*x_axis)/np.sum(projx)
                all_mean_x[n_proj] = mean
            x0 = np.median(all_mean_x)

        elif self.ScreenDirectCalibrationCheck.isChecked():
            x0 = float(self.DirectCalibration.text())*1e-6

        elif self.ScreenLiveCalibrationCheck.isChecked():
            raise NotImplementedError()

        self.screen_calibrated = True
        self.x0 = x0
        print('Screen calibrated: x0 = %i um' % (self.x0*1e6))
        return x0

    def streaker_calibration(self):
        widgets = (self.StreakerDirectCheck,)
        self._check_check(widgets, 'Check streaker calibration checkmarks')

        if self.StreakerDirectCheck.isChecked():
            streaker0 = float(self.StreakerDirect.text())*1e-6

        self.streaker_calibrated = True
        self.streaker0 = streaker0
        print('Streaker calibrated: mean = %i um' % (self.streaker0*1e6))
        return self.streaker0

    def obtain_reconstruction_data(self):
        widgets = (self.ReconstructionDataLoadCheck,)
        self._check_check(widgets, 'Check  checkmarks')



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = StartMain()
    window.show()
    app.exec_()

