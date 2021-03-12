import sys
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import pyqtRemoveInputHook

import tracking
import elegant_matrix
import data_loader
import misc

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
        self.DoReconstruction.clicked.connect(self.do_reconstruction)
        self.SetStreaker.clicked.connect(self.streaker_set)

        self.tracker_initialized = False
        self.screen_calibrated = False
        self.streaker_calibrated = False
        self.streaker_is_set = False
        self.reconstruction_data_obtained = False

    def init_tracker(self):


        sum_checks = self.ImportFileCheck.isChecked()
        assert sum_checks == 1
        if self.ImportFileCheck.isChecked():
            magnet_file = self.ImportFile.text()
        else:
            raise NotImplementedError

        tmp_dir = os.path.expanduser(self.TmpDir.text())
        assert os.path.isdir(tmp_dir)
        assert os.access(tmp_dir, os.W_OK)
        elegant_matrix.set_tmp_dir(tmp_dir)

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
            key = self.ImportCalibrationKey.text()
            index = self.ImportCalibrationIndex.text()
            screen_data = data_loader.load_screen_data(filename, key, index)
            x_axis, projx = screen_data['x_axis'], screen_data['projx']

            median_proj = misc.get_median(projx)
            x0 = np.sum(x_axis*median_proj) / np.sum(median_proj)

            # Debugging
            if False:
                plt.figure()
                for proj in projx:
                    plt.plot(x_axis, proj)
                plt.axvline(x0)
                plt.show()
                import pdb; pdb.set_trace()

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
            streaker0_mean = float(self.StreakerDirect0.text())*1e-6
            streaker1_mean = float(self.StreakerDirect1.text())*1e-6

        self.streaker_calibrated = True
        self.streaker_mean = [streaker0_mean, streaker1_mean]
        print('Streaker calibrated: mean = %i, %i um' % (self.streaker_mean[0]*1e6, self.streaker_mean[1]*1e6))
        return self.streaker_mean

    def streaker_set(self):
        widgets = (self.SetStreakerDirectCheck, self.SetStreakerFromFileCheck, self.SetStreakerFromLiveCheck)
        self._check_check(widgets, 'Check set streaker checkmarks')

        if self.SetStreakerDirectCheck.isChecked():
            gaps = [float(self.StreakerGap0.text())*1e-3, float(self.StreakerGap1.text())*1e-3]
            # beam offset is negative of streaker offset
            beam_offsets = [-float(self.StreakerOffset0.text())*1e-3, -float(self.StreakerOffset1.text())*1e-3]
        elif self.SetStreakerFromFileCheck.isChecked():
            raise NotImplementedError
        elif self.SetStreakerFromLiveCheck.isChecked():
            raise NotImplementedError

        self.gaps = gaps
        self.beam_offsets = beam_offsets
        print('Streaker is set')
        self.streaker_is_set = True
        return gaps, beam_offsets

    def obtain_reconstruction_data(self):
        widgets = (self.ReconstructionDataLoadCheck,)
        self._check_check(widgets, 'Check obtain reconstruction data checkmarks')

        if self.ReconstructionDataLoadCheck.isChecked():
            filename = self.ReconstructionDataLoad.text().strip()
            key = self.ReconstructionDataLoadKey.text()
            index = self.ReconstructionDataLoadIndex.text()
            screen_data = data_loader.load_screen_data(filename, key, index)
            x_axis, projx = screen_data['x_axis'], screen_data['projx']

            if self.ReconstructionDataLoadUseSelect.currentText() == 'Median':
                median_projx = misc.get_median(projx)
            elif self.ReconstructionDataLoadUseSelect.currentText() == 'All':
                raise NotImplementedError

            meas_screen = tracking.ScreenDistribution(x_axis, median_projx)
        else:
            raise NotImplementedError

        self.reconstruction_data_obtained = True
        self.meas_screen = meas_screen
        print('Obtained reconstruction data')
        return meas_screen

    def do_reconstruction(self):
        conditions = [self.tracker_initialized, self.screen_calibrated, self.streaker_calibrated, self.streaker_is_set, self.reconstruction_data_obtained]
        conditions_string = ['tracker_initialized', 'screen_calibrated', 'streaker_calibrated', 'streaker_is_set', 'reconstruction_data_obtained']
        for condition, str_ in zip(conditions, conditions_string):
            if not condition:
                raise ValueError('Condition %s not met' % str_)

        start, stop, step = float(self.SigTfsStart.text()), float(self.SigTfsStop.text()), float(self.SigTfsStep.text())
        stop += 1e-3*step # assert that stop is part of array
        sig_t_range = np.arange(start, stop, step)*1e-15
        tt_halfrange = float(self.ProfileExtent.text())/2*1e-15
        meas_screen = self.meas_screen
        gaps = self.gaps
        beam_offsets = self.beam_offsets
        n_streaker = int(self.StreakerSelect.currentText())
        charge = float(self.Charge.text())*1e-12
        self_consistent = {'True': True, 'False': False}[self.SelfConsistentSelect.currentText()]
        kwargs_json = {
                'sig_t_range': sig_t_range,
                'tt_halfrange': tt_halfrange,
                'gaps': gaps,
                'beam_offsets': beam_offsets,
                'n_streaker': n_streaker,
                'charge': charge,
                'self_consistent': self_consistent,
                }
        kwargs_other = {
                'meas_screen': meas_screen,
                }
        kwargs = {**kwargs_json, **kwargs_other}
        self.tracker.find_best_gauss(**kwargs)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = StartMain()
    window.show()
    app.exec_()

# TBD
# - status, blocking etc

