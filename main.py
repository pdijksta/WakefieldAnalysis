import sys
import os
import re
from datetime import datetime
import numpy as np
import PyQt5.Qt
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import pyqtRemoveInputHook
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
import matplotlib.pyplot as plt

import config
import tracking
import elegant_matrix
import data_loader
import misc
import analysis
#import gaussfit
import h5_storage
import myplotstyle as ms

#TODO
#
# - elog
# - non blocking daq
# - sort out daq pyscan_result_to_dict
# - add info of beamsize with / without assumed screen resolution
# - debug delay after using BsreadPositioner
# - noise reduction from the image
# - add these options to GUI
# - pedestal?
# - lasing
# - add tilt option


try:
    import daq
    always_dryrun = False
except ImportError:
    print('Cannot import daq. Always dry_run True')
    always_dryrun = True
    daq = None

ms.set_fontsizes(8)

# For debug purposes, set this to true
qt5_plot = False

if qt5_plot:
    matplotlib.use('Qt5Agg')
else:
    plt.ion() # Interactive mode
pyqtRemoveInputHook() # for pdb to work
re_time = re.compile('(\\d{4})-(\\d{2})-(\\d{2}):(\\d{2})-(\\d{2})-(\\d{2})')

class StartMain(QtWidgets.QMainWindow):

    def __init__(self):
        super(StartMain, self).__init__()
        uic.loadUi('GUI.ui', self)

        self.DoReconstruction.clicked.connect(self.do_reconstruction)
        self.SaveData.clicked.connect(self.save_data)
        self.LoadData.clicked.connect(self.load_data)
        self.CloseAll.clicked.connect(self.clear_rec_plots)
        self.ObtainStreakerFromLive.clicked.connect(self.obtain_streaker_settings_from_live)
        self.CalibrateStreaker.clicked.connect(self.calibrate_streaker)
        self.ClearCalibPlots.clicked.connect(self.clear_calib_plots)
        self.LoadCalibration.clicked.connect(self.load_calibration)
        self.CalibrateScreen.clicked.connect(self.calibrate_screen)
        self.ClearScreenPlots.clicked.connect(self.clear_screen_plots)
        self.SetXEmittance.clicked.connect(self.set_x_emittance)
        self.LoadScreenCalibration.clicked.connect(self.load_screen_calibration)
        self.ObtainReconstructionData.clicked.connect(self.obtain_reconstruction)

        self.StreakerSelect.activated.connect(self.update_streaker)
        self.BeamlineSelect.activated.connect(self.update_streaker)
        self.update_streaker()

        self.analysis_obj = analysis.Reconstruction()
        self.other_input = {}
        self.tracker_initialized = False


        if qt5_plot:

            def get_new_tab(fig, title):
                new_tab = QtWidgets.QWidget()
                layout = PyQt5.Qt.QVBoxLayout()
                new_tab.setLayout(layout)
                canvas = FigureCanvasQTAgg(fig)
                toolbar = NavigationToolbar2QT(canvas, self)
                layout.addWidget(canvas)
                layout.addWidget(toolbar)
                tab_index = self.tabWidget.addTab(new_tab, title)
                return tab_index, canvas

            fig, self.reconstruction_plot_handles = analysis.reconstruction_figure()
            self.rec_plot_tab_index, self.rec_canvas = get_new_tab(fig, 'Rec plots')

            fig, self.streaker_calib_plot_handles = analysis.streaker_calibration_figure()
            self.streaker_calib_plot_tab_index, self.streaker_calib_canvas = get_new_tab(fig, 'Cal. plots')

            fig, self.screen_calib_plot_handles = analysis.screen_calibration_figure()
            self.screen_calib_plot_tab_index, self.screen_calib_canvas = get_new_tab(fig, 'Screen plots')

        else:
            self.reconstruction_plot_handles = None
            self.streaker_calib_plot_handles = None
            self.screen_calib_plot_handles = None


        #meas_screen = self.obtain_reconstruction_data()
        #import matplotlib.pyplot as plt
        #plt.figure()
        #sp = plt.subplot(1,1,1)
        #meas_screen.plot_standard(sp)
        #plt.show()


    def clear_rec_plots(self):
        analysis.clear_reconstruction(*self.reconstruction_plot_handles)
        self.rec_canvas.draw()
        print('Cleared reconstruction plot')

    def clear_calib_plots(self):
        analysis.clear_streaker_calibration(*self.streaker_calib_plot_handles)
        self.streaker_calib_canvas.draw()

    def clear_screen_plots(self):
        analysis.clear_screen_calibration(*self.screen_calib_plot_handles)
        self.screen_calib_canvas.draw()

    def init_tracker(self):

        widgets = [self.LatticeFromFileCheck, self.LatticeFromLiveCheck]
        self._check_check(widgets, 'Check lattice checkmarks')

        if self.LatticeFromFileCheck.isChecked():
            magnet_file = self.ImportFile.text()
            other_input = {
                    'method': 'from_file',
                    'filename_or_dict': magnet_file,
                    }
        elif self.LatticeFromLiveCheck:
            magnet_file = daq.get_aramis_quad_strengths()
            other_input = {
                    'method': magnet_file,
                    'filename_or_dict': magnet_file,
                    }

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
        self.analysis_obj.add_tracker(self.tracker_kwargs)
        self.tracker_initialized = True
        self.other_input['lattice'] = other_input
        print('Tracker successfully initialized')

    @staticmethod
    def _check_check(widgets, errormessage):
        checks = [x.isChecked() for x in widgets]
        if sum(checks) != 1:
            raise ValueError(errormessage)


    def load_screen_calibration(self):

        filename = self.ImportCalibration.text().strip()
        key = self.ImportCalibrationKey.text()
        index = self.ImportCalibrationIndex.text()
        screen_data = data_loader.load_screen_data(filename, key, index)

        screen_result = analysis.analyze_screen_calibration(screen_data, True, plot_handles=self.screen_calib_plot_handles)

        x0 = screen_result['x0']
        other_input = {
                'method': 'data_loader.load_screen_data',
                'args': (filename, key, index),
                }

        self.screen_x0 = x0
        self.analysis_obj.add_screen_x0(self.screen_x0)
        other_input['x0'] = self.screen_x0
        other_input['beamsize'] = screen_result['beamsize']
        self.other_input['screen_calibration'] = other_input
        self.update_screen_calibration(screen_result)

    def calibrate_screen(self):
        n_images = int(self.CalibrateScreenImages.text())
        image_dict = daq.get_images(self.screen, n_images)
        try:
            result = analysis.analyze_screen_calibration(image_dict, True, self.screen_calib_plot_handles)
        except:
            date = datetime.now()
            basename = date.strftime('%Y_%m_%d-%H_%M_%S_')+'Screen_Calibration_data_%s.h5' % self.screen.replace('.','_')
            filename = os.path.join(self.save_dir, basename)
            h5_storage.saveH5Recursive(filename, image_dict)
            print('Saved screen calibration data %s' % filename)
            raise
        date = datetime.now()
        basename = date.strftime('%Y_%m_%d-%H_%M_%S_')+'Screen_Calibration_%s.h5' % self.screen.replace('.','_')
        filename = os.path.join(self.save_dir, basename)
        h5_storage.saveH5Recursive(filename, image_dict)
        print('Saved screen calibration %s' % filename)
        self.update_screen_calibration(result)

    def update_screen_calibration(self, result):
        x0 = result['x0']
        beamsize = result['beamsize']
        self.DirectCalibration.setText('%.3f' % (x0*1e6))
        self.DirectBeamsizeScreen.setText('%.3f' % (beamsize*1e6))
        print('X0 is %.3f um' % (x0*1e6))
        print('Beamsize is %.3f um' % (beamsize*1e6))
        #self.tabWidget.setCurrentIndex(self.screen_calib_plot_tab_index)

    def set_x_emittance(self):
        if not self.tracker_initialized:
            self.init_tracker()

        tracker = self.analysis_obj.tracker

        tt_halfrange = float(self.ProfileExtent.text())/2*1e-15
        charge = float(self.Charge.text())*1e-12
        len_screen = tracker.len_screen
        smoothen = tracker.smoothen

        bp_test = tracking.get_gaussian_profile(40e-15, tt_halfrange, len_screen, charge, tracker.energy_eV)
        screen_sim = tracker.matrix_forward(bp_test, [10e-3, 10e-3], [0, 0])['screen']
        emit0 = tracker.n_emittances[0]
        sig_real = float(self.DirectBeamsizeScreen.text())*1e-6
        screen_sim2 = tracking.getScreenDistributionFromPoints(screen_sim.real_x, len(screen_sim._xx), smoothen)
        sig_meas = np.sqrt(sig_real**2 - smoothen**2)
        sig_sim = np.sqrt(screen_sim2.gaussfit.sigma**2 - smoothen**2)
        emittance_fit = emit0 * (sig_meas / sig_sim)**2

        print('Emittance fit is %.3f nm' % (emittance_fit*1e9))

    def streaker_calibration(self):
        #widgets = (self.StreakerDirectCheck,)
        #self._check_check(widgets, 'Check streaker calibration checkmarks')

        streaker0_mean = float(self.StreakerDirect0.text())*1e-6
        streaker1_mean = float(self.StreakerDirect1.text())*1e-6
        other_input = {'method': 'direct_input'}

        self.streaker_means = [streaker0_mean, streaker1_mean]
        self.analysis_obj.add_streaker_means(self.streaker_means)
        other_input['streaker_means'] = self.streaker_means
        self.other_input['streaker_calibration'] = other_input
        self.streaker_calibrated = True
        print('Streaker calibrated: mean = %i, %i um' % (self.streaker_means[0]*1e6, self.streaker_means[1]*1e6))
        return self.streaker_means

    def obtain_streaker_settings_from_live(self):
        for n_streaker, gap_widget, offset_widget in [
                (0, self.StreakerGap0, self.StreakerOffset0),
                (1, self.StreakerGap1, self.StreakerOffset1)]:
            streaker = config.streaker_names[self.beamline][n_streaker]
            gap_mm = daq.caget(streaker+':GAP')
            offset_mm = daq.caget(streaker+':CENTER')
            gap_widget.setText('%.3f' % gap_mm)
            offset_widget.setText('%.3f' % offset_mm)

    def streaker_set(self):
        widgets = (self.SetStreakerDirectCheck, self.SetStreakerFromLiveCheck)
        self._check_check(widgets, 'Check set streaker checkmarks')

        if self.SetStreakerDirectCheck.isChecked():
            other_input = {'method': 'direct_input'}
        #elif self.SetStreakerFromFileCheck.isChecked():
        #    raise NotImplementedError
        elif self.SetStreakerFromLiveCheck.isChecked():
            self.obtain_streaker_settings_from_live()
            other_input = {'method': 'live'}

        gaps = [float(self.StreakerGap0.text())*1e-3, float(self.StreakerGap1.text())*1e-3]
        # beam offset is negative of streaker offset
        streaker_offsets = [float(self.StreakerOffset0.text())*1e-3, float(self.StreakerOffset1.text())*1e-3]

        self.gaps = gaps
        self.streaker_offsets = streaker_offsets
        other_input['gaps'] = gaps,
        other_input['streaker_offsets'] = streaker_offsets
        self.other_input['streaker_set'] = other_input
        self.streaker_is_set = True
        print('Streaker is set')
        return gaps, streaker_offsets

    def obtain_reconstruction_data(self):
        widgets = (self.ReconstructionDataLoadCheck,)
        self._check_check(widgets, 'Check obtain reconstruction data checkmarks')

        if self.ReconstructionDataLoadCheck.isChecked():
            filename = self.ReconstructionDataLoad.text().strip()
            key = self.ReconstructionDataLoadKey.text()
            index = self.ReconstructionDataLoadIndex.text()
            screen_data = data_loader.load_screen_data(filename, key, index)
            x_axis, projx = screen_data['x_axis'], screen_data['projx']
            other_input = {'method': 'data_loader.load_screen_data', 'args': (filename, key, index)}

            if self.ReconstructionDataLoadUseSelect.currentText() == 'Median':
                median_projx = misc.get_median(projx)
            elif self.ReconstructionDataLoadUseSelect.currentText() == 'All':
                raise NotImplementedError
            meas_screen = tracking.ScreenDistribution(x_axis, median_projx)

            #import matplotlib.pyplot as plt
            #plt.figure()
            #plt.suptitle('Debug')
            #sp = plt.subplot(1,1,1)
            #sp.plot(meas_screen.x, meas_screen.intensity)
            #meas_screen._yy -= meas_screen._yy.min()
            #gf = meas_screen.gaussfit
            #print(gf.sigma)
            #meas_screen._yy = meas_screen._yy - gf.const
            #sp.plot(meas_screen.x, meas_screen.intensity)
            #sp.plot(gf.xx, gf.reconstruction)
            #sp.plot(gf.xx, gf.fit_func(gf.xx, *gf.p0))
            #plt.show()
            #import pdb; pdb.set_trace()
        else:
            raise NotImplementedError

        self.reconstruction_data_obtained = True
        self.meas_screen = meas_screen
        other_input['screen_x'] = meas_screen.x
        other_input['screen_intensity'] = meas_screen.intensity
        self.other_input['obtain_reconstruction_data'] = other_input
        print('Obtained reconstruction data')
        return meas_screen

    def do_reconstruction(self):

        self.init_tracker()
        self.streaker_calibration()
        self.streaker_set()
        self.obtain_reconstruction_data()
        self.analysis_obj.input_data['screen_x0'] = float(self.DirectCalibration.text())*1e-6

        start, stop, step = float(self.SigTfsStart.text()), float(self.SigTfsStop.text()), float(self.SigTfsStep.text())
        stop += 1e-3*step # assert that stop is part of array
        sig_t_range = np.arange(start, stop, step)*1e-15
        tt_halfrange = float(self.ProfileExtent.text())/2*1e-15
        meas_screen = self.meas_screen
        gaps = self.gaps
        streaker_offsets = self.streaker_offsets
        n_streaker = int(self.StreakerSelect.currentText())
        charge = float(self.Charge.text())*1e-12
        self_consistent = {'True': True, 'False': False}[self.SelfConsistentSelect.currentText()]
        kwargs_recon = {
                'sig_t_range': sig_t_range,
                'tt_halfrange': tt_halfrange,
                'gaps': gaps,
                'streaker_offsets': streaker_offsets,
                'n_streaker': n_streaker,
                'charge': charge,
                'self_consistent': self_consistent,
                'meas_screen': meas_screen,
                }
        self.analysis_obj.input_data['other'] = self.other_input

        kwargs_recon2 = self.analysis_obj.prepare_rec_gauss_args(kwargs_recon)
        print('Analysing reconstruction')

        #import pickle
        #p_file = '/tmp/rec_args.pkl'
        #with open(p_file, 'wb') as f:
        #    pickle.dump((kwargs_recon2, self.analysis_obj), f)
        #    print('Saved %s' % p_file)

        self.analysis_obj.current_profile_rec_gauss(kwargs_recon2, True, self.reconstruction_plot_handles)
        self.rec_canvas.draw()
        self.reconstruction_plot_handles[2].set_ylim(0, None)

    def save_data(self):
        filename = self.analysis_obj.save_data(self.save_dir)
        print('Saved at %s' % filename)

    @property
    def save_dir(self):
        return os.path.expanduser(self.SaveDir.text())

    def load_data(self):
        filename = os.path.expanduser(self.LoadDataFilename.text().strip())
        tmp_dir = os.path.expanduser(self.TmpDir.text().strip())
        analysis.load_reconstruction(filename, tmp_dir, self.reconstruction_plot_handles)
        print('Loaded %s' % filename)
        #self.tabWidget.setCurrentIndex(self.rec_plot_tab_index)

    def update_streaker(self):
        beamline = self.beamline
        self.streaker_name = config.streaker_names[beamline][self.n_streaker]
        self.StreakerName.setText(self.streaker_name)

    def calibrate_streaker(self):
        start, stop, step= float(self.Range1Begin.text()), float(self.Range1Stop.text()), float(self.Range1Step.text())
        range1 = np.linspace(start, stop, step)
        start, stop, step= float(self.Range2Begin.text()), float(self.Range2Stop.text()), float(self.Range2Step.text())
        range2 = np.linspace(start, stop, step)
        range_ = np.concatenate([range1, [0], range2])*1e-3 # Convert mm to m
        range_.sort()
        range_ = np.unique(range_)

        streaker = config.streaker_names[self.beamline][self.n_streaker]
        n_images = int(self.CalibrateStreakerImages.text())

        if daq is None:
            raise ImportError('Daq not available')

        result_dict = daq.data_streaker_offset(streaker, range_, self.screen, n_images, self.dry_run)

        try:
            full_dict = analysis.analyze_streaker_calibration(result_dict, do_plot=True, plot_handles=self.streaker_calib_plot_handles)
        except:
            date = datetime.now()
            basename = date.strftime('%Y_%m_%d-%H_%M_%S_') +'Calibration_data_%s.h5' % streaker.replace('.','_')
            filename = os.path.join(self.save_dir, basename)
            h5_storage.saveH5Recursive(filename, result_dict)
            print('Saved streaker calibration data %s' % filename)
            raise

        date = datetime.now()
        basename = date.strftime('%Y_%m_%d-%H_%M_%S_')+'Calibration_%s.h5' % streaker.replace('.','_')
        filename = os.path.join(self.save_dir, basename)
        h5_storage.saveH5Recursive(filename, full_dict)
        print('Saved streaker calibration %s' % filename)

        streaker_offset = full_dict['meta_data']['streaker_offset']
        self.updateCalibration(streaker_offset)
        #self.tabWidget.setCurrentIndex(self.streaker_calib_plot_tab_index)

    def load_calibration(self):
        filename = self.LoadCalibrationFilename.text().strip()
        saved_dict = h5_storage.loadH5Recursive(filename)

        if 'raw_data' in saved_dict:
            saved_dict = saved_dict['raw_data']
        full_dict = analysis.analyze_streaker_calibration(saved_dict, do_plot=True, plot_handles=self.streaker_calib_plot_handles)

        streaker_offset = full_dict['meta_data']['streaker_offset']
        self.updateCalibration(streaker_offset)
        #self.tabWidget.setCurrentIndex(self.streaker_calib_plot_tab_index)

    def updateCalibration(self, streaker_offset):
        if self.n_streaker == 0:
            widget = self.StreakerDirect0
        elif self.n_streaker == 1:
            widget = self.StreakerDirect1
        old = float(widget.text())
        widget.setText('%.3f' % (streaker_offset*1e6))
        new = float(widget.text())
        print('Updated calibration for streaker %i. Old: %.3f um New: %.3f um' % (self.n_streaker, old, new))

    def obtain_reconstruction(self):
        n_images = int(self.ReconNumberImages.text())
        screen_dict = daq.get_images(self.screen, n_images)
        date = datetime.now()
        basename = date.strftime('%Y_%m_%d-%H_%M_%S_')+'Screen_data_%s.h5' % self.screen.replace('.','_')
        filename = os.path.join(self.save_dir, basename)
        h5_storage.saveH5Recursive(filename, screen_dict)
        print('Saved screen data %s' % filename)

    @property
    def n_streaker(self):
        return int(self.StreakerSelect.currentText())

    @property
    def beamline(self):
        return self.BeamlineSelect.currentText()

    @property
    def dry_run(self):
        return (self.DryRun.isChecked() or always_dryrun)

    @property
    def screen(self):
        return self.ScreenSelect.currentText()

if __name__ == '__main__':
    def my_excepthook(type, value, tback):
        # log the exception here
        # then call the default handler
        sys.__excepthook__(type, value, tback)
        print(type, value, tback)
    sys.excepthook = my_excepthook

    app = QtWidgets.QApplication(sys.argv)
    window = StartMain()
    window.show()
    app.exec_()

