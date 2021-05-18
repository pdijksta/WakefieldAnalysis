import matplotlib.pyplot as plt; plt # Without this line, there is an error...
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
import matplotlib
matplotlib.use('Qt5Agg')

import sys
import os
import re
import socket
from datetime import datetime
import numpy as np
import PyQt5.Qt
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import pyqtRemoveInputHook

import config
import tracking
import elegant_matrix
import data_loader
import misc2 as misc
import analysis
import h5_storage
import image_and_profile as iap

import myplotstyle as ms

#TODO
#
# - add info of beamsize with / without assumed screen resolution
# - debug delay after using BsreadPositioner or any pyscan
# - add tilt option
# - restructure analysis
# - what is the correct beam energy pv?
# - handle feedback in user interface
# - simplify lattice
# - detune undulator button
# - optional provide the pulse energy calibration
# - streaker center calibration: repeat with one data point removed at one side
# - Offset based on centroid, offset based on sizes (?)
# - Script to compare different scans

# Probably fixed:
# - sort out daq pyscan_result_to_dict

# Not so important
# - noise reduction from the image
# - uJ instead of True, False
# - non blocking daq

# Done
# - pulse energy from gas detector in pyscan
# - yum install libhdf5
# - streaker calibration fit guess improvements
# - meta data at begin and end of pyscan
# - lasing
# - y scale of optimization
# - elog
# - charge from pyscan
# - Forward propagation from TDC to screen inside tool
# - plot TDC blmeas next to current reconstruction (optional)
# - save BPM data also
# - Show sizes


try:
    import daq
    always_dryrun = False
except ImportError:
    print('Cannot import daq. Always dry_run True')
    always_dryrun = True
    daq = None

try:
    import elog
except ImportError:
    print('ELOG not available')
    elog = None

ms.set_fontsizes(8)

pyqtRemoveInputHook() # for pdb to work
re_time = re.compile('(\\d{4})-(\\d{2})-(\\d{2}):(\\d{2})-(\\d{2})-(\\d{2})')


class StartMain(QtWidgets.QMainWindow):

    def __init__(self):
        super(StartMain, self).__init__()
        uic.loadUi('GUI.ui', self)

        self.DoReconstruction.clicked.connect(self.reconstruct_current)
        self.SaveData.clicked.connect(self.save_data)
        self.CloseAll.clicked.connect(self.clear_rec_plots)
        self.ObtainStreakerFromLive.clicked.connect(self.obtain_streaker_settings_from_live)
        self.CalibrateStreaker.clicked.connect(self.calibrate_streaker)
        self.ClearCalibPlots.clicked.connect(self.clear_calib_plots)
        self.LoadCalibration.clicked.connect(self.load_calibration)
        self.CalibrateScreen.clicked.connect(self.calibrate_screen)
        self.ClearScreenPlots.clicked.connect(self.clear_screen_plots)
        self.SetXEmittance.clicked.connect(self.set_x_emittance)
        self.ObtainReconstructionData.clicked.connect(self.obtain_reconstruction)
        self.ObtainLasingOnData.clicked.connect(self.obtainLasingOn)
        self.ObtainLasingOffData.clicked.connect(self.obtainLasingOff)
        self.ReconstructLasing.clicked.connect(self.reconstruct_lasing)
        self.ObtainR12.clicked.connect(self.obtain_r12)

        self.StreakerSelect.activated.connect(self.update_streaker)
        self.BeamlineSelect.activated.connect(self.update_streaker)

        self.update_streaker()
        self.other_input = {}

        # Default strings in gui fields
        hostname = socket.gethostname()
        if 'psi' in hostname or 'lc6a' in hostname or 'lc7a' in hostname:
            default_dir = '/sf/data/measurements/2021/04/25/'
            archiver_dir = '/afs/psi.ch/intranet/SF/Beamdynamics/Philipp/data/archiver_api_data/'
            date = datetime.now()
            save_dir = date.strftime('/sf/data/measurements/%Y/%m/%d/')
        elif hostname == 'desktop':
            default_dir = '/storage/data_2021-04-25/'
            archiver_dir = '/storage/Philipp_data_folder/archiver_api_data/'
            save_dir = '/storage/tmp_reconstruction/'
        elif hostname == 'pubuntu':
            default_dir = '/home/work/data_2021-04-25/'
            archiver_dir = '/home/work/archiver_api_data/'
            save_dir = '/home/work/tmp_reconstruction/'

        screen_calib_file = default_dir+'Passive_data_20201003T231958.mat'
        bunch_length_meas_file = default_dir + '117348568_bunch_length_meas.h5'
        recon_data_file = default_dir+'2021_04_25-17_16_30_Lasing_False_SARBD02-DSCR050.h5'
        lattice_file = archiver_dir+'2021-04-25.h5'
        time_str = '2021-04-25:17-22-26'
        lasing_file_off = default_dir + '2021_04_25-17_16_30_Lasing_False_SARBD02-DSCR050.h5'
        lasing_file_on = default_dir + '2021_04_25-17_57_34_Lasing_True_SARBD02-DSCR050.h5'
        streaker_calib_file = default_dir + '2021_04_25-16_55_25_Calibration_SARUN18-UDCP020.h5'

        self.ImportCalibration.setText(screen_calib_file)
        self.ImportFile.setText(lattice_file)
        self.ImportFileTime.setText(time_str)
        self.ReconstructionDataLoad.setText(recon_data_file)
        self.BunchLengthMeasFile.setText(bunch_length_meas_file)
        self.SaveDir.setText(save_dir)
        self.LasingOnDataLoad.setText(lasing_file_on)
        self.LasingOffDataLoad.setText(lasing_file_off)
        self.SaveDir.setText(save_dir)
        self.LoadCalibrationFilename.setText(streaker_calib_file)
        self.ForwardBlmeasFilename.setText(bunch_length_meas_file)

        if elog is not None:
            self.logbook = elog.open('https://elog-gfa.psi.ch/SwissFEL+commissioning+data/')

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

        self.reconstruction_fig, self.reconstruction_plot_handles = analysis.reconstruction_figure()
        self.rec_plot_tab_index, self.rec_canvas = get_new_tab(self.reconstruction_fig, 'Rec plots')

        self.streaker_calib_fig, self.streaker_calib_plot_handles = analysis.streaker_calibration_figure()
        self.streaker_calib_plot_tab_index, self.streaker_calib_canvas = get_new_tab(self.streaker_calib_fig, 'Cal. plots')

        self.screen_calib_fig, self.screen_calib_plot_handles = analysis.screen_calibration_figure()
        self.screen_calib_plot_tab_index, self.screen_calib_canvas = get_new_tab(self.screen_calib_fig, 'Screen plots')

        self.lasing_plot_handles = analysis.lasing_figures()
        self.lasing_figs = [x[0] for x in self.lasing_plot_handles]
        self.lasing_plot_tab_index1, self.lasing_canvas1 = get_new_tab(self.lasing_plot_handles[0][0], 'Lasing 1')
        self.lasing_plot_tab_index2, self.lasing_canvas2 = get_new_tab(self.lasing_plot_handles[1][0], 'Lasing 2')

    def clear_rec_plots(self):
        if self.reconstruction_plot_handles is not None:
            analysis.clear_reconstruction(*self.reconstruction_plot_handles)
            self.rec_canvas.draw()
            print('Cleared reconstruction plot')

    def clear_calib_plots(self):
        if self.streaker_calib_plot_handles is not None:
            analysis.clear_streaker_calibration(*self.streaker_calib_plot_handles)
            self.streaker_calib_canvas.draw()

    def clear_screen_plots(self):
        if self.screen_calib_plot_handles is not None:
            analysis.clear_screen_calibration(*self.screen_calib_plot_handles)
            self.screen_calib_canvas.draw()

    def clear_lasing_plots(self):
        if self.lasing_plot_handles is not None:
            analysis.clear_lasing(self.lasing_plot_handles)

    def obtain_lattice(self):


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
        return magnet_file, other_input

    def obtain_r12(self):
        analysis_obj = self.init_tracker()
        r12 = analysis_obj.tracker.calcR12()[self.n_streaker]
        disp = analysis_obj.tracker.calcDisp()[self.n_streaker]
        print('R12:', r12)
        print('Dispersion:', disp)
        return r12, disp


    def init_tracker(self):

        analysis_obj = analysis.Reconstruction(self.screen_x0, self.streaker_means)
        magnet_file, other_input = self.obtain_lattice()

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
        analysis_obj.add_tracker(self.tracker_kwargs)
        self.other_input['lattice'] = other_input
        print('Tracker successfully initialized')
        return analysis_obj

    @staticmethod
    def _check_check(widgets, errormessage):
        checks = [x.isChecked() for x in widgets]
        if sum(checks) != 1:
            raise ValueError(errormessage)

    def calibrate_screen(self):
        n_images = int(self.CalibrateScreenImages.text())
        image_dict = daq.get_images_and_bpm(self.screen, n_images, dry_run=self.dry_run)
        try:
            result = analysis.analyze_screen_calibration(image_dict, True, self.screen_calib_plot_handles)
        except:
            date = datetime.now()
            basename = date.strftime('%Y_%m_%d-%H_%M_%S_')+'Screen_Calibration_data_%s.h5' % self.screen.replace('.','_')
            filename = os.path.join(self.save_dir, basename)
            h5_storage.saveH5Recursive(filename, image_dict)
            print('Saved screen calibration data %s' % filename)
            raise
        x0, beamsize = self.update_screen_calibration(result)
        date = datetime.now()
        basename = date.strftime('%Y_%m_%d-%H_%M_%S_')+'Screen_Calibration_%s.h5' % self.screen.replace('.','_')
        elog_text = 'Screen calibration\nScreen center at %i um\nBeamsize %i um' % (x0*1e6, beamsize*1e6)
        self.elog_and_H5(elog_text, [self.screen_calib_fig], 'Screen center calibration', basename, image_dict)

    def update_screen_calibration(self, result):
        x0 = result['x0']
        beamsize = result['beamsize']
        self.DirectCalibration.setText('%.3f' % (x0*1e6))
        self.DirectBeamsizeScreen.setText('%.3f' % (beamsize*1e6))
        print('X0 is %.3f um' % (x0*1e6))
        print('Beamsize is %.3f um' % (beamsize*1e6))
        return x0, beamsize
        #self.tabWidget.setCurrentIndex(self.screen_calib_plot_tab_index)

    def set_x_emittance(self):

        analysis_obj = self.init_tracker()
        tracker = analysis_obj.tracker

        tt_halfrange = float(self.ProfileExtent.text())/2*1e-15
        charge = float(self.Charge.text())*1e-12
        len_screen = tracker.len_screen
        smoothen = tracker.smoothen

        bp_test = iap.get_gaussian_profile(40e-15, tt_halfrange, len_screen, charge, tracker.energy_eV)
        screen_sim = tracker.matrix_forward(bp_test, [10e-3, 10e-3], [0, 0])['screen']
        emit0 = tracker.n_emittances[0]
        sig_real = float(self.DirectBeamsizeScreen.text())*1e-6
        screen_sim2 = iap.getScreenDistributionFromPoints(screen_sim.real_x, len(screen_sim._xx), smoothen)
        sig_meas = np.sqrt(sig_real**2 - smoothen**2)
        sig_sim = np.sqrt(screen_sim2.gaussfit.sigma**2 - smoothen**2)
        emittance_fit = emit0 * (sig_meas / sig_sim)**2

        print('Emittance fit is %.3f nm' % (emittance_fit*1e9))

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
        widgets = (self.SetStreakerDirectCheck, self.SetStreakerFromLiveCheck, self.SetStreakerSaveCheck)
        self._check_check(widgets, 'Check set streaker checkmarks')

        if self.SetStreakerDirectCheck.isChecked():
            other_input = {'method': 'direct_input'}
            meta_dict = None
        elif self.SetStreakerFromLiveCheck.isChecked():
            self.obtain_streaker_settings_from_live()
            other_input = {'method': 'live'}
            if daq is None:
                raise RuntimeError('Cannot get settings from live!')
            meta_dict = daq.get_meta_data()

        elif self.SetStreakerSaveCheck:
            other_input = {'method': 'saved'}
            filename = self.ReconstructionDataLoad.text().strip()
            dict_ = h5_storage.loadH5Recursive(filename)
            if 'meta_data' in dict_:
                meta_dict = dict_['meta_data']
            elif 'meta_data_end' in dict_:
                meta_dict = dict_['meta_data_end']

        if meta_dict is not None:
            streaker_dict = config.streaker_names[self.beamline]
            for n_streaker, gap_widget, offset_widget in [
                    (0, self.StreakerGap0, self.StreakerOffset0),
                    (1, self.StreakerGap1, self.StreakerOffset1),
                    ]:
                streaker = streaker_dict[n_streaker]
                offset_mm = meta_dict[streaker+':CENTER']
                gap_mm = meta_dict[streaker+':GAP']
                if offset_mm < .01*gap_mm/2:
                    offset_mm = 0
                gap_widget.setText('%.4f' % gap_mm)
                offset_widget.setText('%.4f' % offset_mm)

        gaps = [float(self.StreakerGap0.text())*1e-3, float(self.StreakerGap1.text())*1e-3]
        # beam offset is negative of streaker offset
        streaker_offsets = self.streaker_offsets

        other_input['gaps'] = gaps,
        other_input['streaker_offsets'] = streaker_offsets
        self.other_input['streaker_set'] = other_input
        print('Streaker is set: gaps: %s, offsets: %s' % (gaps, streaker_offsets))
        return gaps, streaker_offsets

    def reconstruct_current(self):

        analysis_obj = self.init_tracker()
        # Streaker calibration
        other_input = {'method': 'direct_input'}

        streaker_means = self.streaker_means
        other_input['streaker_means'] = streaker_means
        self.other_input['streaker_calibration'] = other_input
        print('Streaker calibrated: mean = %i, %i um' % (streaker_means[0]*1e6, streaker_means[1]*1e6))

        gaps, streaker_offsets = self.streaker_set()
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

        self.meas_screen = meas_screen
        other_input['screen_x'] = meas_screen.x
        other_input['screen_intensity'] = meas_screen.intensity
        self.other_input['obtain_reconstruction_data'] = other_input
        print('Obtained reconstruction data')

        analysis_obj.input_data['screen_x0'] = self.screen_x0
        analysis_obj.input_data['streaker_offsets'] = self.streaker_offsets

        if self.ShowBlmeasCheck.isChecked():
            blmeas_file = self.BunchLengthMeasFile.text()
        else:
            blmeas_file = None

        start, stop, step = float(self.SigTfsStart.text()), float(self.SigTfsStop.text()), float(self.SigTfsStep.text())
        stop += 1e-3*step # assert that stop is part of array
        sig_t_range = np.arange(start, stop, step)*1e-15
        tt_halfrange = float(self.ProfileExtent.text())/2*1e-15
        meas_screen = self.meas_screen
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
        analysis_obj.input_data['other'] = self.other_input

        kwargs_recon2 = analysis_obj.prepare_rec_gauss_args(kwargs_recon)
        print('Analysing reconstruction')

        #import pickle
        #p_file = '/tmp/rec_args.pkl'
        #with open(p_file, 'wb') as f:
        #    pickle.dump((kwargs_recon2, analysis_obj), f)
        #    print('Saved %s' % p_file)

        self.clear_rec_plots()
        analysis_obj.current_profile_rec_gauss(kwargs_recon2, True, self.reconstruction_plot_handles, blmeas_file)
        self.current_rec_obj = analysis_obj

        if self.rec_canvas is not None:
            self.rec_canvas.draw()

    def save_data(self):
        save_path = self.save_dir
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        date = datetime.now()
        basename = date.strftime('%Y_%m_%d-%H_%M_%S_PassiveReconstruction.h5')
        save_dict = {
                'input': self.current_rec_obj.input_data,
                'gaussian_reconstruction': self.current_rec_obj.gauss_dict,
                }
        elog_text = 'Passive current reconstruction'
        self.elog_and_H5(elog_text, [self.reconstruction_fig], 'Passive current reconstruction', basename, save_dict)

    @property
    def save_dir(self):
        return os.path.expanduser(self.SaveDir.text())

    def update_streaker(self):
        beamline = self.beamline
        self.streaker_name = config.streaker_names[beamline][self.n_streaker]
        self.StreakerName.setText(self.streaker_name)

    def calibrate_streaker(self):
        start, stop, step= float(self.Range1Begin.text()), float(self.Range1Stop.text()), int(float(self.Range1Step.text()))
        range1 = np.linspace(start, stop, step)
        start, stop, step= float(self.Range2Begin.text()), float(self.Range2Stop.text()), int(float(self.Range2Step.text()))
        range2 = np.linspace(start, stop, step)
        range_ = np.concatenate([range1, [0], range2])*1e-3 # Convert mm to m
        range_.sort()
        range_ = np.unique(range_)

        streaker = config.streaker_names[self.beamline][self.n_streaker]
        n_images = int(self.CalibrateStreakerImages.text())

        if daq is None:
            raise ImportError('Daq not available')

        result_dict = daq.bpm_data_streaker_offset(streaker, range_, self.screen, n_images, self.dry_run)

        try:
            full_dict = self._analyze_streaker_calib(result_dict)
        except:
            date = datetime.now()
            basename = date.strftime('%Y_%m_%d-%H_%M_%S_') +'Calibration_data_%s.h5' % streaker.replace('.','_')
            filename = os.path.join(self.save_dir, basename)
            h5_storage.saveH5Recursive(filename, result_dict)
            print('Saved streaker calibration data %s' % filename)
            raise

        date = datetime.now()
        basename = date.strftime('%Y_%m_%d-%H_%M_%S_')+'Calibration_%s.h5' % streaker.replace('.','_')
        streaker_offset = full_dict['meta_data']['streaker_offset']
        self.updateCalibration(streaker_offset)

        elog_text = 'Streaker calibration streaker %s\nCenter: %i um' % (streaker, streaker_offset*1e6)
        self.elog_and_H5(elog_text, [self.streaker_calib_fig], 'Streaker center calibration', basename, full_dict)

    def _analyze_streaker_calib(self, result_dict):
        forward_blmeas = self.ForwardBlmeasCheck.isChecked()
        if forward_blmeas:
            tracker = self.init_tracker().tracker
            blmeasfile = self.ForwardBlmeasFilename.text()
        else:
            tracker = None
            blmeasfile = None

        full_dict = analysis.analyze_streaker_calibration(result_dict, do_plot=True, plot_handles=self.streaker_calib_plot_handles, forward_propagate_blmeas=forward_blmeas, tracker=tracker, blmeas=blmeasfile, beamline=self.beamline)
        return full_dict

    def load_calibration(self):
        filename = self.LoadCalibrationFilename.text().strip()
        saved_dict = h5_storage.loadH5Recursive(filename)

        if 'raw_data' in saved_dict:
            saved_dict = saved_dict['raw_data']
        full_dict = self._analyze_streaker_calib(saved_dict)

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
        screen_dict = daq.get_images_and_bpm(self.screen, n_images, dry_run=self.dry_run)
        date = datetime.now()
        basename = date.strftime('%Y_%m_%d-%H_%M_%S_')+'Screen_data_%s.h5' % self.screen.replace('.','_')
        elog_text = 'Screen %s data taken' % self.screen
        self.elog_and_H5(elog_text, [], 'Screen data', basename, screen_dict)

    @property
    def n_streaker(self):
        return int(self.StreakerSelect.currentText())

    @property
    def streaker(self):
        return config.streaker_names[self.beamline][self.n_streaker]

    @property
    def beamline(self):
        return self.BeamlineSelect.currentText()

    @property
    def dry_run(self):
        return (self.DryRun.isChecked() or always_dryrun)

    @property
    def screen_x0(self):
        return float(self.DirectCalibration.text())*1e-6

    @property
    def streaker_offsets(self):
        return [float(self.StreakerOffset0.text())*1e-3, float(self.StreakerOffset1.text())*1e-3]

    @property
    def streaker_means(self):
        streaker0_mean = float(self.StreakerDirect0.text())*1e-6
        streaker1_mean = float(self.StreakerDirect1.text())*1e-6
        return [streaker0_mean, streaker1_mean]

    @property
    def screen(self):
        if self.dry_run:
            return 'simulation'
        else:
            return self.ScreenSelect.currentText()

    def obtainLasing(self, lasing_on_off):
        if lasing_on_off:
            n_images = int(self.LasingOnNumberImages.text())
        else:
            n_images = int(self.LasingOffNumberImages.text())

        image_dict = daq.get_images_and_bpm(self.screen, n_images, dry_run=self.dry_run)
        date = datetime.now()
        screen_str = self.screen.replace('.','_')
        lasing_str = str(lasing_on_off)
        basename = date.strftime('%Y_%m_%d-%H_%M_%S_')+'Lasing_%s_%s.h5' % (lasing_str, screen_str)
        if lasing_on_off:
            elog_text = 'Saved lasing ON'
        else:
            elog_text = 'Saved lasing OFF'
        filename = self.elog_and_H5(elog_text, [], 'Saved lasing images', basename, image_dict)
        if lasing_on_off:
            self.LasingOnDataLoad.setText(filename)
        else:
            self.LasingOffDataLoad.setText(filename)

    def obtainLasingOn(self):
        return self.obtainLasing(True)

    def obtainLasingOff(self):
        return self.obtainLasing(False)

    def reconstruct_lasing(self):
        file_on = self.LasingOnDataLoad.text()
        file_off = self.LasingOffDataLoad.text()
        lasing_energy_txt = self.LasingEnergyInput.text()
        if lasing_energy_txt == 'None':
            lasing_energy = None
        else:
            lasing_energy = float(lasing_energy_txt)*1e-6
        file_current = self.LasingCurrentProfileDataLoad.text()
        screen_center = self.screen_x0

        if self.n_streaker == 0:
            structure_center = float(self.StreakerDirect0.text())*1e-6
        elif self.n_streaker == 1:
            structure_center = float(self.StreakerDirect1.text())*1e-6

        streaker_name = config.streaker_names[self.beamline][self.n_streaker]
        structure_length = [float(self.StructLength1.text()), float(self.StructLength1.text())][self.n_streaker]

        assert os.path.isfile(file_on)
        assert os.path.isfile(file_off)
        r12, disp = self.obtain_r12()
        energy_eV = 'file'
        charge = float(self.Charge.text())*1e-12

        if self.lasing_plot_handles is not None:
            analysis.clear_lasing(self.lasing_plot_handles)

        analysis.reconstruct_lasing(file_on, file_off, screen_center, structure_center, structure_length, file_current, r12, disp, energy_eV, charge, streaker_name, self.lasing_plot_handles, lasing_energy)

        if self.lasing_plot_handles is not None:

            self.tabWidget.setCurrentIndex(self.lasing_plot_tab_index2)

    def elog_and_H5(self, text, figs, title, basename, data_dict):

        filename = os.path.join(self.save_dir, basename)
        h5_storage.saveH5Recursive(filename, data_dict)
        print('Saved %s' % filename)

        attachments = []
        for num, fig in enumerate(figs):
            fig_title = filename.replace('.h5', '_%i.png' % num)
            fig_filename = os.path.join(self.save_dir, fig_title+'.png')
            fig.savefig(fig_filename, bbox_inches='tight', pad_inches=0)
            print('Saved %s' % fig_filename)
            attachments.append(fig_filename)

        text += '\nData saved in %s' % filename
        text += '\nStreaker: %s' % self.streaker
        text += '\nScreen: %s' % self.screen

        if elog is None:
            print('Cannot save to ELOG')
            print('I would post:')
            print(text)
        elif self.ElogSaveCheck.isChecked():
            dict_att = {'Application': 'PostUndulatorStreakerAnalysis', 'Category': 'Measurement', 'Title': title}
            self.logbook.post(text, attributes=dict_att, attachments=attachments)

            print('ELOG entry saved.')
        else:
            print('Save to ELOG is not checked in GUI')

        return filename


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

