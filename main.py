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
import analysis
import lasing
import h5_storage
import streaker_calibration as sc

import myplotstyle as ms

#TODO
#
# - add info of beamsize with / without assumed screen resolution
# - debug delay after using BsreadPositioner or any pyscan
# - add tilt option
# - handle feedback in user interface
# - detune undulator button
# - streaker center calibration: repeat with one data point removed at one side
# - Offset based on centroid, offset based on sizes (?)
# - Dispersion (?)
# - Plot centroid of forward propagated
# - One-sided plate
# - add blmeas option to lasing rec
# - Mean of square instead of square of mean of squareroot

#Problematic / cannot be done easily:
# - save BPM data also
# - One-sided plate

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
# - Show sizes
# - simplify lattice
# - restructure analysis
# - Rec plot legends
# - Comments to elog
# - optional provide the pulse energy calibration
# - png.png

# Other comments
# - Data for paper
# - 33713
# - One for big streaking
# - Resolution for big streaking

# - Figure 2: Current profile reconstruction (early offset scan)
# - Maybe add wake function
# - Time profile reconstructed and measured
# - Screen profile: measured, reconstructed, TDC forward
# - Unstreaked and streaked
# - Resolution

# - Figure 3: Lasing reconstruction of full, short, two-color beam
# - Images (in time) 3x2
# - FEL power profile (average and shot-to-shot) 3x1
# - Slice energy spread per slice and slice current 1

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

ms.set_fontsizes(config.fontsize)

pyqtRemoveInputHook() # for pdb to work
re_time = re.compile('(\\d{4})-(\\d{2})-(\\d{2}):(\\d{2})-(\\d{2})-(\\d{2})')


class StartMain(QtWidgets.QMainWindow):

    def __init__(self):
        super(StartMain, self).__init__()
        uic.loadUi('GUI.ui', self)

        self.DoReconstruction.clicked.connect(self.reconstruct_current)
        self.SaveCurrentRecData.clicked.connect(self.save_current_rec_data)
        self.SaveLasingRecData.clicked.connect(self.save_lasing_rec_data)
        self.CloseAll.clicked.connect(self.clear_rec_plots)
        self.ObtainStreakerFromLive.clicked.connect(self.obtain_streaker_settings_from_live)
        self.CalibrateStreaker.clicked.connect(self.calibrate_streaker)
        self.GapReconstruction.clicked.connect(self.gap_reconstruction)
        self.ClearCalibPlots.clicked.connect(self.clear_calib_plots)
        self.ClearGapRecPlots.clicked.connect(self.clear_gap_recon_plots)
        self.LoadCalibration.clicked.connect(self.load_calibration)
        self.CalibrateScreen.clicked.connect(self.calibrate_screen)
        self.ClearScreenPlots.clicked.connect(self.clear_screen_plots)
        self.ObtainReconstructionData.clicked.connect(self.obtain_reconstruction)
        self.ObtainLasingOnData.clicked.connect(self.obtainLasingOn)
        self.ObtainLasingOffData.clicked.connect(self.obtainLasingOff)
        self.ReconstructLasing.clicked.connect(self.reconstruct_all_lasing)
        self.ObtainR12.clicked.connect(self.obtain_r12_0)

        self.StreakerSelect.activated.connect(self.update_streaker)
        self.BeamlineSelect.activated.connect(self.update_streaker)

        self.update_streaker()

        # Default strings in gui fields
        hostname = socket.gethostname()
        if 'psi' in hostname or 'lc6a' in hostname or 'lc7a' in hostname:
            default_dir = '/sf/data/measurements/2021/05/18/'
            date = datetime.now()
            save_dir = date.strftime('/sf/data/measurements/%Y/%m/%d/')
        elif hostname == 'desktop':
            default_dir = '/storage/data_2021-05-18/'
            save_dir = '/storage/tmp_reconstruction/'
        elif hostname == 'pubuntu':
            default_dir = '/home/work/data_2021-05-18/'
            save_dir = '/home/work/tmp_reconstruction/'

        screen_calib_file = default_dir+'Passive_data_20201003T231958.mat'
        bunch_length_meas_file = default_dir + '119325494_bunch_length_meas.h5'
        #recon_data_file = default_dir+'2021_05_18-17_41_02_PassiveReconstruction.h5'
        lasing_file_off = default_dir + '2021_05_18-21_45_00_Lasing_False_SARBD02-DSCR050.h5'
        lasing_file_on = default_dir + '2021_05_18-21_41_35_Lasing_True_SARBD02-DSCR050.h5'
        streaker_calib_file = default_dir + '2021_05_18-22_11_36_Calibration_SARUN18-UDCP020.h5'
        lasing_current_profile = default_dir + '2021_05_18-17_41_02_PassiveReconstruction.h5'
        screen_X0 = 898.02e-6
        streaker_offsets = 0, 364e-6
        delta_gap = 0, -62e-6
        pulse_energy = 180e-6

        self.DirectCalibration.setText('%i' % (screen_X0*1e6))
        self.StreakerDirect0.setText('%i' % (streaker_offsets[0]*1e6))
        self.StreakerDirect1.setText('%i' % (streaker_offsets[1]*1e6))
        self.StreakerGapDelta0.setText('%i' % (delta_gap[0]*1e6))
        self.StreakerGapDelta1.setText('%i' % (delta_gap[1]*1e6))
        self.LasingEnergyInput.setText('%i' % (pulse_energy*1e6))

        self.ImportCalibration.setText(screen_calib_file)
        self.ReconstructionDataLoad.setText(lasing_file_off)
        self.BunchLengthMeasFile.setText(bunch_length_meas_file)
        self.SaveDir.setText(save_dir)
        self.LasingOnDataLoad.setText(lasing_file_on)
        self.LasingOffDataLoad.setText(lasing_file_off)
        self.LasingCurrentProfileDataLoad.setText(lasing_current_profile)
        self.SaveDir.setText(save_dir)
        self.LoadCalibrationFilename.setText(streaker_calib_file)
        self.ForwardBlmeasFilename.setText(bunch_length_meas_file)

        ds = config.get_default_tracker_settings()
        gs = config.get_default_gauss_recon_settings()
        self.StructLength1.setText('%.2f' % ds['struct_lengths'][0])
        self.StructLength2.setText('%.2f' % ds['struct_lengths'][1])
        self.N_Particles.setText('%i' % ds['n_particles'])
        self.TransEmittanceX.setText('%i' % round(ds['n_emittances'][0]*1e9))
        self.TransEmittanceY.setText('%i' % round(ds['n_emittances'][1]*1e9))
        self.ScreenSmoothen.setText('%i' % round(ds['smoothen']*1e6))
        self.ProfileSmoothen.setText('%i' % round(ds['bp_smoothen']*1e15))
        self.SelfConsistentCheck.setChecked(gs['self_consistent'])
        self.UseQuadCheck.setChecked(ds['quad_wake'])
        self.OverrideQuadCheck.setChecked(ds['override_quad_beamsize'])
        self.QuadBeamsize1.setText('%.2f' % (ds['quad_x_beamsize'][0]*1e6))
        self.QuadBeamsize2.setText('%.2f' % (ds['quad_x_beamsize'][1]*1e6))
        self.SigTfsStart.setText('%i' % round(gs['sig_t_range'][0]*1e15))
        self.SigTfsStop.setText('%i' % round(gs['sig_t_range'][-1]*1e15))
        self.SigTSize.setText('%i' % len(gs['sig_t_range']))
        self.TmpDir.setText(config.tmp_elegant_dir)
        self.ScreenBins.setText('%i' % ds['screen_bins'])
        self.ScreenLength.setText('%i' % ds['len_screen'])
        self.ScreenCutoff.setText('%.4f' % ds['screen_cutoff'])
        self.ProfileCutoff.setText('%.4f' % ds['profile_cutoff'])
        self.ProfileExtent.setText('%i' % round(gs['tt_halfrange']*2*1e15))
        self.Charge.setText('%i' % round(gs['charge']*1e12))

        if elog is not None:
            self.logbook = elog.open('https://elog-gfa.psi.ch/SwissFEL+commissioning+data/')

        self.current_rec_dict = None
        self.lasing_rec_dict = None

        ## Handle plots
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
        self.rec_plot_tab_index, self.rec_canvas = get_new_tab(self.reconstruction_fig, 'I Rec.')

        self.streaker_calib_fig, self.streaker_calib_plot_handles = sc.streaker_calibration_figure()
        self.streaker_calib_plot_tab_index, self.streaker_calib_canvas = get_new_tab(self.streaker_calib_fig, 'Calib.')

        self.gap_recon_fig, self.gap_recon_plot_handles = sc.gap_recon_figure()
        self.gap_recon_tab_index, self.gap_recon_canvas = get_new_tab(self.gap_recon_fig, 'Gap rec.')

        self.screen_calib_fig, self.screen_calib_plot_handles = analysis.screen_calibration_figure()
        self.screen_calib_plot_tab_index, self.screen_calib_canvas = get_new_tab(self.screen_calib_fig, 'Screen')

        self.all_lasing_fig, self.all_lasing_plot_handles = lasing.lasing_figure()
        self.all_lasing_tab_index, self.all_lasing_canvas = get_new_tab(self.all_lasing_fig, 'All lasing')

    def clear_rec_plots(self):
        analysis.clear_reconstruction(*self.reconstruction_plot_handles)
        self.rec_canvas.draw()

    def clear_gap_recon_plots(self):
        sc.clear_gap_recon(*self.gap_recon_plot_handles)
        self.gap_recon_canvas.draw()

    def clear_calib_plots(self):
        sc.clear_streaker_calibration(*self.streaker_calib_plot_handles)
        self.streaker_calib_canvas.draw()

    def clear_screen_plots(self):
        analysis.clear_screen_calibration(*self.screen_calib_plot_handles)
        self.screen_calib_canvas.draw()

    def clear_all_lasing_plots(self):
        lasing.clear_lasing_figure(*self.all_lasing_plot_handles)

    def obtain_r12_0(self):
        return self.obtain_r12()

    def obtain_r12(self, meta_data=None):
        if meta_data is None:
            meta_data = daq.get_meta_data(self.screen)
        #print('obtain_r12', meta_data)
        tracker = self.get_tracker(meta_data)
        r12 = tracker.calcR12()[self.n_streaker]
        disp = tracker.calcDisp()[self.n_streaker]
        print('R12:', r12)
        print('Dispersion:', disp)
        return r12, disp

    @property
    def delta_gaps(self):
        return float(self.StreakerGapDelta0.text())*1e-6, float(self.StreakerGapDelta1.text())*1e-6

    def get_gauss_kwargs(self):
        start, stop, size = float(self.SigTfsStart.text()), float(self.SigTfsStop.text()), float(self.SigTSize.text())
        sig_t_range = np.exp(np.linspace(np.log(start), np.log(stop), size))*1e-15
        tt_halfrange = float(self.ProfileExtent.text())/2*1e-15
        n_streaker = int(self.StreakerSelect.currentText())
        charge = self.charge
        self_consistent = self.SelfConsistentCheck.isChecked()
        delta_gap = self.delta_gaps
        kwargs_recon = {
                'sig_t_range': sig_t_range,
                'tt_halfrange': tt_halfrange,
                'n_streaker': n_streaker,
                'charge': charge,
                'self_consistent': self_consistent,
                'method': 'centroid',
                'delta_gap': delta_gap,
                }
        return kwargs_recon

    def get_tracker_kwargs(self, magnet_dict=None):
        tmp_dir = os.path.expanduser(self.TmpDir.text())
        assert os.path.isdir(tmp_dir)
        assert os.access(tmp_dir, os.W_OK)
        elegant_matrix.set_tmp_dir(tmp_dir)

        #time_match = re_time.match(self.ImportFileTime.text().strip())
        #if time_match is None:
        #    raise ValueError('Wrong format of ImportFileTime: %s\nMust be: yyyy-MM-dd:hh-mm-ss' % self.ImportFileTime.text())
        timestamp = None
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

        tracker_kwargs = {
                'magnet_file': magnet_dict,
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
        return tracker_kwargs

    def get_tracker(self, magnet_dict=None):
        tracker_kwargs = self.get_tracker_kwargs(magnet_dict)
        tracker = tracking.Tracker(**tracker_kwargs)
        return tracker

    @staticmethod
    def _check_check(widgets, errormessage):
        checks = [x.isChecked() for x in widgets]
        if sum(checks) != 1:
            raise ValueError(errormessage)

    def calibrate_screen(self):
        self.clear_screen_plots()
        n_images = int(self.CalibrateScreenImages.text())
        image_dict = daq.get_images(self.screen, n_images, dry_run=self.dry_run)
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
            meta_dict = None
        elif self.SetStreakerFromLiveCheck.isChecked():
            self.obtain_streaker_settings_from_live()
            if daq is None:
                raise RuntimeError('Cannot get settings from live!')
            meta_dict = daq.get_meta_data(self.screen)

        elif self.SetStreakerSaveCheck:
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

        print('Streaker is set: gaps: %s, offsets: %s' % (gaps, streaker_offsets))
        return gaps, streaker_offsets

    def reconstruct_current(self):
        self.clear_rec_plots()
        filename = self.ReconstructionDataLoad.text().strip()
        streaker_means = self.streaker_means
        print('Streaker calibrated: mean = %i, %i um' % (streaker_means[0]*1e6, streaker_means[1]*1e6))

        rec_mode = self.ReconstructionDataLoadUseSelect.currentText()

        print('Obtained reconstruction data')

        if self.ShowBlmeasCheck.isChecked():
            blmeas_file = self.BunchLengthMeasFile.text()
        else:
            blmeas_file = None

        gauss_kwargs = self.get_gauss_kwargs()
        tracker_kwargs = self.get_tracker_kwargs()
        self.current_rec_dict = analysis.reconstruct_current(filename, self.n_streaker, self.beamline, tracker_kwargs, rec_mode, gauss_kwargs, self.screen_x0, self.streaker_means, blmeas_file, self.reconstruction_plot_handles)

        self.rec_canvas.draw()
        self.tabWidget.setCurrentIndex(self.rec_plot_tab_index)

    def save_current_rec_data(self):
        if self.current_rec_dict is None:
            raise ValueError('No current reconstruction to save!')

        save_path = self.save_dir
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        date = datetime.now()
        basename = date.strftime('%Y_%m_%d-%H_%M_%S_PassiveReconstruction.h5')
        elog_text = 'Passive current reconstruction'
        elog_text +='\nComment: %s' % self.CurrentElogComment.text()
        self.elog_and_H5(elog_text, [self.reconstruction_fig], 'Passive current reconstruction', basename, self.current_rec_dict)

    @property
    def save_dir(self):
        return os.path.expanduser(self.SaveDir.text())

    def update_streaker(self):
        beamline = self.beamline
        self.streaker_name = config.streaker_names[beamline][self.n_streaker]
        self.StreakerName.setText(self.streaker_name)

    def calibrate_streaker(self):

        self.clear_calib_plots()
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

        result_dict = daq.data_streaker_offset(streaker, range_, self.screen, n_images, self.dry_run)

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
        self.updateStreakerCenter(streaker_offset)

        elog_text = 'Streaker calibration streaker %s\nCenter: %i um' % (streaker, streaker_offset*1e6)
        self.elog_and_H5(elog_text, [self.streaker_calib_fig], 'Streaker center calibration', basename, full_dict)
        self.tabWidget.setCurrentIndex(self.streaker_calib_plot_tab_index)

    def _analyze_streaker_calib(self, result_dict):
        forward_blmeas = self.ForwardBlmeasCheck.isChecked()
        tracker = self.get_tracker(result_dict['meta_data_begin'])
        if forward_blmeas:
            blmeasfile = self.ForwardBlmeasFilename.text()
        else:
            blmeasfile = None

        streaker = result_dict['streaker']
        #gap0 = result_dict['meta_data_begin'][streaker+':GAP']*1e-3
        #gap_arr = [gap0-100e-6, gap0+50e-6]
        #gauss_kwargs = self.get_tracker_kwargs()
        #gap = sc.gap_reconstruction2(gap_arr, tracker, gauss_kwargs)
        beamline, n_streaker = analysis.get_beamline_n_streaker(streaker)

        full_dict = sc.analyze_streaker_calibration(result_dict, do_plot=True, plot_handles=self.streaker_calib_plot_handles, forward_propagate_blmeas=forward_blmeas, tracker=tracker, blmeas=blmeasfile, beamline=beamline)
        return full_dict

    def gap_reconstruction(self):
        self.clear_gap_recon_plots()

        filename = self.LoadCalibrationFilename.text().strip()
        saved_dict = h5_storage.loadH5Recursive(filename)

        if 'raw_data' in saved_dict:
            saved_dict = saved_dict['raw_data']

        tracker = self.get_tracker(saved_dict['meta_data_begin'])
        gauss_kwargs = self.get_gauss_kwargs()
        gap_recon_dict = sc.reconstruct_gap(saved_dict, tracker, gauss_kwargs, plot_handles=self.gap_recon_plot_handles)
        n_streaker = gap_recon_dict['n_streaker']
        delta_gap = gap_recon_dict['delta_gap']
        gap = gap_recon_dict['gap']
        self.updateDeltaGap(delta_gap, n_streaker)
        print('Reconstructed gap: %.3f mm' % (gap*1e3))
        self.gap_recon_canvas.draw()

    def load_calibration(self):
        self.clear_calib_plots()
        filename = self.LoadCalibrationFilename.text().strip()
        saved_dict = h5_storage.loadH5Recursive(filename)

        if 'raw_data' in saved_dict:
            saved_dict = saved_dict['raw_data']
        full_dict = self._analyze_streaker_calib(saved_dict)

        streaker_offset = full_dict['meta_data']['streaker_offset']
        self.updateStreakerCenter(streaker_offset)
        #self.tabWidget.setCurrentIndex(self.streaker_calib_plot_tab_index)
        if self.streaker_calib_canvas is not None:
            self.streaker_calib_canvas.draw()

    def updateStreakerCenter(self, streaker_offset, n_streaker=None):
        if n_streaker is None:
            n_streaker = self.n_streaker
        if self.n_streaker == 0:
            widget = self.StreakerDirect0
        elif self.n_streaker == 1:
            widget = self.StreakerDirect1
        old = float(widget.text())
        widget.setText('%.3f' % (streaker_offset*1e6))
        new = float(widget.text())
        print('Updated center calibration for streaker %i. Old: %.3f um New: %.3f um' % (self.n_streaker, old, new))

    def updateDeltaGap(self, delta_gap, n_streaker=None):
        if n_streaker is None:
            n_streaker = self.n_streaker
        if self.n_streaker == 0:
            widget = self.StreakerGapDelta0
        elif self.n_streaker == 1:
            widget = self.StreakerGapDelta1
        old = float(widget.text())
        widget.setText('%.3f' % (delta_gap*1e6))
        new = float(widget.text())
        print('Updated gap calibration for streaker %i. Old: %.3f um New: %.3f um' % (self.n_streaker, old, new))

    def obtain_reconstruction(self):
        n_images = int(self.ReconNumberImages.text())
        screen_dict = daq.get_images(self.screen, n_images, dry_run=self.dry_run)
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
    def charge(self):
        return float(self.Charge.text())*1e-12

    @property
    def streaker_offsets(self):
        return np.array([float(self.StreakerOffset0.text()), float(self.StreakerOffset1.text())])*1e-3

    @property
    def streaker_means(self):
        streaker0_mean = float(self.StreakerDirect0.text())*1e-6
        streaker1_mean = float(self.StreakerDirect1.text())*1e-6
        return np.array([streaker0_mean, streaker1_mean])

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

        image_dict = daq.get_images(self.screen, n_images, dry_run=self.dry_run)
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

    @staticmethod
    def get_energy_from_meta(meta_data):
        if 'SARBD01-MBND100:ENERGY-OP' in meta_data:
            energy_eV = meta_data['SARBD01-MBND100:ENERGY-OP']*1e6
        elif 'SARBD01-MBND100:P-SET' in meta_data:
            energy_eV = meta_data['SARBD01-MBND100:P-SET']*1e6
        return energy_eV

    def reconstruct_all_lasing(self):
        self.clear_all_lasing_plots()

        screen_x0 = self.screen_x0
        beamline, n_streaker = self.beamline, self.n_streaker
        charge = self.charge
        streaker_offset = self.streaker_means[n_streaker]
        delta_gap = self.delta_gaps[n_streaker]
        pulse_energy = float(self.LasingEnergyInput.text())*1e-6
        slice_factor = int(self.LasingReconstructionSliceFactor.text())

        file_on = self.LasingOnDataLoad.text()
        file_off = self.LasingOffDataLoad.text()
        lasing_off_dict = h5_storage.loadH5Recursive(file_off)
        lasing_on_dict = h5_storage.loadH5Recursive(file_on)

        tracker_kwargs = self.get_tracker_kwargs()
        recon_kwargs = self.get_gauss_kwargs()
        las_rec_images = {}

        for main_ctr, (data_dict, title) in enumerate([(lasing_off_dict, 'Lasing Off'), (lasing_on_dict, 'Lasing On')]):
            rec_obj = lasing.LasingReconstructionImages(screen_x0, beamline, n_streaker, streaker_offset, delta_gap, tracker_kwargs, recon_kwargs=recon_kwargs, charge=charge, subtract_median=True, slice_factor=slice_factor)

            rec_obj.add_dict(data_dict)
            if main_ctr == 1:
                rec_obj.profile = las_rec_images['Lasing Off'].profile
                rec_obj.ref_slice_dict = las_rec_images['Lasing Off'].ref_slice_dict
            rec_obj.process_data()
            las_rec_images[title] = rec_obj
            #rec_obj.plot_images('raw', title)
            #rec_obj.plot_images('tE', title)

        las_rec = lasing.LasingReconstruction(las_rec_images['Lasing Off'], las_rec_images['Lasing On'], pulse_energy, current_cutoff=1.5e3)
        las_rec.plot(plot_handles=self.all_lasing_plot_handles)
        self.all_lasing_canvas.draw()

    def save_lasing_rec_data(self):
        if self.lasing_rec_dict is None:
            raise ValueError('No lasing reconstruction data to save')
        elog_text = 'Lasing reconstruction'
        elog_text +='\nComment: %s' % self.LasingElogComment.text()
        date = datetime.now()
        screen_str = self.screen.replace('.','_')
        basename = date.strftime('%Y_%m_%d-%H_%M_%S_')+'Lasing_reconstruction_%s.h5' % screen_str
        self.elog_and_H5(elog_text, self.lasing_figs, 'Lasing reconstruction', basename, self.lasing_rec_dict)

    def elog_and_H5(self, text, figs, title, basename, data_dict):

        filename = os.path.join(self.save_dir, basename)
        h5_storage.saveH5Recursive(filename, data_dict)
        print('Saved %s' % filename)

        attachments = []
        for num, fig in enumerate(figs):
            fig_title = filename.replace('.h5', '_%i.png' % num)
            fig_filename = os.path.join(self.save_dir, fig_title)
            fig.savefig(fig_filename, bbox_inches='tight', pad_inches=0)
            print('Saved %s' % fig_filename)
            attachments.append(fig_filename)

        text += '\nData saved in %s' % filename
        text += '\nBeamline: %s' % self.beamline
        text += '\nStreaker: %s' % self.streaker
        text += '\nScreen: %s' % self.screen

        if elog is None:
            print('Cannot save to ELOG')
            print('I would post:')
            print(text)
        elif self.ElogSaveCheck.isChecked():
            dict_att = {'Author': 'Application: PostUndulatorStreakerAnalysis', 'Application': 'PostUndulatorStreakerAnalysis', 'Category': 'Measurement', 'Title': title}
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

