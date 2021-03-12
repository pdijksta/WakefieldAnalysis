"""
No SwissFEL / PSI specific imports in this file.
Should handle analysis, saving and reloading of data.
"""
import matplotlib.pyplot as plt

import tracking

class Reconstruction:
    def __init__(self, tracker_args):
        self.tracker = tracking.Tracker(**tracker_args)
        self.tracker_args = tracker_args

    def current_profile_rec_gauss(self, args_json, args_other, do_plot, plot_handles=None):
        kwargs = {**args_json, **args_other}
        gauss_dict = self.tracker.find_best_gauss(**kwargs)

        if do_plot:
            pass

        return gauss_dict

    def save(self):
        pass

    def load(self):
        pass

