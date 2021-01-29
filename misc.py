import re
import numpy as np

import tracking
import elegant_matrix

def find_rising_flank(arr, method='Size'):
    """
    Method can be 'Length' or 'Size'
    """
    arr = arr.copy()
    #arr[arr<arr.max()*0.01] = 0
    prev_val = -np.inf
    start_index = None
    len_ctr = 0
    pairs = []
    for index, val in enumerate(arr):
        if val > prev_val:
            if start_index is None:
                start_index = index - 1
                len_ctr = 1
                start_val = val
            else:
                len_ctr += 1
        else:
            if start_index is not None:
                if method == 'Length':
                    pairs.append((len_ctr, start_index, index))
                elif method == 'Size':
                    pairs.append((prev_val-start_val, start_index, index))
                start_index = None
                start_val = None
            len_ctr = 0
        prev_val = val
    #import pdb
    #pdb.set_trace()
    end_longest_streak = sorted(pairs)[-1][-1]
    return end_longest_streak


def avergage_BeamProfiles(bp_list, align='Max'):
    all_profiles_time, all_profiles_current = [], []
    for profile in bp_list:
        if align == 'Max':
            center_index = np.argmax(profile.current)
        elif align == 'Left':
            center_index = find_rising_flank(profile.current)
        elif align == 'Right':
            center_index = len(profile.current) - find_rising_flank(profile.current[::-1])

        all_profiles_time.append(profile.time - profile.time[center_index])
    else:
        new_time = np.linspace(min((x.min() for x in all_profiles_time)), max((x.max() for x in all_profiles_time)), len(bp_list[0]._xx))
        for profile in bp_list:
            all_profiles_current.append(np.interp(new_time, (profile.time), (profile.current), left=0, right=0))
        else:
            all_profiles_current = np.array(all_profiles_current)
            average_profile = tracking.BeamProfile(new_time, np.mean(all_profiles_current, axis=0), bp_list[0].energy_eV, bp_list[0].charge)
            error_bar = np.std(bp_list, axis=0)
            return (average_profile, error_bar)


def fit_nat_beamsize(screen_meas, screen_sim, emittance, print_=False):
    sig_meas = screen_meas.gaussfit.sigma
    sig_sim = screen_sim.gaussfit.sigma
    emittance_fit = emittance * (sig_meas / sig_sim) ** 2
    if print_:
        print('Old emittance: %.2e, New emittance: %.2e Old beamsize %.2e New beamsize %.2e' % (emittance, emittance_fit, sig_sim, sig_meas))
    return emittance_fit


re_file = re.compile('Passive_data_(\\d{4})(\\d{2})(\\d{2})T(\\d{2})(\\d{2})(\\d{2}).mat')

def get_timestamp(filename):
    match = re_file.match(filename)
    if match is None:
        print(filename)
        raise ValueError
    args = [int(x) for x in match.groups()]
    return (elegant_matrix.get_timestamp)(*args)


