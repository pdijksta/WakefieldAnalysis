import datetime
import elegant_matrix

#data_timestamps = list(elegant_matrix.mag_data.values())[0][:,0]
simulator = elegant_matrix.get_simulator('/afs/psi.ch/intranet/SF/Beamdynamics/Philipp/data/archiver_api_data/2020-10-03.h5')


year, month = 2020, 10
times = [
        (3, 15, 0, 0),
        (3, 18, 0, 0),
        (3, 21, 0, 0),
        (4, 1, 0, 0),
        (4, 15, 0, 0),
        (4, 18, 0, 0),
        (4, 21, 0, 0),
        (4, 23, 0, 0),
        ]

for day, hour, minute, second in times:
    date = datetime.datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
    timestamp = int(date.strftime('%s'))
    #mat0 = elegant_matrix.get_elegant_matrix(0, timestamp, del_sim=False, print_=False)
    mat1, disp_dict = simulator.get_elegant_matrix(1, timestamp, del_sim=False, print_=False)


    dscr0 = 'SARBD01.DSCR050'
    dscr1 = 'SARBD02.DSCR050'

    for dscr in (dscr1,):
        #print('Streaker 1 to %s at %02i:%02i:%02i R12=%.3e' % (dscr, hour, minute, second, mat0[dscr][0,1]))
        print('Streaker 2 to %s at %04i-%02i-%02i-%02i:%02i:%02i R12=%.3e, R36=%.3e' % (dscr, year, month, day, hour, minute, second, mat1[dscr][0,1], disp_dict[dscr]))





