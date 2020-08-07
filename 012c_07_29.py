import datetime
import elegant_matrix_for_alex as elegant_matrix

data_timestamps = list(elegant_matrix.mag_data.values())[0][:,0]


year, month, day = 2020, 7, 26
times = [
        (10, 42, 53),
        (11, 0, 0),
        (11, 21, 00),
        (17, 34, 33),
        (17, 57, 47),
        ]

for hour, minute, second in times:
    date = datetime.datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
    timestamp = int(date.strftime('%s'))
    mat0 = elegant_matrix.get_elegant_matrix(0, timestamp, del_sim=False, print_=False)
    mat1 = elegant_matrix.get_elegant_matrix(1, timestamp, del_sim=False, print_=False)


    dscr0 = 'SARBD01.DSCR050'
    dscr1 = 'SARBD02.DSCR050'

    for dscr in (dscr0, dscr1):
        print('Streaker 1 to %s at %02i:%02i:%02i R12=%.3e' % (dscr, hour, minute, second, mat0[dscr][0,1]))
        print('Streaker 2 to %s at %02i:%02i:%02i R12=%.3e' % (dscr, hour, minute, second, mat1[dscr][0,1]))





