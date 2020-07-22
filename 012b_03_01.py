import datetime
import elegant_matrix_for_alex as elegant_matrix

data_timestamps = list(elegant_matrix.mag_data.values())[0][:,0]


year, month, day = 2020, 3, 1
times = [
        (23, 9, 30),
        (23, 50, 52),
        (23, 52, 32),
        (23, 53, 58),
        (16, 48, 57),
        (21, 48, 6),
        (21, 48, 27),
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





