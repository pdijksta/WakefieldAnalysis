import datetime
import elegant_matrix


data_timestamps = list(elegant_matrix.mag_data.values())[0][:,0]

year, month, day = 2020, 8, 26
times = [
        (21, 55, 0),
        (21, 57, 0),
        (22, 0, 0),
        #(11, 0, 0),
        #(11, 21, 00),
        #(17, 34, 33),
        #(17, 57, 47),
        ]

for hour, minute, second in times:
    date = datetime.datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
    timestamp = int(date.strftime('%s'))
    mat0 = elegant_matrix.get_elegant_matrix(99, timestamp, del_sim=False, print_=False, branch='Athos')


    dscr0 = 'SATBD02.DSCR050'

    print('Streaker 1 to %s at %02i:%02i:%02i R12=%.3e' % (dscr0, hour, minute, second, mat0[dscr0][0,1]))


