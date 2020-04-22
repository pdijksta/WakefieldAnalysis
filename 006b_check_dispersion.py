from ElegantWrapper.watcher import FileViewer

disp = 0.4284*1e6 # for delta / y
disp1 = 0.198*1000 # for delta1 / y1 = BPM1 = DBPM040
disp2 = 0.3978*1000 # for delta2 / y2 = BPM2 = DBPM040


viewer = FileViewer('./SwissFEL3.mat.h5')

elements = viewer['ElementName']

screens = [(index, x) for index, x in enumerate(elements) if ('DSCR' in x or 'DBPM' in x)]

disp = viewer['R36']
r12 = viewer['R12']


for index, screen in screens:
    print(screen, disp[index], r12[index])
    # Why is dispersion so low?

