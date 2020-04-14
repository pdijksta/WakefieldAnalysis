from ElegantWrapper.watcher import FileViewer

viewer = FileViewer('./default.twi.h5')
s_names = []

for s, name in zip(viewer['s'], viewer['ElementName']):
    if s > 566 and ('DSCR' in name or 'BPM' in name or 'UDCP' in name or 'MQUA.COR' in name):
        print(round(s-567.85, 2), name)

