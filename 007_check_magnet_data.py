from datetime import datetime
import data_loader
import numpy as np

dl = data_loader.DataLoader(file_json='/storage/data_2020-02-03/2020-02-03.json1')


tt0 = list(dl.values())[0][0,0]


tt = datetime.fromtimestamp(tt0)

tt1 = datetime.now()


# Timezone seems to be correct...

last_quad = dl['SARBD02-MQUA030:I-SET']
#sarbd01_quad = dl['SARBD01-MQUA020:I-SET']

eduard_optics1 = {
        #'SARUN15.MQUA080.Q1.K1': -9.672893217266694e-01,
        #'SARUN16.MQUA080.Q1.K1': -2.443535112150988e+00,
        #'SARUN17.MQUA080.Q1.K1': +1.608546947532094e+00,
        'SARUN18.MQUA080.Q1.K1': +1.360154558769963e+00,
        'SARUN19.MQUA080.Q1.K1': -1.495693035627149e+00,
        'SARUN20.MQUA080.Q1.K1': -1.072774681910800e+00,
        'SARBD01.MQUA020.Q1.K1': -1.136049185308167e-01,
        }

eduard_optics2 = {
        #'SARUN15.MQUA080.Q1.K1': -2.810125636006008e-01,
        #'SARUN16.MQUA080.Q1.K1': -1.820840559288582e+00,
        #'SARUN17.MQUA080.Q1.K1': +1.581672326954900e+00,
        'SARUN18.MQUA080.Q1.K1': +1.203212725100745e+00,
        'SARUN19.MQUA080.Q1.K1': -8.416072303887925e-01,
        'SARUN20.MQUA080.Q1.K1': -9.584983188383855e-01,
        'SARBD01.MQUA020.Q1.K1': +4.661779344251262e-01,
}

eduard_optics1_k1l = {}
eduard_optics2_k1l = {}
for key, val in eduard_optics1.items():
    if 'SARBD01' in key:
        length = 0.3
    else:
        length = 0.08

    eduard_optics1_k1l[key] = val*length
    eduard_optics2_k1l[key] = eduard_optics2[key]*length


for n_dict, dict_ in enumerate([eduard_optics1_k1l, eduard_optics2_k1l]):

    for key, val in dict_.items():
        other_key = key.replace('.Q1.K1','').replace('.','-')+':K1L-SET'
        other_vals = np.abs(dl[other_key][:,1])
        abs_val = abs(val)
        truth_arr = np.logical_and(other_vals*0.80 < abs_val, other_vals*1.2 > abs_val)
        print(n_dict, key, np.any(truth_arr))

