import numpy as np
import wf_model


xx = np.exp(np.linspace(np.log(0.1e-6), np.log(40e-6), int(1000)))
xx -= xx.min()

wf_model.generate_elegant_wf('./test.sdds', xx, 5e-3, 2e-3, L=1)


