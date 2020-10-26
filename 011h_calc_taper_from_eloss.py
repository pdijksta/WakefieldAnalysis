
K = 1.7
E_eV = 4542e6
eloss_eV = 2.5e-3*E_eV
n_module = 13


lambda_u = 1.5e-2
gamma = E_eV/511e3
lambda_ = lambda_u/(2*gamma**2) * (1 + K**2/2)

rel_eloss = eloss_eV/E_eV
del_k = (4 * lambda_) / (K*lambda_u) * gamma**2 * rel_eloss

rel_taper = del_k/K/n_module
print(rel_taper*1e4)


