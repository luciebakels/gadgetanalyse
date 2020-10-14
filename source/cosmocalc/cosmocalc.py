import numpy as np
from scipy import integrate

G_cm3_gi_si2 = 6.672599e-8
G_kpc_km2_Msi_si2 = 4.301313e-6
G_Mpc_km2_Msi_si2 = 4.301313e-9
rhocrit_Ms_kpci3 = 2.7755e2
rhocrit_g_cmi3 = 1.9790e-29
kpc_to_cm = 3.085678e21
Mpc_to_m = 3.085678e22
Mpc_to_cm = 3.085678e24
Mpc_to_km = 3.085678e19
s_to_yr = 2.893777e-8
Msun_kg = 1.9891e30
Msun_g = 1.9891e33
Lsun_erg_s = 3.846e33
Rsun_cm	= 6.9551e10
sigmaSB_erg_cm2_si_K4 = 5.6704e-5
atomicMass_kg = 1.660539e-27
h_eV_s = 4.135668e-15
h_erg_s = 1.054573e-27
c_cm_si = 2.997925e10
kB_eV_Ki = 8.613303e-5
kB_erg_Ki = 1.380649e-16
erg_to_eV = 6.2415e11
hydrogen_g = 1.673724e-24

def dt(a, H0, Om0, Ol0):
	return 1/H0*np.sqrt(Om0/a + a*a*Ol0)*Mpc_to_km

def timeDifference(z1, z2, H0, Om0, Ol0):
	a = np.sort([1./(1+z1), 1./(1+z2)])
	return integrate.quad(dt, a[0], a[1], args=(H0, Om0, Ol0))[0]