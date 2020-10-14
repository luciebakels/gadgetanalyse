import numpy as np
from collections import OrderedDict

G_cm3_gi_si2 = 6.672599e-8
G_kpc_km2_Msi_si2 = 4.301313e-6
G_Mpc_km2_Msi_si2 = 4.301313e-9

kpc_to_cm = 3.085678e21
kpc_to_km = 3.085678e16
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
nH_fraction = (12.0/27.0)
deltaVir = 200.
rhocrit_Ms_Mpci3_com = 3*(100./Mpc_to_km)**2/(8*np.pi*G_Mpc_km2_Msi_si2/(Mpc_to_km)**2)
MassTable8 = [0.0026613709153710937,0.014255408365429689,0.0,0.0,0.0,0.0]
MassTable = [3.326713644213867E-4,0.0017819260456787111,0.0,0.0,0.0,0.0]
h = 0.6751
Om0 = 0.3121
Ob0 = 0.0491
class constant:
	def __init__(self, redshift=0, H0=67.51, Om0=0.3121, Ob0=0.0491):
		self.H0 = H0
		self.Om0 = Om0
		self.Ob0 = Ob0
		self.redshift = redshift
		self.H = self.H0*np.sqrt(self.Om0*(1.+self.redshift)**3 + (1.-self.Om0))
		#self.h = self.H/100.
		self.rhocrit_Ms_kpci3 = 3*(self.H0/kpc_to_km/1e3)**2/(8*np.pi*G_kpc_km2_Msi_si2/(kpc_to_km)**2) #2.7755e2*h^2
		self.rhocrit_Ms_Mpci3 = 3*(self.H0/Mpc_to_km)**2/(8*np.pi*G_Mpc_km2_Msi_si2/(Mpc_to_km)**2)
		self.rhocrit_g_cmi3 = 3*(self.H0/Mpc_to_cm*1e5)**2/(8*np.pi*G_cm3_gi_si2)

	def change_constants(self, redshift):
		self.redshift = redshift
		self.H = self.H0*np.sqrt(self.Om0*(1.+self.redshift)**3 + (1.-self.Om0)) #km/s/Mpc
		#self.h = self.H/100.
		self.rhocrit_Ms_kpci3 = 3*(self.H/kpc_to_km/1e3)**2/(8*np.pi*G_kpc_km2_Msi_si2/(kpc_to_km)**2)
		self.rhocrit_Ms_Mpci3 = 3*(self.H/Mpc_to_km)**2/(8*np.pi*G_Mpc_km2_Msi_si2/(Mpc_to_km)**2)
		self.rhocrit_g_cmi3 = 3*(self.H/Mpc_to_cm*1e5)**2/(8*np.pi*G_cm3_gi_si2)
		self.rhocrit_Ms_kpci3_com = self.rhocrit_Ms_kpci3/(1.+self.redshift)**3
		self.rhocrit_Ms_Mpci3_com = self.rhocrit_Ms_Mpci3/(1.+self.redshift)**3
		self.rhocrit_g_cmi3_com = self.rhocrit_g_cmi3/(1.+self.redshift)**3
		self.rhocrit_Ms_kpci3_com_h = self.rhocrit_Ms_kpci3_com*h/(h**3)
		self.rhocrit_Ms_Mpci3_com_h = self.rhocrit_Ms_Mpci3_com*h/(h**3)
		self.rhocrit_g_cmi3_com_h = self.rhocrit_g_cmi3_com*h/(h**3)
		self.rhocrit_Ms_kpci3_h = self.rhocrit_Ms_kpci3*h/(h**3)
		self.rhocrit_Ms_Mpci3_h = self.rhocrit_Ms_Mpci3*h/(h**3)
		self.rhocrit_g_cmi3_h = self.rhocrit_g_cmi3*h/(h**3)

linestyles = OrderedDict(
    [('solid',               (0, ())),
     ('loosely dotted',      (0, (1, 10))),
     ('dotted',              (0, (1, 5))),
     ('densely dotted',      (0, (1, 1))),

     ('loosely dashed',      (0, (5, 10))),
     ('dashed',              (0, (5, 5))),
     ('densely dashed',      (0, (5, 1))),

     ('loosely dashdotted',  (0, (3, 10, 1, 10))),
     ('dashdotted',          (0, (3, 5, 1, 5))),
     ('densely dashdotted',  (0, (3, 1, 1, 1))),

     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])