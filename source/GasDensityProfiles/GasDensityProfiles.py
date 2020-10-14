import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as colors
from scipy.optimize import curve_fit
from scipy import integrate as intgr
from matplotlib import gridspec
import matplotlib as mpl
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
import sys
import os

G_cm3_gi_si2 = 6.672599e-8
G_kpc_km2_Msi_si2 = 4.3e-6
rhocrit_Ms_kpci3 = 2.7755e2
rhocrit_g_cmi3 = 1.9790e-29
kpc_to_cm = 3.085678e21
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
delta_vir = 200.

def A(cNFW):
	return 1./cNFW*np.sqrt(2.*(np.log(1.+cNFW) - cNFW/(1.+cNFW)))

def cH(cNFW):
	return (1 - A(cNFW))/A(cNFW)

def r_vir(Mvir):
	return (3./4./np.pi*Mvir/delta_vir/rhocrit_Ms_kpci3)**(1./3.)

# def cnfw2(Mvir):
# 	masses = np.array([1e6, 5e6, 1e7, 5e7, 1e8, 5e8, 1e9, 5e9, 1e10, 5e10, 1e11, 5e11, 1e12, 5e12, 
# 		1e13, 5e13, 1e14, 5e14, 1e15])
# 	CNFWs = np.array([1948, 1822, 1768, 1639, 1583, 1456, 1399, 1267, 1210, 1073, 1016, 890, 835, 711, 659, 
# 		544, 499, 406, 371])/100.
# 	result = interp1d(masses, CNFWs)
# 	return result(Mvir)

def cnfw(Mvir, z=0):
	if z > 3:
		sys.exit('Error: redshift out of bounds')
	data = np.genfromtxt('Planck_cmz_z0z1z2z3.dat', names=True)
	redshift = data['z']
	mass = 10**(10.+data['mass'])

	cnfw = data['cnfw']
	if z == 0:
		result = interp1d(mass[np.where(redshift==0)[0]], cnfw[np.where(redshift==0)[0]])
		return result(Mvir)
	else:
		Mtemp = np.zeros(4)
		for i in range(4):
			temp = interp1d(mass[np.where(redshift==i)[0]], cnfw[np.where(redshift==i)[0]])
			Mtemp[i] = temp(Mvir)
		ztemp = np.array([0, 10, 100, 1000])
		result = interp1d(ztemp, Mtemp)
		return result(10**z)


def mH(x):
	return x**2/(1+x)**2

def mNFW(x):
	return np.log(1 + x) - x/(1 + x)

def mDM_Hernquist(Mvir, cNFW = False, z=0):
	if not cNFW:
		cNFW = cnfw(Mvir, z=z)
	return Mvir/(1. - A(cNFW))**2

def dmDensityProfileHernquist(r, Mvir, integrate=False, cNFW = False, DMFraction = 0.84, a=False, z=0):
	if not cNFW:
		cNFW = cnfw(Mvir, z=z)
	r_v = r_vir(Mvir) * kpc_to_cm
	if not a:
		a = r_v/cH(cNFW)
	else:
		a *= kpc_to_cm
	Mdm = mDM_Hernquist(Mvir, cNFW)
	factor = 1
	if integrate:
		factor = 4.*np.pi*r*kpc_to_cm*r*kpc_to_cm*kpc_to_cm*hydrogen_g/Msun_g
	return DMFraction/0.96*factor*Mdm*Msun_g/hydrogen_g /2./np.pi*a/(r*kpc_to_cm*(r*kpc_to_cm+a)**3)

def starMassProfileHernquist(r, Mvir, Mstar, scalefraction=0.01, gasdistr='NFW', cNFW = False, z=0):
	"""
	In Msun kpc^-3
	"""
	if not cNFW:
		cNFW = cnfw(Mvir, z=0)
	r_v = r_vir(Mvir)
	if gasdistr=='NFW':
		r_s = r_v/cNFW
	elif gasdistr=='Hern':
		r_s = r_v/cH(cNFW)
	a = scalefraction*r_s
	return Mstar*( r/a )**2 / ( 1 + r/a )**2

def radius_particle_hern(a, Npart):
	particlefrac = Npart/np.arange(1, int(Npart + 1), 1)
	particlefrac[len(particlefrac)-1] = ( particlefrac[len(particlefrac)-2] - 
		0.99 * ( particlefrac[len(particlefrac)-2] - particlefrac[len(particlefrac)-1] ) )
	radius = np.zeros(len(particlefrac))
	radius[:] = (2.*a*np.random.rand(len(radius)) + a/(np.sqrt(particlefrac) - 1.))
	return np.array(sorted(radius))

def dmMassNFW(r, Mvir, Rvir=None, cNFW=None, z=0):
	if cNFW is None:
		if isinstance(z, np.ndarray):
			cNFW = np.zeros(len(z))
			for i in range(len(z)):
				cNFW[i] = cnfw(Mvir[i], z=np.min([z[i], 3]))
		else:
			cNFW = cnfw(Mvir, z=z)
	r_v = r_vir(Mvir) * kpc_to_cm
	r_s = r_v/cNFW 
	return (mNFW(r/r_s)) * Mvir/(mNFW(cNFW))

def dmDensityProfileNFW(r, Mvir, integrate=False, cNFW = False, DMFraction = 0.84, z=0):
	if not cNFW:
		cNFW = cnfw(Mvir, z=z)
	r_v = r_vir(Mvir) * kpc_to_cm
	r_s = r_v/cNFW
	rho_0 = DMFraction*Mvir*Msun_g/hydrogen_g/(4*np.pi*r_s**3*(mNFW(cNFW)))
	factor = 1
	if integrate:
		factor = 4.*np.pi*(r*kpc_to_cm)**2 * kpc_to_cm * hydrogen_g/Msun_g
	return factor*rho_0/(r*kpc_to_cm/r_s * (1 + r*kpc_to_cm/r_s)**2)

def gasDensityProfileHernquist(r, Mvir, integrate=False, cNFW = False, DMFraction = 0.84, gamma=False, z=0):#particles/cm^3
	if not cNFW:
		cNFW = cnfw(Mvir, z=z)
	r_v = r_vir(Mvir) * kpc_to_cm
	a = r_v/cH(cNFW)
	eta0 = 0.3435*cNFW**0.9020 + 1.075
	if gamma == False:
		gamma = -0.1637*np.log(cNFW)**-0.6614 + 1.312
	gasDensHernNorm = 1.4280*cNFW**(0.04213) - 1.4001

	rho_0 = (1.-DMFraction)/DMFraction * Mvir * Msun_g / hydrogen_g / (4*np.pi*a**3*(mH(r_v/a))) / gasDensHernNorm
	y = 1. - 3. / eta0 * (gamma - 1) / gamma / mH(cH(cNFW)) * cH(cNFW) * (1.-1./(1+r*kpc_to_cm/a))

	factor = 1
	if integrate:
		factor = 4.*np.pi*r*kpc_to_cm*r*kpc_to_cm*kpc_to_cm*hydrogen_g/Msun_g
	return factor*rho_0*y**(1/(gamma-1))

def eta_0_Hernquist(x, cH, gamma):
	s_star = -(1 + 3*x/(1+x))
	return 1./gamma * ( -3./s_star * mH(x)*cH/mH(cH)/x + 3.*(gamma - 1)*cH/mH(cH) * ( 1. - 1./(1.+x)) )

def eta_0_NFW(x, cNFW, gamma):
	s_star = -(1 + 2.*x/(1+x))
	return 1./gamma * ( -3./s_star * mNFW(x)*cNFW/mNFW(cNFW)/x + 3.*(gamma - 1)*cNFW/mNFW(cNFW) * ( 1. - np.log(1. + x)/x ) )

def gasDensityProfileNFW_Integrate(r, Mvir, cNFW, shape):
	r_v = r_vir(Mvir) * kpc_to_cm
	r_s = r_v/cNFW
	gamma = 1.15 + 0.01*(cNFW - 6.5)
	eta_0 = eta_0_NFW(shape*cNFW, cNFW, gamma)

	rho_0 = Mvir*Msun_g/hydrogen_g / (4*np.pi*r_s**3*mNFW(cNFW))
	y = 1. - 3. / eta_0 * (gamma - 1) / gamma / mNFW(cNFW) * cNFW * (1 - np.log(1 + r*kpc_to_cm/r_s)/(r*kpc_to_cm/r_s))

	factor = 4.*np.pi*r*kpc_to_cm*r*kpc_to_cm*kpc_to_cm*hydrogen_g/Msun_g
	result = factor*rho_0 * y**(1/(gamma-1))
	if np.isnan(result):
		result = 0.
	return result


def gasDensityProfileNFW(r, Mvir, integrate=False, cNFW = False, DMFraction = 0.84, shape = 1, z=0): #particles/cm^3
	"""
	Output: particles/cm^3
	Input:	r in kpc
			Mvir in Msun
	"""
	if not isinstance(Mvir, (list, tuple, np.ndarray)):
		if not cNFW:
			cNFW = cnfw(Mvir, z=z)

	r_v = r_vir(Mvir) * kpc_to_cm
	r_s = r_v/cNFW
	gamma = 1.15 + 0.01*(cNFW - 6.5)

	if shape:
		gasDensNFWnorm = intgr.quad(gasDensityProfileNFW_Integrate, 0, r_v/kpc_to_cm, args=(Mvir, cNFW, shape))[0]/Mvir
		eta_0 = eta_0_NFW(shape*cNFW, cNFW, gamma)
	else:
		gasDensNFWnorm = 5.5224*np.exp(0.2464*cNFW - 4.7292) + 0.3003
		eta_0 = 0.00676*(cNFW - 6.5)**2 + 0.206*(cNFW - 6.5) + 2.48

	rho_0 = (1.-DMFraction)/DMFraction*Mvir*Msun_g/hydrogen_g / (4*np.pi*r_s**3*mNFW(cNFW)) / gasDensNFWnorm
	#rho_0 = Mvir*Msun_g/hydrogen_g / (4*np.pi*r_s**3*mNFW(cNFW))
	y = 1 - 3. / eta_0 * (gamma - 1) / gamma / mNFW(cNFW) * cNFW * (1 - np.log(1 + r*kpc_to_cm/r_s)/(r*kpc_to_cm/r_s))

	factor = 1
	if integrate:
		factor = 4.*np.pi*r*kpc_to_cm*r*kpc_to_cm*kpc_to_cm*hydrogen_g/Msun_g
	if not isinstance(r, (list, tuple, np.ndarray)):
		if r == 0:
			return factor * rho_0
	else:
		y[np.where(r == 0)[0]] = 0
	return factor*rho_0 * y**(1/(gamma-1))

def gasTemperatureProfileNFW(r, Mvir, cNFW = False, shape = False, z=0):
	if not cNFW:
		cNFW = cnfw(Mvir, z=z)

	r_v = r_vir(Mvir) * kpc_to_cm
	r_s = r_v/cNFW
	gamma = 1.15 + 0.01*(cNFW - 6.5)

	if shape:
		eta_0 = eta_0_NFW(shape*cNFW, cNFW, gamma)
	else:
		eta_0 = 0.00676*(cNFW-6.5)**2 + 0.206*(cNFW-6.5) + 2.48

	T_0 = G_cm3_gi_si2*Mvir*Msun_g*hydrogen_g/r_v/3.0/kB_erg_Ki*eta_0
	y = 1 - 3. / eta_0 * (gamma - 1) / gamma / mNFW(cNFW) * cNFW * (1 - np.log(1 + r*kpc_to_cm/r_s)/(r*kpc_to_cm/r_s))
	if not isinstance(r, (list, tuple, np.ndarray)):
		if r == 0:
			return T_0
	else:
		y[np.where(r == 0)[0]] = 0
	return T_0*y

def gasTemperatureProfileHernquist(r, Mvir, cNFW = False, shape = False, z=0):
	if not cNFW:
		cNFW = cnfw(Mvir, z=z)
	r_v = r_vir(Mvir) * kpc_to_cm
	a = r_v/cH(cNFW)
	gamma = -0.1637*np.log(cNFW)**-0.6614 + 1.312
	if shape:
		eta_0 = eta_0_Hernquist(shape*cH(cNFW), cH(cNFW), gamma)
	else:
		eta_0 = 0.3435*cNFW**0.9020 + 1.075
		

	T_0 = G_cm3_gi_si2*Mvir*Msun_g*hydrogen_g/r_v/3.0/kB_erg_Ki*eta_0
	y = 1. - 3. / eta_0 * (gamma - 1) / gamma / mH(cH(cNFW)) * cH(cNFW) * (1.-1./(1+r*kpc_to_cm/a))

	return T_0*y

def gasUProfileNFW(r, Mvir, cNFW = False, z=0):
	if not cNFW:
		cNFW = cnfw(Mvir, z=z)
	gamma = 1.15 + 0.01*(cNFW - 6.5)
	return gasTemperatureProfileNFW(r, Mvir, cNFW = cNFW) * kB_erg_Ki / hydrogen_g * gamma

def gasUProfileHernquist(r, Mvir, cNFW = False, z=0):
	if not cNFW:
		cNFW = cnfw(Mvir, z=z)
	gamma = -0.1637*np.log(cNFW)**-0.6614 + 1.312
	return gasTemperatureProfileHernquist(r, Mvir, cNFW = cNFW) * kB_erg_Ki / hydrogen_g * gamma

def cooling(r, n, T):
	rnieuw = r*kpc_to_cm
	TLam = np.genfromtxt("/home/luciebakels/Code/Radiation/MPIPopMake/InputFiles/TemperatureCoolingTF.txt", names=True)
	Lambda = interp1d(TLam['T'], TLam['F'])	
	nhier = interp1d(rnieuw, n)
	T[np.where(T>np.max(TLam['T']))[0]] = np.max(TLam['T'])
	T[np.where(T<np.min(TLam['T']))[0]] = np.min(TLam['T'])
	Thier = interp1d(rnieuw, T)
	return Lambda(Thier(rnieuw))*nhier(rnieuw)**2*nH_fraction**2

def coolingTime(r, n, T):
	return 3.0 * n * kB_erg_Ki * T/ 2.0 / cooling(r, n, T)

def hubbleCoolingRadius(Mvir):
	hubbletime = 4.55e17 #s
	r = np.logspace(-2, 1, 100)*r_vir(Mvir)
	n = gasDensityProfileNFW(r, Mvir)
	n[np.where(np.isnan(n))] = 0.0
	T = gasTemperatureProfileNFW(r, Mvir)
	radius = interp1d(coolingTime(r, n, T), r)
	return radius(hubbletime)

def setCooling(r, n, T):
	rnieuw = r*kpc_to_cm
	TLam = np.genfromtxt("/home/luciebakels/Code/Radiation/MPIPopMake/InputFiles/TemperatureCoolingTF.txt", names=True)
	Lambda = interp1d(TLam['T'], TLam['F'])
	nhier = interp1d(rnieuw, n)
	Thier = interp1d(rnieuw, T)
	#cooling = Lambda(T)*n*n*nH_fraction*nH_fraction
	def coolingint(rnieuw, nhier, Thier, Lambda):
		if nhier(rnieuw) < 0.0 or Thier(rnieuw) < 0.0:
			return 0.0
		elif Thier(rnieuw) > np.max(TLam['T']):
			return Lambda(np.max(TLam['T']))*(nhier(rnieuw)*nH_fraction*rnieuw)**2 * 4*np.pi
		elif Thier(rnieuw) < np.min(TLam['T']):
			return Lambda(np.min(TLam['T']))*(nhier(rnieuw)*nH_fraction*rnieuw)**2 * 4*np.pi
		else:
			return Lambda(Thier(rnieuw))*nhier(rnieuw)*nhier(rnieuw)*nH_fraction*nH_fraction*4.*np.pi*rnieuw*rnieuw
	coolintegrated = intgr.quad(coolingint, rnieuw[0], rnieuw[np.argmin(T*T)-1], args=(nhier, Thier, Lambda))
	return coolintegrated

def totalcooling(Mvir, shape, cNFW = False, distr = 'NFW'):
	r = np.logspace(-2, 1, 100)*r_vir(Mvir)
	n = gasDensityProfileNFW(r, Mvir, cNFW = cNFW, shape = shape)
	n[np.where(np.isnan(n))] = 0.0
	T = gasTemperatureProfileNFW(r, Mvir, cNFW = cNFW, shape = shape)
	return setCooling(r, n, T)[0]

def integratedcooling(rmax, Mvir, shape, cNFW = False, distr = 'NFW'):
	r = np.logspace(-2, 0, 100)*rmax
	n = gasDensityProfileNFW(r, Mvir, cNFW = cNFW, shape = shape)
	n[np.where(np.isnan(n))] = 0.0
	T = gasTemperatureProfileNFW(r, Mvir, cNFW = cNFW, shape = shape)
	return setCooling(r, n, T)[0]