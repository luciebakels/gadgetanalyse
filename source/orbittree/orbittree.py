import numpy as np
import sys
import os
from constants import *
from snapshot import *
import haloanalyse as ha
from scipy.interpolate import interp1d
import velociraptor_python_tools as vpt
from scipy.optimize import brentq, curve_fit
import GasDensityProfiles as gdp


def snapshot_to_redshift_2048(snapshot):
	snaps = 1./np.array([1.000000,0.997025,0.981025,0.965281,0.949790,0.934547,0.919550,0.904792,0.890272,0.875985,0.861927,0.848094,
		0.834484,0.821092,0.807915,0.794949,0.782192,0.769639,0.757287,0.745134,0.733176,0.721410,0.709833,0.698441,
		0.687232,0.676203,0.665351,0.654674,0.644167,0.633830,0.623658,0.613649,0.603801,0.594111,0.584577,0.575195,
		0.565964,0.556882,0.547945,0.539151,0.530499,0.521985,0.513608,0.505366,0.497255,0.489275,0.481423,0.473697,
		0.466095,0.458615,0.451255,0.444013,0.436888,0.429877,0.422978,0.416190,0.409511,0.402939,0.396472,0.390110,
		0.383849,0.377689,0.371628,0.365664,0.359795,0.354021,0.348340,0.342750,0.337249,0.331837,0.326511,0.321271,
		0.316116,0.311043,0.306051,0.301139,0.296307,0.291551,0.286872,0.282269,0.277739,0.273281,0.268896,0.264580,
		0.260334,0.256157,0.252046,0.248001,0.244021,0.240105,0.236251,0.232460,0.228729,0.225059,0.221447,0.217893,
		0.214396,0.210956,0.207570,0.204239,0.200961,0.197736,0.194563,0.191441,0.188368,0.185345,0.182371,0.179444,
		0.176564,0.173731,0.170943,0.168199,0.165500,0.162844,0.160231,0.157659,0.155129,0.152640,0.150190,0.147780,
		0.145408,0.143075,0.140778,0.138519,0.136296,0.134109,0.131957,0.129839,0.127755,0.125705,0.123688,0.121703,
		0.119750,0.117828,0.115937,0.114076,0.112246,0.110444,0.108672,0.106928,0.105212,0.103523,0.101862,0.100227,
		0.098619,0.097036,0.095479,0.093947,0.092439,0.090955,0.089496,0.088060,0.086646,0.085256,0.083888,0.082541,
		0.081217,0.079913,0.078631,0.077369,0.076127,0.074906,0.073703,0.072521,0.071357,0.070212,0.069085,0.067976,
		0.066885,0.065812,0.064756,0.063717,0.062694,0.061688,0.060698,0.059724,0.058765,0.057822,0.056894,0.055981,
		0.055083,0.054199,0.053329,0.052473,0.051631,0.050803,0.049987,0.049185,0.048396,0.047619])[::-1] - 1
	return snaps[snapshot]

def snapshot_to_redshift(snapshot, zstart=30, numsnap=200):
	if zstart == 20:
		return snapshot_to_redshift_2048(snapshot)
	tstart = (1/(1+zstart))
	tdelta = (1/tstart)**(1/numsnap)

	if isinstance(snapshot, np.ndarray):
		antwoord = np.zeros(len(snapshot))
		antwoord[snapshot!=numsnap] = 1./(tstart*tdelta**snapshot[snapshot!=numsnap]) -1
		return antwoord
	if snapshot == numsnap:
		return 0
	return 1/(tstart*tdelta**snapshot) - 1

def redshift_to_snapshot(redshift, zstart=30, numsnap=200):
	tstart = (1/(1+zstart))
	tdelta = (1/tstart)**(1/numsnap)
	if isinstance(redshift, np.ndarray):
		antwoord = np.zeros(len(redshift))
		antwoord = np.log10(1./(redshift + 1.)/tstart)/np.log10(tdelta)
		return np.rint(antwoord).astype(int)
	return np.int(np.rint(np.log10(1./(redshift + 1.)/tstart)/np.log10(tdelta)))


def find_outputtemp_massfractions(d_output, d_nbtree, d_mhtree, massfracmin, massfracmax, keys=None, atmax=False,
	nparttype='npart_bound', preprocessed=False, unbound = False, atinfall=False, nofrac=False):
	hosts = np.array(list(d_output.keys()))
	masses = 0
	aantal = 0
	nparthost = np.zeros(len(hosts))
	i = -1

	d_outputtemp = {}
	j = 0
	if keys is None:
		keys = d_output[hosts[0]].keys()

	nhosts = 0
	for jj in hosts:
		d_outputtemp[jj] = {}
		nieterin = True
		for key in keys:
			if atinfall:
				massfraction = np.zeros(len(d_output[jj][key]))
				ww = np.where(d_nbtree[jj]['snap(R200)'][d_output[jj][key]] > 0)[0]
				if nofrac:
					massfraction[ww] = np.array([d_nbtree[jj][nparttype][i, d_nbtree[jj]['snap(R200)'][i]] for i in d_output[jj][key][ww]])
				else:	
					massfraction[ww] = [d_nbtree[jj][nparttype][i, d_nbtree[jj]['snap(R200)'][i]]/d_mhtree[jj][nparttype][d_nbtree[jj]['snap(R200)'][i]] for i in d_output[jj][key][ww]]
			elif atmax:
				massfraction = np.zeros(len(d_output[jj][key]))
				if nofrac:
					massfraction = np.array([np.max(d_nbtree[jj][nparttype][i, :]) for i in d_output[jj][key]])
				else:	
					massfraction = [d_nbtree[jj][nparttype][i, i_max]/d_mhtree[jj][nparttype][i_max] for i in np.argmax(d_nbtree[jj][nparttype][i, :])]

			else:
				if len(d_nbtree[jj][nparttype].shape)>1:
					if nofrac:
						massfraction = d_nbtree[jj][nparttype][d_output[jj][key], -1]
					else:
						massfraction = d_nbtree[jj][nparttype][d_output[jj][key], -1]/d_mhtree[jj][nparttype][-1]
				else:
					if nofrac:
						massfraction = d_nbtree[jj][nparttype][d_output[jj][key]]
					else:
						massfraction = d_nbtree[jj][nparttype][d_output[jj][key]]/d_mhtree[jj][nparttype][-1]
					
			if unbound:
				waar = np.where(
					(d_nbtree[jj]['SpecificEorb(z0)'][d_output[jj][key]]>0)&
					(massfraction>massfracmin)&
					(massfraction<=massfracmax))[0]

			elif preprocessed:
				waar = np.where(
					(d_nbtree[jj]['Preprocessed'][d_output[jj][key]]!=-1)&
					(massfraction>massfracmin)&
					(massfraction<=massfracmax))[0]
			else:
				waar = np.where(
					(massfraction>massfracmin)&
					(massfraction<=massfracmax))[0]
			if len(waar) > 0:
				nieterin = False
			d_outputtemp[jj][key] = d_output[jj][key][waar]
			aantal += len(waar)
			masses += np.sum(massfraction[waar])
		if nieterin == False:
			nhosts += 1
		j += 1
	return d_outputtemp, masses/aantal, np.arange(len(hosts)), nhosts

def find_outputtemp(hostmasses, d_output, d_nbtree, d_mhtree, massmin, massmax, min_particle_limit=50, keys=None,
	nparttype='npart', minhosts=10, preprocessed=False, bound = False, groupinfall=False, minfrac_limit=True):
	hosts = np.array(list(d_output.keys()))
	host_i = np.where((hostmasses >= massmin)&(hostmasses <= massmax))[0]
	hosts = hosts[host_i]
	if len(hosts)<minhosts:
		return None, 0, 0
	medmasses = np.median(hostmasses[host_i])

	nparthost = np.zeros(len(hosts))
	i = -1
	for jj in hosts:
		i += 1
		nparthost[i] = d_mhtree[jj][nparttype][-1]
	if minfrac_limit:
		minfrac = min_particle_limit/np.median(nparthost)
		print(minfrac, np.median(nparthost), np.min(nparthost))
	else:
		minfrac = 0

	d_outputtemp = {}
	j = 0
	if keys is None:
		keys = d_output[hosts[0]].keys()
	for jj in hosts:
		d_outputtemp[jj] = {}
		for key in keys:
			if bound:
				waar = d_output[jj][key][np.where(
					(d_nbtree[jj]['SpecificEorb_cnfw(z0)'][d_output[jj][key]]<0)&
					(d_nbtree[jj][nparttype][d_output[jj][key], -1]>=min_particle_limit)&
					(d_nbtree[jj][nparttype][d_output[jj][key], -1]>=d_mhtree[jj][nparttype][-1]*minfrac))[0]]
			elif groupinfall != False:
				waar = d_output[jj][key][np.where(
					(d_nbtree[jj][groupinfall][d_output[jj][key]]==True)&
					(d_nbtree[jj][nparttype][d_output[jj][key], -1]>=min_particle_limit)&
					(d_nbtree[jj][nparttype][d_output[jj][key], -1]>=d_mhtree[jj][nparttype][-1]*minfrac))[0]]
			elif preprocessed:
				waar = d_output[jj][key][np.where(
					(d_nbtree[jj]['Preprocessed'][d_output[jj][key]]>0)&
					(d_nbtree[jj][nparttype][d_output[jj][key], -1]>=min_particle_limit)&
					(d_nbtree[jj][nparttype][d_output[jj][key], -1]>=d_mhtree[jj][nparttype][-1]*minfrac))[0]]
			else:
				waar = d_output[jj][key][np.where(
					(d_nbtree[jj][nparttype][d_output[jj][key], -1]>=min_particle_limit)&
					(d_nbtree[jj][nparttype][d_output[jj][key], -1]>=d_mhtree[jj][nparttype][-1]*minfrac))[0]]
			d_outputtemp[jj][key] = waar
		j += 1
	return d_outputtemp, medmasses, host_i

def select_by_TreeLength(ot):
	d_output = {}
	hosts = np.array(list(ot.d_mhtree.keys()))
	for host in hosts:
		d_output[host] = {}
		for key in ot.d_categorise[hosts[0]]:
			d_output[host][key] = {}

	redshift_arr = snapshot_to_redshift(np.arange(len(ot.d_mhtree[host]['M200'])), numsnap=len(ot.d_mhtree[host]['M200'])-1)
	lbt = np.zeros(len(redshift_arr))
	for i in range(len(redshift_arr)):
		lbt[i] = timeDifference(redshift_arr[i], 0)

	for host in hosts:
		v200 = np.sqrt(G_Mpc_km2_Msi_si2*ot.d_mhtree[host]['M200'][-1]*1e10/ot.d_mhtree[host]['R200'][-1])
		tLmin = 0.5*np.pi*ot.d_mhtree[host]['R200'][-1]*Mpc_to_km/v200
		if host == hosts[0]:
			print(tLmin, np.abs(tLmin-lbt).argmin())
		waarniet = np.zeros(0)
		for key in ot.d_categorise[host].keys():
			tL = lbt[189 - ot.d_nbtree[host]['TreeLength'][ot.d_categorise[host][key].astype(int)]]
			waar =np.where(tL > tLmin)[0]

			d_output[host][key] = ot.d_categorise[host][key][waar]
			waarniet = np.append(waarniet, np.delete(ot.d_categorise[host][key],waar))
		d_output[host]['too short'] = waarniet.astype(int)

	return d_output

def select_by_npart_bound(ot, minpart=50, nparttype='npart', atinfall=False, min_snapshots=8):
	d_output = {}
	hosts = np.array(list(ot.d_mhtree.keys()))
	for host in hosts:
		d_output[host] = {}
		for key in ot.d_categorise[hosts[0]]:
			d_output[host][key] = {}

	for host in hosts:
		waarniet1 = np.zeros(0)
		waarniet2 = np.zeros(0)
		for key in ot.d_categorise[host].keys():
			if atinfall:
				waar1 =np.where(ot.d_nbtree[host][nparttype][ot.d_categorise[host][key].astype(int), 
					ot.d_nbtree[host]['snap(R200)'][ot.d_categorise[host][key].astype(int)].astype(int)] >= minpart)[0]
			else:
				if len(ot.d_nbtree[host][nparttype].shape)>1:
					waar1 =np.where(ot.d_nbtree[host][nparttype][ot.d_categorise[host][key].astype(int), -1] >= minpart)[0]
				else:
					waar1 =np.where(ot.d_nbtree[host][nparttype][ot.d_categorise[host][key].astype(int)] >= minpart)[0]
			d_output[host][key] = ot.d_categorise[host][key][waar1]
			waarniet1 = np.append(waarniet1, np.delete(ot.d_categorise[host][key],waar1))
			waar2 = np.where(ot.d_nbtree[host]['TreeLength'][d_output[host][key].astype(int)]>min_snapshots)[0]
			d_output[host][key] = d_output[host][key][waar2]
			#waar2 = waar1[waar2]
			waarniet2 = np.append(waarniet2, np.delete(d_output[host][key], waar2))
		d_output[host]['too small'] = waarniet1.astype(int)
		d_output[host]['too short'] = waarniet2.astype(int)

	return d_output

def select_by_shark_stellarmass(ot, d_output=None, minmass=1e8, maxmass=None):
	if d_output is None:
		d_output = {}
		for host in ot.hosts:
			d_output[host] = {}
			for key in ot.d_categorise[host]:
				d_output[host][key] = np.zeros(0).astype(int)

	for host in ot.d_output.keys():
		for key in ot.d_output[host].keys():
			waar1 = np.where(ot.d_nbtree[host]['mstars_bulge_main'][ot.d_categorise[host][key]] + 
				ot.d_nbtree[host]['mstars_disk_main'][ot.d_categorise[host][key]] > minmass)[0]
			if len(waar1) == 0:
				continue
			if maxmass is not None:
				waar1 = waar1[np.where(ot.d_nbtree[host]['mstars_bulge_main'][ot.d_categorise[host][key]] + 
				ot.d_nbtree[host]['mstars_disk_main'][ot.d_categorise[host][key]] < maxmass)[0]]
			d_output[host][key] = ot.d_categorise[host][key][waar1]

	return d_output
	
def select_by_npart_merged(ot, minpart=50, nparttype='npart', min_snapshots=8):
	d_output = {}
	hosts = np.array(list(ot.d_mhtree.keys()))
	for host in hosts:
		d_output[host] = {}

		waar1 =np.where(ot.d_nbtree_merger[host][nparttype][np.arange(len(ot.d_nbtree_merger[host][nparttype])).astype(int), 
			ot.d_nbtree_merger[host]['snap(R200)'].astype(int)] > minpart)[0]

		d_output[host]['all'] = waar1

		waar2 = np.where((ot.d_nbtree_merger[host]['born_within_1.0'][d_output[host]['all'].astype(int)]==False)&
			(ot.d_nbtree_merger[host]['no_crossing_1.0'][d_output[host]['all'].astype(int)]==False)&#)[0]
			(ot.d_nbtree_merger[host]['TreeLength'][d_output[host]['all'].astype(int)]>min_snapshots))[0]

		d_output[host]['all'] = d_output[host]['all'][waar2]

	return d_output

def read_profiles(path = '/home/luciebakels/DMO11/Profiles/', datasets=['Radius', 'HaloIndex', 'Mass_profile', 'Density', 'MassTable']):
	filenames = []
	for filename in os.listdir(path):
		if filename.endswith(".hdf5") == False:
			continue
		filenames.append(filename)


	profiles = {}

	for filename in filenames:
		haloprop = h5py.File(path+filename, 'r')
		if datasets is None:
			datasets = []
			for key in haloprop.id:
				datasets.append(key.decode('utf-8'))
		for ds in datasets:
			if ds not in profiles:
				profiles[ds] = haloprop[ds.encode('utf-8')][:]
			elif ds in ['Radius', 'MassTable']:
				continue
			elif ds in ['HaloID', 'HaloIndex']:
				waarvervang = np.where(profiles[ds] == -1)[0]
				profiles[ds][waarvervang] = haloprop[ds.encode('utf-8')][:][waarvervang]
			elif ds in ['AngularMomentum', 'Velrad']:
				npart = haloprop['Npart_profile'.encode('utf-8')][:]
				npartprevious = profiles['Npart_profile']
				nietnul = np.where(npart+npartprevious>0)[0]
				profiles[ds][nietnul] = (haloprop[ds.encode('utf-8')][:][nietnul]*npart[nietnul] + 
					profiles[ds][nietnul]*npartprevious[nietnul])/(npart[nietnul]+npartprevious[nietnul])
			else:
				profiles[ds] += haloprop[ds.encode('utf-8')]

	profiles['Radius'] = np.array([5.00000000e-04, 1.11949519e-03, 1.38704376e-03, 1.71853386e-03,
       2.12924690e-03, 2.63811641e-03, 3.26860083e-03, 4.04976495e-03,
       5.01761978e-03, 6.21678259e-03, 7.70253377e-03, 9.54336517e-03,
       1.18241375e-02, 1.46499926e-02, 1.81511997e-02, 2.24891617e-02,
       2.78638548e-02, 3.45230478e-02, 4.27737238e-02, 5.29962319e-02,
       6.56618210e-02, 8.13543639e-02, 1.00797274e-01, 1.24886851e-01,
       1.54733606e-01, 1.91713448e-01, 2.37531116e-01, 2.94298766e-01,
       3.64633337e-01, 4.51777194e-01, 5.00000000e-01, 5.50000000e-01,
       6.50000000e-01, 7.50000000e-01, 8.50000000e-01, 9.50000000e-01,
       1.05000000e+00, 1.15000000e+00, 1.25000000e+00, 1.35000000e+00,
       1.45000000e+00, 1.55000000e+00, 1.65000000e+00, 1.75000000e+00,
       1.85000000e+00, 1.95000000e+00, 2.05000000e+00, 2.15000000e+00,
       2.25000000e+00, 2.35000000e+00, 2.45000000e+00, 2.55000000e+00,
       2.65000000e+00, 2.75000000e+00, 2.85000000e+00, 2.95000000e+00,
       3.05000000e+00, 3.15000000e+00, 3.25000000e+00, 3.35000000e+00,
       3.45000000e+00, 3.55000000e+00, 3.65000000e+00, 3.75000000e+00,
       3.85000000e+00])
	return profiles

def Einasto_Mass(r, alpha, r2, rho2):
	return integrate.quad(Einasto, 0, r, args=(alpha, r2, rho2, True))[0]
def Einasto(r, alpha, r2, rho2, integrate = False):
	if integrate == False:
		return rho2 * np.exp(-2/alpha * ((r/r2)**alpha - 1))
	else:
		return 4*np.pi* r**2 * rho2 * np.exp(-2/alpha * ((r/r2)**alpha - 1))

def fit_Einasto(r, density, r200):

	#Find r2
	waar = np.where(r > r200)[0]
	i_r2 = []#waar[np.where(np.log10(np.abs(density[waar][1:] - density[waar][:-1]))/np.log10(np.abs(r[waar][1:] - r[waar][:-1])) <= -2)[0]]

	# if len(i_r2) == 0:
	# 	print('No r2 fit')
	# 	bla = np.log10(np.abs(density[waar][1:] - density[waar][:-1]))/np.log10(np.abs(r[waar][1:] - r[waar][:-1]))
	# 	if bla[-1] > bla[-2]:
	# 		i_r2 = [waar[np.argmin(bla)]]
	# 	else:
	# 		i_r2 = []

	if len(i_r2) == 0:
		print('No r2 fit')
		def Einasto_profile(r_temp, alpha, r2, rho2):
			rho = np.log10(rho2 * np.exp(-2/alpha * ((r_temp/r2)**alpha - 1)))
			return rho

		[alpha, r2, rho2], bla = curve_fit(Einasto_profile, r, np.log10(density), bounds=([0, 0, 0], [2, np.inf,np.inf]))

		return alpha, r2, rho2

	else:
		i_r2 = i_r2[0]
		rho2 = density[i_r2]
		r2 = r[i_r2]	

		def Einasto_profile(r_temp, alpha):
			rho = rho2 * np.exp(-2/alpha * ((r_temp/r2)**alpha - 1))
			return rho

		alpha, bla = curve_fit(Einasto_profile, r, density, bounds=[0,1])

		return alpha, r2, rho2


def R200mean(r, R200crit, firstterm, c):
	#firstterm = (3*M200crit*rhom/(800*np.pi*(np.log(1+cNFW)-cNFW/(1+cNFW))))
	return firstterm*(np.log(1 + r * c / R200crit) - r * c / (R200crit + r * c))**(1/3) - r

def R200crit_from_R200mean(r, R200mean, firstterm, rhoc):
	M200 = 800./3.*np.pi*rhoc*r**3
	c = gdp.cnfw(M200)
	temp = R200mean * c / r
	return firstterm*((np.log(1+c) - c/(1+c)) / (np.log(1 + temp) - temp / (1 + temp)))**(1./3.) - r

def computeR200mean(ot):
	z = snapshot_to_redshift_2048(np.arange(190))
	rhom = 2048**3*MassTable11[1]*1e10/((105/(1+z))**3)
	for host in ot.d_mhtree.keys():
		waar = np.where(ot.d_mhtree[host]['R200']>0)[0]
		if 'R200mean' not in ot.d_mhtree[host].keys():
			ot.d_mhtree[host]['R200mean'] = np.zeros(len(ot.d_mhtree[host]['R200']))
		firstterm = (3*ot.d_mhtree[host]['M200_inter']*1e10/(800*np.pi*rhom*(np.log(1+ot.d_mhtree[host]['cNFW'])
			-ot.d_mhtree[host]['cNFW']/(1+ot.d_mhtree[host]['cNFW']))))**(1/3)
		for i in waar:

			ot.d_mhtree[host]['R200mean'][i] = brentq(R200mean, 0.5*ot.d_mhtree[host]['R200_inter'][i]/(1+z[i]), 
				10*ot.d_mhtree[host]['R200_inter'][i]/(1+z[i]),
				args=(ot.d_mhtree[host]['R200_inter'][i]/(1+z[i]), firstterm[i], ot.d_mhtree[host]['cNFW'][i]))

def computeR200mean_cNFW(R200, cNFW):
	c=constant()
	c.change_constants(redshift=0)
	rhom = Om0*c.rhocrit_Ms_Mpci3_h
	r200mean = np.zeros(len(R200))
	M200 = c.rhocrit_Ms_Mpci3_h * 200 * 4./3.* np.pi * R200**3
	firstterm = (3*M200/(800*np.pi*rhom*(np.log(1.+cNFW)-cNFW/(1.+cNFW))))**(1/3)
	for i in range(len(R200)):
		r200mean[i] = brentq(R200mean, 0.5*R200[i], 10*R200[i], args=(R200[i], firstterm[i], cNFW))

	return r200mean, 4./3.*np.pi* 200 * rhom * r200mean**3

def computeR200BN_cNFW(R200, cNFW):
	c=constant()
	c.change_constants(redshift=0)
	rhoc = c.rhocrit_Ms_Mpci3_h
	q = (1-Om0)
	delta = 18*np.pi**2 - 82*q - 39*q*q
	r200mean = np.zeros(len(R200))
	M200 = c.rhocrit_Ms_Mpci3_h * delta * 4./3.* np.pi * R200**3
	firstterm = (3*M200/(4*delta*np.pi*rhoc*(np.log(1.+cNFW)-cNFW/(1.+cNFW))))**(1/3)
	for i in range(len(R200)):
		r200mean[i] = brentq(R200mean, 0.5*R200[i], 10*R200[i], args=(R200[i], firstterm[i], cNFW))

	return r200mean, 4./3.*np.pi* delta * rhoc * r200mean**3

def computeR200crit(R200mean):
	c =	constant()
	rhoc = c.rhocrit_Ms_Mpci3
	rhom = 2048**3*MassTable11[1]*1e10/((105)**3)
	M200mean = 4./3*200*np.pi*rhom*R200mean**3
	R200crit = np.zeros(len(R200mean))
	firstterm = (3*M200mean/(800*np.pi*rhoc))**(1/3)
	for i in range(len(R200mean)):
		R200crit[i] = brentq(R200crit_from_R200mean, 0.5*R200mean[i], 3*R200mean[i],
			args=(R200mean[i], firstterm[i], rhoc))

	return R200crit

def find_rmax_equals_rconv(boxsize=105, particles=2048, hh=0.6751, factor=1): #boxsize/h
	l = boxsize/hh/particles
	rconv = factor * 0.77*(3*Om0/800/np.pi)**(1./3.)*l

	c = constant()
	c.change_constants(0)

	mdm = MassTable11[1]*1e10#Om0 * c.rhocrit_Ms_Mpci3 * l**3

	def Rmax_Ludlow_hier(npart):
		M200 = npart * mdm
		R200 = (M200/(4./3*np.pi*200*c.rhocrit_Ms_Mpci3))**(1./3.)

		cLud = gdp.cnfw(M200/hh, z=0)

		Rs = R200/cLud
		Rmax = 2.16258 * Rs
		return Rmax - rconv

	antwoord = brentq(Rmax_Ludlow_hier, 10, 5000)
	return antwoord, mdm*antwoord

def find_rmax_equals_rconv_L13(M200, boxsize=105, particles=2048, hh=0.6751): #boxsize/h
	l = boxsize/particles
	rconv = 0.77*(3*Om0/800/np.pi)**(1./3.)*l

	c = constant()
	c.change_constants(0)

	mdm = MassTable11[1]*1e10

	Mhalf = M200/2
	R200 = (M200/(4./3*np.pi*200*c.rhocrit_Ms_Mpci3_h))**(1./3.)
	rconv_cNFW = np.zeros(len(M200))
	rconv_rhalf = np.zeros(len(M200))

	for i in range(len(M200)):

		def Rhalf_P_hier(cNFW):
			rho0 = M200[i]/((np.log(1.+ cNFW) - cNFW/(1. + cNFW)))
			R50 = brentq(MassProfileNorm_fit, 0, R200[i],
				args = (rho0, Mhalf[i], M200[i], R200[i], cNFW), maxiter=500)

			return R50 - rconv

		rconv_cNFW[i] = brentq(Rhalf_P_hier, 0.01, 100)

		rho0 = M200[i]/((np.log(1.+ rconv_cNFW[i]) - rconv_cNFW[i]/(1. + rconv_cNFW[i])))
		rconv_rhalf[i] = brentq(MassProfileNorm_fit, 0, R200[i], 
			args = (rho0, Mhalf[i], M200[i], R200[i], rconv_cNFW[i]), maxiter=500)

	return M200, rconv_rhalf

def find_rmax_equals_rconv_P03(M200, boxsize=105, particles=2048, hh=0.6751): #boxsize/h
	l = boxsize/particles
	kappa = 0.177
	rconv_C = (3*kappa**2*Om0/800/np.pi)**(1./3.)*l

	c = constant()
	c.change_constants(0)

	mdm = MassTable11[1]*1e10

	Mhalf = M200/2
	R200 = (M200/(4./3*np.pi*200*c.rhocrit_Ms_Mpci3_h))**(1./3.)
	C = 4 * (np.log(Mhalf/mdm)/np.sqrt(Mhalf/mdm)) ** (2./3.)
	rconv = C * rconv_C
	rconv_cNFW = np.zeros(len(M200))
	rconv_rhalf = np.zeros(len(M200))

	for i in range(len(M200)):

		def Rhalf_P_hier(cNFW):
			rho0 = M200[i]/((np.log(1.+ cNFW) - cNFW/(1. + cNFW)))
			R50 = brentq(MassProfileNorm_fit, 0, R200[i],
				args = (rho0, Mhalf[i], M200[i], R200[i], cNFW), maxiter=500)

			return R50 - rconv[i]

		rconv_cNFW[i] = brentq(Rhalf_P_hier, 0.01, 100)

		rho0 = M200[i]/((np.log(1.+ rconv_cNFW[i]) - rconv_cNFW[i]/(1. + rconv_cNFW[i])))
		rconv_rhalf[i] = brentq(MassProfileNorm_fit, 0, R200[i], 
			args = (rho0, Mhalf[i], M200[i], R200[i], rconv_cNFW[i]), maxiter=500)

	return M200, rconv_rhalf

def Vmax_Rmax_cNFW(M200, cNFW = np.arange(1, 15, 0.1), redshift=0, h_divide=False):
	c = constant()
	c.change_constants(redshift=redshift)

	if h_divide:
		R200 = (M200/h/(4./3.*np.pi*200*c.rhocrit_Ms_Mpci3_h))**(1./3.)
	else:	
		R200 = (M200/h/(4./3.*np.pi*200*c.rhocrit_Ms_Mpci3))**(1./3.)

	Rs = R200/cNFW

	Rmax = 2.16258 * Rs

	een = G_Mpc_km2_Msi_si2 * M200 / (Rmax)
	Rtemp = Rmax * cNFW / R200
	twee = np.log(1. + Rtemp)
	drie = Rtemp / (1. + Rtemp)
	vier = np.log(1. + cNFW) - cNFW / (1. + cNFW)

	Vmax = np.sqrt(een * (twee - drie) / vier)

	return Vmax, Rmax

def Vmax_Rmax_Duffy(M200, no_h=True, redshift=0):
	c = constant()
	c.change_constants(redshift)
	R200 = (M200/(4./3.*np.pi*200*c.rhocrit_Ms_Mpci3))**(1./3.)

	cDuff = c_Duffy(M200, no_h = no_h, z=redshift)

	Rs = R200/cDuff

	Rmax = 2.16258 * Rs


	een = G_Mpc_km2_Msi_si2 * M200 / (Rmax)
	Rtemp = Rmax * cDuff / R200
	twee = np.log(1. + Rtemp)
	drie = Rtemp / (1. + Rtemp)
	vier = np.log(1. + cDuff) - cDuff / (1. + cDuff)

	Vmax = np.sqrt(een * (twee - drie) / vier)

	return Vmax, Rmax

def c_Duffy(M200, no_h=True, z=0):
	if z==0:
		A = 5.74
		B = -0.097
		C = 0
	else:
		A = 5.71
		B = -0.084
		C = -0.47
	Mpivot = 2.e12
	if no_h:
		Mpivot /= h
	return A * (M200 / Mpivot)**B * (1+z)**C
def Vmax_Rmax_Ludlow(M200, redshift=0, physical=True, h_divide = False):
	c = constant()
	c.change_constants(redshift)

	if physical and h_divide==False:
		R200 = (M200/(4./3*np.pi*200*c.rhocrit_Ms_Mpci3))**(1./3.)
	elif physical and h_divide:
		R200 = (M200/(4./3*np.pi*200*c.rhocrit_Ms_Mpci3_h))**(1./3.)
	elif physical==False and h_divide==False:
		R200 = (M200/(4./3*np.pi*200*c.rhocrit_Ms_Mpci3_com))**(1./3.)
	elif physical==False and h_divide:
		R200 = (M200/(4./3*np.pi*200*c.rhocrit_Ms_Mpci3_com_h))**(1./3.)

	cnfw = gdp.cnfw(M200, z=0)

	Rs = R200/cnfw

	Rmax = 2.16258 * Rs

	een = G_Mpc_km2_Msi_si2 * M200 / (Rmax)
	Rtemp = Rmax * cnfw / R200
	twee = np.log(1. + Rtemp)
	drie = Rtemp / (1. + Rtemp)
	vier = np.log(1. + cnfw) - cnfw / (1. + cnfw)

	Vmax = np.sqrt(een * (twee - drie) / vier)

	return Vmax, Rmax

class OrbitTree:
	def __init__(self, path, zstart=20):
		self.path = path
		self.zstart = zstart
		self.filenames = []
		self.file_halo = {}
		self.d_hostinfo = {}
		self.out_notmerged = {}
		self.out_merged = {}
		self.d_mhtree = {}
		self.d_nbtree = {}
		self.d_nbtree_merger = {}
		self.haloprop = {}
		self.d_categorise = {}
		self.only_add_existing_hosts = False

		for filename in os.listdir(self.path):
			if filename.endswith(".hdf5") == False:
				continue
			self.filenames.append(filename)

	def convert_to_Mpc(self, in_Mpc = 1):
		self.readNeighbourTree(datasets=['Distance', 'R200', 'X', 'Y', 'Z'])
		for host in self.d_nbtree.keys():
			for i in range(len(self.d_nbtree[host]['Distance'])):
				waar = np.where(self.d_nbtree[host]['Distance'][i,:]>0)[0]
				self.d_nbtree[host]['Distance'][i, waar] = self.d_nbtree[host]['Distance'][i, waar]*in_Mpc
				self.d_nbtree[host]['X'][i, waar] = self.d_nbtree[host]['X'][i, waar]*in_Mpc
				self.d_nbtree[host]['Y'][i, waar] = self.d_nbtree[host]['Y'][i, waar]*in_Mpc
				self.d_nbtree[host]['Z'][i, waar] = self.d_nbtree[host]['Z'][i, waar]*in_Mpc
			waar = np.where(self.d_nbtree[host]['R200']>0)[0]
			self.d_nbtree[host]['R200'][waar] = self.d_nbtree[host]['R200'][waar]*in_Mpc

		self.rewriteNeighbourTree(datasets=['Distance', 'R200', 'X', 'Y', 'Z'])

	def convert_to_Mpc_host(self, in_Mpc = 1):
		self.readHostHalo(datasets=['R200', 'Coord'])
		for host in self.d_mhtree.keys():
			for i in range(len(self.d_mhtree[host]['Coord'])):
				waar = np.where(self.d_mhtree[host]['Coord'][i]>0)[0]
				self.d_mhtree[host]['Coord'][i, waar] = self.d_mhtree[host]['Coord'][i, waar]*in_Mpc
			waar = np.where(self.d_mhtree[host]['R200']>0)[0]
			self.d_mhtree[host]['R200'][waar] = self.d_mhtree[host]['R200'][waar]*in_Mpc

		self.rewriteHostHalo(datasets=['Distance', 'R200', 'X', 'Y', 'Z'])

	def doealles(self, firstsnap = 101, read_velocities=False, read_positions=False, include_merged=False):
		print("Find good haloes...")
		self.readHostInfo()
		self.readHostHalo(datasets=['M200', 'R200', 'hostHaloIndex'])
		treelength = np.zeros(len(self.d_mhtree.keys()))
		
		self.readNeighbourTree(datasets=['R200'])

		weghost = []
		weginfo = np.zeros(0)
		i = -1
		for host in self.d_mhtree.keys():
			i += 1
			snaps = np.where(self.d_mhtree[host]['M200']>0)[0]
			if self.d_mhtree[host]['hostHaloIndex'][-1] != -1:
				weghost.append(host)
				weginfo = np.append(weginfo, i)
			elif len(self.d_nbtree[host]['R200']) == 0:
				weghost.append(host)
				weginfo = np.append(weginfo, i)
			elif len(snaps) > 0:
				if snaps[0] > firstsnap:
					weghost.append(host)
					weginfo = np.append(weginfo, i)
			else:
				weghosts.append(host)
				weginfo = np.append(weginfo, i)

		for key in self.d_hostinfo.keys():
			self.d_hostinfo[key] = np.delete(self.d_hostinfo[key], weginfo)

		print("Read hosts...")
		self.readHostHalo(datasets=['M200', 'R200', 'npart', 'hostHaloIndex', 'cNFW', 'HaloIndex'])#'GroupM200', 'GroupMnbMhost(R200)', 
			#'GroupRedshift(R200)', 'GroupVelRad(R200)','InfallingGroup', , 'Vel', 'Coord' ])
		
		for host in weghost:
			del self.d_mhtree[host]

		print("Read neighbours...")
		if read_velocities:
			self.readNeighbourTree(datasets=['VX', 'VY', 'VZ', 'Distance'])
		elif read_positions:
			self.readNeighbourTree(datasets=['X', 'Y', 'Z', 'Distance'])
		else:
			self.readNeighbourTree(datasets=['npart', 'Preprocessed', 'MnbMhost(R200)', 'Apocenter(z0)', 'hostHaloIndex', 'SpecificEorb(z0)',# 'SpecificAngularMomentum_hub', 'SpecificEorb_hub',
				'Pericenter(z0)', 'orbits', 'Distance', 'born_within_1.0', 'snapshot(R200)', 'VelRad', 'interaction2',# 'npart_bound(R200)', 
				'N_peri', #'Vmax', 'Rmax','1.0xR200', 'TimeAcc', 'eccentricity', 'Postprocessed', #'Vmax$_{peak}$', 'npart_bound$_{peak}$', 'LengthPreprocessed', 'LengthPostprocessed', 
				'TreeLength', 'i_apo_all', 'i_peri_all', 'D_apo_phys_all', 'D_peri_phys_all'])#'X', 'Y', 'Z', 'VX', 'VY', 'VZ'])#
			# self.readNeighbourTree(datasets=['npart', 'Apocenter(z0)', 'hostHaloIndex',# 'SpecificAngularMomentum_hub', 'SpecificEorb_hub',
			# 	'orbits', 'Distance', 'born_within_1.0', 'snapshot(R200)', 'VelRad',# 'npart_bound(R200)', 
			# 	'X', 'Y', 'Z'#'Vmax$_{peak}$', 'npart_bound$_{peak}$', 'LengthPreprocessed', 'LengthPostprocessed', 
			# 	])#'X', 'Y', 'Z', 'VX', 'VY', 'VZ'])#
			if include_merged:
				self.readMergedTree(datasets=['npart', 'hostHaloIndex', 'HaloIndex', 'Distance', 'born_within_1.0', 'TreeLength'])
		for host in weghost:
			if include_merged:
				del self.d_nbtree_merger[host]
			del self.d_nbtree[host]

		print("Smooth R200")
		self.smooth_R200()
		self.fix_formation_times()
		
		print("Find crossings")
		for host in self.d_nbtree.keys():
			self.find_crossing(self.d_nbtree[host], self.d_mhtree[host], R200type='R200_inter')
			if include_merged:
				self.find_crossing(self.d_nbtree_merger[host], self.d_mhtree[host], R200type='R200_inter')

		if (read_velocities == False)&(read_positions == False):
			print("Find peaks")
			#self.peak_fix(nbtree_indices_label='snap(R200)', datasets=['Vmax', 'Rmax', 'npart'])

			print("Find orbits after infall")
			self.find_numorbits_afterinfall()

			print("Categorise")
			self.categorise(R200type='R200_inter')

		self.only_add_existing_hosts = True

		self.hosts = np.array(list(self.d_nbtree.keys()))
		for host in self.hosts:
			self.d_nbtree[host]['snap(R200)'] = self.d_nbtree[host]['snap(R200)'].astype(int)

		print("Done!")

	def find_and_write_encounters(self, firstsnap=101, boxsize=105):
		self.doealles(firstsnap = firstsnap, read_positions=True, include_merged=False)
		self.readNeighbourTree(datasets=['npart'])
		print("Finding encounters...")
		self.npart_R200()
		for host in ot.hosts:
			self.find_interaction_subhalo(host, boxsize=boxsize)
		print("Finished finding encounters")

		print("Writing encounter data do files: interaction2")
		self.rewriteNeighbourTree(datasets=['interaction2'], rewrite=False)
		print("Done!")

	def npart_R200(self):
		hosts = np.array(list(self.d_mhtree.keys()))
		snapshots = np.arange(len(self.d_mhtree[hosts[0]]['M200']))
		rhocrit = np.zeros(len(self.d_mhtree[hosts[0]]['M200']))
		redshifts = snapshot_to_redshift(snapshots, zstart=self.zstart, numsnap=len(self.d_mhtree[hosts[0]]['M200'])-1)
		c = constant()
		for i in range(len(rhocrit)):
			c.change_constants(redshifts[i])
			rhocrit[i] = c.rhocrit_Ms_Mpci3_com_h

		for host in self.d_mhtree.keys():
			mhtree = self.d_nbtree[host]

			m200_oud = np.copy(mhtree['npart'])
			mhtree['M200_npart'] = np.where(mhtree['npart']>0, mhtree['npart']*MassTable11[1], 0)
			mhtree['R200_npart'] = np.ones_like(len(mhtree['npart']))*-1
			mhtree['R200_npart'] = (mhtree['M200_npart']*1e10/(4./3*np.pi*200*rhocrit))**(1./3.)

	def find_interaction_subhalo(self, host, boxize=105):
		htree = {}

		#Making a tree for all the neighbours around the host
		for i in range(190):
			waar = np.where(self.d_nbtree[host]['X'][:, i]>0)[0]
			if len(waar) == 0:
				continue
			coord = np.zeros((len(waar), 3))
			coord[:, 0] = self.d_nbtree[host]['X'][waar, i]
			coord[:, 1] = self.d_nbtree[host]['Y'][waar, i]
			coord[:, 2] = self.d_nbtree[host]['Z'][waar, i]
			htree[i] = cKDTree(coord%boxsize, boxsize=boxsize)

		self.d_nbtree[host]['interaction2'] = np.zeros_like(self.d_nbtree[host]['X'])

		#Looping over the neighbours
		for i in range(len(self.d_nbtree[host]['X'])):
			# welkebinnen = np.where((ot.d_nbtree[host]['hostHaloIndex'][i] - ot.d_mhtree[host]['HaloIndex'] ==0)&
			# 	(ot.d_mhtree[host]['HaloIndex'] > 0))[0]
			
			#Did the neighbour fall into the host?
			if self.d_nbtree[host]['snap(R200)'][i] <= 0:
				continue

			#Selecting all snapshots after first infall
			welkebinnen = (np.arange(190)[int(ot.d_nbtree[host]['snap(R200)'][i]):]).astype(int)

			#Looping over snapshots
			for wb in welkebinnen:
				#Selecting all neighbours that exist at snapshot wb
				waar = np.where(self.d_nbtree[host]['X'][:, wb]>0)[0]
				coord = np.zeros((len(waar), 3))
				coord[:, 0] = self.d_nbtree[host]['X'][waar, wb]
				coord[:, 1] = self.d_nbtree[host]['Y'][waar, wb]
				coord[:, 2] = self.d_nbtree[host]['Z'][waar, wb]

				#Coordinates of neighbour i
				ctemp = np.array([self.d_nbtree[host]['X'][i, wb], self.d_nbtree[host]['Y'][i, wb], self.d_nbtree[host]['Z'][i, wb]])
				#Neighbours within R200 of host of neighbour i
				buren = np.array(htree[wb].query_ball_point(ctemp, r=self.d_mhtree[host]['R200_inter'][wb])).astype(int)

				#Relative distance of the neighbours to neighbour i
				coordrel = coord[buren] - ctemp
				coordrel = np.where(np.abs(coordrel) > 0.5*boxsize, coordrel - coordrel/np.abs(coordrel)*boxsize, coordrel)
				dist = np.sqrt(np.sum(coordrel**2, axis=1))

				#Ordering neighbours to distance to neighbour i
				sortorder = np.argsort(dist)
				for ii in sortorder:
					#Neighbour other
					i_other = waar[buren[ii]]
					#If neighbour i is smaller than neighbour other, and it is within the estimated r200 of neighbour other, it is listed as an encounter
					if (self.d_nbtree[host]['npart'][i, wb] < self.d_nbtree[host]['npart'][i_other, wb]) & (dist[ii] < self.d_nbtree[host]['R200_npart'][i_other, wb]):
						self.d_nbtree[host]['interaction2'][i, wb] = i_other#dist[ii]
						break


	def fix_formation_times(self):
		hosts = np.array(list(self.d_mhtree.keys()))
		z = snapshot_to_redshift(np.arange(len(self.d_mhtree[hosts[0]]['M200_inter'])), zstart=self.zstart, numsnap=len(self.d_mhtree[hosts[0]]['M200_inter'])-1)
		for i in range(len(hosts)):
			host = hosts[i]
			ww = np.where(self.d_mhtree[host]['M200_inter'] > 0)[0]
			temp = interp1d(self.d_mhtree[host]['M200_inter'][ww], z[ww])
			at0 = self.d_mhtree[host]['M200_inter'][-1]
			self.d_hostinfo['z$_{0.50}$'][i] = temp(0.5*at0)
			self.d_hostinfo['z$_{0.75}$'][i] = temp(0.75*at0)
			if 0.25*at0 > np.min(self.d_mhtree[host]['M200_inter'][ww]):
				self.d_hostinfo['z$_{0.25}$'][i] = temp(0.25*at0)


	def peak_fix(self, nbtree_indices_label='snap(R200)', datasets=['Vmax', 'Rmax', 'npart_bound']):
		for host in self.d_nbtree.keys():
			nbtree = self.d_nbtree[host]
			mhtree = self.d_mhtree[host]

			welke = np.where((nbtree[nbtree_indices_label]>0)&(nbtree['born_within_1.0'] == False))[0]
			nbtree_indices = nbtree[nbtree_indices_label][welke].astype(int)
			for ds in datasets:
				if ds not in nbtree.keys():
					continue
				if ds+'$_{peak}$' not in nbtree.keys():
					nbtree[ds+'$_{peak}$'] = np.ones(len(nbtree[ds]))*-1
					nbtree[ds+'(R200)'] = np.ones(len(nbtree[ds]))*-1
					nbtree['MnbMhost(R200)'] = np.ones(len(nbtree[ds]))*-1

				for i in range(len(welke)):
					nbtree[ds+'(R200)'][welke[i]] = nbtree[ds][welke[i], nbtree_indices[i]]
					nbtree[ds+'$_{peak}$'][welke[i]] = np.max(nbtree[ds][welke[i], :(nbtree_indices[i]+1)])
					if ds == 'npart_bound':
						nbtree['MnbMhost(R200)'][welke[i]] = nbtree[ds][welke[i], nbtree_indices[i]]/mhtree[ds][nbtree_indices[i]]

	def find_crossing(self, nbtree, mhtree, R200type='R200_inter'):
		if 'Distance_R200' not in nbtree.keys():
			nbtree['Closest_R200'] = np.zeros(len(nbtree['Distance']))
			nbtree['Distance_R200'] = np.zeros_like(nbtree['Distance'])
			nbtree['snap(R200)'] = np.ones(len(nbtree['Distance']))*-1
			nbtree['no_crossing_1.0'] = np.array([True]*len(nbtree['Distance'])).astype(bool)
			for i in range(len(nbtree['Closest_R200'])):
				nbtree['Distance_R200'][i, :] = np.where(((nbtree['Distance'][i, :]>0)&(mhtree[R200type]>0)),
					nbtree['Distance'][i, :]/mhtree[R200type], 99)

				temp = np.where((nbtree['Distance_R200'][i, :]<1)&(mhtree[R200type]>0))[0]
				if (len(temp) > 0):# and (temp[0] >= np.where(mhtree['R200_inter']>0)[0][0]):
					nbtree['snap(R200)'][i] = temp[0]-1
					nbtree['no_crossing_1.0'][i] = False
				nbtree['Closest_R200'][i] = np.min(nbtree['Distance_R200'][i, :])

	def find_numorbits_afterinfall(self):
		hosts = np.array(list(self.d_nbtree.keys()))
		for host in hosts:
			if 'orbits_after' in self.d_nbtree[host].keys():
				continue
			self.d_nbtree[host]['orbits_after'] = np.zeros(len(self.d_nbtree[host]['D_apo_phys_all']))
			for i in range(len(self.d_nbtree[host]['D_apo_phys_all'])):
				i_apo = np.where(self.d_nbtree[host]['D_apo_phys_all'][i, :] > 0)[0]
				i_peri = np.where(self.d_nbtree[host]['D_peri_phys_all'][i, :] > 0)[0]
				if len(i_peri) == 0:
					continue
				snap_peri = self.d_nbtree[host]['i_peri_all'][i, i_peri]
				snap_apo = self.d_nbtree[host]['i_apo_all'][i, i_apo]
				eerste = np.where(snap_peri > self.d_nbtree[host]['snap(R200)'][i])[0]
				if len(eerste) == 0:
					continue
				self.d_nbtree[host]['orbits_after'][i] = len(eerste)/2.

				eerste = np.where(snap_apo > self.d_nbtree[host]['snap(R200)'][i])[0]
				if len(eerste) == 0:
					continue
				self.d_nbtree[host]['orbits_after'][i] += len(eerste)/2.

	def categorise(self, R200type='R200_inter'):
		"""
		- first infall satellites: (orbits==0 and subhalo==True) or (orbits==0.5 and subhalo=True and velrad<0)
		- orbital satellites: orbits>0 and subhalo==True
		- ex-satellites: notcrossed==False and subhalo==False
		- orbital halo: (orbits>0 and notcrossed==True and (orbits!=0.5 and vrad<0.5))
		- first infall: preprocessed==False and notcrossed==True and ((orbits==0) or (orbits==0.5 and velrad<0))
		- preprocessed infall: preprocessed==True and notcrossed=True and ((orbits==0) or (orbits==0.5 and velrad<0))
		"""

		# find_closest_approach(self.d_nbtree, self.d_mhtree, physical=False,zstart=20)
		for host in self.d_nbtree.keys():
			nbtree = self.d_nbtree[host]
			mhtree = self.d_mhtree[host]

			velrad = nbtree['VelRad'][:, -1]
			velradever = np.sum(np.where((nbtree['VelRad']<0)&(nbtree['VelRad']!=-1), 1, 0), axis=1)
			temp = np.where((nbtree['VelRad']<0)&(nbtree['VelRad']!=-1), 1, 0)
			temp2 = np.where(nbtree['VelRad']>0, 1, 0)
			turnaround = np.ones(len(temp)).astype(int)*(len(temp[0, :])-1)
			plusafter = np.zeros(len(turnaround))
			for i in range(len(temp)):
				ww = np.where(temp[i, :]>0)[0]
				if len(ww)>0:
					turnaround[i] = ww[0]
				plusafter[i] = np.sum(temp2[i, turnaround[i]:])

			orbits = nbtree['orbits']
			nperi = nbtree['N_peri']
			orbits_ai = nbtree['orbits_after']
			preprocessed = nbtree['Preprocessed']
			notcrossed = nbtree['no_crossing_1.0']
			subhalo = np.where(nbtree['Distance'][:, -1] - mhtree[R200type][-1]<0, True, False)
			subander = np.where((nbtree['hostHaloIndex'][:, -1]>-1)&(nbtree['hostHaloIndex'][:, -1]!=mhtree['HaloIndex'][-1]), True, False)
			borninside = nbtree['born_within_1.0']
			afstandnu = nbtree['Distance'][:, -1]/mhtree[R200type][-1]

			#mindist = np.zeros(len(nbtree['Distance']))
			self.find_crossing(nbtree, mhtree)
			mindist_r200 = nbtree['Closest_R200']
			#ca = nbtree['closest_approach']/mhtree[R200type][-1]

			self.d_categorise[host] = {}
			self.d_categorise[host]['infalling subhaloes'] = np.where((subhalo==True)&(borninside==False)&(nperi==0)&(orbits_ai==0))[0]#&(ca>=0.9*afstandnu))[0]
			self.d_categorise[host]['orbital subhaloes'] = np.where((subhalo==True)&(borninside==False)&((orbits_ai>0)))[0]#|(ca<0.9*afstandnu)))[0]

			self.d_categorise[host]['ex-satellites'] = np.where((mindist_r200<=4)&(borninside==False)&(notcrossed==False)&(subhalo==False))[0]
			self.d_categorise[host]['orbital haloes ($>$r$_{200}$)'] = np.where((mindist_r200<=4)&(borninside==False)&(subhalo==False)&(notcrossed==True)&(nperi>0))[0]
			self.d_categorise[host]['orbital haloes ($<$r$_{200}$)'] = np.where((subhalo==True)&(borninside==False)&(nperi>0)&(orbits_ai==0))[0]#&(ca>=0.9*afstandnu))[0]

			self.d_categorise[host]['pristine haloes'] = np.where((subhalo==False)&(mindist_r200<=4)&(borninside==False)&
				(preprocessed==-1)&(notcrossed==True)&(nperi==0))[0]

			self.d_categorise[host]['secondary ex-satellites'] = np.where((subander==False)&(subhalo==False)&(mindist_r200<=4)&(borninside==False)&
				(preprocessed!=-1)&(notcrossed==True)&(nperi==0))[0]
			self.d_categorise[host]['secondary subhaloes'] = np.where((subander==True)&(subhalo==False)&(mindist_r200<=4)&(borninside==False)&(preprocessed!=-1)&
				(notcrossed==True)&(nperi==0))[0]

			alles = np.zeros(0).astype(int)
			for i in self.d_categorise[host].keys():
				alles = np.append(alles, self.d_categorise[host][i])
			self.d_categorise[host]['other'] = np.delete(np.arange(len(orbits)).astype(int), alles)
			self.d_categorise[host]['other'] = np.delete(self.d_categorise[host]['other'], 
				np.where((mindist_r200[self.d_categorise[host]['other']]>4)|(borninside[self.d_categorise[host]['other']]==True))[0])

	def smooth_R200(self):
		hosts = np.array(list(self.d_mhtree.keys()))
		snapshots = np.arange(len(self.d_mhtree[hosts[0]]['M200']))
		rhocrit = np.zeros(len(self.d_mhtree[hosts[0]]['M200']))
		redshifts = snapshot_to_redshift(snapshots, zstart=self.zstart, numsnap=len(self.d_mhtree[hosts[0]]['M200'])-1)
		c = constant()
		for i in range(len(rhocrit)):
			c.change_constants(redshifts[i])
			rhocrit[i] = c.rhocrit_Ms_Mpci3_com_h

		for host in self.d_mhtree.keys():
			mhtree = self.d_mhtree[host]

			m200_oud = np.copy(mhtree['M200'])
			mhtree['M200_inter'] = np.ones(len(mhtree['M200']))*-1
			mhtree['R200_inter'] = np.ones(len(mhtree['R200']))*-1
			mhtree_temp = np.copy(mhtree['M200'])
			waar = np.where((mhtree_temp >0)&(mhtree['hostHaloIndex']==-1))[0]

			if len(waar)>1:
				mhtreeM200 = interp1d(1./(1+redshifts[waar]), mhtree_temp[waar])

				mhtree['M200_inter'][waar[0]:waar[-1]+1] = mhtreeM200(1./(1+redshifts[waar[0]:waar[-1]+1]))
				mhtree['R200_inter'][waar[0]:waar[-1]+1] = (mhtree['M200_inter'][waar[0]:waar[-1]+1]*1e10
					/(4./3*np.pi*200*rhocrit[waar[0]:waar[-1]+1]))**(1./3.)
			else:
				mhtree['M200_inter'] = mhtree['M200']
				mhtree['R200_inter'] = mhtree['R200']
			mhtree['M200_inter'][-1] = mhtree['M200'][-1]
			mhtree['R200_inter'][-1] = mhtree['R200'][-1]

	def select_hosts(self, massrange=None, z025range=None, z050range=None, z075range=None):
		for filename in filenames:
			
			self.readHostInfo(filename)

			allhaloes = np.arange(len(self.d_hostinfo['M200'])).astype(int)
			if massrange is not None:
				allhaloes = allhaloes[np.where((self.d_hostinfo['M200'][allhaloes] >= massrange[0]) & 
					(self.d_hostinfo['M200'][allhaloes] <=massrange[1]))[0]]
				if len(allhaloes) == 0:
					continue
			if z025range is not None:
				allhaloes = allhaloes[np.where((self.d_hostinfo['z$_{0.25}$'][allhaloes] >= z025range[0]) & 
					(self.d_hostinfo['z$_{0.25}$'][allhaloes] <= z025[1]))[0]]
				if len(masstemp) == 0:
					continue
			if z050range is not None:
				allhaloes = allhaloes[np.where((self.d_hostinfo['z$_{0.50}$'][allhaloes] >= z050range[0]) & 
					(self.d_hostinfo['z$_{0.50}$'][allhaloes] <= z050[1]))[0]]
				if len(masstemp) == 0:
					continue
			if z075range is not None:
				allhaloes = allhaloes[np.where((self.d_hostinfo['z$_{0.75}$'][allhaloes] >= z075range[0]) & 
					(self.d_hostinfo['z$_{0.75}$'][allhaloes] <= z075[1]))[0]]
				if len(masstemp) == 0:
					continue
			self.file_halo[filename] = allhaloes

	def matchSharkData(self, d_output=None, sharkpath='/home/luciebakels/DMO11/Shark/189/', nfolders=128, datasets=[],
		hd_path ='/home/luciebakels/DMO11/Velcopy/', hd_name= '11DM.snapshot_189.quantities.hdf5', snapshot=189, totnumsnap=189, boxsize=105,
		old_shark_version=False):

		if d_output is None:
			d_output = self.d_categorise

		for key in ['id_subhalo_tree', 'id_halo_tree', 'mvir_hosthalo', 'type', 'mvir_subhalo', 'mgas_bulge', 'mgas_disk', 'mhot', 'mstars_bulge', 'mstars_disk']:
			datasets.append(key)
		
		self.readNeighbourTree(datasets=['HaloIndex'])
		
		keys = (['infalling subhaloes', 'orbital subhaloes', 'ex-satellites', 'pristine haloes', 'secondary ex-satellites', 
			'secondary subhaloes', 'orbital haloes ($<$r$_{200}$)', 'orbital haloes ($>$r$_{200}$)'])
		start_time = time.time()
		hd = ha.HaloData(hd_path, hd_name, snapshot=snapshot, totzstart=self.zstart, totnumsnap=totnumsnap, boxsize=boxsize)
		hd.readSharkData(sharkpath=sharkpath, nfolders=nfolders, datasets=datasets)
		hd.readData(datasets=['Tail', 'Head', 'HaloID'])

		allH = (hd.sharkdata['mgas_bulge'] + hd.sharkdata['mgas_disk'] + 
			hd.sharkdata['mstars_bulge'] + hd.sharkdata['mstars_disk'])

		shark_haloes_match = hd.sharkdata['id_subhalo_tree']
		replace = np.where(shark_haloes_match > (hd.snapshot+1)*hd.THIDVAL)[0]
		shark_haloes_match[replace] = shark_haloes_match[replace]%(1000*hd.THIDVAL)

		allindices = np.zeros(0).astype(int)

		hosts = np.array(list(self.d_nbtree.keys()))
		print("--- %s seconds ---" % (time.time() - start_time), 'read data')

		start_time = time.time()
		for host in hosts:
			for key in keys:
				allindices = np.append(allindices, self.d_nbtree[host]['HaloIndex'][d_output[host][key], -1])
		
		if old_shark_version:
			allindices = hd.hp['Tail'][allindices] + 1
		else:
			allindices = hd.hp['HaloID'][allindices]
		waarmatch = np.where(np.in1d(shark_haloes_match, allindices))[0]
		shark_haloes_match = shark_haloes_match[waarmatch]
		print("--- %s seconds ---" % (time.time() - start_time), 'found matching indices')

		start_time = time.time()
		for host in hosts:
			#print(host)
			subhaloids = np.zeros(0).astype(int)
			welke = np.zeros(0).astype(int)
			for key in keys:
				subhaloids = np.append(subhaloids, self.d_nbtree[host]['HaloIndex'][d_output[host][key], -1])
				welke = np.append(welke, d_output[host][key])
			haloid = self.d_mhtree[host]['HaloIndex'][-1]

			self.d_nbtree[host]['SharkDM'] = np.zeros(len(self.d_nbtree[host]['HaloIndex']))
			self.d_nbtree[host]['SharkH'] = np.zeros(len(self.d_nbtree[host]['HaloIndex']))
			for ds in datasets:
				self.d_nbtree[host][ds] = np.zeros(len(self.d_nbtree[host]['HaloIndex']))
				if ds == 'type':
					self.d_nbtree[host][ds] = np.ones(len(self.d_nbtree[host]['HaloIndex']))*-1
				elif ds not in ['mvir_subhalo', 'mvir_hosthalo', 'position_x', 'position_y', 'position_z']:
					self.d_nbtree[host][ds+'_main'] = np.zeros(len(self.d_nbtree[host]['HaloIndex']))
			self.d_nbtree[host]['Ngalaxy'] = np.zeros(len(self.d_nbtree[host]['HaloIndex']))

			if old_shark_version:
				my_haloes_match = hd.hp['Tail'][subhaloids] + 1
			else:
				my_haloes_match = hd.hp['HaloID'][subhaloids]

			replace = np.where(my_haloes_match == 0)[0]
			my_haloes_match[replace] = subhaloids[replace] + hd.snapshot*hd.THIDVAL + 1

			shark_found = np.zeros(0).astype(int)
			my_found = np.zeros(0).astype(int)
			
			#start_time = time.time()
			h_in_shark = np.where(np.in1d(my_haloes_match, shark_haloes_match))[0]
			#df = pd.DataFrame({'A': self.sharkdata['id_halo_tree']})
			temparange = np.where(np.in1d(shark_haloes_match, my_haloes_match))[0]
			#print("--- %s seconds ---" % (time.time() - start_time), 'find matching indices')
			tempid = np.array(shark_haloes_match[temparange])
			#time1 = 0
			for halo in h_in_shark:
				#start_time = time.time()
				waar = np.where(my_haloes_match[halo] == tempid)[0]
				if len(waar) == 0:
					continue
				allgalaxies = waarmatch[temparange[waar]]#np.in1d(tempid, my_haloes_match[halo])]
				#time1 += time.time() - start_time

				#self.hp['SharkID'][halo_i[i]] = halo
				maingal = np.argmin(np.array(hd.sharkdata['type'][allgalaxies]))
				self.d_nbtree[host]['SharkDM'][welke[halo]] = hd.sharkdata['mvir_subhalo'][allgalaxies[maingal]]
				self.d_nbtree[host]['SharkH'][welke[halo]] = np.sum(allH[allgalaxies]) + hd.sharkdata['mhot'][allgalaxies][0]
				self.d_nbtree[host]['Ngalaxy'][welke[halo]] = len(allgalaxies)
				for ds in datasets:
					if ds == 'type':
						self.d_nbtree[host][ds][welke[halo]] = np.min(hd.sharkdata[ds][allgalaxies])
					elif ds in ['mvir_subhalo', 'mvir_hosthalo', 'position_x', 'position_y', 'position_z']:
						self.d_nbtree[host][ds][welke[halo]] = hd.sharkdata[ds][allgalaxies[maingal]]
					else:
						self.d_nbtree[host][ds][welke[halo]] = np.sum(hd.sharkdata[ds][allgalaxies])
						self.d_nbtree[host][ds+'_main'][welke[halo]] = hd.sharkdata[ds][allgalaxies[maingal]]

			#print("--- %s seconds ---" %(time1), 'bla')
			my_haloes_match = hd.hp['Tail'][haloid] + 1
			if my_haloes_match == 0:
				my_haloes_match = haloid + hd.snapshot*hd.THIDVAL + 1

			h_in_shark = np.where(my_haloes_match == hd.sharkdata['id_subhalo_tree'])[0]
			if len(h_in_shark) == 0:
				continue
			#self.hp['SharkID'][halo_i[i]] = halo
			self.d_mhtree[host]['SharkDM'] = hd.sharkdata['mvir_hosthalo'][h_in_shark][0]
			self.d_mhtree[host]['SharkH'] = np.sum(allH[h_in_shark]) + hd.sharkdata['mhot'][h_in_shark][0]
			self.d_mhtree[host]['Ngalaxy'] = len(h_in_shark)
			for ds in datasets:
				self.d_mhtree[host][ds] = np.sum(hd.sharkdata[ds][h_in_shark])
		print("--- %s seconds ---" % (time.time() - start_time), 'finished matching shark')

	def readHostInfo(self):
		for filename in self.filenames:
			self.readHostInfo_onefile(filename)

	def readHostHalo(self, datasets=None):
		for filename in self.filenames:
			self.readHostHalo_onefile(filename, datasets=datasets, only_add_existing_hosts=self.only_add_existing_hosts)

	def readNeighbourTree(self, datasets=None):
		for filename in self.filenames:
			self.readNeighbourTree_onefile(filename, datasets=datasets, only_add_existing_hosts=self.only_add_existing_hosts)

	def readMergedTree(self, datasets=None):
		for filename in self.filenames:
			self.readMergedTree_onefile(filename, datasets=datasets, only_add_existing_hosts=self.only_add_existing_hosts)

	def rewriteHostHalo(self, datasets=None, rewrite=True):
		for filename in self.filenames:
			self.rewriteHostHalo_onefile(filename, datasets=datasets, rewrite=rewrite)

	def rewriteNeighbourTree(self, datasets=None, rewrite=True):
		for filename in self.filenames:
			self.rewriteNeighbourTree_onefile(filename, datasets=datasets, rewrite=rewrite)

	def rewriteMergedTree(self, datasets=None, rewrite=True):
		for filename in self.filenames:
			self.rewriteMergedTree_onefile(filename, datasets=datasets, rewrite=rewrite)

	def readFile(self, filename, readtype='r'):
		if filename not in self.haloprop.keys():
			self.haloprop[filename] = h5py.File(self.path+filename, readtype)

	def closeFile(self, filename):
		if filename in self.haloprop.keys():
			self.haloprop[filename].close()
			del self.haloprop[filename]

	def readHostInfo_onefile(self, filename):
		if filename in self.d_hostinfo.keys():
			return 0

		self.readFile(filename)

		# Header = self.haloprop[filename]['Header']
		# if filename not in self.d_hostinfo.keys():
		# 	self.d_hostinfo[filename] = {}

		# if 'massrange' not in self.d_hostinfo.keys():
		# 	self.d_hostinfo['massrange'] = {}
		# if 'numhosts' not in self.d_hostinfo.keys():
		# 	self.d_hostinfo['numhosts'] = {}

		# self.d_hostinfo['massrange'][filename] = Header.attrs['MassRange'][:]
		# self.d_hostinfo['numhosts'][filename] = Header.attrs['NumHosts']

		for key in self.haloprop[filename].id:
			if isinstance(self.haloprop[filename][key].id, h5py.h5d.DatasetID):
				if key.decode('utf-8') == 'MaskVelIndices':
					continue
				else:
					if key.decode('utf-8') not in self.d_hostinfo.keys():
						self.d_hostinfo[key.decode('utf-8')] = self.haloprop[filename][key][:]
					else:
						self.d_hostinfo[key.decode('utf-8')] = np.append(self.d_hostinfo[key.decode('utf-8')], self.haloprop[filename][key][:])

	def readMergedSort_onefile(self, filename, closefile=True):
		self.readFile(filename)

		outnbm = self.haloprop[filename]['MergedSort'.encode('utf-8')]
		if filename in self.file_halo.keys():
			hosts = outnbm['host'.encode('utf-8')]
			waar = np.in1d(hosts, self.d_hostinfo['HostsRootDescen'])

		for key in outnbm.id:
			if key.decode('utf-8') not in self.out_merged.keys():
				if filename in self.file_halo.keys():
					self.out_merged[key.decode('utf-8')] = outnbm[key][waar]
				else:
					self.out_merged[key.decode('utf-8')] = outnbm[key][:]
			else:
				if filename in self.file_halo.keys():
					self.out_merged[key.decode('utf-8')] = np.append(self.out_merged[key.decode('utf-8')], outnbm[key][waar])
				else:
					self.out_merged[key.decode('utf-8')] = np.append(self.out_merged[key.decode('utf-8')], outnbm[key][:])
		
		if closefile:
			self.readFile(filename)

	def readNotMergedSort_onefile(self, filename, closefile=True):
		self.readFile(filename)

		outnb = self.haloprop[filename]['NotMergedSort'.encode('utf-8')]
		if filename in self.file_halo.keys():
			hosts = outnbm['host'.encode('utf-8')]
			waar = np.in1d(hosts, self.d_hostinfo['HostsRootDescen'])

		for key in outnb.id:
			if key.decode('utf-8') not in self.out_merged.keys():
				if filename in self.file_halo.keys():
					self.out_merged[key.decode('utf-8')] = outnb[key][waar]
				else:
					self.out_merged[key.decode('utf-8')] = outnb[key][:]
			else:
				if filename in self.file_halo.keys():
					self.out_merged[key.decode('utf-8')] = np.append(self.out_merged[key.decode('utf-8')],outnb[key][waar])
				else:
					self.out_merged[key.decode('utf-8')] = np.append(self.out_merged[key.decode('utf-8')],outnb[key][:])

		if closefile:
			self.closefile(filename)

	def check_if_exists(self, dictionary, filename, datasets=None):
		check = True
		if datasets is None:
			return False
		if (filename in self.file_halo.keys()):
			hosts = np.array([filename[:-5]+'_'+str(i) for i in self.file_halo[filename]])
			for host in hosts:
				for ds in datasets:
					if ds not in dictionary[host].keys():
						return False
		else:
			return False
		return True
	
	def readHostHalo_onefile(self, filename, datasets=None, closefile=True, only_add_existing_hosts=False):
		if self.check_if_exists(self.d_mhtree, filename, datasets):
			return 0
		self.readFile(filename)
		for key in self.haloprop[filename].id:
			if isinstance(self.haloprop[filename][key].id, h5py.h5g.GroupID):
				if key.decode('utf_8') in ['Header', 'MergedSort', 'NotMergedSort']:
					continue
				if (filename in self.file_halo.keys()) and (int(key.decode('utf_8')) not in self.file_halo[filename]):
					continue
				hp2 = self.haloprop[filename][key]
				mh = hp2['MainHaloTree'.encode('utf-8')]
				if filename[:-5]+'_'+key.decode('utf-8') not in self.d_mhtree.keys():
					if only_add_existing_hosts:
						continue
					self.d_mhtree[filename[:-5]+'_'+key.decode('utf-8')] = {}

				for key2 in mh.id:
					if key2.decode('utf-8') in self.d_mhtree[filename[:-5]+'_'+key.decode('utf-8')]:
						continue
					if datasets is None:
						self.d_mhtree[filename[:-5]+'_'+key.decode('utf-8')][key2.decode('utf-8')] = mh[key2][:]
					elif key2.decode('utf-8') in datasets:
						self.d_mhtree[filename[:-5]+'_'+key.decode('utf-8')][key2.decode('utf-8')] = mh[key2][:]
		if closefile:
			self.closeFile(filename)


	def rewriteHostHalo_onefile(self, filename, datasets=None, rewrite=True):
		if datasets is None:
			print('Error: datasets field imput needed')
			return 0

		self.readFile(filename, readtype='r+')
		hp = self.haloprop[filename]
		for host_hdf in hp.id:
			host = host_hdf.decode('utf_8')
			if host not in self.d_mhtree.keys():
				continue
			mh = hp[host_hdf]['MainHaloTree'.encode('utf-8')]
			for ds in datasets:
				if ds not in self.d_mhtree[filename[:-5]+'_'+host].keys():
					print(ds + ' not present in the loaded catalog.')
					continue
				if (rewrite==False) and (ds.encode('utf-8') in mh.id):
					print(ds + ' already present in saved catalog. If you would like to overwrite, set rewrite=True.')
					continue
				if (rewrite==True) and (ds.encode('utf-8') in mh.id):
					del mh[ds.encode('utf-8')]
				newset = mh.create_dataset(ds, data = self.d_mhtree[filename[:-5]+'_'+host][ds])

		self.closeFile(filename)

	def readNeighbourTree_onefile(self, filename, datasets=None, closefile=True, only_add_existing_hosts=False):
		if self.check_if_exists(self.d_nbtree, filename, datasets):
			return 0
		self.readFile(filename)
		for key in self.haloprop[filename].id:
			if isinstance(self.haloprop[filename][key].id, h5py.h5g.GroupID):
				if key.decode('utf_8') in ['Header', 'MergedSort', 'NotMergedSort']:
					continue
				if (filename in self.file_halo.keys()) and (int(key.decode('utf_8')) not in self.file_halo[filename]):
					continue
				hp2 = self.haloprop[filename][key]
				mh = hp2['NeighbourTree'.encode('utf-8')]
				if filename[:-5]+'_'+key.decode('utf-8') not in self.d_nbtree.keys():
					if only_add_existing_hosts:
						continue
					self.d_nbtree[filename[:-5]+'_'+key.decode('utf-8')] = {}

				for key2 in mh.id:
					if key2.decode('utf-8') in self.d_nbtree[filename[:-5]+'_'+key.decode('utf-8')]:
						continue
					if datasets is None:
						self.d_nbtree[filename[:-5]+'_'+key.decode('utf-8')][key2.decode('utf-8')] = mh[key2][:]
					elif key2.decode('utf-8') in datasets:
						self.d_nbtree[filename[:-5]+'_'+key.decode('utf-8')][key2.decode('utf-8')] = mh[key2][:]	
		if closefile:
			self.closeFile(filename)

	def readMergedTree_onefile(self, filename, datasets=None, closefile=True, only_add_existing_hosts=False):
		if self.check_if_exists(self.d_nbtree_merger, filename, datasets):
			return 0
		self.readFile(filename)
		for key in self.haloprop[filename].id:
			if isinstance(self.haloprop[filename][key].id, h5py.h5g.GroupID):
				if key.decode('utf_8') in ['Header', 'MergedSort', 'NotMergedSort']:
					continue
				if (filename in self.file_halo.keys()) and (int(key.decode('utf_8')) not in self.file_halo[filename]):
					continue
				hp2 = self.haloprop[filename][key]
				mh = hp2['MergedTree'.encode('utf-8')]
				if filename[:-5]+'_'+key.decode('utf-8') not in self.d_nbtree_merger.keys():
					if only_add_existing_hosts:
						continue
					self.d_nbtree_merger[filename[:-5]+'_'+key.decode('utf-8')] = {}

				for key2 in mh.id:
					if key2.decode('utf-8') in self.d_nbtree_merger[(filename[:-5]+'_'+key.decode('utf-8'))]:
						continue
					if datasets is None:
						self.d_nbtree_merger[filename[:-5]+'_'+key.decode('utf-8')][key2.decode('utf-8')] = mh[key2][:]
					elif key2.decode('utf-8') in datasets:
						self.d_nbtree_merger[filename[:-5]+'_'+key.decode('utf-8')][key2.decode('utf-8')] = mh[key2][:]		
		if closefile:
			self.closeFile(filename)


	def rewriteNeighbourTree_onefile(self, filename, datasets=None, rewrite=True):
		if datasets is None:
			print('Error: datasets field imput needed')
			return 0

		self.readFile(filename, readtype='r+')
		hp = self.haloprop[filename]
		for host_hdf in hp.id:
			host = filename[:-5]+'_'+host_hdf.decode('utf_8')
			if host not in self.d_nbtree.keys():
				continue
			nb = hp[host_hdf]['NeighbourTree'.encode('utf-8')]
			for ds in datasets:
				if ds not in self.d_nbtree[host].keys():
					print(ds + ' not present in the loaded catalog.')
					continue
				if (rewrite==False) and (ds.encode('utf-8') in nb.id):
					print(ds + ' already present in saved catalog. If you would like to overwrite, set rewrite=True.')
					continue
				if (rewrite==True) and (ds.encode('utf-8') in nb.id):
					del nb[ds.encode('utf-8')]
				newset = nb.create_dataset(ds, data = self.d_nbtree[host][ds])

		self.closeFile(filename)


	def rewriteMergedTree_onefile(self, filename, datasets=None, rewrite=True):
		if datasets is None:
			print('Error: datasets field imput needed')
			return 0

		self.readFile(filename, readtype='r+')
		hp = self.haloprop[filename]
		for host_hdf in hp.id:
			host = host_hdf.decode('utf_8')
			if host not in self.d_nbtree_merged.keys():
				continue
			nb = hp[host_hdf]['MergedTree'.encode('utf-8')]
			for ds in datasets:
				if ds not in self.d_nbtree_merger[filename[:-5]+'_'+host].keys():
					print(ds + ' not present in the loaded catalog.')
					continue
				if (rewrite==False) and (ds.encode('utf-8') in nb.id):
					print(ds + ' already present in saved catalog. If you would like to overwrite, set rewrite=True.')
					continue
				if (rewrite==True) and (ds.encode('utf-8') in nb.id):
					del nb[ds.encode('utf-8')]
				newset = nb.create_dataset(ds, data = self.d_nbtree_merger[filename[:-5]+'_'+host][ds])

		self.closeFile(filename)


	def read_VelData(self, datasets=['cNFW'], velpath='/home/luciebakels/DMO11/VELz0/', snapshot=189):
		catalog, np2, at = vpt.ReadPropertyFile(velpath + 'snapshot_%03d' %(snapshot), ibinary=2, desiredfields=datasets)
		self.readNeighbourTree(datasets=['HaloIndex'])
		for host in self.hosts:
			hi = self.d_nbtree[host]['HaloIndex'][:, snapshot]
			for ds in datasets:
				if ds in self.d_nbtree[host].keys():
					if len(self.d_nbtree[host][ds].shape) > 1:
						self.d_nbtree[host][ds][:, snapshot] = catalog[ds][hi]
				else:
					self.d_nbtree[host][ds] = catalog[ds][hi]

	def read_HaloData(self, datasets=['R_HalfMass'], hd_path ='/home/luciebakels/DMO11/Velcopy/11DM.', snapshot=189,
		totzstart=20, totnumsnap=189, boxsize=105):

		hd = ha.HaloData(hd_path, 'snapshot_%3d.quantities.hdf5' %snapshot, snapshot=snapshot, totzstart=self.zstart, totnumsnap=totnumsnap, 
			boxsize=boxsize)
		hd.readData(datasets=datasets)

		self.readNeighbourTree(datasets=['HaloIndex'])

		for host in self.hosts:
			hi = self.d_nbtree[host]['HaloIndex'][:, snapshot]
			for ds in datasets:
				if ds in self.d_nbtree[host].keys():
					if len(self.d_nbtree[host][ds].shape) > 1:
						self.d_nbtree[host][ds][:, snapshot] = hd.hp[ds][hi]
				else:
					self.d_nbtree[host][ds] = hd.hp[ds][hi]

	def readAllData(self, datasets=None, datasets_host=None):
		for filename in self.filenames:
			self.readFile(filename)

			self.readHostInfo(filename, closefile=False)

			self.readMergedSort(filename, closefile=False)

			self.readHostInfo_onefile(filename, datasets=datasets_host, closefile=False)

			self.readNeighbourTree_onefile(filename, datasets=datasets, closefile=False)

			self.readMergedTree_onefile(filename, datasets=datasets, closefile=False)

			self.closeFile(filename)
