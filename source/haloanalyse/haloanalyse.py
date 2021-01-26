import numpy as np

from scipy.optimize import curve_fit
from scipy.optimize import fsolve, brentq
from scipy.interpolate import interp1d
from scipy import integrate
from scipy import stats

import sys
import os
import GasDensityProfiles as gdp
import velociraptor_python_tools as vpt
from scipy.spatial import cKDTree
import h5py
import re
import pandas as pd
from constants import *
from snapshot import *


def version():
	print("Version-1.6")

def offsetRadius(rad):
	return np.logspace(np.log10(rad[0]) - 0.5*(np.log10(rad[-1])-np.log10(rad[0]))/len(rad), 
		np.log10(rad[-1]) - 0.5*(np.log10(rad[-1])-np.log10(rad[0]))/len(rad), len(rad))

#Snapshot to redshifts
def snapshot_to_redshift_2048(snapshot):
	"""The redshift belonging to each snapshot of the GENESIS 105 Mpc^3 box

	Parameters
	----------
	snapshot : int, int list, or int array
		snapshot number(s) of which you want to know the redshift

	Returns
	-------
	float or array of floats
		redshift(s) belonging to imput snapshot
	"""
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
	"""The redshift belonging to each snapshot for a logarithmic snapshot spacing

	Parameters
	----------
	snapshot : int or int array
		snapshot number(s) of which you want to know the redshift
	zstart : float
		redshift of first snapshot
	numsnap : int
		number of final snapshot (at z=0)

	Returns
	-------
	float or array of floats
		redshift(s) belonging to imput snapshot
	"""
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
	"""Closest snapshot at a given redshift for a logarithmic snapshot spacing
	
	Parameters
	----------
	snapshot : int or int array
		snapshot number(s) of which you want to know the redshift
	zstart : float
		redshift of first snapshot
	numsnap : int
		number of final snapshot (at z=0)

	Returns
	-------
	float or array of floats
		redshift(s) belonging to imput snapshot
	"""
	tstart = (1/(1+zstart))
	tdelta = (1/tstart)**(1/numsnap)
	if isinstance(redshift, np.ndarray):
		antwoord = np.zeros(len(redshift))
		antwoord = np.log10(1./(redshift + 1.)/tstart)/np.log10(tdelta)
		return np.rint(antwoord).astype(int)
	return np.int(np.rint(np.log10(1./(redshift + 1.)/tstart)/np.log10(tdelta)))

class HaloData:
	"""
	Class that contains, reads, and computes halo properties of one snapshot

	Attributes
	----------
	boxsize : float
		length of a side of the simulation box in units of the Coord field of 
		the catalogue (often in Mpc/h)
	haloarray : list
		list of the fields within the halo data file
	halocoordtree : cKDTree
		cKDTree of the halo positions
	hp : dict
		dictionary that contains the properties of each halo
	hpvel : dict
		dictionary that contains the VR catalogue data
	hydro : bool
		a flag when set, reads in hydrosimulation fields
	infofile : hdf5 file
		private (used to open and close hdf5 files)
	name : str
		name of the halo property file
	path : str
		path to the halo property file
	physical : bool
		a flag when set to False, will convert the physical units of the catalogue
		to comoving units (default is False).
	redshift : float
		redshift of the snapshot (default is None)
	snapshot : int
		snapshot number of the catalogue (default is None)
	THIDVAL : int
		TEMPORALHALOIDVAL from VELOCIraptor (default is 1e12)

	Functions
	---------
	--Private--
	allocateSizes(key, lengte)
		Allocates the 'hp' dictionary when necessary
	propertyDictionary(datasets=[])
		Organises the right datatypes for each 'hp' dictionary field
	openFile(write=False)
		Opens the catalogue file
	closeFile()
		Closes the catalogue file
	vel_correction(vel, z=0)
		Can be changed if a velocity correction is necessary (because of mess ups...)
	makeAllHaloCoordTree(halodata, indices=None)
		cKDTree used for the twopointcorrelation function
	readSharkFile(path, name, datasets=None)
		Reads a single file within the specified SHARK catalogue

	--Reading and Writing--
	makeHaloCoordTree(indices = None)
		Computes a cKDTree for all given haloes and stores it in halocoordtree
	readData(datasets=[], closeFile=True, haloIndex=None)
		Reads data for given dataset fields and stores it in the 'hp' dictionary
	reWriteData(datasets=[])
		Rewrites given datasets that already exist on file
	addData(datasets=[])
		Adds datasets that don't exist on file yet
	replaceWithVel(halodata, datasets = None, velsets=None)
		Replacing a halodata properties dataset with, or adding, a VR dataset
	readVel(halodata, velsets=None)
		Saves VR catalogue datasets to 'hpvel' dictionary
	readSharkData(self, sharkpath = '/home/luciebakels/SharkOutput/Shark-Lagos18-final-vcut30-alpha_v-0p85/', 
		datasets= ['id_subhalo_tree', 'mvir_hosthalo', 'mgas_bulge', 'mgas_disk', 'mhot', 'mstars_bulge', 'mstars_disk'],
		nfolders=32)
		Reads all galaxies.hdf5 files within the specified SHARK catalogue and saves it
		to the dictionary 'sharkdata'
	matchSharkData(self, haloes=None, sharkpath='/home/luciebakels/SharkOutput/Shark-Lagos18-final-vcut30-alpha_v-0p85/',
		nfolders=32, datasets=[])
		Matches Shark information to the correct haloes within the hp catalogue and saves it 
		to the hp catalogue

	--Matching haloes between run and computing gas fractions--
	match_VELOCIraptor(velpath, matchname='IndexMatch')
		Matches VR data of different runs by position and number of particles
	find_bound_gas_fraction(velpath)
		Finds gas from a hydro sim and matches it to the halodata catalogue
	find_bound_gasparticles_withinR200(velpath, snappath)
		Finds bound gas within R200 of each halo and matches it to the
		halodata catalogue
	get_DMFraction_boundgas_R200()
		Computes the gas fractions within each halo

	--Misc--
	findR200profileIndex()
		Returns the index of each profile closest to its R200 value
	exactDMFractionEstimation()
		Interpolate density profiles to find dark matter fractions in haloes
	interpolate(dataset)
		Interpolate a given profile
	findDMmassWithinR200()
		Find the DM mass within R200
	twopointcorrelation(halodata=None, bins=10, average_over=10, minmax=[0, 16])
		Computes the two point correlation function for a given range
	twopointcorrelation_kde(haloes=None, velpath=None, bins=10, average_over=10, minmax=[0, 16], boxsize=32, z=0)
		Computes the two point correlation function for a given range using a 
		Gaussian kde for the estimation of the densities
	neighbourDensity(maxdist=5)
		Assuming an NFW profile, computes the expected background density and temperature 
		at the location of each halo, resulting from its neighbours

	--Replaced by orbittree--
	readNeighbourData(haloes, closeFile=True)
		Reads halo properties of neighbouring haloes
	getVelRadNeighbourHaloes(radius = 10, masslim = 1, addHubbleFlow=True)
		Calculating neighbour halo properties
	
	"""
	def __init__(self, path, name, Hydro=None, snapshot=None, partType = None, new=True, TEMPORALHALOIDVAL=1000000000000, 
		boxsize=32, extra=True, physical=False, redshift=None, totzstart=30, totnumsnap=200, VELOCIraptor=False,
		TreeFrog_path=None):
		"""
		Parameters
		----------
		path : str
			path to the halo property file
		name : str
			name of the halo property file
		Hydro : bool
			a flag when set, reads in hydrosimulation fields
		snapshot : int
			snapshot number of the catalogue (default is None)
		partType : int
		new : bool
		TEMPORALHALOIDVAL : int
			from VELOCIraptor (default is 1e12)
		boxsize : float
			length of a side of the simulation box in units of the Coord field of 
			the catalogue (often in Mpc/h) (default is 32 Mpc/h)
		extra : bool
		physical : bool
			a flag when set to False, will convert the physical units of the catalogue
			to comoving units (default is False).
		redshift : float
			redshift of the snapshot, if not given, the redshift can be calculated
			using totzstart and totnumsnap (default is None)
		totzstart : float
			redshift of the first snapshot (default is 30)
		totnumsnap : int
			number of the last snapshot (default is 200)
		"""

		self.hp = {}
		self.hpvel = {}
		self.hydro = Hydro
		self.boxsize = boxsize
		self.THIDVAL = TEMPORALHALOIDVAL
		self.infofile = None
		self.halocoordtree = None
		self.path = path
		self.name = name
		self.snapshot = snapshot
		self.physical = physical
		self.redshift = redshift
		self.VELOCIraptor = VELOCIraptor
		self.tf_path = TreeFrog_path
		if redshift is None:
			self.redshift = snapshot_to_redshift(self.snapshot, zstart=totzstart, numsnap=totnumsnap)

		if partType is not None:
			if partType in [0, 2, 3, 4, 5]:
				sys.exit("Bestaat nog niet voor partType = %i" %partType)
			elif partType == 7:
				Hydro = True
			elif partType == 8:
				Hydro = True
		self.haloarray = (['HaloIndex', 'HaloID', 'Coord', 'R200', 'M200', 'R500', 'M500', 'R_HalfMass', 'Efrac', 'Ekin', 'cNFW',
				'Epot', 'redshift', 'snapshot', 'lambda', 'Density', 'Npart', 'Vmax', 'Rmax', 'hostHaloID',
				'AngularMomentum', 'Npart_profile', 'Radius', 'Velrad', 'Vel', 'Mass_profile', 'Partindices', 'n_part', 'MaxRadIndex', 
				'Virial_ratio', 'COM_offset', 'Msub', 'CrossTime', 'hostHaloIndex', 'Tail', 'Head', 'MassTable', 'Vmax_Duffy',
				'Efrac', 'dmpart_bound', 'npart'])
		if Hydro:
			self.haloarray.extend(['lambdaDM', 'lambdaH', 'DensityDM', 'DensityH', 'NpartH', 'NpartDM', 'n_gas', 'n_star',
				'NpartH_profile', 'DMFraction', 'DMFraction_profile', 'HFraction', 'HFraction_profile', 'MassH_profile', 'MassDM_profile', 
				'VelradDM', 'VelradH', 'Temperature', 'AngularMomentumDM', 'AngularMomentumH',
				'hpart_bound'])
		if partType == 8:
			self.haloarray.extend(['lambdaS', 'DensityS', 'NpartS',
				'NpartS_profile', 'SFraction', 'SFraction_profile', 'MassS_profile',
				'VelradB', 'VelradS', 'AgeS', 'AngularMomentumS'])

		if extra is not None:
			self.haloarray.extend(['RadiusFix', 'DMFractionAdjust', 'SharkDM', 'SharkH',
				'RootHead', 'RootTail', 'hostHaloIndexRoot', 'Mass_bound',
				'Neighbour_VelRad', 'Neighbour_Index', 'Neighbour_M200', 'Neighbour_R200', 'Neighbour_Distance', 
				'DMFraction_bound', 'DMFraction_bound_gasR200', 
				'Vol_ellips', 'Vol_Vc_ellips'])

		for key in self.haloarray:
			self.hp[key] = None

	def allocateSizes(self, key, lengte):
		"""(Private) Allocates the 'hp' dictionary when necessary
		"""
		if key in ['R200', 'M200', 'redshift', 'lambda', 'Vmax', 'Rmax', 'Efrac', 'R500', 'M500', 
				'R_HalfMass', 'Efrac', 'Ekin', 'Epot', 'cNFW', 
				'Virial_ratio', 'COM_offset', 'Msub', 'CrossTime', 'lambdaDM', 'lambdaH', 
				'DMFraction', 'HFraction', 'lambdaS', 'SFraction']:
			print(lengte[0], np.ones(lengte[0])*-1)
			return np.ones(lengte[0])*-1

		if key in ['HaloIndex', 'HaloID', 'hostHaloID', 'snapshot', 'Npart', 'NpartH', 'NpartDM', 'NpartS', 'npart',
				'n_part', 'MaxRadIndex', 'hostHaloIndex', 'Tail', 'Head', 'RootTail', 'RootHead','n_gas', 'n_star',
				'RootHead', 'RootTail', 'SharkDM', 'SharkH', 'dmpart_bound', 'hpart_bound']:
			return np.ones(lengte[0]).astype(int)*-1

		elif key in ['Coord', 'Vel']:
			return np.ones((lengte[0], 3))*-1

		elif key in ['Density', 'AngularMomentum', 'Velrad', 'Mass_profile',
				'DensityDM', 'DensityH', 'DMFraction_profile', 'HFraction_profile', 'MassH_profile', 'MassDM_profile', 
				'VelradDM', 'VelradH', 'Temperature', 'AngularMomentumDM', 'AngularMomentumH', 'lambdaS', 'DensityS',
				'SFraction_profile', 'MassS_profile','VelradB', 'VelradS', 'AgeS', 'AngularMomentumS']:
			return np.zeros((lengte[0], lengte[1]))

		elif key in ['Npart', 'Npart_profile', 'NpartDM_profile', 'NpartH_profile', 'NpartS_profile']:
			return np.zeros((lengte[0], lengte[1])).astype(int)

	def propertyDictionary(self, datasets=[]):
		"""(Private) Organises the right datatypes for each 'hp' dictionary field
		"""
		if len(datasets) == 0:
			datasets = self.hp.keys()
		d_prop = {}
		d_prop['None'] = []
		d_prop['single'] = []
		d_prop['scalar'] = []
		d_prop['array'] = []
		d_prop['dict'] = []
		for key in datasets:
			if self.hp[key] is None:
				d_prop['None'].append(key)
			elif key in ['Radius', 'RadiusFix', 'MassTable', 'redshift', 'snapshot']:
				d_prop['single'].append(key)
			elif isinstance(self.hp[key], dict):
				d_prop['dict'].append(key)
			elif key in ['Coord', 'DensityDM', 'DensityH', 'Density', 'NpartDM_profile', 'Npart_profile', 'DMFraction_profile', 'MassH_profile',
				'VelradDM', 'Velrad', 'VelradH', 'Vel', 'Temperature', 'Mass_profile', 'MassDM_profile'
				'AngularMomentumDM', 'AngularMomentumH', 'AngularMomentum']:
				d_prop['array'].append(key)
			else:
				d_prop['scalar'].append(key)
		return d_prop

	def makeHaloCoordTree(self, indices = None):
		"""Computes a cKDTree for all given haloes and stores it in halocoordtree

		Parameters
		----------
		indices : array of ints, optional
			if given, the function returns a cKDTree of the listed haloes instead
			of saving the full tree to 'halocoordtree'

		Returns
		-------
		None
			if indices is not given
		cKDTree
			if list of indices is given
		"""
		if indices is None:
			if self.halocoordtree is None:
				if self.hp['Coord'] is None:
					self.readData(datasets=['Coord'])
				if len(self.hp['Coord']) == 0:
					self.noHalo = True
					return 0
				self.halocoordtree = cKDTree(self.hp['Coord'], boxsize=self.boxsize)
		else:
			if self.hp['Coord'] is None:
				self.readData(datasets=['Coord'])
			return cKDTree(self.hp['Coord'][indices], boxsize=self.boxsize)

	def openFile(self, write=False):
		"""(Private) Opens the catalogue file
		"""
		if self.infofile is not None:
			return 0
		if write==True:
			wr = 'r+'
		else:
			wr = 'r'
		if os.path.isfile(self.path + self.name):
			self.infofile = h5py.File(self.path+self.name, wr)
		else:
			sys.exit("Error: file "+self.path+self.name+" not found.")		

	def closeFile(self):
		"""(Private) Closes the catalogue file
		"""
		if self.infofile is None:
			return 0
		self.infofile.close()
		self.infofile = None

	def openFileVR(self, datasets=[]):
		"""(Private) Opens the VR files and stores it in the haloproperties dictionary
		"""
		if self.infofile is not None:
			return 0

		if os.path.isfile(self.path + '/snapshot_%03d.properties' %self.snapshot) or os.path.isfile(self.path + '/snapshot_%03d.properties.0' %self.snapshot):
			temp_name = self.path + '/snapshot_%03d' %self.snapshot
		elif os.path.isfile(self.path + '/snapshot_%03d/snapshot_%03d.properties' %(self.snapshot, self.snapshot)) or os.path.isfile(self.path + '/snapshot_%03d/snapshot_%03d.properties.0' %(self.snapshot, self.snapshot)):
			self.path = self.path + '/snapshot_%03d/' %self.snapshot
			temp_name = self.path + '/snapshot_%03d' %self.snapshot
		elif os.path.isfile(self.path + '/snapshot_%03d.VELOCIraptor.properties' %self.snapshot) or os.path.isfile(self.path + '/snapshot_%03d.VELOCIraptor.properties.0' %self.snapshot):
			temp_name = self.path + '/snapshot_%03d.VELOCIraptor' %self.snapshot
		elif os.path.isfile(self.path + '/snapshot_%03d/snapshot_%03d.VELOCIraptor.properties' %(self.snapshot, self.snapshot)) or os.path.isfile(self.path + '/snapshot_%03d/snapshot_%03d.VELOCIraptor.properties.0' %(self.snapshot, self.snapshot)):
			self.path = self.path + '/snapshot_%03d/' %self.snapshot
			temp_name = self.path + '/snapshot_%03d.VELOCIraptor' %self.snapshot
		else:
			sys.exit("The VELOCIraptor path is incorrect, or the name convention is different for the version you are using.\n Change this in the openFileVR function in haloanalyse.py.")

		desiredfields = []
		for ds in datasets:
			if ds in [ 'redshift', 'snapshot', 'Density', 'Npart', 
				'Npart_profile', 'Radius', 'Velrad', 'Mass_profile', 
				'Partindices', 'n_part', 'MaxRadIndex', 'Virial_ratio', 'COM_offset', 'Msub', 'CrossTime', 
				'hostHaloIndex', 'Tail', 'Head', 'MassTable', 'Vmax_Duffy','dmpart_bound', 
				'lambdaDM', 'lambdaH', 'DensityDM', 'DensityH', 'NpartH', 'NpartDM', 'n_star',
				'NpartH_profile', 'DMFraction', 'DMFraction_profile', 'HFraction', 'HFraction_profile', 
				'MassH_profile', 'MassDM_profile', 'VelradDM', 'VelradH', 'Temperature', 'AngularMomentumDM', 
				'AngularMomentumH',	'hpart_bound', 'lambdaS', 'DensityS', 'NpartS',
				'NpartS_profile', 'SFraction', 'SFraction_profile', 'MassS_profile',
				'VelradB', 'VelradS', 'AgeS', 'AngularMomentumS']:
				continue
			elif ds == 'M200':
				desiredfields.append('Mass_200crit')
			elif ds == 'R200':
				desiredfields.append('R_200crit')
			elif ds == 'M500':
				desiredfields.append('SO_Mass_500_rhocrit')
			elif ds == 'R500':
				desiredfields.append('SO_R_200_rhocrit')
			elif ds in ['HaloID', 'HaloIndex']:
				desiredfields.append('ID')
			elif ds == 'Coord':
				desiredfields.append('Xcminpot')
				desiredfields.append('Ycminpot')
				desiredfields.append('Zcminpot')
			elif ds == 'Vel':
				desiredfields.append('VXc')
				desiredfields.append('VYc')
				desiredfields.append('VZc')
			elif ds == 'AngularMomentum':
				desiredfields.append('Lx')
				desiredfields.append('Ly')
				desiredfields.append('Lz')
			elif ds == 'lambda':
				desiredfields.append('lambda_B')
			elif ds == 'hostHaloID':
				desiredfields.append('hostHaloID')
			elif ds in (['R_HalfMass', 'Efrac', 'Ekin', 'cNFW',	'Epot', 'R_size', 'Rmax', 'Vmax', 'npart', 'n_gas']):
				desiredfields.append(ds)

		if len(desiredfields) == 0:
			return 0

		vel, numhalo, atime = vpt.ReadPropertyFile(temp_name, desiredfields=desiredfields)

		Nhalo = numhalo

		self.hp['redshift'] = 1/atime - 1
		for velset in desiredfields:
			if velset in (['Yc', 'Zc', 'Ycminpot', 'Zcminpot', 'VYc', 'VZc', 'VXcminpot', 
				'VYcminpot', 'VZcminpot', 'Ly', 'Lz']):
				continue
			elif velset == 'Xcminpot':
				ds = 'Coord'
			elif velset == 'VXc':
				ds = 'Vel'
			elif velset == 'Lx':
				ds = 'AngularMomentum'
			elif velset == 'lambda_B':
				ds = 'lambda'
			elif velset in (['hostHaloID', 'R_HalfMass', 'Efrac', 'Ekin', 'cNFW', 'Epot', 
				'R_size', 'Rmax', 'Vmax', 'npart', 'n_gas']):
				ds = velset
			elif velset == 'Mass_200crit':
				ds = 'M200'
			elif velset == 'R_200crit':
				ds = 'R200'
			elif velset == 'SO_Mass_500_rhocrit':
				ds = 'M500'
			elif velset == 'SO_R_200_rhocrit':
				ds = 'R500'
			elif velset == 'ID':
				ds = 'HaloID'
			else:
				continue
				
			self.allocateSizes(ds, [int(Nhalo), 0])
			print(ds, self.hp[ds])
			if Nhalo == 0:
				continue
			print(ds, velset)
			if ds == 'Coord':
				self.hp[ds][:, 0] = vel['Xcminpot']
				self.hp[ds][:, 1] = vel['Ycminpot']
				self.hp[ds][:, 2] = vel['Zcminpot']
			elif ds == 'Vel':
				self.hp[ds][:, 0] = vel['VXc']
				self.hp[ds][:, 1] = vel['VYc']
				self.hp[ds][:, 2] = vel['VZc']
			elif ds == 'AngularMomentum':
				self.hp[ds][:, 0] = vel['Lx']
				self.hp[ds][:, 1] = vel['Ly']
				self.hp[ds][:, 2] = vel['Lz']
			elif velset == 'ID':
				if 'HaloID' in datasets:
					self.hp['HaloID'] = vel['ID']
				if 'HaloIndex' in datasets:
					self.hp['HaloIndex'] = vel['ID'] - 1
			else:
				self.hp[ds] = vel[velset]

	def openFileTF(snapshot):
		hdffile = h5py.File(self.tf_path, 'r')
		numsnaps = hdffile['Header'].attrs["NSnaps"]

		halodata = {}

		for key in hdffile['Snapshots']['Snap_%03d' %snapshot].keys():
			halodata[key] = np.array(
				hdffile['Snapshots']['Snap_%03d' %snapshot][key])
		hdffile.close()

		return halodata

	def readDataVELOCIraptor(self, datasets=[]):

		#Only reading in datasets that aren't already read in
		datasetsnew = datasets.copy()
		
		if len(datasetsnew) == 0:
			datasetsnew = list(self.hp.keys())

		removelist = []

		for ds in datasetsnew:
			if ds not in self.haloarray:
				removelist.append(ds)
			elif self.hp[ds] is not None:
				removelist.append(ds)

		for ds in removelist:
			datasetsnew.remove(ds)

		#If a dataset is flagged to be completely read already,
		#it is removed from the new reading list
		if hasattr(self, "completely_read"):
			for rm in self.completely_read:
				if rm in datasetsnew:
					datasetsnew.remove(rm)

		#If all desired fields are already read, return 0
		if len(datasetsnew) == 0:
			self.non_complete = None
			return 0

		#Reading the new datafields from the VR catalogue and storing it in self.hp
		self.openFileVR(datasets=datasetsnew)

		#If the output is required to be in comoving units, it
		#converts the datasets here
		if self.physical == False:
			for ds in datasetsnew:
				if ds in ['Vel', 'VX', 'VY', 'VZ']:
					self.hp[ds] *= (1+self.redshift)
				elif ds == 'M200':
					self.hp[ds] *= h
				elif ds == 'R200':
					self.hp[ds] *= h*(1+self.redshift)
				elif ds in ['Coord', 'X', 'Y', 'Z']:
					self.hp[ds] *= h*(1+self.redshift)
		
		#Reading the relevant TreeFrog data
		if ('Head' in datasetsnew) or ('Tail' in datasetsnew) or ('RootHead' in datasetsnew) or ('RootTail' in datasetsnew):
			tree = openFileTF(self.snapshot)
			if 'Head' in datasetsnew:
				self.hp['Head'] = tree['Head'] - 1
			if 'Tail' in datasetsnew:
				self.hp['Tail'] = tree['Tail'] - 1
			if 'RootHead' in datasetsnew:
				self.hp['RootHead'] = tree['RootHead'] - 1
			if 'RootTail' in datasetsnew:
				self.hp['RootTail'] = tree['RootTail']	- 1


	def readData(self, datasets=[], closeFile=True, haloIndex=None):
		"""
		Reads data for given dataset fields and stores it in the 'hp' dictionary

		This function checks for each dataset if it is already read in,
		and only reads it in if it hasn't already. 

		Properties
		----------
		datasets : list of str, optional
			desired fields to be read in
		closeFile : bool
			a flag that can be set to False if the user wants to keep the
			hdf5 files open. This is used as a 'private' flag for some of
			the other functions
		haloIndex : array of int
			only reads in information for selected 'haloIndex' haloes
		VELOCIraptor : bool
			if set, directly reads in VELOCIraptor data instead of haloproperties dataset
			Note that the file naming convention might be different for different versions of VR

		Returns
		-------
		None
		"""

		if self.VELOCIraptor:
			self.readDataVELOCIraptor(datasets=datasets)
			return 0

		#Only reading in datasets that aren't already read in
		datasetsnew = datasets.copy()
		
		if len(datasetsnew) == 0:
			datasetsnew = list(self.hp.keys())

		removelist = []

		#If only selected haloes are read in, they
		#will need to be overwritten when this function is
		#called again
		overwrite = False
		if getattr(self, "not_complete", None) is None:
			if haloIndex is not None:
				self.not_complete = True
		else:
			overwrite = True

		if overwrite == False:
			for ds in datasetsnew:
				if ds not in self.haloarray:
					removelist.append(ds)
				elif self.hp[ds] is not None:
					removelist.append(ds)
				elif ds in ['Neighbour_VelRad', 'Neighbour_Index', 'Neighbour_M200', 'Neighbour_R200', 'Neighbour_Distance']:
					removelist.append(ds)

			for ds in removelist:
				datasetsnew.remove(ds)

		#If a dataset is flagged to be completely read already,
		#is is removed from the new reading list
		if hasattr(self, "completely_read"):
			for rm in self.completely_read:
				if rm in datasetsnew:
					datasetsnew.remove(rm)

		#If all desired fields are already read, return 0
		if len(datasetsnew) == 0:
			self.non_complete = None
			return 0

		#Reading the new datafields from the property file
		self.openFile()

		if haloIndex is not None:
			ylen = 0
			#The Radius and RadiusFix are read in regardless if the user
			#asks for it (only exist in profile files)
			if 'Radius'.encode('utf-8') in self.infofile.id:
				self.hp['Radius'] = self.infofile['Radius'.encode('utf-8')][:]
				ylen = len(self.hp['Radius'])
				if 'Radius' in datasetnew:
					removelist.append('Radius')
				if 'RadiusFix' in datasetnew:
					removelist.append('RadiusFix')
				for ds in removelist:
					datasetnew.remove(ds)
			Nhalo = self.infofile['HaloIndex'].size

		for key in self.infofile.id:
			if key.decode('utf-8') in datasetsnew:
				if isinstance(self.infofile[key].id, h5py.h5g.GroupID):
					self.hp[key.decode('utf-8')] = {}
					temp = self.infofile[key]
					if haloIndex is None:
						for key2 in self.infofile[key].id:
							haloindex = [int(s) for s in re.findall(r'\d+', key2.decode('utf-8'))][0]
							self.hp[key.decode('utf-8')][haloindex] = temp[key2][:]
					else:
						for haloindex in haloIndex:
							key2 = str(haloindex).encode('utf-8')
							if key2 in self.infofile[key].id:
								self.hp[key.decode('utf-8')][haloindex] = temp[key2][:]

				elif key.decode('utf-8') in ['snapshot', 'redshift']:
					self.hp[key.decode('utf-8')] = self.infofile[key].value
				elif isinstance(self.infofile[key].id, h5py.h5d.DatasetID):
					if haloIndex is None:
						self.hp[key.decode('utf-8')] = self.infofile[key][:]
					else:
						if self.hp[key.decode('utf-8')] is None:
							self.hp[key.decode('utf-8')] = self.allocateSizes(key.decode('utf-8'), [Nhalo, ylen])
						self.hp[key.decode('utf-8')][haloIndex] = self.infofile[key][list(haloIndex)]

		#Flag datasets to be already read, for next time the function is called
		if haloIndex is None:
			if not hasattr(self, "completely_read"):
				self.completely_read = datasetsnew
			else:
				self.completely_read.extend(datasetsnew)
				self.completely_read = list(set(self.completely_read))

		if closeFile:
			self.closeFile()

		#If the output is required to be in comoving units, it
		#converts the datasets here
		if self.physical == False:
			for ds in datasetsnew:
				if ds in ['Vel', 'VX', 'VY', 'VZ']:
					self.hp[ds] *= (1+self.redshift)
				elif ds == 'M200':
					self.hp[ds] *= h
				elif ds == 'R200':
					self.hp[ds] *= h*(1+self.redshift)
				elif ds in ['Coord', 'X', 'Y', 'Z']:
					self.hp[ds] *= h*(1+self.redshift)

	def readNeighbourData(self, haloes, closeFile=True):
		"""reads halo properties of neighbouring haloes
		"""
		datasets=['Neighbour_VelRad', 'Neighbour_Index', 'Neighbour_M200', 'Neighbour_R200', 'Neighbour_Distance']
		
		self.openFile()	

		for key in datasets:
			temp = self.infofile[key.encode('utf-8')]
			if self.hp[key] is None:
				self.hp[key] = {}
			for halo in haloes:
				if halo in self.hp[key].keys():
					continue
				self.hp[key][halo] = temp[str(halo).encode('utf-8')][:]
		
		if closeFile:
			self.closeFile()

	def reWriteData(self, datasets=[]):
		"""Rewrites given datasets that already exist in file
		"""
		self.openFile(write=True)

		if len(datasets)==0:
			itlist = self.hp.keys()
		else:
			itlist = datasets

		for ds in itlist:
			if isinstance(self.hp[ds], dict):
				if str(ds).encode('utf-8') not in self.infofile.id:
					self.closeFile()
					sys.exit('The group does not exist!')
				else:
					del self.infofile[str(ds).encode('utf-8')]
					temp = self.infofile.create_group(ds)
				for ds2 in self.hp[ds].keys():
					del temp[str(ds2).encode('utf-8')]
					temp.create_dataset(str(ds2), data=self.hp[ds][ds2])
			else:
				del self.infofile[str(ds).encode('utf-8')]
				self.infofile.create_dataset(ds, data=self.hp[ds])

		self.closeFile()

	def addData(self, datasets=[]):
		"""Adds datasets that don't exist in the file yet
		"""
		self.openFile(write=True)

		for ds in datasets:
			if isinstance(self.hp[ds], dict):
				if str(ds).encode('utf-8') not in self.infofile.id:
					temp = self.infofile.create_group(ds)
				else:
					temp = self.infofile[str(ds).encode('utf-8')]
				for ds2 in self.hp[ds].keys():
					temp.create_dataset(str(ds2), data=self.hp[ds][ds2])
			else:
				self.infofile.create_dataset(str(ds), data = self.hp[ds])

		self.closeFile()

	def match_VELOCIraptor(self, velpath, matchname='IndexMatch'):
		"""matches VR data of different runs by position and number of particles

		Parameters
		----------
		velpath : str
			path to the VELOCIraptor files
		matchname : str
			key name where the matched VR indices will be stored in the 'hp' dictionary

		Returns
		-------
		None
		"""
		vel = vpt.ReadPropertyFile(velpath, ibinary=2, desiredfields=['Xcmbp', 'Xc', 
			'Ycmbp', 'Yc', 'Zcmbp', 'Zc', 'npart', 'n_gas', 'Mass_200crit'])[0]
		if len(vel['Xc']) == 0:
			self.hp[matchname] = np.ones(0)*-1
			return 0
		self.readData(datasets=['Coord', 'n_part', 'redshift', 'M200', 'hostHaloIndex'])
		if len(self.hp['n_part']) == 0:
			self.hp[matchname] = np.ones(0)*-1
			return 0
		z = self.hp['redshift'][0]
		self.hp[matchname] = np.ones(len(self.hp['n_part'])).astype(int)*-1
		coords = np.zeros((len(vel['Xc']), 3))
		coords[:, 0] = (vel['Xcmbp']+vel['Xc'])*h*(1+z)
		coords[:, 1] = (vel['Ycmbp']+vel['Yc'])*h*(1+z)
		coords[:, 2] = (vel['Zcmbp']+vel['Zc'])*h*(1+z)
		coords = coords%self.boxsize	
		veltree = cKDTree(coords, boxsize=self.boxsize)
		self.hp['n_part'] = self.hp['n_part'].astype(int)
		vel['npart'] = vel['npart'].astype(int)
		vel['n_gas'] = vel['n_gas'].astype(int)
		for halo in range(len(self.hp[matchname])):
			dist, posmatch = veltree.query(self.hp['Coord'][halo], k=5)
			#print(dist, posmatch, (vel['npart'][posmatch] - vel['n_gas'][posmatch]), hd.hp['n_part'][halo], vel['Mass_200crit'][posmatch], hd.hp['M200'][halo])
			temp = np.abs(self.hp['n_part'][halo] - 
				(vel['npart'][posmatch] - vel['n_gas'][posmatch])).argmin()
			bestmatch = posmatch[temp]
			bestdist = dist[temp]
			if bestmatch in self.hp[matchname]:
				print("Halo already assigned", bestmatch)
				continue
			if (self.hp['hostHaloIndex'][halo] == -1 and 0.1*self.hp['M200'][halo] > vel['Mass_200crit'][bestmatch]) or (bestdist > 0.02 and 
				np.abs(self.hp['n_part'][halo] - (vel['npart'][bestmatch] - vel['n_gas'][bestmatch])) > 0.5*self.hp['n_part'][halo]):
				#print(bestdist,hd.hp['n_part'][halo], (vel['npart'][bestmatch] - vel['n_gas'][bestmatch]), np.abs(hd.hp['n_part'][halo] - (vel['npart'][bestmatch] - vel['n_gas'][bestmatch])))
				print("No matching halo for", halo)
				continue
			self.hp[matchname][halo] = bestmatch

	def find_bound_gas_fraction(self, velpath):
		"""Finds gas from a hydro sim and matches it to the halodata catalogue

		Parameters
		----------
		velpath : str
			path to the VELOCIraptor files

		Returns
		-------
		None
		"""
		if 'IndexMatch' not in self.hp.keys():
			self.match_VELOCIraptor(velpath)
			if len(self.hp['IndexMatch']) == 0:
				self.hp['DMFraction_bound'] = np.ones(0)*-1
				return 0
		if len(self.hp['IndexMatch']) == 0:
			self.hp['DMFraction_bound'] = np.ones(0)*-1
			return 0
		velp = vpt.ReadParticleTypes(velpath, unbound=False)
		self.readData(datasets=['Npart' ,'DMFraction', 'MassTable'])
		self.hp['DMFraction_bound'] = np.ones(len(self.hp['IndexMatch']))*-1
		self.hp['Mass_bound'] = np.ones(len(self.hp['IndexMatch']))*-1
		self.hp['NumPart'] = np.ones(len(self.hp['IndexMatch'])).astype(int)*-1
		self.hp['dmpartvel'] = np.ones(len(self.hp['IndexMatch'])).astype(int)*-1
		self.hp['hpartvel'] = np.ones(len(self.hp['IndexMatch'])).astype(int)*-1
		self.hp['dmpart'] = np.ones(len(self.hp['IndexMatch'])).astype(int)*-1
		self.hp['hpart'] = np.ones(len(self.hp['IndexMatch'])).astype(int)*-1
		for halo in range(len(self.hp['DMFraction_bound'])):
			if self.hp['IndexMatch'][halo] == -1 or len(velp['Particle_Types'][self.hp['IndexMatch'][halo]]) == 0:
				continue
			dmpart = len(np.where(velp['Particle_Types'][self.hp['IndexMatch'][halo]])[0])
			allpart = len(velp['Particle_Types'][self.hp['IndexMatch'][halo]])
			gaspart = allpart - dmpart
			self.hp['NumPart'][halo] = allpart
			self.hp['dmpartvel'][halo] = dmpart
			self.hp['hpartvel'][halo] = gaspart
			self.hp['dmpart'][halo] = int(self.hp['Npart'][halo]*self.hp['MassTable'][0]*self.hp['DMFraction'][halo] /
				(self.hp['MassTable'][0]*self.hp['DMFraction'][halo] + self.hp['MassTable'][1]*(1 - self.hp['DMFraction'][halo])))
			self.hp['hpart'][halo] = self.hp['Npart'][halo] - self.hp['dmpart'][halo]
			self.hp['DMFraction_bound'][halo] = dmpart*self.hp['MassTable'][1]/(self.hp['MassTable'][0]*gaspart+self.hp['MassTable'][1]*dmpart)
			self.hp['Mass_bound'][halo] = self.hp['MassTable'][0]*gaspart+self.hp['MassTable'][1]*dmpart

	def find_bound_gasparticles_withinR200(self, velpath, snappath):
		"""Finds bound gas within R200 of each halo and matches it to the
			halodata catalogue

			The 'IndexMatch' field in the hp dictionary that links the desired
			VR catalogue to the catalogue in this class, needs to exist.
			This can be done with the function 'match_VELOCIraptor()'.

		Parameters
		----------
		velpath : str
			path to the VELOCIraptor files
		snappath : str
			path to the snapshot files

		Returns
		-------
		None
		"""
		vel = vpt.ReadParticleDataFile(velpath, ibinary=2, iparttypes=1, unbound=False)
		snap200 = Snapshot('/media/luciebakels/DATA1/snapshots/', 200, partType=0)
		snap200.makeCoordTree()
		self = HaloData('/home/luciebakels/HaloInfo_9p/', 'snapshot_200.info.hdf5', Hydro=True, new=True)
		self.readData(datasets=['Coord', 'R200', 'DMFraction', 'Npart'])

		snapID = pd.DataFrame({'A': snap200.get_IDs()})
		self.hp['Ngas_r200_b'] = np.zeros(len(self.hp['R200'])).astype(int)
		for halo in range(len(self.hp['IndexMatch'])):
			if self.hp['IndexMatch'][halo] == -1:
				self.hp['Ngas_r200_b'][halo] = -1
				continue
			snapID = pd.DataFrame({'A': vel['Particle_IDs'][self.hp['IndexMatch'][halo]]})
			print(halo)
			indices = np.array(snap200.tree.query_ball_point(self.hp['Coord'][halo], r=self.hp['R200'][halo]))
			if len(indices) == 0:
				self.hp['Ngas_r200_b'][halo] = 0
				continue
			IDs = snap200.IDs[indices]
			self.hp['Ngas_r200_b'][halo] = len(np.where(snapID['A'].isin(IDs))[0])

	def get_DMFraction_boundgas_R200(self):
		"""Computes the gas fractions within each halo

			The fields 'Ngas_r200_b' and 'dmpart' need to be loaded
			'Ngas_r200_b' can be computed using the function 'find_bound_gasparticles_wihinR200'
		"""
		self.hp['DMFraction_bound_gasR200'] = (self.hp['dmpart']*self.hp['MassTable'][1]/(self.hp['dmpart']*self.hp['MassTable'][1] + 
			self.hp['Ngas_r200_b']*self.hp['MassTable'][0]))
		self.hp['DMFraction_bound_gasR200'][np.where(self.hp['Ngas_r200_b']==-1)] = -1

	def vel_correction(self, vel, z=0):
		#return vel*np.sqrt(1/(1+z))
		return vel

	def getVelRadNeighbourHaloes(self, radius = 10, masslim = 1, addHubbleFlow=True):
		"""(Replaced by orbittree.py) Calculating neighbour halo properties.

		Find radial velocity, indices, M200, R200, and distances from
		all neighbouring halos. Then attaching the data to either the 
		large halo or the small halo, so the information can be read 
		from the sattellite or cluster point of view.

		Parameters
		----------
		radius : float
			this number times R200 of the host haloes is the radius within neighbours
			are being selected
		masslim : float
			the minimum mass in units of hp['M200'] to find neighbours for
		addHubbleFlow : bool
			a flag if set, includes the Hubble flow when computing radial velocities

		Returns
		-------
		None
		"""
		self.makeHaloCoordTree()

		self.readData(datasets = ['M200', 'R200', 'Coord', 'Vel', 'HaloIndex', 'redshift'])
		self.hp['Neighbour_VelRad'] = {}
		self.hp['Neighbour_Index'] = {}
		self.hp['Neighbour_M200'] = {}
		self.hp['Neighbour_R200'] = {}
		self.hp['Neighbour_Distance'] = {}

		if getattr(self, "noHalo", None) is not None:
			print("No haloes in "+self.name)
			return 0
		z = self.hp['redshift']
		c = constant()
		c.change_constants(z)

		print("Computing neighbour properties for "+self.name)
		for halo in range(len(self.hp['M200'])):
			self.hp['Neighbour_VelRad'][halo] = np.zeros(0)
			self.hp['Neighbour_Index'][halo] = np.zeros(0).astype(int)
			self.hp['Neighbour_M200'][halo] = np.zeros(0)
			self.hp['Neighbour_R200'][halo] = np.zeros(0)
			self.hp['Neighbour_Distance'][halo] = np.zeros(0)
		for halo in range(len(self.hp['M200'])):
			if self.hp['M200'][halo]<masslim or self.hp['M200'][halo] == -1 or self.hp['R200'][halo] > 0.8:
				continue

			buren = self.halocoordtree.query_ball_point(self.hp['Coord'][halo], r = radius*self.hp['R200'][halo])
			if len(buren) <= 1:
				continue

			buren.remove(halo)

			buren = np.array(buren)

			velocity = self.hp['Vel'][buren] - self.hp['Vel'][halo]
			coords = self.hp['Coord'][buren] - self.hp['Coord'][halo]
			coords = np.where(np.abs(coords) > 0.5*self.boxsize, coords - coords/np.abs(coords)*self.boxsize, coords)
			ndisttemp = np.sqrt(np.sum(coords*coords, axis=1))
			if addHubbleFlow:
				velocity = self.vel_correction(velocity, z=z) + c.H * np.array([ndisttemp]).T/h
			else:
				self.vel_correction(velocity, z=z)
			vel = (velocity[:, 0]*(coords[:, 0])*Mpc_to_km + velocity[:, 1]*(coords[:, 1])*Mpc_to_km  + 
				velocity[:, 2]*(coords[:, 2])*Mpc_to_km)

			#if bighalo:
			self.hp['Neighbour_M200'][halo] = np.append(self.hp['Neighbour_M200'][halo], self.hp['M200'][buren])
			self.hp['Neighbour_R200'][halo] = np.append(self.hp['Neighbour_R200'][halo], self.hp['R200'][buren])
			self.hp['Neighbour_Index'][halo] = np.append(self.hp['Neighbour_Index'][halo], buren)
			self.hp['Neighbour_Distance'][halo] = np.append(self.hp['Neighbour_Distance'][halo], ndisttemp)
			self.hp['Neighbour_VelRad'][halo] = np.append(self.hp['Neighbour_VelRad'][halo], vel/(ndisttemp*Mpc_to_km))
			#	continue

			distances = np.sqrt(np.sum(coords*coords, axis=1))
			velrads = vel/(distances*Mpc_to_km)
			for x in range(len(buren)):
				if halo in self.hp['Neighbour_Index'][buren[x]]:
					continue
				if buren[x] == halo:
					continue
				self.hp['Neighbour_M200'][buren[x]] = np.append(self.hp['Neighbour_M200'][buren[x]], self.hp['M200'][halo])
				self.hp['Neighbour_R200'][buren[x]] = np.append(self.hp['Neighbour_R200'][buren[x]], self.hp['R200'][halo])
				self.hp['Neighbour_Index'][buren[x]] = np.append(self.hp['Neighbour_Index'][buren[x]], self.hp['HaloIndex'][halo]%self.THIDVAL)
				self.hp['Neighbour_Distance'][buren[x]] = np.append(self.hp['Neighbour_Distance'][buren[x]], distances[x])
				self.hp['Neighbour_VelRad'][buren[x]] = np.append(self.hp['Neighbour_VelRad'][buren[x]], velrads[x])

	def replaceWithVel(self, halodata, datasets = None, velsets=None):
		"""Replacing a 'hp' properties dataset with, or adding, a VR dataset

		Parameters
		----------
		halodata : dict
			VELOCIraptor catalogue
		datasets : list of str, optional
			list of datasets to be added or replaced by the VR catalogue,
			if set, will replace; if not, will add (default is None)
		velsets : list of str
			list of VR catalogue fields that the user wants to add.
			if the user wants to replace 'hp' datasets, it needs to have the
			same shape and order as the 'datasets' parameter.

		Returns
		-------
		None
		"""

		if (velsets is None) and (datasets is None):
			return 0

		if velsets is None:
			sys.exit("No VELOCIraptor given")

		self.readData(datasets=['HaloID'])
		haloids = halodata['ID']

		if datasets is None:
			for vs in velsets:
				self.hp[vs] = np.zeros(len(self.hp['HaloID']))
			for j in range(len(self.hp['HaloID'])):
				indexhd = np.where(haloids == self.hp['HaloID'][j])[0]
				for vs in velsets:
					self.hp[vs][j] = halodata[vs][indexhd]
	
		elif len(datasets) != len(velsets):
			sys.exit("Datasets and Velsets are not compatible")		


		else:
			self.readData(datasets=datasets)
			for j in range(len(self.hp['HaloID'])):
				indexhd = np.where(haloids == self.hp['HaloID'][j])[0]
				for dsi in range(len(ds)):
					self.hp[datasets[dsi]][j] = halodata[vs[dsi]][indexhd]

	def readVel(self, halodata, velsets=None):
		"""Saves VR catalogue datasets to the 'hpvel' dictionary

		Parameters
		----------
		halodata : VR catalogue
		velsets : list of str
			the VR fields to be copied

		Returns
		-------
		None
		"""
		if velsets is None:
			sys.exit("No VELOCIraptor given")

		for vs in velsets:
			self.hpvel[vs] = halodata[vs]

	def makeAllHaloCoordTree(self, halodata, indices=None):
		"""(Private) cKDTree used for the twopointcorrelation function
		"""
		if getattr(self, 'allcoordtree', None) is not None:
			return 0
		if indices is None:
			indices = np.arange(len(self.hpvel['Xc'])).astype(int)
		self.readVel(halodata, velsets=['Xc', 'Yc', 'Zc', 'Xcmbp', 'Ycmbp', 'Zcmbp'])
		self.readData(datasets=['redshift'])
		z = self.hp['redshift']
		coords = np.zeros((len(indices), 3))
		coords[:, 0] = (self.hpvel['Xcmbp'][indices]+self.hpvel['Xc'][indices])*h*(1+z)
		coords[:, 1] = (self.hpvel['Ycmbp'][indices]+self.hpvel['Yc'][indices])*h*(1+z)
		coords[:, 2] = (self.hpvel['Zcmbp'][indices]+self.hpvel['Zc'][indices])*h*(1+z)
		coords = coords%self.boxsize	
		self.allcoordtree = cKDTree(coords, boxsize=self.boxsize)

	def findR200profileIndex(self):
		"""Returns the index of each profile closest to its R200 value

		Returns
		-------
		array
		"""
		self.readData(datasets=['Radius', 'R200'])
		return np.abs(np.subtract.outer(self.hp['Radius'], self.hp['R200'])).argmin(0)

	def exactDMFractionEstimation(self):
		"""Interpolate density profiles to find dark matter fractions in haloes
		"""
		self.readData(datasets=['DMFraction', 'Radius', 'RadiusFix', 'R200', 'DensityDM', 
			'DensityH', 'MassDM_profile', 'MassH_profile'])
		
		if len(self.hp['R200']) == 0:
			self.hp['DMFractionAdjust'] = np.zeros(0)
			return 0

		densityR = self.findR200profileIndex()
		densityR2 = densityR + 1
		densitydm = self.hp['DensityDM'][np.arange(len(self.hp['R200'])).astype(int), densityR]
		densityh = self.hp['DensityH'][np.arange(len(self.hp['R200'])).astype(int), densityR]
		densitydm2 = self.hp['DensityDM'][np.arange(len(self.hp['R200'])).astype(int), densityR2]
		densityh2 = self.hp['DensityH'][np.arange(len(self.hp['R200'])).astype(int), densityR2]

		massdm = 4./3.*np.pi*self.hp['R200']**3*(densitydm+ densitydm2)/2.
		massh = 4./3.*np.pi*self.hp['R200']**3*(densityh+ densityh2)/2.
		self.hp['DMFractionAdjust'] = np.zeros(len(self.hp['R200']))
		for i in range(len(self.hp['R200'])):
			if self.hp['R200'][i] == -1:
				self.hp['DMFractionAdjust'][i] = -1
				continue
			elif self.hp['R200'][i] < self.hp['RadiusFix'][0]:
				massDMn = np.max([self.hp['MassDM_profile'][i][0] - massdm[i], 0])
				massHn = np.max([self.hp['MassH_profile'][i][0] - massh[i], 0])
			else:
				massDMinter = interp1d(self.hp['RadiusFix'], self.hp['MassDM_profile'][i])
				massHinter = interp1d(self.hp['RadiusFix'], self.hp['MassH_profile'][i])
				massDMn = np.max([massDMinter(self.hp['R200'][i]) - massdm[i], 0])
				massHn = np.max([massHinter(self.hp['R200'][i]) - massh[i], 0])
			if massDMn == 0 and massHn == 0:
				self.hp['DMFractionAdjust'][i] = self.hp['DMFraction'][i]
				continue
			self.hp['DMFractionAdjust'][i] = massDMn/(massDMn + massHn)

	def interpolate(self, dataset):
		"""Interpolate a given profile, saving the interp1d function to self.result

		Parameters
		----------
		dataset : str
			the name of the dataset in the profile data to interpolate over
		"""
		self.readData(datasets=['Radius'])
		self.readData(datasets=[dataset])
		self.result = interp1d(self.hp['Radius'], self.hp[dataset])

	def findDMmassWithinR200(self):
		"""Find the DM mass within R200 and saves it within the hp dictionary as 'DM_M200'
		"""
		self.hp['DM_M200'] = np.zeros(len(self.hp['HaloIndex']))
		self.hp['DM_M200'] = np.where(self.hp['M200'] == -1, -1, self.hp['M200']*self.hp['DMFraction'])

	def twopointcorrelation(self, halodata=None, bins=10, average_over=10, minmax=[0, 16]):
		"""Computes the two point correlation function for a given range

		The resulting 2-point correlation function found for each halo in the hp catalogue
		will be saved to the hp dictionary with the key name: 'twopointcorrelation'.

		Parameters
		----------
		halodata : list of dict, optional
			VELOCIraptor catalogue: if selected, the 2-point correlation function of the areas
			haloes in this class catalogue (hp) will be computed using ALL haloes from the 
			VELOCIraptor catalogues, if not selected, only the subselection of haloes within
			the hp catalogue will be used (default is None)
		bins : int
			number of bins to compute the 2-point correlation function for (default is 10)
		average_over : int
			number of times to do the random selection to get an average 'background function'
			necessary for the computation of the 2-point correlation (default is 10)
		minmax : list(2)
			the radius over which the correlation function is computed

		Returns
		-------
		None
		"""
		if halodata is None:
			self.readData(datasets=['M200'])
			self.allcoordtree = self.makeHaloCoordTree(indices = np.where(self.hp['M200'] != -1)[0])
			self.hpvel = {}
			self.hpvel['Xc'] = self.hp['Coord'][:, 0]
		else:
			self.makeAllHaloCoordTree(halodata, indices = np.where(halodata['hostHaloID'] == -1)[0])

		if getattr(self, "noHalo", None) is not None:
			return 0

		self.readData(datasets=['HaloIndex', 'Coord'])


		def makeHistfixedRange(a, bins, minmax=[0, 16], log=True):
			if log:
				a = np.delete(a, np.where(a==0.0)[0])
				binar = np.logspace(np.log10(0.1), np.log10(minmax[1]), bins)
				offset = np.logspace(np.log10(binar[0])+(np.log10(binar[1]/binar[0]))*0.5, 
					np.log10(binar[-2])-(np.log10(binar[-1]/binar[-2]))*0.5, bins-1)
			else:
				binar = np.arange(minmax[0], minmax[1], (minmax[1]-minmax[0])/bins)
				offset = np.arange(binar[0] + 0.5*(binar[1]-binar[0]), 
					binar[-2] + (binar[-1]-binar[-2]), binar[1]-binar[0])
			histres = np.zeros(len(binar) - 1)
			binsize = np.zeros(len(binar) - 1)
			for i in range(len(binar)-1):
				tijdelijk = np.where((a > binar[i]) & (a < binar[i+1]))[0]
				histres[i] = len(tijdelijk)
				binsize[i] = binar[i+1] - binar[i]
			return offset, histres, binsize

		
		if getattr(self, "treerand", None) is None:
			self.npart = len(self.hpvel['Xc'])

			xrand = np.random.rand(self.npart)*self.boxsize
			yrand = np.random.rand(self.npart)*self.boxsize
			zrand = np.random.rand(self.npart)*self.boxsize

			self.coordrand = np.zeros((self.npart, 3))

			self.coordrand[:, 0] = xrand
			self.coordrand[:, 1] = yrand
			self.coordrand[:, 2] = zrand

			self.treerand = cKDTree(self.coordrand, boxsize=self.boxsize)

		alldistances = np.zeros(self.npart)
		for i in range(average_over):
			aantal = len(self.treerand.query_ball_point(self.coordrand[i], r = minmax[1]))
			alldistancestemp, indices = self.treerand.query(self.coordrand[i], k=aantal)
			alldistances = np.append(alldistances, alldistancestemp)

		offset, histres, binsize = makeHistfixedRange(alldistances, bins=bins, minmax=minmax)
		histres = histres/average_over
		nR = np.sum(histres)

		self.hp['twopointcorrelation'] = np.zeros((len(self.hp['Coord']), bins-1))
		self.hp['twopointradius'] = offset
		self.hp['twopointrand'] = histres

		for halo in range(len(self.hp['Coord'])):
			aantal = len(self.allcoordtree.query_ball_point(self.hp['Coord'][halo], r = minmax[1]))
			ad, temp = self.allcoordtree.query(self.hp['Coord'][halo], k=aantal)
			temp, hr_halo, bs = makeHistfixedRange(ad, bins=bins, minmax=minmax)
			nD = np.sum(hr_halo)

			self.hp['twopointcorrelation'][halo] = hr_halo/histres - 1

	def twopointcorrelation_kde(self, haloes=None, velpath=None, bins=10, average_over=10, minmax=[0, 16], boxsize=32, z=0):
		"""Computes the two point correlation function for a given range using a 
		Gaussian kde for the estimation of the densities

		The resulting 2-point correlation function found for each halo in the hp catalogue
		will be saved to the hp dictionary with the key name: 'twopointcorrelation'.

		Parameters
		----------
		halodata : list of dict, optional
			VELOCIraptor catalogue: if selected, the 2-point correlation function of the areas
			haloes in this class catalogue (hp) will be computed using ALL haloes from the 
			VELOCIraptor catalogues, if not selected, only the subselection of haloes within
			the hp catalogue will be used (default is None)
		bins : int
			number of bins to compute the 2-point correlation function for (default is 10)
		average_over : int
			number of times to do the random selection to get an average 'background function'
			necessary for the computation of the 2-point correlation (default is 10)
		minmax : list(2)
			the radius over which the correlation function is computed

		Returns
		-------
		None
		"""
		self.readData(datasets=['Coord'])
		coords = self.hp['Coord']

		if velpath is None:
			self.readData(datasets=['hostHaloIndex'])
			self.allcoordtree = self.makeHaloCoordTree(indices = np.where(self.hp['hostHaloIndex'] == -1)[0])
			npart = len(self.hp['Coord'][:, 0])
		else:
			vel = vpt.ReadPropertyFile(velpath, ibinary=2, desiredfields=['Xcmbp', 'Xc', 
				'Ycmbp', 'Yc', 'Zcmbp', 'Zc', 'hostHaloID'])[0]
			self.makeAllHaloCoordTree(vel, indices = np.where(halodata['hostHaloID'] == -1)[0])

			coordsvel = np.zeros((len(vel['Xc']), 3))
			coordsvel[:, 0] = (vel['Xcmbp']+vel['Xc'])*h*(1+z)%boxsize
			coordsvel[:, 1] = (vel['Ycmbp']+vel['Yc'])*h*(1+z)%boxsize
			coordsvel[:, 2] = (vel['Zcmbp']+vel['Zc'])*h*(1+z)%boxsize
			coordsvel = coordsvel[vel['hostHaloID'] == -1]
			self.allcoordtree = cKDTree(coordsvel, boxsize=boxsize)
			npart = len(voordsvel[:, 0])

		self.hp['twopointradius'] = np.logspace(np.log10(np.max([0.1, minmax[0]])), np.log10(minmax[1]), bins)

		xrand = np.random.rand(npart)*boxsize
		yrand = np.random.rand(npart)*boxsize
		zrand = np.random.rand(npart)*boxsize

		coordrand = np.zeros((npart, 3))

		coordrand[:, 0] = xrand
		coordrand[:, 1] = yrand
		coordrand[:, 2] = zrand

		treerand = cKDTree(coordrand, boxsize=boxsize)

		alldistances = np.zeros(npart)
		for i in range(average_over):
			aantal = len(treerand.query_ball_point(coordrand[i], r = minmax[1]))
			distrand, indices = treerand.query(coordrand[i], k=aantal)
			distrand = np.delete(distrand, [0])
			densityrandom = stats.kde.gaussian_kde(distrand, bw_method='scott')
			if i == 0:
				antrand = densityrandom(self.hp['twopointradius'])*len(distrand)
			else:
				antrand += densityrandom(self.hp['twopointradius'])*len(distrand)
		antrand /= average_over

		self.hp['twopointcorrelation'] = np.zeros((len(coords), bins))
		self.hp['twopointrand'] = antrand

		if haloes is None:
			haloes = np.arange(len(coords))

		for halo in haloes:
			aantal = len(coordtree.query_ball_point(coords[halo], r = minmax[1]))
			dist, temp = coordtree.query(coords[halo], k=aantal)
			dist = np.delete(dist, [0])

			density = stats.kde.gaussian_kde(dist, bw_method='scott')

			self.hp['twopointcorrelation'][halo] = density(self.hp['twopointradius'])*len(dist)/antrand - 1

	def neighbourDensity(self, maxdist=5):
		"""Assuming an NFW profile, computes the expected background density and temperature 
		at the location of each halo, resulting from its neighbours

		The resulting densities and temperatures are saved to 'Neighbour_Density' and
		'Neighbour_Temperature', respectively, in the 'hp' dictionary.

		Parameters
		----------
		maxdist : float
			The maximum radius in units of hp['Coord'] to find neighbouring haloes for that
			are used for calculating the bacground densities and temperatures

		Returns
		-------
		None
		"""
		if 'Neighbour_Density' in list(self.hp.keys()):
			if getattr(self, "maxdist", None) == maxdist:
				return 0

		self.maxdist = maxdist

		self.makeHaloCoordTree()
		self.readData(datasets=['HaloIndex', 'M200', 'R200', 'Coord'])

		def densityProfileNFW(r, Mvir, Rvir):
			cNFW = 10
			r_v = Rvir
			r_s = r_v/cNFW
			rho_0 = Mvir*(4*np.pi*r_s**3*(np.log(1 + cNFW) - cNFW/(1 + cNFW)))
			return rho_0/(r/r_s * (1 + r/r_s)**2)

		def temperatureProfileNFW(r, Mvir, Rvir):
			def mNFW(x):
				return np.log(1 + x) - x/(1 + x)
			cNFW = 10
			r_v = Rvir * Mpc_to_cm
			r_s = r_v/cNFW
			gamma = 1.15 + 0.01*(cNFW - 6.5)

			eta_0 = 0.00676*(cNFW-6.5)**2 + 0.206*(cNFW-6.5) + 2.48

			T_0 = G_cm3_gi_si2*Mvir*Msun_g*hydrogen_g/r_v/3.0/kB_erg_Ki*eta_0
			y = 1 - 3. / eta_0 * (gamma - 1) / gamma / mNFW(cNFW) * cNFW * (1 - np.log(1 + r*Mpc_to_cm/r_s)/(r*Mpc_to_cm/r_s))
			if not isinstance(r, (list, tuple, np.ndarray)):
				if r == 0:
					return T_0
			else:
				y[np.where(r == 0)[0]] = 0
			return T_0*y

		self.hp['Neighbour_Density'] = np.zeros(len(self.hp['M200']))
		self.hp['Neighbour_Temperature'] = np.zeros(len(self.hp['M200']))

		for halo in range(len(self.hp['Neighbour_Density'])):
			aantal = len(self.halocoordtree.query_ball_point(self.hp['Coord'][halo], r = maxdist))
			rad, index = self.halocoordtree.query(self.hp['Coord'][halo], k=aantal)
			waar = np.where(self.hp['M200'][index] > self.hp['M200'][halo])[0]

			if len(waar) == 0:
				continue
			else:
				self.hp['Neighbour_Density'][halo] = np.sum(densityProfileNFW(rad[waar], 
					self.hp['M200'][index[waar]]*1e10, self.hp['R200'][index[waar]]))
				self.hp['Neighbour_Temperature'][halo] = np.sum(temperatureProfileNFW(rad[waar], 
					self.hp['M200'][index[waar]]*1e10, self.hp['R200'][index[waar]]))

	def readSharkFile(self, path, name, datasets=None):
		"""(Private) Reads a single file within the specified SHARK catalogue
		"""
		sharkdata = h5py.File(path+name, 'r')
		d_data = {}
		haloprop = sharkdata['galaxies'.encode('utf-8')]
		if datasets is None:
			datasets = haloprop.id
		for key in datasets:
			if key not in self.sharkdata.keys():
				key = key.encode('utf-8')
				d_data[key.decode('utf-8')] = haloprop[key][:]
		return d_data

	def readSharkData(self, sharkpath = '/home/luciebakels/SharkOutput/Shark-Lagos18-final-vcut30-alpha_v-0p85/', 
		datasets= ['id_subhalo_tree', 'mvir_hosthalo', 'mgas_bulge', 'mgas_disk', 'mhot', 'mstars_bulge', 'mstars_disk'],
		nfolders=32):
		"""Reads all galaxies.hdf5 files within the specified SHARK catalogue and saves it
		to the dictionary 'sharkdata'


		Parameters
		----------
		sharkpath : str
			path to the SHARK catalogues
		datasets : list of str
			list of the datasets within SHARK that are being read
		nfolders : int
			number of subvolumes within the SHARK data

		Returns
		-------
		None

		"""
		snaps = np.array(os.listdir(sharkpath)).astype(int)
		sharkdata = {}
		
		if not hasattr(self, "sharkdata"):
			self.sharkdata = {}

		for folder in range(nfolders):
			if os.path.isdir(sharkpath+'/'+str(folder)) == False:
				continue
			sharkdatatemp = self.readSharkFile(sharkpath+'/'+str(folder)+'/', 'galaxies.hdf5', datasets=datasets)
			
			for key in sharkdatatemp.keys():
				if key not in sharkdata.keys():
					sharkdata[key] = sharkdatatemp[key]
				else:
					#print(snap, folder, len(sharkdata[snap][key]))
					sharkdata[key] = np.append(sharkdata[key], sharkdatatemp[key])

		for key in sharkdata.keys():
			self.sharkdata[key] = sharkdata[key]

		# for snap in snaps:
		# 	sharkdata[snap] = {}
		# 	for folder in range(nfolders):
		# 		sharkdatatemp = readSharkFile(sharkpath+str(snap)+'/'+str(folder)+'/', 'galaxies.hdf5')
			
		# 		for key in sharkdatatemp.keys():
		# 			if key not in sharkdata[snap].keys():
		# 				sharkdata[snap][key] = sharkdatatemp[key]
		# 			else:
		# 				#print(snap, folder, len(sharkdata[snap][key]))
		# 				sharkdata[snap][key] = np.append(sharkdata[snap][key], sharkdatatemp[key])

		# self.sharkdata = sharkdata

	def matchSharkData(self, haloes=None, sharkpath='/home/luciebakels/SharkOutput/Shark-Lagos18-final-vcut30-alpha_v-0p85/',
		nfolders=32, datasets=[]):
		"""Matches Shark information to the correct haloes within the hp catalogue and saves it 
		to the hp catalogue

		Parameters
		----------
		haloes : array of ints, optional
			indices of haloes within the hp catalogue that are being matched to the SHARK data,
			if not set, it will consider all haloes within the catalogue (default is None)
		sharkpath : str
			path to the SHARK data
		nfolders : str
			number of SHARK subvolumes
		dataset : list of str
			list of the datasets within SHARK that are being read

		Returns
		-------
		None

		"""
		for key in ['id_subhalo_tree', 'mvir_hosthalo', 'mgas_bulge', 'mgas_disk', 'mhot', 'mstars_bulge', 'mstars_disk']:
			datasets.append(key)
		self.readSharkData(sharkpath=sharkpath, nfolders=nfolders, datasets=datasets)
		self.readData(datasets=['Tail', 'Head', 'HaloID'])

		self.hp['SharkDM'] = np.zeros(len(self.hp['Tail']))
		self.hp['SharkH'] = np.zeros(len(self.hp['Tail']))
		#self.hp['SharkID'] = np.zeros(len(self.hp['Tail'])).astype(int)
		for ds in datasets:
			self.hp[ds] = np.zeros(len(self.hp['Tail']))
		self.hp['Ngalaxy'] = np.zeros(len(self.hp['Tail']))

		shark_haloes = np.array(list(set(self.sharkdata['id_subhalo_tree'])))

		my_haloes_match = self.hp['Tail'] + 1
		replace = np.where(my_haloes_match == 0)[0]
		my_haloes_match[replace] = replace + self.snapshot*self.THIDVAL + 1

		shark_haloes_match = self.sharkdata['id_subhalo_tree']
		replace = np.where(shark_haloes_match > (self.snapshot+1)*self.THIDVAL)[0]
		shark_haloes_match[replace] = shark_haloes_match[replace]%(1000*self.THIDVAL)

		shark_found = np.zeros(0).astype(int)
		my_found = np.zeros(0).astype(int)

		if haloes is None:
			h_in_shark = np.where(np.in1d(my_haloes_match, self.sharkdata['id_subhalo_tree']))[0]
		else:
			h_in_shark = haloes[np.in1d(my_haloes_match[haloes], self.sharkdata['id_subhalo_tree'])]

		allH = (self.sharkdata['mgas_bulge'] + self.sharkdata['mgas_disk'] + 
			self.sharkdata['mstars_bulge'] + self.sharkdata['mstars_disk'])


		#df = pd.DataFrame({'A': self.sharkdata['id_halo_tree']})
		tempid = np.copy(self.sharkdata['id_subhalo_tree'])
		temparange = np.arange(len(tempid))
		for halo in h_in_shark:
			numbers_to_delete = np.where(np.in1d(tempid, my_haloes_match[halo]))[0]
			
			if len(numbers_to_delete) == 0:
				continue

			allgalaxies = temparange[numbers_to_delete]
			#self.hp['SharkID'][halo_i[i]] = halo
			self.hp['SharkDM'][halo] = self.sharkdata['mvir_hosthalo'][allgalaxies][0]
			self.hp['SharkH'][halo] = np.sum(allH[allgalaxies]) + self.sharkdata['mhot'][allgalaxies][0]
			self.hp['Ngalaxy'][halo] = len(allgalaxies)
			for ds in datasets:
				self.hp[ds][halo] = np.sum(self.sharkdata[ds][allgalaxies])

			tempid = np.delete(tempid, numbers_to_delete)
			temparange = np.delete(temparange, numbers_to_delete)

			# allgalaxies = np.array(df['A'].isin([self.hp['Tail'][halo]])).astype(bool)
			# #self.hp['SharkID'][halo_i[i]] = halo
			# self.hp['SharkDM'][halo] = self.sharkdata['mvir_hosthalo'][allgalaxies][0]
			# self.hp['SharkH'][halo] = np.sum(allH[allgalaxies])
			# print(halo, self.hp['SharkDM'][halo])

class HaloTree:
	"""
	Class that contains and computes properties of haloes over time

	Container class of HaloData, and can compute properties over time

	Attributes
	----------
	halotree : dict
		dictionary containing a HaloData class for each snapshot
	snapstart : int
		first snapshot to be considered
	zstart : float
		redshift of the first snapshot in the simulation
	snapend : int
		number of the last snapshot
	THIDVAL : int
		VELOCIraptor TEMPORALHALOIDVAL
	trees : dict
		dictionary of the tree per halo
	mergers : dict
		dictionary containing the information of all merged systems per halo
	nbtree : dict
		dictionary containing the information of all non-merged systems
		per host halo
	boxsize : float
		length of a side of the simulation box

	Functions
	---------
	--Private--
	dt_private(self, a, H0, Om0, Ol0)
	timeDifference_private(self, z1, z2, H0=h*100, Om0=Om0)

	--Reading and Writing--
	makeHaloCoordTree(self)
	readData(self, datasets=[], haloIDs=None)
	addData(self, datasets=[])
	readVelRadNeighbourHaloes(self, haloes = None)
	updateTree(self, path_halodata)
	reWriteTree(self, path_halodata)
	reWriteSatelliteInfo(self)
	writeNeighbourInfo(self)
	add_Volume_ellips(self, path_halodata, snapstart=None, snapend=None)

	--Simulation properties--
	timeDifference(self, z1, z2, H0=h*100, Om0=Om0)
	redshiftDifference(self, t1, t2, H0=h*100, Om0=Om0)

	--Halo properties--
	halo_key(self, halo, snapshot)
	key_halo(self, halokey)
	exactDMFractionEstimation(self)
	findDMmassWithinR200(self)

	--Tree properties--
	findRootHead(self, halo)
	findAllRootHead(self)

	--Replaced by orbittree--
	getVelRadNeighbourHaloes(self, radius = 10)
	treemaker(self, haloes = None, datasets=[], overwrite=False, snapshot=None, full_tree=False, mergers_only=False, 
	readNBinfo=True, read_only_used_haloes=False)
	getMergerProperties(self, halo, snapshot, dataset='M200')
	neighbourtreemaker2(self, main_halo, neighbour_haloes, datasets=None, read_only_used_haloes=False, enclosedMass=True,
		calculate_eccentricities=False)
	overwriteSatelliteProperties(self)
	loopOverNeighbourInfo(self, halo, neighbours, snapshot, 
		datasets= ['DMFraction_bound', 'M200', 'n_part', 'redshift', 'snapshot', 'HaloIndex'])
	getNeighbourTrajectories(self, halo, snapshot, neighbours=None, reupdate=False, full_tree = True, 
		datasets = ['DMFraction_bound', 'M200', 'n_part', 'redshift', 'snapshot', 'HaloIndex'])
	readNeighbourData(self, haloes, closeFile=True)
	addNeighbourInfo(self, halo, snapshot, datasets=None)
	add_DMFraction_bound(self, path_velociraptor, snapstart=None, snapend=None)

	--Misc--
	initial_fixeverything(self, path_halodata)

	findMassAccrTime(self, massFraction=0.5)
	computeFormationTime(self, massFraction=0.5)
	computePeakDMMassFraction(self, haloes)
	
	findhalodivide(self, radius=3, main_halo_frac_mass=2, min_mass_frac_sat=0.25, min_distance=3)
	hostHaloIndex_toRootHead(self)
	constructSubset(self, halo, radius = 10, snapshot=200)
	constructWhole(self, radius = 10, snapshot = 200, neighbourTrajectories=False, full_tree=False)

	"""
	def __init__(self, path, snapstart=0, snapend=200, Hydro=True, new=True, TEMPORALHALOIDVAL=1000000000000, 
		new_format=False, boxsize=32., physical=False, zstart=30, VELOCIraptor=False, TreeFrog_path=None):
		"""
		Parameters
		----------
		path : str
			path to the halo property files
		Hydro : bool
			a flag when set, reads in hydrosimulation fields
		snapshot : int
			snapshot number of the catalogue (default is None)
		partType : int
		new : bool
		TEMPORALHALOIDVAL : int
			from VELOCIraptor (default is 1e12)
		boxsize : float
			length of a side of the simulation box in units of the Coord field of 
			the catalogue (often in Mpc/h) (default is 32 Mpc/h)
		extra : bool
		physical : bool
			a flag when set to False, will convert the physical units of the catalogue
			to comoving units (default is False).
		redshift : float
			redshift of the snapshot, if not given, the redshift can be calculated
			using totzstart and totnumsnap (default is None)
		totzstart : float
			redshift of the first snapshot (default is 30)
		VELOCIraptor : bool
			if set True, reads in VELOCIraptor directly instead of the copied files
			If set, you need to provide TreeFrog_path
		"""
		self.halotree = {}
		self.snapstart = snapstart
		self.zstart = zstart
		self.snapend = snapend
		self.THIDVAL = TEMPORALHALOIDVAL
		self.trees = {}
		self.mergers = {}
		self.nbtree = {}
		self.boxsize = boxsize
		#Saving all HaloData object in the halotree dictionary
		if new_format:
			for snap in range(self.snapstart, self.snapend+1):
				self.halotree[snap] = HaloData(path, 'snapshot_%03d.quantities.hdf5'%snap,
					Hydro=Hydro, new=new, snapshot=snap, TEMPORALHALOIDVAL=TEMPORALHALOIDVAL, boxsize=boxsize, 
					physical=physical, totzstart=self.zstart, totnumsnap=self.snapend, VELOCIraptor=VELOCIraptor,
					TreeFrog_path = TreeFrog_path)
		else:
			for snap in range(self.snapstart, self.snapend+1):
				self.halotree[snap] = HaloData(path, 'snapshot_%03d.info.hdf5'%snap, 
					Hydro=Hydro, new=new, snapshot=snap, TEMPORALHALOIDVAL=TEMPORALHALOIDVAL, boxsize=boxsize, 
					physical=physical, totzstart=self.zstart, totnumsnap=self.snapend, VELOCIraptor=VELOCIraptor,
					TreeFrog_path = TreeFrog_path)
	
	def dt_private(self, a, H0, Om0, Ol0):
		"""Returns time in units of seconds

		Parameters
		----------
		a : float
			time in 1/(1+redshift)
		H0 : float
			Hubble constant
		Om0 : float
			Omega matter at z=0
		Ol0 : float
			Omega Lambda at z=0

		Returns
		-------
		float
			time in seconds
		"""
		return 1./H0*np.sqrt(Om0/a + a*a*Ol0)*Mpc_to_km 

	def timeDifference_private(self, z1, z2, H0=h*100, Om0=Om0):
		"""Computes the time difference between two redshifts in seconds

		Parameters
		----------
		z1, z2 : floats
			the redshift interval
		H0 :float
			Hubble constant
		Om0 : float
			Omega matter at z=0

		Returns
		-------
		float
			time in seconds
		"""
		a = np.sort([1./(1+z1), 1./(1+z2)])
		return integrate.quad(self.dt_private, a[0], a[1], args=(H0, Om0, 1-Om0))[0]

	def timeDifference(self, z1, z2, H0=h*100, Om0=Om0):
		"""Fast way of converting (a list of) redshift intervals to time by 
		storing time intervals as a function of redshift interval

		Parameters
		----------
		z1, z2 : floats
			the redshift interval
		H0 : float
			Hubble constant
		Om0 : float
			Omega matter at z=0

		Returns
		-------
		float or array
			time in seconds
		"""

		#Checking if time intervals are already calculated, and if not,
		#doing it here.
		if hasattr(self, "lbt_dt") == False:
			self.redshift_arr = np.logspace(np.log10(1), np.log10(self.zstart + 1), 100000)[::-1] - 1
			self.lbt_dt = np.zeros(len(self.redshift_arr))
			for i in range(len(self.redshift_arr) - 1):
				self.lbt_dt[i] = self.timeDifference_private(self.redshift_arr[i], self.redshift_arr[i+1], H0=H0, Om0=Om0)
		if hasattr(self, "lbt_cum") == False:
			self.lbt_cum = np.cumsum(self.lbt_dt)

		if isinstance(z1, np.ndarray) and isinstance(z2, np.ndarray):
			if len(z1) != len(z2):
				print("Error, z1 and z2 don't have the same length")
				return 0
			i1 = np.zeros(len(z1)).astype(int)
			i2 = np.zeros(len(z1)).astype(int)
			for i in range(len(z1)):
				i1[i] = np.abs(self.redshift_arr - z1[i]).argmin()		
				i2[i] = np.abs(self.redshift_arr - z2[i]).argmin()
		elif isinstance(z1, np.ndarray):
			i1 = np.zeros(len(z1)).astype(int)
			for i in range(len(z1)):
				i1[i] = np.abs(self.redshift_arr - z1[i]).argmin()
			i2 = np.abs(self.redshift_arr - z2).argmin()
		elif isinstance(z2, np.ndarray):
			i2 = np.zeros(len(z2)).astype(int)
			for i in range(len(z2)):
				i2[i] = np.abs(self.redshift_arr - z2[i]).argmin()
			i1 = np.abs(self.redshift_arr - z1).argmin()
		else:
			i2 = np.abs(self.redshift_arr - z2).argmin()
			i1 = np.abs(self.redshift_arr - z1).argmin()

		return np.abs(self.lbt_cum[i1] - self.lbt_cum[i2])

	def redshiftDifference(self, t1, t2, H0=h*100, Om0=Om0):
		"""Time interval (lookback time in seconds) to redshift

		Parameters
		----------
		t1, t2 : floats
			the time interval in lookback time (seconds)
		H0 : float
			Hubble constant
		Om0 : float
			Omega matter at z=0

		Returns
		-------
		float or array
			redshift at given t1 and t2
		"""
		if hasattr(self, "lbt_dt") == False:
			self.redshift_arr = np.arange(0, self.zstart, 0.0001)[::-1]
			self.lbt_dt = np.zeros(len(self.redshift_arr))
			for i in range(len(self.redshift_arr) - 1):
				self.lbt_dt[i] = self.timeDifference_private(self.redshift_arr[i], self.redshift_arr[i+1], H0=H0, Om0=Om0)
		i1 = np.abs(np.cumsum(self.lbt_dt[::-1]) - t1).argmin()
		i2 = np.abs(np.cumsum(self.lbt_dt[::-1]) - t2).argmin()

		return np.array(self.redshift[np.min([-i1, -i2])], self.redshift[np.max([-i2, -i1])])
	
	def halo_key(self, halo, snapshot):
		"""Combines the index of halo(es) and snapshot number to get a unique 'key'

		Note that although these keys are similar to VELOCIraptor's IDs, they start
		at 0 instead of 1

		Parameters
		----------
		halo : int, list of int
			halo indices
		snapshot: int, list of int
			snapshots belonging to each halo index

		Returns
		-------
		int, array
			unique halo keys
		"""
		if isinstance(halo, list):
			return np.array(halo).astype(int) + self.THIDVAL*snapshot
		return halo + self.THIDVAL*snapshot

	def key_halo(self, halokey):
		"""Extracts the snapshot(s) and halo(es) from the unique halo 'key'

		Parameters
		----------
		halokey : int, array of int, list of int
			unique key of the halo(es) (e.g. 189000000000123)

		Returns
		-------
		int, array of int
			index of the halo
		int, array of int
			snapshot of the halo
		"""
		if isinstance(halokey, np.ndarray):
			return halokey%self.THIDVAL, (halokey/self.THIDVAL).astype(int)
		elif isinstance(halokey, list):
			return np.array(halokey)%self.THIDVAL, (np.array(halokey)/self.THIDVAL).astype(int)
		else:
			return halokey%self.THIDVAL, int(halokey/self.THIDVAL)	
	
	def readData(self, datasets=[], haloIDs=None):
		"""Reading data from the catalogues and saving it to the
		hp dictionary

		Parameters
		----------
		datasets : list of str, optional
			datasets to be read, if not set, will read everything
		haloIDs : array
			array of the haloes that are going to be read
			if not set, will read all haloes

		Returns
		-------
		None
		"""
		if haloIDs is None:
			for snap in self.halotree.keys():
				self.halotree[snap].readData(datasets=datasets)
			return 0
		hids = np.copy(haloIDs)
		#Tail always needs to be read to be able to construct the trees
		if 'Tail' not in datasets:
			datasets.append('Tail')
		for snap in np.arange(self.snapstart, self.snapend+1).astype(int)[::-1]:
			hi, sn = self.key_halo(hids)
			welke = np.where(sn == snap)[0]
			if len(welke) == 0:
				continue
			#print(len(welke), len(hi), sn[0], snap)
			self.halotree[snap].readData(datasets=datasets, haloIndex=np.array(hi[welke]).astype(int))
			tailtemp = self.halotree[snap].hp['Tail'][np.array(hi[welke]).astype(int)]
			hids = np.delete(hids, welke)
			hids = np.append(tailtemp, hids)

	def addData(self, datasets=[]):
		"""Writing additional fields to the catalogue

		Parameters
		----------
		datasets : list of str
			datasets to be written to file

		Returns
		-------
		None
		"""
		for snap in self.halotree.keys():
			self.halotree[snap].addData(datasets=datasets)

	def getVelRadNeighbourHaloes(self, radius = 10):
		"""Calculating neighbour halo properties over the whole tree
		"""
		for snap in self.halotree.keys():
			self.halotree[snap].getVelRadNeighbourHaloes(radius = radius)

	def readNeighbourData(self, haloes, closeFile=True):
		"""Reads halo properties of neighbouring haloes over the whole tree
		for given haloes
		"""
		allhaloes = self.key_halo(haloes)
		if len(haloes) == 1:
			self.halotree[allhaloes[1][0]].readNeighbourData(haloes=allhaloes[0], closeFile=closeFile)
			return 0
		for snap in self.halotree.keys():#range(np.min(allhaloes[1]), np.max(allhaloes[1])+1):
			htemp = np.where(allhaloes[1] == snap)[0]
			if len(htemp) == 0:
				continue
			self.halotree[snap].readNeighbourData(haloes=allhaloes[0][htemp], closeFile=closeFile)

	def readVelRadNeighbourHaloes(self, haloes = None):
		if haloes is None:
			self.readData(datasets=['Neighbour_VelRad', 'Neighbour_Index', 'Neighbour_M200', 'Neighbour_R200', 'Neighbour_Distance'])
		else:
			self.readNeighbourData(haloes = haloes)

	def updateTree(self, path_halodata):
		"""(Redundant) If 'Tail' is not correctly linked, this can be used to correctly add 'Tail' and 'Head' to the catalogue
		"""
		updateAlreadyDone = True
		for snap in self.halotree.keys():
			self.halotree[snap].readData(datasets=['Tail', 'HaloIndex']) 
			if len(self.halotree[snap].hp['Tail']) != len(self.halotree[snap].hp['HaloIndex']):
				updateAlreadyDone = False
		if updateAlreadyDone:
			return 0
		atime,tree,numhalos,halodata,cosmodata,unitdata = vpt.ReadUnifiedTreeandHaloCatalog(path_halodata, 
			icombinedfile=1, iverbose=1, desiredfields=['ID', 'Tail', 'Head'])
		for i in range(self.snapstart, self.snapend+1):
			self.halotree[i].readData(datasets = ['HaloID', 'M200', 'Coord'])
			self.halotree[i].hp['Tail'] = np.zeros(len(self.halotree[i].hp['HaloID']), dtype=np.int64)
			self.halotree[i].hp['Head'] = np.zeros(len(self.halotree[i].hp['HaloID']), dtype=np.int64)
		for i in range(self.snapstart, self.snapend+1):
			print('snapshot',i)
			haloids = halodata[i]['ID']
			tailids = halodata[i]['Tail']
			headids = halodata[i]['Head']
			print('Number of haloes:',len(self.halotree[i].hp['HaloID']))
			for j in range(len(self.halotree[i].hp['HaloID'])):
				indexhd = np.where(haloids==self.halotree[i].hp['HaloID'][j])[0]
				tailtemp = tailids[indexhd]
				headtemp = headids[indexhd]
				if int(tailtemp/self.THIDVAL) == i:
					self.halotree[i].hp['Tail'][j] = j + self.THIDVAL*i
				else:
					indextemp2 = np.where(tailtemp==self.halotree[int(tailtemp/self.THIDVAL)].hp['HaloID'])[0]
					lim = 0
					while len(indextemp2) == 0 and lim < 5:
						tailtemp = halodata[int(tailtemp/self.THIDVAL)]['Tail'][int(tailtemp%self.THIDVAL - 1)]
						indextemp2 = np.where(tailtemp==self.halotree[int(tailtemp/self.THIDVAL)].hp['HaloID'])[0]
						lim += 1
					if len(indextemp2)==0:
						self.halotree[i].hp['Tail'][j] = j + self.THIDVAL*i
					else:
						self.halotree[i].hp['Tail'][j] = indextemp2 + (self.THIDVAL*(int(tailtemp/self.THIDVAL)))
				if int(headtemp/self.THIDVAL) == i:
					self.halotree[i].hp['Head'][j] = j + self.THIDVAL*i
				else:
					indextemp2 = np.where(headtemp==self.halotree[int(headtemp/self.THIDVAL)].hp['HaloID'])[0]
					lim = 0
					while len(indextemp2) == 0 and lim < 5:
						headtemp = halodata[int(headtemp/self.THIDVAL)]['Head'][int(headtemp%self.THIDVAL - 1)]
						indextemp2 = np.where(headtemp==self.halotree[int(headtemp/self.THIDVAL)].hp['HaloID'])[0]
						lim += 1
					if len(indextemp2) == 0:
						self.halotree[i].hp['Head'][j] = j + self.THIDVAL*i
					else:
						self.halotree[i].hp['Head'][j] = indextemp2 + (self.THIDVAL*(int(headtemp/self.THIDVAL)))

	def reWriteTree(self, path_halodata):
		"""Redundant
		"""
		self.updateTree(path_halodata=path_halodata)
		for snap in self.halotree.keys():
			self.halotree[snap].hp['HaloIndex'] = np.arange(len(self.halotree[snap].hp['HaloID'])).astype(np.int64) + self.THIDVAL*snap
			self.halotree[snap].addData(datasets=['HaloIndex', 'Tail', 'Head'])

	def reWriteSatelliteInfo(self):
		"""Redundant
		"""
		needschange = ['DMFraction', 'M200', 'R200', 'lambda', 'lambdaDM', 'lambdaH']
		self.overwriteSatelliteProperties()
		for snap in self.halotree.keys():
			self.halotree[snap].reWriteData(datasets=needschange)

	def add_DMFraction_bound(self, path_velociraptor, snapstart=None, snapend=None):
		"""Runs HaloData.find_bound_gas_fraction for all snapshots and saves it to file
		"""
		if snapstart is None:
			snapstart = self.snapstart
		if snapend is None:
			snapend = self.snapend
		for snap in self.halotree.keys():
			if (snap < snapstart) or (snap > snapend):
				continue
			print("Processing snapshot %i" %snap)
			self.halotree[snap].find_bound_gas_fraction(path_velociraptor+'/snapshot_%03d/snapshot_%03d' %(snap, snap))
			self.halotree[snap].addData(datasets=['IndexMatch', 'DMFraction_bound'])

	def add_Volume_ellips(self, path_halodata, snapstart=None, snapend=None):
		"""Compute the volume of each ellips halo by using the VR q and s fields
		"""
		if snapstart is None:
			snapstart = self.snapstart
		if snapend is None:
			snapend = self.snapend

		self.readData(datasets=['redshift'])
		atime,tree,numhalos,halodata,cosmodata,unitdata = vpt.ReadUnifiedTreeandHaloCatalog(path_halodata, 
			icombinedfile=1, iverbose=1, desiredfields=['ID', 'R_size', 'q', 's', 'Rmax', 'RVmax_q', 'RVmax_s'])
		for snap in self.halotree.keys():
			if (snap < snapstart) or (snap > snapend):
				continue
			print("Processing snapshot %i" %snap)
			self.halotree[snap].replaceWithVel(halodata[snap], 
				velsets=['R_size', 'q', 's', 'Rmax', 'RVmax_q', 'RVmax_s'])
			if len(self.halotree[snap].hp['redshift']) == 0:
				continue
			z = self.halotree[snap].hp['redshift']
			self.halotree[snap].hp['Vol_ellips'] = 4./3.*np.pi*self.halotree[snap].hp['q']*self.halotree[snap].hp['s']*(self.halotree[snap].hp['R_size']*h*(1+z))**3
			self.halotree[snap].hp['Vol_Vc_ellips'] = 4./3.*np.pi*self.halotree[snap].hp['RVmax_q']*self.halotree[snap].hp['RVmax_s']*(self.halotree[snap].hp['Rmax']*h*(1+z))**3
			#self.halotree[snap].addData(datasets=['Vol_ellips', 'Vol_Vc_ellips'])

	def initial_fixeverything(self, path_halodata):
		"""Redundant
		"""
		print("Fixing tree")
		self.reWriteTree(path_halodata)
		print("Fixing radius")
		for snap in self.halotree.keys():
			fixedRadius = np.logspace(-3, 0, 60)*0.75
			self.halotree[snap].hp['RadiusFix'] = fixedRadius
			fout = np.logspace(np.log10(fixedRadius[0]) + 0.5, np.log10(fixedRadius[-1]) + 0.5, len(fixedRadius))
			self.halotree[snap].hp['Radius'] = offsetRadius(fout)
			self.halotree[snap].reWriteData(datasets=['Radius'])
		print("Fixing satellite properties")
		self.reWriteSatelliteInfo()

		print("Making DMFraction adjustments")
		self.exactDMFractionEstimation()
		self.addData(datasets=['RadiusFix', 'DMFractionAdjust'])

		print("Write neighbour information")
		self.writeNeighbourInfo()

		print("Write RootHeads")
		self.findAllRootHead()
		self.addData(datasets=['RootHead'])

		print("Write volume ellipsoids")
		self.add_Volume_ellips(path_halodata)

	def exactDMFractionEstimation(self):
		"""Runs HaloData.exactDMFractionEstimation for all snapshots
		"""
		for snap in self.halotree.keys():
			print("snapshot %03d" %snap)
			self.halotree[snap].exactDMFractionEstimation()

	def writeNeighbourInfo(self):
		"""Redundant
		"""
		for snap in self.halotree.keys():
			self.halotree[snap].getVelRadNeighbourHaloes(radius = 10)
			self.halotree[snap].addData(datasets=['Neighbour_VelRad', 'Neighbour_Index', 'Neighbour_M200', 'Neighbour_R200', 'Neighbour_Distance'])

	def makeHaloCoordTree(self):
		"""Computing the cKDTree of each HaloData object (snapshot) within the HaloTree
		"""
		for snap in self.halotree.keys():
			self.halotree[snap].makeHaloCoordTree()

	def findDMmassWithinR200(self):
		"""Runs HaloData.findDMmassWithinR200 for all snapshots
		"""
		for snap in self.halotree.keys():
			self.halotree[snap].findDMmassWithinR200()

	def findRootHead(self, halo):
		"""Walking the trees to find the unique halo keys at z=0 for a given list of haloes

		Parameters
		----------
		halo : int, array of int
			unique halo keys of all haloes the keys at z=0 will be searched for
		Returns
		-------
		array
			the unique halo keys of all given haloes at z=0
		"""
		self.readData(datasets=['Head'])
		headoud, snap_i = self.key_halo(halo)
		headnieuw = self.halotree[snap_i].hp['Head'][headoud]
		while (headnieuw != headoud+self.THIDVAL*snap_i):
			headoud, snap_i = self.key_halo(headnieuw)
			headnieuw = self.halotree[snap_i].hp['Head'][headoud]
		return headnieuw

	def findAllRootHead(self):
		"""Walking the trees to find the unique halo keys at z=0 of all haloes at each snapshot

		Returns
		-------
		None
		"""
		self.readData(datasets=['Head'])
		for snap in self.halotree.keys():
			self.halotree[snap].hp['RootHead'] = np.zeros(len(self.halotree[snap].hp['Head'])).astype(int)
			i = 0
			for halo in self.halotree[snap].hp['Head']:
				headoud, snap_i = self.key_halo(halo)
				headnieuw = self.halotree[snap_i].hp['Head'][headoud]
				while(headnieuw != headoud+self.THIDVAL*snap_i):
					headoud, snap_i = self.key_halo(headnieuw)
					headnieuw = self.halotree[snap_i].hp['Head'][headoud]
				self.halotree[snap].hp['RootHead'][i] = headnieuw
				i += 1

	def treemaker(self, haloes = None, datasets=[], overwrite=False, snapshot=None, full_tree=False, mergers_only=False, 
		readNBinfo=True, read_only_used_haloes=False):
		"""Redundant
		"""
		if not mergers_only:
			start_time = time.time()

		if snapshot is None:
			snapshot=self.snapend
		alreadyUpdated = True

		if not mergers_only:
			start_time = time.time()
			if (full_tree==True) or (haloes is None) or (read_only_used_haloes==False):
				self.readData(datasets=['Tail', 'Head', 'snapshot', 'HaloIndex'])
				self.readData(datasets=datasets)
			else:
				datasets.extend(['Tail', 'snapshot', 'HaloIndex'])
				datasets = list(set(datasets))
				self.readData(datasets=datasets, HaloIDs=haloes)
			
			for snap in self.halotree.keys():
				if len(self.halotree[snap].hp['Tail']) != len(self.halotree[snap].hp['HaloIndex']):
					alreadyUpdated = False
			if not alreadyUpdated:
				sys.exit("You need to run 'updateTree()' first")
			
			if haloes is None:
				haloes = self.halotree[snapshot].hp['HaloIndex']
		
			for snap in self.halotree.keys():
				self.halotree[snap].readData(datasets=datasets)
		
			if len(datasets) == 0:
				datasets = list(self.halotree[snapshot].hp.keys())
		

		d_prop = self.halotree[snapshot].propertyDictionary(datasets=datasets)
		
		for datatype in d_prop.keys():
			if datatype == 'single' and len(d_prop[datatype]) >0:
				for ds in d_prop[datatype]:
					self.trees[ds] = self.halotree[snapshot].hp[ds]
		

		if readNBinfo:
			if 'Neighbour_Index' in datasets:
				self.readNeighbourData(haloes, closeFile=False)

		for halo in haloes:
			halo_index, snapshot = self.key_halo(halo)
			if full_tree:
				if halo not in self.mergers.keys():
					self.mergers[halo] = np.zeros(0).astype(int)
				if overwrite == False and halo in self.trees.keys():
					continue
			self.trees[halo] = {}

			for datatype in d_prop.keys():
				if datatype == 'None':
					for ds in d_prop[datatype]:
						if ds == 'snapshot':
							self.trees[halo][ds] = snapshot
					continue
				elif datatype == 'dict' and len(d_prop[datatype])>0:
					for ds in d_prop[datatype]:
						self.trees[halo][ds] = {}
						self.trees[halo][ds][snapshot] = self.halotree[snapshot].hp[ds][halo_index]
				elif datatype == 'array' and len(d_prop[datatype])>0:
					for ds in d_prop[datatype]:
						self.trees[halo][ds] = [self.halotree[snapshot].hp[ds][halo_index]]
				elif datatype == 'scalar' and len(d_prop[datatype])>0:
					for ds in d_prop[datatype]:
						self.trees[halo][ds] = self.halotree[snapshot].hp[ds][halo_index]
				elif datatype == 'single' and len(d_prop[datatype])>0:
					for ds in d_prop[datatype]:
						if ds == 'snapshot':
							self.trees[halo][ds] = snapshot
						else:
							self.trees[halo][ds] = self.halotree[snapshot].hp[ds]

			tailtemp = self.halotree[snapshot].hp['Tail'][halo_index]
			tailtempoud = self.halotree[snapshot].hp['HaloIndex'][halo_index]
			while(tailtempoud != tailtemp):
				if readNBinfo:
					if 'Neighbour_Index' in datasets:
						self.readNeighbourData([tailtemp], closeFile=False)
				for datatype in d_prop.keys():
					if datatype == 'None':
						for ds in d_prop[datatype]:
							if ds == 'snapshot':
								self.trees[halo][ds] = np.append(self.trees[halo][ds], int(tailtemp/self.THIDVAL))
						continue
					elif datatype == 'dict' and len(d_prop[datatype])>0:
						for ds in d_prop[datatype]:
							self.trees[halo][ds][int(tailtemp/self.THIDVAL)] = self.halotree[int(tailtemp/self.THIDVAL)].hp[ds][tailtemp%self.THIDVAL]
					elif datatype == 'array' and len(d_prop[datatype])>0:
						for ds in d_prop[datatype]:
							self.trees[halo][ds] = np.concatenate((self.trees[halo][ds], 
								[self.halotree[int(tailtemp/self.THIDVAL)].hp[ds][tailtemp%self.THIDVAL]]))
					elif datatype == 'scalar' and len(d_prop[datatype])>0:
						for ds in d_prop[datatype]:
							self.trees[halo][ds] = np.append(self.trees[halo][ds], 
									self.halotree[int(tailtemp/self.THIDVAL)].hp[ds][tailtemp%self.THIDVAL])
					elif datatype == 'single' and len(d_prop[datatype])>0:
						for ds in d_prop[datatype]:
							if ds == 'snapshot':
								self.trees[halo][ds] = np.append(self.trees[halo][ds], int(tailtemp/self.THIDVAL))
							else:
								self.trees[halo][ds] = np.append(self.trees[halo][ds], 
									self.halotree[int(tailtemp/self.THIDVAL)].hp[ds])				
				if full_tree:
					allprogenitors = np.zeros(0).astype(int)
					for snapterug in range(5):
						allprogenitors = np.append(allprogenitors, 
							self.halo_key(np.where(self.halotree[int(tailtemp/self.THIDVAL)-snapterug].hp['Head'] == tailtempoud)[0], 
								int(tailtemp/self.THIDVAL)-snapterug))
						allprogenitors = allprogenitors[allprogenitors != tailtemp]
					self.mergers[halo] = np.append(self.mergers[halo], allprogenitors)
					if len(allprogenitors) > 0:
						self.treemaker(haloes = allprogenitors, datasets=datasets, full_tree=True, mergers_only=True, overwrite=True)
				tailtempoud = tailtemp
				tailtemp = self.halotree[int(tailtemp/self.THIDVAL)].hp['Tail'][tailtemp%self.THIDVAL]
		if not mergers_only:
			print("--- %s seconds ---" % (time.time() - start_time), 'walked tree')

	def getMergerProperties(self, halo, snapshot, dataset='M200'):
		"""Redundant
		"""
		H_i = self.mergers[snapshot][halo]['HaloIndex']
		snap = self.mergers[snapshot][halo]['snapshot']
		datanew = np.zeros(len(snap))

		for i in range(len(snap)):
			datanew[i] = self.halotree[snap[i]].hp[dataset][H_i[i]]

		return snap, datanew

	def overwriteSatelliteProperties(self):
		"""Redundant
		"""
		needschange = ['M200', 'R200', 'lambda', 'lambdaDM', 'lambdaH']
		self.readData(datasets = needschange)
		self.readData(datasets = ['DMFraction', 'Radius', 'DMFraction_profile', 'R200'])
		self.treemaker(snapshot = self.snapend, datasets=['hostHaloIndex', 'snapshot', 'HaloIndex'], full_tree=True)

		for halo in self.trees.keys():
			change = np.where(self.trees[halo]['hostHaloIndex'] != -1)[0]
			if len(change) == 0:
				continue
			for i in change[::-1]:
				stemp = self.trees[halo]['snapshot'][i]
				htemp = self.trees[halo]['HaloIndex'][i]%self.THIDVAL

				if len(self.trees[halo]['snapshot']) <= (i+1):
					continue

				sprev = self.trees[halo]['snapshot'][i+1]
				hprev = self.trees[halo]['HaloIndex'][i+1]%self.THIDVAL
				# if self.halotree[sprev].hp['hostHaloIndex'][hprev] != -1:	
				# 	continue
				for ds in needschange:
					self.halotree[stemp].hp[ds][htemp] = self.halotree[sprev].hp[ds][hprev]
				r200_i = np.abs(self.halotree[stemp].hp['RadiusFix'] - self.halotree[sprev].hp['R200'][hprev]).argmin()
				self.halotree[stemp].hp['DMFraction'][htemp] = self.halotree[stemp].hp['DMFraction_profile'][htemp][r200_i]
				if halo%self.THIDVAL == 770:
					print(halo, r200_i, self.halotree[stemp].hp['DMFraction'][htemp], self.halotree[sprev].hp['DMFraction'][hprev], stemp, sprev, htemp, hprev, self.halotree[stemp].hp['M200'][htemp], self.halotree[sprev].hp['M200'][hprev])

		for snap in self.mergers.keys():
			for halo in self.mergers[snap].keys():
				for snap2 in self.mergers[snap][halo]['Progenitors'].keys():
					progs = self.halo_key(self.mergers[snap][halo]['Progenitors'][snap2], snap2)
					if len(progs) == 0:
						continue

					for prog in progs:
						change = np.where(self.trees[prog]['hostHaloIndex'] != -1)[0]
						if len(change) == 0:
							continue
						for i in change[::-1]:
							stemp = self.trees[prog]['snapshot'][i]
							htemp = self.trees[prog]['HaloIndex'][i]%self.THIDVAL

							if len(self.trees[prog]['snapshot']) <= (i+1):
								continue

							sprev = self.trees[prog]['snapshot'][i+1]
							hprev = self.trees[prog]['HaloIndex'][i+1]%self.THIDVAL
							# if self.halotree[sprev].hp['hostHaloIndex'][hprev] != -1:	
							# 	continue
							if halo%self.THIDVAL == 633:
								print(halo, stemp, sprev, htemp, hprev, self.halotree[stemp].hp['M200'][htemp], self.halotree[sprev].hp['M200'][hprev])
							for ds in needschange:
								self.halotree[stemp].hp[ds][htemp] = self.halotree[sprev].hp[ds][hprev]
							r200_i = np.abs(self.halotree[stemp].hp['RadiusFix'] - self.halotree[stemp].hp['R200'][htemp]).argmin()
							self.halotree[stemp].hp['DMFraction'][htemp] = self.halotree[stemp].hp['DMFraction_profile'][htemp][r200_i]

	def findMassAccrTime(self, massFraction=0.5):
		"""Redundant
		"""
		z = np.zeros(len(self.trees.keys()))
		Mf = np.zeros(len(self.trees.keys()))
		bf = np.zeros(len(self.trees.keys()))
		index = np.zeros(len(self.trees.keys())).astype(int)
		j = 0
		for i in self.trees.keys():
			# if len(trees[i]['M200']) < 50:
			# 	continue
			if not isinstance(self.trees[i]['M200'], np.ndarray):
				continue
			if self.trees[i]['M200'][0] == -1:# or len(trees[i]['M200']) < 50:
				j += 1
				continue
			temp = np.where(self.trees[i]['M200'][self.trees[i]['M200']!=-1] > self.trees[i]['M200'][0]*massFraction)[0]
			if len(temp) > 0:
				if len(np.where(self.trees[i]['M200'][self.trees[i]['M200']!=-1] < self.trees[i]['M200'][0]*massFraction)[0]) == 0:
					j+=1
					continue
				z[j] = self.trees[i]['redshift'][self.trees[i]['M200']!=-1][temp[-1]]
				Mf[j] = self.trees[i]['M200'][0]
				bf[j] = self.trees[i]['DMFraction'][0]
				index[j] = i#self.trees[i]['HaloIndex'][0]
			j += 1
		return Mf, bf, z, index

	def computeFormationTime(self, massFraction=0.5):
		"""Redundant
		"""
		self.massFraction = massFraction
		Mf, bf, z, index = self.findMassAccrTime(massFraction=massFraction)
		for snap in self.halotree.keys():
			self.halotree[snap].hp['FormationTime'] = np.zeros(len(self.halotree[snap].hp['M200']))
			self.halotree[snap].hp['FormationRedshift'] = np.zeros(len(self.halotree[snap].hp['M200']))
		for i in range(len(index)):
			if Mf[i] == 0:
				continue
			halo_i, snap_i = self.key_halo(index[i])
			self.halotree[snap_i].hp['FormationTime'][halo_i] = timeDifference(0, z[i], 67.51, 0.3121, 0.6879)*s_to_yr/1.e6
			self.halotree[snap_i].hp['FormationRedshift'][halo_i] = z[i]

	def computePeakDMMassFraction(self, haloes):
		self.halotree[self.snapend].hp['DMPeakFraction'] = np.zeros(len(self.halotree[self.snapend].hp['DM_M200']))
		if haloes is None:
			haloes = range(len(self.halotree[self.snapend].hp['DM_M200']))
		for halo_i in haloes:
			halo = self.halo_key(halo_i, self.snapend)
			if self.trees[halo]['DM_M200'][0] == 0:
				continue
			self.halotree[self.snapend].hp['DMPeakFraction'][halo_i] = self.trees[halo]['DM_M200'][0]/np.max(self.trees[halo]['DM_M200'])

	def addNeighbourInfo(self, halo, snapshot, datasets=None):
		"""Redundant
		"""
		if datasets is None:
			sys.exit('No datasets entered')

		self.readData(datasets = datasets)
		d_prop = self.halotree[self.snapend].propertyDictionary(datasets=datasets)

		self.getNeighbourTrajectories(halo, snapshot)
		for nb in self.nbtree[halo].keys():
			for datatype in d_prop.keys():
				if datatype == 'None':
					continue
				elif datatype == 'dict' and len(d_prop[datatype])>0:
					for ds in d_prop[datatype]:
						self.nbtree[halo][nb][ds] = {}
				elif datatype == 'array' and len(d_prop[datatype])>0:
					for ds in d_prop[datatype]:
						self.nbtree[halo][nb][ds] = np.zeros((len(self.nbtree[halo][nb]['HaloIndex']),
							len(self.halotree[snapshot].hp[ds][self.key_halo[nb][0]])))
				elif datatype == 'scalar' and len(d_prop[datatype])>0:
					for ds in d_prop[datatype]:
						self.nbtree[halo][nb][ds] = np.zeros(len(self.nbtree[halo][nb]['HaloIndex']))

				i=0
				halonieuw, snapshotnieuw = self.key_halo(self.nbtree[halo][nb]['HaloIndex'])
				for hn, sn in zip(halonieuw, snapshotnieuw):
					#nb_nieuw = self.nbtree[halo][nb]['HaloIndex'][i]
					if datatype == 'None':
						continue
					elif datatype == 'dict' and len(d_prop[datatype]) > 0:
						for ds in d_prop[datatype]:
							self.nbtree[halo][nb][ds][i] = self.halotree[sn].hp[ds][hn]
					elif len(d_prop[datatype]) > 0:
						for ds in d_prop[datatype]:
							self.nbtree[halo][nb][ds][i] = self.halotree[sn].hp[ds][hn]
					i+=1

	def loopOverNeighbourInfo(self, halo, neighbours, snapshot, 
		datasets= ['DMFraction_bound', 'M200', 'n_part', 'redshift', 'snapshot', 'HaloIndex']):
		"""Redundant
		"""
		for nb in neighbours:
			if nb == halo:
				continue
			self.nbtree[halo][nb] = {}
			self.nbtree[halo][nb]['vrad'] = np.ones(200)*-3
			self.nbtree[halo][nb]['r'] = np.ones(200)*-3
			for ds in datasets:
				if ds in ['snapshot', 'HaloIndex']:
					self.nbtree[halo][nb][ds] = np.ones(200).astype(int)*-3
				else:	
					self.nbtree[halo][nb][ds] = np.ones(200)*-3
			
			i_sn = -1
			for sn in self.trees[halo]['Neighbour_Index'].keys():
				i_sn += 1
				if sn > snapshot:
					continue
				if nb not in self.trees.keys():
					continue
				if not isinstance(self.trees[halo]['snapshot'], np.ndarray):
					continue
				if not isinstance(self.trees[nb]['snapshot'], np.ndarray):
					continue			
				sn_in_nb = np.where(self.trees[halo]['snapshot'][i_sn] == self.trees[nb]['snapshot'])[0]
				
				if len(sn_in_nb) == 0:
					continue

				sn_in_nb = sn_in_nb[0]

				halo_index = self.trees[halo]['HaloIndex'][i_sn]%self.THIDVAL
				if not isinstance(self.trees[nb]['HaloIndex'], np.ndarray):
					continue
				i_in_halotree = self.trees[nb]['HaloIndex'][sn_in_nb]%self.THIDVAL
				i_in_neighbours = np.where(self.halotree[sn].hp['Neighbour_Index'][i_in_halotree] == halo_index)[0]
				if len(i_in_neighbours) == 0:
					continue

				i_in_neighbours = i_in_neighbours[0]
				self.nbtree[halo][nb]['vrad'][i_sn] = self.halotree[sn].hp['Neighbour_VelRad'][i_in_halotree][i_in_neighbours]
				self.nbtree[halo][nb]['r'][i_sn] = self.halotree[sn].hp['Neighbour_Distance'][i_in_halotree][i_in_neighbours]
				
				for ds in datasets:
					if ds  == 'M200':
						self.nbtree[halo][nb][ds][i_sn] = self.halotree[sn].hp['Neighbour_M200'][i_in_halotree][i_in_neighbours]
					elif ds == 'R200':
						self.nbtree[halo][nb][ds][i_sn] = self.halotree[sn].hp['Neighbour_R200'][i_in_halotree][i_in_neighbours]
					elif ds in ['redshift', 'snapshot']:
						self.nbtree[halo][nb][ds][i_sn] = self.halotree[sn].hp[ds]
					else:				
						self.nbtree[halo][nb][ds][i_sn] = self.halotree[sn].hp[ds][i_in_halotree]

			self.nbtree[halo][nb]['vrad'] =self.nbtree[halo][nb]['vrad'][self.nbtree[halo][nb]['vrad']!=-3]
			self.nbtree[halo][nb]['r'] =self.nbtree[halo][nb]['r'][self.nbtree[halo][nb]['r']!=-3]

			for ds in datasets:
				self.nbtree[halo][nb][ds] = self.nbtree[halo][nb][ds][self.nbtree[halo][nb][ds] !=-3]

	def getNeighbourTrajectories(self, halo, snapshot, neighbours=None, reupdate=False, full_tree = True, 
		datasets = ['DMFraction_bound', 'M200', 'n_part', 'redshift', 'snapshot', 'HaloIndex']):
		"""Redundant
		"""
		if (halo in self.nbtree.keys()) and (reupdate==False):
			return 0

		self.nbtree[halo] = {}

		halo_i = self.key_halo(halo)[0]

		# start_time = time.time()
		if neighbours is None:
			self.readData(datasets=['RootHead'])
			neighbours = set()
			for snap in self.trees[halo]['Neighbour_Index'].keys():
				nbtemp = self.trees[halo]['Neighbour_Index'][snap] + self.THIDVAL*snap
				for nb in nbtemp:
					if nb in self.trees.keys():
						neighbours.add(nb)
					else:
						neighbours.add(self.halotree[int(nb/self.THIDVAL)].hp['RootHead'][nb%self.THIDVAL])
			neighbours = list(neighbours)
		# print("--- %s seconds ---" % (time.time() - start_time), '1' )
		
		# start_time = time.time()
		self.loopOverNeighbourInfo(halo, neighbours, snapshot, datasets=datasets)
		# print("--- %s seconds ---" % (time.time() - start_time), '2' )
		
		if not full_tree:
			return
		
		# start_time= time.time()
		prog = self.mergers[halo]
		# print("--- %s seconds ---" % (time.time() - start_time), '4' )
		
		# start_time = time.time()
		self.loopOverNeighbourInfo(halo, prog, snapshot, datasets=datasets)
		# print("--- %s seconds ---" % (time.time() - start_time), '3' )

	def findhalodivide(self, radius=3, main_halo_frac_mass=2, min_mass_frac_sat=0.25, min_distance=3):
		"""Redundant
		"""
		self.All = np.zeros(0).astype(int)
		self.orbit1 = np.zeros(0).astype(int)
		self.orbit2 = np.zeros(0).astype(int)
		self.orbitm = np.zeros(0).astype(int)
		self.noorbit = np.zeros(0).astype(int)
		

		self.readData(datasets=['M200', 'Tail', 'Coord', 'R200', 'snapshot', 'hostHaloIndex', 'HaloIndex', 
			'DMFraction', 'redshift', 'n_part', 'Temperature', 'Radius', 'Vel', 'Mass_profile'])

		self.getVelRadNeighbourHaloes(radius = radius)

		self.neighbourDensity()

		self.treemaker(datasets = ['M200', 'DMFraction', 'redshift', 'snapshot', 'HaloIndex', 
			'R200', 'hostHaloIndex', 'n_part', 'Coord', 'Mass_profile', 'Neighbour_VelRad', 
			'Neighbour_Distance', 'Neighbour_Index', 'Neighbour_M200', 'Neighbour_R200', 
			'Neighbour_Density', 'Neighbour_Temperature'])

		for nb in self.trees.keys():
			self.trees[nb]['orbits'] = 0
			self.trees[nb]['orbits_tree'] = np.zeros(len(self.trees[nb]['snapshot']))
			self.trees[nb]['surrounding_density'] = np.zeros(len(self.trees[nb]['snapshot']))

		self.halotree[self.snapend].hp['bound_param'] = np.zeros(len(self.halotree[self.snapend].hp['M200']))
		self.halotree[self.snapend].hp['orbits'] = np.zeros(len(self.halotree[self.snapend].hp['M200']))
		self.allboundparam = np.zeros(len(self.halotree[self.snapend].hp['M200']))
		for halo in range(len(self.halotree[self.snapend].hp['M200'])):
			if self.halotree[self.snapend].hp['M200'][halo] == -1:
				continue
			self.getNeighbourTrajectories(halo, self.snapend)
			for nb in self.nbtree[halo].keys():
				if len(self.nbtree[halo][nb]['vrad']) <= 2:
					continue
				if self.nbtree[halo][nb]['n_part'][0] > self.halotree[self.snapend].hp['n_part'][halo]/main_halo_frac_mass:
					continue
				if len(np.where(self.nbtree[halo][nb]['r'] < min_distance*self.halotree[self.snapend].hp['R200'][halo])[0]) == 0:
					continue
				vradpos = np.where((self.nbtree[halo][nb]['n_part'] > min_mass_frac_sat*self.nbtree[halo][nb]['n_part'][0]))[0]#&
				#	(self.nbtree[halo][nb]['r'] < min_distance*self.halotree[self.snapend].hp['R200'][halo]))[0]
				if len(vradpos) <= 2:
					continue
				snapshotstemp = self.nbtree[halo][nb]['snapshot'][vradpos]
				vradtemp = self.nbtree[halo][nb]['vrad'][vradpos]
				orbittemp = np.zeros(len(vradtemp))
				orbitindices = np.where(vradtemp[1:]*vradtemp[:-1] < 0)[0]

				snaptot = pd.DataFrame({'A': self.trees[nb]['snapshot']})
				indicestemp = np.where(snaptot['A'].isin(snapshotstemp))[0]

				orbittemp[orbitindices] = 0.5

				self.trees[nb]['orbits_tree'][indicestemp] += orbittemp

				self.trees[nb]['orbits'] += len(np.where(vradtemp[1:]*vradtemp[:-1] < 0)[0])*0.5


		for halo in self.trees.keys():
			if self.trees[halo]['M200'][0] == -1 or len(self.trees[halo]['M200']) < 10:
				continue
			self.All = np.append(self.All, halo)

			massa = self.trees[halo]['M200']
			for i in range(1, len(massa)):
				if massa[i] == -1:
					massa[i] = massa[i-1]
			self.halotree[self.snapend].hp['bound_param'][halo] = self.halotree[self.snapend].hp['Neighbour_Density'][halo]/massa[0]
			self.allboundparam[halo]= self.halotree[self.snapend].hp['bound_param'][halo]
			self.halotree[self.snapend].hp['orbits'][halo] = self.trees[halo]['orbits']
			
			if self.halotree[self.snapend].hp['orbits'][halo] == 0:
				self.noorbit = np.append(self.noorbit, halo)
			elif self.halotree[self.snapend].hp['orbits'][halo] == 0.5:
				self.orbit1 = np.append(self.orbit1, halo)
			elif self.halotree[self.snapend].hp['orbits'][halo] == 1:
				self.orbit2 = np.append(self.orbit2, halo)
			else:
				self.orbitm = np.append(self.orbitm, halo)

	def hostHaloIndex_toRootHead(self):
		self.treemaker(datasets=['HaloIndex'], full_tree = True)
		self.readData(datasets=['hostHaloIndex', 'HaloIndex', 'RootHead'])
		for snap in self.halotree.keys():
			self.halotree[snap].hp['hostHaloIndexRoot'] = np.zeros(len(self.halotree[snap].hp['RootHead']))
			if len(self.halotree[snap].hp['RootHead']) == 0:
				continue
			ii = np.where(self.halotree[snap].hp['hostHaloIndex'] != -1)[0]
			hh = self.halo_key(self.halotree[snap].hp['hostHaloIndex'][ii], snap)
			rr = self.halotree[snap].hp['RootHead'][self.key_halo(hh)[0]]
			for i in range(len(rr)):
				if hh[i] in self.trees[rr[i]]['HaloIndex']:
					self.halotree[snap].hp['hostHaloIndexRoot'][ii[i]] = hh[i]
				else:
					for prog in self.mergers[rr[i]]:
						if self.halo_key(hh[i], snap) in self.trees[prog]['HaloIndex']:
							self.halotree[snap].hp['hostHaloIndexRoot'][ii[i]] = prog
							break
					print(hh[i], ii[i], rr[i])

	def constructSubset(self, halo, radius = 10, snapshot=200):
		start_time = time.time()
		self.readData(datasets=['M200', 'Tail', 'Coord', 'R200', 'snapshot', 'hostHaloIndex', 'HaloIndex', 
			'DMFraction', 'redshift', 'n_part', 'Temperature','lambdaDM', 'Radius', 'Vel', 'Mass_profile'])
		print("--- %s seconds ---" % (time.time() - start_time), 'read data')

		start_time = time.time()
		#self.getVelRadNeighbourHaloes(radius = radius, bighalo=False)
		self.readNeighbourData(haloes=[self.halo_key(halo, snapshot)])
		nb = self.halo_key(self.halotree[snapshot].hp['Neighbour_Index'][halo], snapshot)
		self.readNeighbourData(haloes=nb)
		print("--- %s seconds ---" % (time.time() - start_time), 'computed neighbour properties')

		allemaal = np.append(nb, self.halo_key(halo, snapshot))

		self.treemaker(haloes = allemaal, datasets = ['M200', 'DMFraction', 'redshift', 'snapshot', 'HaloIndex', 
			'R200', 'hostHaloIndex', 'n_part', 'Coord', 'Mass_profile', 'Neighbour_VelRad', 
			'Neighbour_Distance', 'Neighbour_Index', 'Neighbour_M200', 'lambdaDM',
			'Neighbour_R200', 'Vel'], 
			full_tree = True, readNBinfo=True)
		for snap in self.halotree.keys():
			self.halotree[snap].closeFile()

		start_time = time.time()
		self.getNeighbourTrajectories(self.halo_key(halo, snapshot), snapshot, radius=radius, full_tree=True)
		print("--- %s seconds ---" % (time.time() - start_time), 'computed trajectories')

	def constructWhole(self, radius = 10, snapshot = 200, neighbourTrajectories=False, full_tree=False):
		self.getVelRadNeighbourHaloes(radius=radius)
		#self.findDMmassWithinR200()
		self.readData(datasets=['Head', 'Tail', 'RootHead', 'M200', 'DMFraction', 'redshift', 'snapshot', 
			'HaloIndex', 'R200', 'hostHaloIndex', 'n_part', 'Coord', 'Neighbour_VelRad', 'DMFraction_bound', 
			'hostHaloIndexRoot', 'lambdaDM',	'Vel'])
		self.treemaker(datasets=['M200', 'DMFraction', 'redshift', 'snapshot', 'HaloIndex', 'R200', 
			'hostHaloIndex', 'n_part', 'Coord', 'Neighbour_VelRad', 'DMFraction_bound', 'hostHaloIndexRoot',
			'Neighbour_Distance', 'Neighbour_Index', 'Neighbour_M200', 'lambdaDM', 'Neighbour_R200',
			'Vel'], full_tree=full_tree, readNBinfo=False)
		#self.computePeakDMMassFraction()
		if neighbourTrajectories:
			for halo in self.trees.keys():
				self.getNeighbourTrajectories(halo, int(halo/self.THIDVAL), full_tree=full_tree)

	def neighbourtreemaker2(self, main_halo, neighbour_haloes, datasets=None, read_only_used_haloes=False, enclosedMass=True,
		calculate_eccentricities=False):
		start_time = time.time()

		mhid, sn = self.key_halo(main_halo)
		nhid, nsn = self.key_halo(neighbour_haloes)
		wel = np.where(nsn == sn)[0]

		#neighbour_haloes = neighbour_haloes[wel]

		if datasets is None:
			datasets = ['Coord', 'Vel']

		datasets.extend(['Tail',  'snapshot', 'redshift'])
		datasets = list(set(datasets))
		if read_only_used_haloes and len(wel)==len(neighbour_haloes):
			self.readData(datasets=['cNFW'], haloIDs = np.append(neighbour_haloes, main_halo))
			self.readData(datasets=datasets, haloIDs = np.append(neighbour_haloes, main_halo))
		else:
			self.readData(datasets=['cNFW'])
			self.readData(datasets=datasets)
		
		mh_tree = {}
		mh_tree['HaloIndex'] = np.ones(len(list(self.halotree.keys()))).astype(int)*-1
		mh_tree['snapshot'] = np.ones(len(list(self.halotree.keys()))).astype(int)*-1
		mh_tree['redshift'] = np.ones(len(list(self.halotree.keys())))*-1
		mh_tree['cNFW'] = np.ones(len(list(self.halotree.keys())))*-1
		hi, sn = self.key_halo(main_halo)
		mh_tree['HaloIndex'][sn] = hi
		mh_tree['snapshot'][sn] = sn
		mh_tree['redshift'][sn] = self.halotree[sn].hp['redshift'][0]
		mh_tree['cNFW'][sn] = self.halotree[sn].hp['cNFW'][hi]

		hubbleconst = np.zeros(len(list(self.halotree.keys())))

		d_prop = self.halotree[sn].propertyDictionary(datasets=datasets)

		for datatype in d_prop.keys():
			if datatype in ['None', 'dict', 'single']:
				continue
			elif datatype == 'array':
				for ds in d_prop[datatype]:
					mh_tree[ds] = np.ones((len(list(self.halotree.keys())), 3))*-1
					mh_tree[ds][sn] = self.halotree[sn].hp[ds][hi]
			elif datatype == 'scalar':
				for ds in d_prop[datatype]:
					mh_tree[ds] = np.ones(len(list(self.halotree.keys())))*-1
					mh_tree[ds][sn] = self.halotree[sn].hp[ds][hi]
		
		c = constant()
		c.change_constants(self.halotree[sn].hp['redshift'][0])
		hubbleconst[sn] = c.H

		tailtemp = self.halotree[sn].hp['Tail'][hi]
		tailtempoud = main_halo
		while((tailtempoud != tailtemp) and (tailtemp!=-1)):
			hi, sn = self.key_halo(tailtemp)
			c.change_constants(self.halotree[sn].hp['redshift'][0])
			hubbleconst[sn] = c.H		
			mh_tree['HaloIndex'][sn] = hi
			mh_tree['snapshot'][sn] = sn
			mh_tree['redshift'][sn] = self.halotree[sn].hp['redshift'][0]
			mh_tree['cNFW'][sn] = self.halotree[sn].hp['cNFW'][hi]
			for datatype in d_prop.keys():
				if datatype in ['None', 'dict', 'single']:
					continue
				else:
					for ds in d_prop[datatype]:
						mh_tree[ds][sn] = self.halotree[sn].hp[ds][hi]
			tailtempoud = tailtemp
			tailtemp = self.halotree[sn].hp['Tail'][hi]

		nbtree = {}
		nbtree['HaloIndex'] = np.ones((len(neighbour_haloes), len(list(self.halotree.keys())))).astype(int)*-1
		nbtree['TreeLength'] = np.ones(len(neighbour_haloes)).astype(int)*-1
		
		for datatype in d_prop.keys():
			if datatype in ['None', 'dict', 'single']:
				continue
			elif datatype == 'array':
				for ds in d_prop[datatype]:
					if ds not in ['Coord', 'Vel']:
						print(ds, 'is not an option yet for this structure.')
						return 0
					if ds == 'Coord':
						nbtree['X'] = np.ones((len(neighbour_haloes), len(list(self.halotree.keys()))))*-1
						nbtree['Y'] = np.ones((len(neighbour_haloes), len(list(self.halotree.keys()))))*-1
						nbtree['Z'] = np.ones((len(neighbour_haloes), len(list(self.halotree.keys()))))*-1
					if ds == 'Vel':
						nbtree['VX'] = np.ones((len(neighbour_haloes), len(list(self.halotree.keys()))))*-1
						nbtree['VY'] = np.ones((len(neighbour_haloes), len(list(self.halotree.keys()))))*-1
						nbtree['VZ'] = np.ones((len(neighbour_haloes), len(list(self.halotree.keys()))))*-1						
			elif datatype == 'scalar':
				for ds in d_prop[datatype]:
					nbtree[ds] = np.ones((len(neighbour_haloes), len(list(self.halotree.keys()))))*-1
					if ds in ['npart', 'hostHaloID', 'hostHaloIndex']:
						nbtree[ds] = nbtree[ds].astype(int)
		if 'Coord' in datasets:
			nbtree['Distance'] = np.ones((len(neighbour_haloes), len(list(self.halotree.keys()))))*-1
			nbtree['Phi'] = np.ones((len(neighbour_haloes), len(list(self.halotree.keys()))))*-1
			nbtree['Theta'] = np.ones((len(neighbour_haloes), len(list(self.halotree.keys()))))*-1
		if ('Vel' in datasets) and ('Coord' in datasets):
			nbtree['VelRad'] = np.zeros((len(neighbour_haloes), len(list(self.halotree.keys()))))
			nbtree['SpecificAngularMomentum'] = np.ones((len(neighbour_haloes), len(list(self.halotree.keys()))))*-1
			nbtree['SpecificEorb'] = np.zeros((len(neighbour_haloes), len(list(self.halotree.keys()))))
			if calculate_eccentricities:
				nbtree['Epsilon'] = np.ones((len(neighbour_haloes), len(list(self.halotree.keys()))))*-1
				nbtree['Peri'] = np.zeros((len(neighbour_haloes), len(list(self.halotree.keys()))), dtype=bool)
				nbtree['Apo'] = np.zeros((len(neighbour_haloes), len(list(self.halotree.keys()))), dtype=bool)
			if ('M200' in datasets):
				nbtree['AngularMomentum'] = np.ones((len(neighbour_haloes), len(list(self.halotree.keys()))))*-1
				nbtree['Eta'] = np.ones((len(neighbour_haloes), len(list(self.halotree.keys()))))*-1
				nbtree['ReducedMass'] = np.ones((len(neighbour_haloes), len(list(self.halotree.keys()))))*-1
				nbtree['rcirc'] = np.ones((len(neighbour_haloes), len(list(self.halotree.keys()))))*-1
				nbtree['Vcirc'] = np.zeros((len(neighbour_haloes), len(list(self.halotree.keys()))))
				nbtree['Eorb'] = np.zeros((len(neighbour_haloes), len(list(self.halotree.keys()))))
		
		for i in range(len(neighbour_haloes)):
			nb = neighbour_haloes[i]
			halo_index, snapshot = self.key_halo(nb)

			nbtree['HaloIndex'][i, snapshot] = halo_index
			for datatype in d_prop.keys():
				if datatype in ['None', 'dict', 'single']:
					continue
				elif datatype == 'array':
					for ds in d_prop[datatype]:
						if ds == 'Coord':
							nbtree['X'][i, snapshot] = self.halotree[snapshot].hp[ds][halo_index, 0]
							nbtree['Y'][i, snapshot] = self.halotree[snapshot].hp[ds][halo_index, 1]
							nbtree['Z'][i, snapshot] = self.halotree[snapshot].hp[ds][halo_index, 2]
						elif ds == 'Vel':
							nbtree['VX'][i, snapshot] = self.halotree[snapshot].hp[ds][halo_index, 0]
							nbtree['VY'][i, snapshot] = self.halotree[snapshot].hp[ds][halo_index, 1]
							nbtree['VZ'][i, snapshot] = self.halotree[snapshot].hp[ds][halo_index, 2]							
				elif datatype == 'scalar':
					for ds in d_prop[datatype]:
						nbtree[ds][i, snapshot] = self.halotree[snapshot].hp[ds][halo_index]

			tailtemp = self.halotree[snapshot].hp['Tail'][halo_index]
			tailtempoud = nb
			while((tailtempoud != tailtemp) and (tailtemp!=-1)):
				hi, sn = self.key_halo(tailtemp)
				nbtree['HaloIndex'][i, sn] = hi
				for datatype in d_prop.keys():
					if datatype in ['None', 'dict', 'single']:
						continue
					else:
						for ds in d_prop[datatype]:
							if ds == 'Coord':
								nbtree['X'][i, sn] = self.halotree[sn].hp[ds][hi, 0]
								nbtree['Y'][i, sn] = self.halotree[sn].hp[ds][hi, 1]
								nbtree['Z'][i, sn] = self.halotree[sn].hp[ds][hi, 2]
							elif ds == 'Vel':
								nbtree['VX'][i, sn] = self.halotree[sn].hp[ds][hi, 0]
								nbtree['VY'][i, sn] = self.halotree[sn].hp[ds][hi, 1]
								nbtree['VZ'][i, sn] = self.halotree[sn].hp[ds][hi, 2]								
							else:
								nbtree[ds][i, sn] = self.halotree[sn].hp[ds][hi]

				tailtempoud = tailtemp
				tailtemp = self.halotree[sn].hp['Tail'][hi]
			nbtree['TreeLength'][i] = len(np.where(nbtree['HaloIndex'][i, :]>0)[0])
			if 'Coord' in datasets:
				#What indices do they have in common?
				waar = np.where((nbtree['HaloIndex'][i]!=-1)&(mh_tree['HaloIndex']!=-1))[0]

				coord = np.zeros_like(mh_tree['Coord'])
				coord[:, 0] = nbtree['X'][i]
				coord[:, 1] = nbtree['Y'][i]
				coord[:, 2] = nbtree['Z'][i]
				#Relative coordinates
				coords = coord[waar] - mh_tree['Coord'][waar]

				coords = np.where(np.abs(coords) > 0.5*self.boxsize, coords - coords/np.abs(coords)*self.boxsize, coords)
				nbtree['Distance'][i, waar] = np.sqrt(np.sum(coords*coords, axis=1))

				#Physical distance
				coords = coords/h/(1. + mh_tree['redshift'][waar, None])
				dist = np.sqrt(np.sum(coords*coords, axis=1))

				#Angles
				xsigns = np.sign(coords[:, 0])
				ysigns = np.sign(coords[:, 1])
				nbtree['Phi'][i, waar] = np.arctan(np.abs(coords[:, 1]/coords[:, 0]))
				nbtree['Phi'][i, waar] = np.where((xsigns<0)&(ysigns>0), np.pi - nbtree['Phi'][i, waar], nbtree['Phi'][i, waar]) 
				nbtree['Phi'][i, waar[(xsigns>0)&(ysigns<0)]] *= -1
				nbtree['Phi'][i, waar[(xsigns<0)&(ysigns<0)]] -= np.pi
				nbtree['Theta'][i, waar] = np.arccos(coords[:, 2]/dist)

			if ('Vel' in datasets) and ('Coord' in datasets):
				#Physical relative velocity
				vel = np.zeros_like(mh_tree['Coord'])
				vel[:, 0] = nbtree['VX'][i]
				vel[:, 1] = nbtree['VY'][i]
				vel[:, 2] = nbtree['VZ'][i]
				velocity = vel[waar] - mh_tree['Vel'][waar]
				velocity = velocity/(1. +  mh_tree['redshift'][waar, None])
				velocityhub = velocity + hubbleconst[waar, None]*coords

				#Radial velocity
				vel = (velocityhub[:, 0]*(coords[:, 0])*Mpc_to_km + velocityhub[:, 1]*(coords[:, 1])*Mpc_to_km  + 
					velocityhub[:, 2]*(coords[:, 2])*Mpc_to_km)
				nbtree['VelRad'][i, waar] = vel/(dist*Mpc_to_km)

				Jx = (coords[:, 1]*velocityhub[:, 2] - coords[:, 2]*velocityhub[:, 1])
				Jy = (coords[:, 2]*velocityhub[:, 0] - coords[:, 0]*velocityhub[:, 2])
				Jz = (coords[:, 0]*velocityhub[:, 1] - coords[:, 1]*velocityhub[:, 0])
				nbtree['SpecificAngularMomentum'][i, waar] = np.sqrt(Jx*Jx+Jy*Jy+Jz*Jz)
				#Total Energy Msun*(km/s)^2
				Mvir = mh_tree['M200'][waar]*1.e10/h
				Rvir = mh_tree['R200'][waar]/h
				cNFW = mh_tree['cNFW'][waar]
				r_s = Rvir/cNFW
				rho_0 = Mvir/(4*np.pi*r_s**3*(np.log(1 + cNFW) - cNFW/(1 + cNFW)))
				nbtree['SpecificEorb'][i, waar] = (0.5 * np.sum(velocity**2, axis=1) - 
					4. * np.pi * G_Mpc_km2_Msi_si2 * rho_0 * r_s * r_s * np.log(1. + dist/r_s)/(dist/r_s))

				#Peri- and apocentres
				if calculate_eccentricities:
					if len(waar) > 1:
						veltemp = nbtree['VelRad'][i, waar]
						veltemp = np.sign(veltemp)
						crossings = np.where(np.diff(veltemp))[0]
						apo = waar[crossings[veltemp[crossings] == 1]]
						peri = waar[crossings[veltemp[crossings] == -1]]
						if len(apo) > 0 and len(peri) > 0:
							if apo[0] < peri[0]:
								apo = apo[1:]
						nbtree['Apo'][i, peri] = True
						nbtree['Peri'][i, apo] = True
						if (len(apo) != 0) and (len(peri) != 0):
							distap = nbtree['Distance'][i, apo]
							distper = nbtree['Distance'][i, peri]
							if peri[-1] < apo[-1]:
								if len(peri) != len(apo):
									print("Deze halo heeft niet evenveel peris en apos:", nbtree['HaloIndex'][i, -1])
									print(peri, apo)
									continue
								nbtree['Epsilon'][i, apo] = (distap - distper)/(distap + distper)
								if len(peri) > 1:
									nbtree['Epsilon'][i, peri[1:]] = (distap[:-1] - distper[1:])/(distap[:-1] + distper[1:])
								nbtree['Epsilon'][i, peri[0]] = nbtree['Epsilon'][i, apo[0]]
							elif len(peri) > 1 and (len(apo) == len(peri)-1):
								nbtree['Epsilon'][i, apo] = (distap - distper[:-1])/(distap + distper[:-1])
								nbtree['Epsilon'][i, peri[1:]] = (distap - distper[1:])/(distap + distper[1:])
								nbtree['Epsilon'][i, peri[0]] = nbtree['Epsilon'][i, apo[0]]

				# if ('M200' in datasets):
				# 	#Reduced Mass
				# 	waartemp = np.where((nbtree['M200'][i, waar] != -1) & (mh_tree['M200'][waar] != -1))[0]
				# 	if len(waartemp) == 0:
				# 		print(nb)
				# 		continue
				# 	else:
				# 		velocity = velocity[waartemp]
				# 		coords = coords[waartemp]
				# 		dist = dist[waartemp]
				# 		vel = vel[waartemp]
				# 		waar = waar[waartemp]
				# 	nbtree['ReducedMass'][i, waar] = 1./h*1.e10 * (nbtree['M200'][i, waar] * mh_tree['M200'][waar])/(nbtree['M200'][i, waar] + mh_tree['M200'][waar])

				# 	#Angular Momentum Msun*Mpc*km/s
				# 	Jx = (coords[:, 1]*velocity[:, 2] - coords[:, 2]*velocity[:, 1])*nbtree['ReducedMass'][i, waar]
				# 	Jy = (coords[:, 2]*velocity[:, 0] - coords[:, 0]*velocity[:, 2])*nbtree['ReducedMass'][i, waar]
				# 	Jz = (coords[:, 0]*velocity[:, 1] - coords[:, 1]*velocity[:, 0])*nbtree['ReducedMass'][i, waar]
				# 	nbtree['AngularMomentum'][i, waar] = np.sqrt(Jx*Jx+Jy*Jy+Jz*Jz)
				# 	#Total Energy Msun*(km/s)^2
				# 	E = (0.5 * nbtree['ReducedMass'][i, waar] * np.sum(velocity**2, axis=1) -
				# 	 G_Mpc_km2_Msi_si2 * 1.e10/h * nbtree['M200'][i, waar] * 1.e10/h * mh_tree['M200'][waar] / dist)

				# 	waar = waar[E<0]
				# 	dist = dist[E<0]
				# 	E = E[E<0]

				# 	nbtree['Eorb'][i, waar] = E

				# 	#Circular Velocity (km/s)^2
				# 	Vcirc = np.sqrt(-2. * E / nbtree['ReducedMass'][i, waar])

				# 	# Mpc
				# 	if enclosedMass:
				# 		masshost = gdp.dmMassNFW(dist*Mpc_to_km*1.e5, mh_tree['M200'][waar]*1e10/h, 
				# 			cNFW = cNFW[waar], z=mh_tree['redshift'][waar])
				# 	else:
				# 		masshost = mh_tree['M200'][waar]*1.e10/h
				# 	rcirc = np.abs(G_Mpc_km2_Msi_si2 * 1.e10/h * nbtree['M200'][i, waar] * masshost / (2 * E))
				# 	nbtree['rcirc'][i, waar] = rcirc
				# 	nbtree['Vcirc'][i, waar] = Vcirc
					
				# 	#Epsilon
				# 	nbtree['Eta'][i, waar] = (nbtree['AngularMomentum'][i, waar] / 
				# 		(rcirc * nbtree['ReducedMass'][i, waar] * Vcirc))


		print("--- %s seconds ---" % (time.time() - start_time), 'walked tree')

		return mh_tree, nbtree