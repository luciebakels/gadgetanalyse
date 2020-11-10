import numpy as np
import sys
import os
import h5py
import pandas as pd
from constants import *
from scipy.spatial import cKDTree
import time
import copy


class Unbuffered(object):
	"""
	Copied from stackoverflow
	This forces HPC facilities to print when print is called
	"""
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def writelines(self, datas):
       self.stream.writelines(datas)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)


class Snapshots:
	"""
	Class made of snapshots
	"""
	def __init__(self, path, snaplist, partType=1, d_partType = None, useIDs=True, conversions=[1, 1, 1, 1], nfiles=None, nfilestart=0, debug=False,
		softeningLength = 0.002, bigFile=False, read_only_coords=False, read_only_header=False, readIDs=True, physical=False):
		self.snapshot = {}
		for i in snaplist:
			self.snapshot[i] = Snapshot(path, i, partType=partType, d_partType = d_partType, useIDs=useIDs, 
				conversions=conversions, nfiles=nfiles, nfilestart=nfilestart, debug=debug,
				softeningLength = softeningLength, bigFile=bigfile, read_only_coords=read_only_coords, 
				read_only_header=read_only_header, readIDs=readIDs, physical=physical)


class Snapshot:
	"""
	Class used to read in and analyse GADGET or SWIFT snapshots.

	This class can be used to read in complete snapshots, or just headers, IDS, coordinates, or velocities.
	Other functionality can be used for computing properties of given regions/haloes, if either 
	coordinates, radius, or a selection of particle IDs are given. Properties that can be computed are:
	Centre of mass, M200, R200, central velocity, spin parameter, virial ratio, Vmax, Rmax,
	density profile, mass profile, radial velocity profile, and the angular momemtum profile.

	Attributes
	----------

	-- unit conversion --
	lengte_to_Mpc : float
		conversion factor of snapshot length to Mpc (default is 1)
	snelheid_to_kms : float
		conversion factor of snapshot speed to kms (default is 1)
	dichtheid_to_Mpc3 : float
		conversion factor of snapshot density to 10*10Msun/Mpc3
		(default is 1)
	massa_to_10msun : float
		conversion factor of snapshot mass to 10*10Msun (default is 1)

	-- snapshot properties --
	snapshottype : str
		the type of simulation ('gadget', 'swift') (default is 'gadget')
	softeningLength : float
		softening length of the simulation (default is 0.002)
	bigFile : bool
		WARNING: NOT PROPERLY TESTED!
		a flag that can be set if the snapshot is too large 
		to load on your computer (very slow) (default is False)
	snappath : str
		the location of the directory of the snapshot	
	nsnap : int
		the number of the snapshot
	nfiles : int
		the number of snapshot files that are read in (default is None)
	nfilestart : int
		the starting snapshot file that is read in (default is 0)
	partType : int, obsolete
		a number that sets the particle types to be read in from the snapshot
		replaced by d_partType (default is 1)
	namePrefix : str array
		array with the names of particles (DM, H, S)
	readParticles : int array
		accompanying array with namePrefix, specifying the ParticleType number used in the snapshot
	partOffset : int array
		accompanying array with namePrefix, specifying the number of particles of each type

	-- snapshot header --
	npart : int array
		the number of particles of each type
	time : float
		atime of the snapshot
	redshift : float
		redshift of the snapshot
	boxsize : float
		length of the sides of the simulated box
	mass : float array
		masses of each particle type

	-- snapshot data --
	coordinates : float array
		coordinates of each particle type
	velocities : float array
		velocities of each particle type
	IDs : float array
		IDs of each particle type
	internalEnergy : float array
		internalEnergy of gas particles
	density : float array
		density of gas particles
	temperature : float array
		temperature of gas particles
	starFormationRate : float array
		starFormationRate of star particles
	stellarFormationTime : float array
		stellarFormationTime of star particles		

	-- misc -- 
	physical : bool
		a flag that specifies if the outputs are converted to 
		physical units (default is False)
	factorh : float
		constant to convert units with a factor h or 1
	factorz : float
		constant to convert units to physical (or not)
	useIDs : bool
		a flag specifying if particle IDs are used 
		to compute properties with (default is True)
	debug : bool
		a flag to print out computing times of different routines
		(default is False)
	tree : cKDTree
		KDTree of the positions of all particles that are read in
	temphalo : dict
		dictionary containing properties of a specified halo/regions
	dataopen : dict
		dictionary of the snapshot file(s)


	Functions
	---------

	-- reading and getting datasets --
	open_snap(self, snappath, nsnap, filenumber=None)
		Opening snapshot
	open_property(prop)
		Opening and reading given dataset 'prop'
	get_IDs()
		Returns the IDs dataset
	get_coordinates()
		Returns the coordinates dataset
	get_velocities()
		Returns the velocities dataset
	get_internalEnergy()
		Returns the internal energy dataset
	get_temperature()
		Returns the temperature dataset
	get_density()
		Returns the density dataset
	get_masses()
		Returns the masses dataset
	get_time()
		Returns the a-time
	get_boxsize()
		Returns the boxsize

	-- bulk property computations --
	makeCoordTree()
		Computes cKDTree of all particle positions and stores it in self.tree
	get_indices(coords = None, IDs = None, radius=None)
		Returns particle indices for within a given region or for a given list of IDs
	get_number_of_particles()
		Returns the total number of particles
	get_radius(point=np.array([0, 0, 0]), coords = np.zeros((0, 3)))
		Computes the radius of all particles relative to a given point of reference
	get_average_velocity()
		Computes average velocity of all particles
	delete_used_indices(indices)
		Deletes particles from the datasets


	-- Halo properties (temphalo)--
	get_temphalo(coord, radius, fixedRadius=np.logspace(-3, 0, 1000), 
		r200fac=1, partlim=20, satellite=False, particledata=None, mass=False, 
		initialise_profiles=True, use_existing_r200=False)
		Initialises a halo for a given position and radius, or particle IDs and
		computes properties
	get_temphalo_profiles()
		Computes density, mass, radial velocity, temperature, and stellar age profiles
	get_virial_ratio(number)
		Computes the virial ratio for a random selection of 'number' particles within temphalo
	get_specific_angular_momentum_radius(coords, radius)
		Computes the specific angular momentum profile of temphalo
	get_Vmax_Rmax()
		Computes Vmax and Rmax of temphalo
	get_spin_parameter(coords)
		Computes the Bullock spin parameter of temphalo


	get_masscenter_temphalo(particles)
		Computes the centre of mass for given particles
	get_radialvelocity(coord, IDs=None, indices=None)
		Computes radial velocities of each (given) particle for a given 
		point of reference 'coord'
	get_angular_momentum_from_coords(coords, radius)
		WARNING: NOT PROPERLY TESTED!
		Computes the angular momentum 'vector length' for all particles
		within radius 'radius' surrounding the point of reference 'coords'
	get_spherical_coords(coords, indices)
		WARNING: NOT PROPERLY TESTED!
		Computes the spherical coordinates of all particles 'indices' surrounding
		a point of reference 'coords'
	get_spherical_coord_velocity(coords, indices)
		WARNING: NOT PROPERLY TESTED!
		Computes the spherical velocities of all particles 'indices' surrounding
		a point of reference 'coords'
	get_number_density_profile(coords, IDs, radius)
		Computes the number density profile
	get_mass(coords, IDs, radius=False)
		Computes total mass of particles within a given region or for a list of IDs
	get_IDs_within_radius(coords, radius)
		Returns particle IDs within a given region
	get_indici_within_radius(coords, radius)
		Returns particle indices within a given region
	get_mass_within_radius(coords, radius)
		Computes the mass within a given radius


	"""
	def __init__(self, path, nsnap, partType=1, d_partType = None, useIDs=True, 
		conversions=[1, 1, 1, 1], nfiles=None, nfilestart=0, debug=False,
		softeningLength = 0.002, bigFile=False, read_only_coords=False, 
		read_only_header=False, readIDs=True, physical=False,
		snapshottype='GADGET'):
		"""
		Parameters
		----------
		path : str
			the location of the directory of the snapshot	
		nsnap : int
			the number of the snapshot

		partType : int, obsolete
			a number that sets the particle types to be read in from the snapshot
			replaced by d_partType (default is 1)
		d_partType : dict
			field 'namePrefix' = str array
				array with the names of particles (DM, H, S)
			field 'readParticles' = int array
				accompanying array with namePrefix, specifying the ParticleType number used in the snapshot

		useIDs : bool
			a flag specifying if particle IDs are used 
			to compute properties with (default is True)

		conversions : array
			conversions[0] = snapshot length unit to Mpc default is 1)
			conversions[1] = snapshot speed unit to kms (default is 1)
			conversions[2] = snapshot density unit to 10*10Msun/Mpc3 (default is 1)
			conversions[3] = snapshot mass unit to 10*10Msun (default is 1)

		nfiles : int
			the number of snapshot files that are read in (default is None)
		nfilestart : int
			the starting snapshot file that is read in (default is 0)

		debug : bool
			a flag to print out computing times of different routines
			(default is False)

		softeningLength : float
			softening length of the simulation (default is 0.002)
		bigFile : bool
			a flag that can be set if the snapshot is too large 
			to load on your computer (very slow) (default is False)

		read_only_coords : bool
			only coordinates will be read if this flag is set (default is False) 
		read_only_header : bool
			only the header of the snapshot will be read (default is False)
		readIDs : bool
			IDs will be read if this flag is set (default is True)

		physical : bool
			a flag that specifies if the outputs are converted to 
			physical units (default is False)

		snapshottype : str
			the type of simulation ('gadget', 'swift') (default is 'gadget')

		"""

		#Set unit conversions
		self.lengte_to_Mpc = conversions[0]
		self.snelheid_to_kms = conversions[1]
		self.dichtheid_to_Mpc3 = conversions[2]
		self.massa_to_10msun = conversions[3]
		
		#Make attributes of input parameters
		self.softeningLength = softeningLength
		self.bigFile=bigFile
		self.physical=physical
		self.snappath = path
		self.nsnap = nsnap
		self.partType = partType
		self.useIDs = useIDs
		self.debug = debug

		self.snapshottype = 'gadget'
		if snapshottype in ['SWIFT', 'Swift', 'swift']:
			self.snapshottype = 'swift'

		if nfiles is None:
			self.nfiles = 1
		else:
			self.nfiles = nfiles
		if nfilestart is None:
			self.nfilestart = 0
		else:
			self.nfilestart = nfilestart

		#Initialise cKDTree for function 'makeCoordTree()'
		self.tree = []

		#Initialise temporal halo for function 'get_temphalo()'
		self.temphalo = {}
		self.temphalo['exists'] = False

		#Reading header information
		self.npart = None
		self.dataopen = {}
		print("Opening Headers")
		for n in range(self.nfiles):
			if nfiles is None:
				filenumber = None
			else:
				filenumber = n
			self.dataopen[n] = self.open_snap(self.snappath, nsnap, filenumber=filenumber)
			Header = self.dataopen[n]['Header']
			self.time = Header.attrs['Time']
			if self.snapshottype == 'swift':
				self.boxsize = Header.attrs['BoxSize'][0]*self.lengte_to_Mpc
				#self.redshift = Header.attrs['Redshift'][0]
			else:
				self.boxsize = Header.attrs['BoxSize']*self.lengte_to_Mpc
			self.redshift = Header.attrs['Redshift']
			#print('If ', self.boxsize, ' is a list, you need to edit line 208 in snapshot.py to make it not a list')
			if self.npart is None:
				self.npart = Header.attrs['NumPart_ThisFile'][:].astype("int64")
			else:
				self.npart += Header.attrs['NumPart_ThisFile'][:].astype("int64")
			
			self.mass = Header.attrs['MassTable'][:]*self.massa_to_10msun
			print(np.sum(self.npart), n+self.nfilestart)
			if read_only_header:
				self.dataopen[n].close()

		#Set conversion factors in case units need to be converted to physical
		self.factorh = 1.
		self.factorz = 1.
		if self.physical:
			if self.snapshottype == 'gadget':
				self.factorh = h
				self.factorz = 1. + self.redshift
			elif self.snapshottype == 'swift':
				self.factorz = 1. + self.redshift
		self.boxsize = self.boxsize/(self.factorh*self.factorz)
		self.mass = self.mass/self.factorh

		if read_only_header:
			return None
		print('There are %i particles in total'%(np.sum(self.npart)))

		#Allocating arrays to store particle data
		if d_partType is None:
			if partType < 6:
				self.readParticles = np.array([partType]).astype(int)
				self.partOffset = np.array([self.npart[partType]])
				self.namePrefix = ['DM']
			elif partType == 6:
				self.readParticles = np.array([1, 5]).astype(int)
				self.partOffset = np.array([self.npart[1], self.npart[5]])
				self.namePrefix = ['DM', 'S']
			elif partType == 7:
				self.readParticles = np.array([1, 0]).astype(int)
				self.partOffset = np.array([self.npart[1], self.npart[0]])
				self.namePrefix = ['DM', 'H']
			elif partType == 8:
				self.readParticles = np.array([1, 0, 5]).astype(int)
				self.namePrefix = ['DM', 'H', 'S']
				self.partOffset = np.array([self.npart[1], self.npart[0], self.npart[5]])
			else:
				sys.exit("No valid partType given: %i" %partType)
		else:
			self.namePrefix = d_partType['particle_type']
			self.readParticles = d_partType['particle_number']
			self.partOffset = np.zeros(len(self.readParticles)).astype(int)
			for i in range(len(self.readParticles)):
				self.partOffset[i] = self.npart[self.readParticles[i]]

		self.coordinates = np.zeros((np.sum(self.npart), 3))
		if not read_only_coords:
			self.velocities = np.zeros((np.sum(self.npart), 3))
			self.masses = np.zeros(np.sum(self.npart))
			if 'H' in self.namePrefix:
				self.internalEnergy = np.zeros(np.sum(self.npart))
				self.density = np.zeros(np.sum(self.npart))
				self.temperature = np.zeros(np.sum(self.npart))
			if 'S' in self.namePrefix:
				self.starFormationRate = np.zeros(np.sum(self.npart))
				self.stellarFormationTime = np.zeros(np.sum(self.npart))

		if readIDs:
			self.IDs = np.zeros(np.sum(self.npart)).astype(int)		

		#Reading particle data
		i = 0
		startnumber = 0
		print(self.namePrefix)
		for pT in self.readParticles:
			print(pT)
			for n in self.dataopen.keys():
				print(n+self.nfilestart)
				IDs = None
				if readIDs:
					IDs = self.dataopen[n]['PartType{}/ParticleIDs'.format(int(pT))][:]

				coordinates = self.dataopen[n]['PartType{}/Coordinates'.format(int(pT))][:,:]*self.lengte_to_Mpc/(self.factorh*self.factorz)
				endnumber = startnumber + len(coordinates)
				
				if not read_only_coords:
					if self.snapshottype == 'gadget':
						velocities = self.dataopen[n]['PartType{}/Velocities'.format(int(pT))][:,:]*self.snelheid_to_kms*np.sqrt(1./(1.+self.redshift))/self.factorz
					else:
						velocities = self.dataopen[n]['PartType{}/Velocities'.format(int(pT))][:,:]*self.snelheid_to_kms/self.factorz

					if IDs is None:
						IDs = self.dataopen[n]['PartType{}/ParticleIDs'.format(int(pT))][:]

					if self.mass[int(pT)] != 0:
						masses = np.ones(len(IDs))*self.mass[int(pT)]
					else:
						masses = self.dataopen[n]['PartType{}/Masses'.format(int(pT))][:]*self.massa_to_10msun/self.factorh
					if self.namePrefix[i] == 'H':
						internalEnergy = self.dataopen[n]['PartType{}/InternalEnergy'.format(int(pT))][:]*self.massa_to_10msun*self.snelheid_to_kms**2/(self.factorh*self.factorz**2)
						density = self.dataopen[n]['PartType{}/Density'.format(int(pT))][:]*self.dichtheid_to_Mpc3/(self.factorh/(self.factorh*self.factorz)**3)
						temperature = internalEnergy*80.8
					elif self.namePrefix[i] == 'S':
						starFormationRate = self.dataopen[n]['PartType{}/StarFormationRate'.format(int(pT))][:]
						stellarFormationTime = self.dataopen[n]['PartType{}/StellarFormationTime'.format(int(pT))][:]

				self.coordinates[startnumber:endnumber, :] = coordinates
				if not read_only_coords:
					self.velocities[startnumber:endnumber, :] = velocities
					self.IDs[startnumber:endnumber] = IDs
					self.masses[startnumber:endnumber] = masses
					if self.namePrefix[i] == 'H':
						self.internalEnergy[startnumber:endnumber] = internalEnergy
						self.density[startnumber:endnumber] = density
						self.temperature[startnumber:endnumber] = temperature
					elif self.namePrefix[i] == 'S':
						self.starFormationRate[startnumber:endnumber] = FormationRate
						self.stellarFormationTime[startnumber:endnumber] = stellarFormationTime
				if readIDs and read_only_coords:
					self.IDs[startnumber:endnumber] = IDs
				startnumber = endnumber
			i += 1						

		for n in self.dataopen.keys():
			self.dataopen[n].close()

	def open_property(self, prop):
		"""Opens dataset

		Parameters
		----------
		prop: str
			name of property dataset to be read in

		Returns
		-------
		array
			an array of all particle properties
		"""
		if self.partType < 6:
			return self.dataopen['PartType{}/'.format(int(self.partType))+prop][:]
		elif self.partType == 6:
			een = self.dataopen['PartType1/'+prop][:]
			return np.append(een, self.dataopen['PartType5/'+prop][:], axis=0)
		elif self.partType == 7:
			een = self.dataopen['PartType1/'+prop][:]
			return np.append(een, self.dataopen['PartType0/'+prop][:], axis=0)			
	
	def get_IDs(self):
		"""Gets particle IDs

		Returns
		-------
		array
			an array of all particle IDs
		"""
		if self.bigFile:
			return self.open_property('ParticleIDs')
		else:
			return self.IDs

	def get_coordinates(self):
		"""Gets particle coordinates

		Returns
		-------
		array
			an array of all particle coordinates
		"""
		if self.bigFile:
			return self.open_property('Coordinates')*self.lengte_to_Mpc/(self.factorh*self.factorz)
		else:
			return self.coordinates

	def get_velocities(self):
		"""Gets particle velocities

		Returns
		-------
		array
			an array of all particle velocities
		"""
		if self.bigFile:
			if self.snapshottype == 'gadget':
				return self.open_property('Velocities')*self.snelheid_to_kms*np.sqrt(1./(1.+self.redshift))/self.factorz
			else:
				return self.open_property('Velocities')*self.snelheid_to_kms/self.factorz
		else:
			return self.velocities

	def get_internalEnergy(self):
		"""Gets gas particle internal energies

		Returns
		-------
		array
			an array of all gas particle internal energies
		"""
		if 'H' in self.namePrefix:
			return self.internalEnergy
		else:
			return []

	def get_temperature(self):
		"""Gets gas particle temperatures

		Returns
		-------
		array
			an array of all gas particle temperatures
		"""
		if 'H' in self.namePrefix:
			return self.temperature
		else:
			return []

	def get_density(self):
		"""Gets gas particle densities

		Returns
		-------
		array
			an array of all gas particle densities
		"""
		if 'H' in self.namePrefix:
			return self.density
		else:
			return []

	def get_masses(self):
		"""Gets particle masses

		Returns
		-------
		array
			an array of all particle masses
		"""
		if self.bigFile:
			i = 0
			for pT in self.readParticles:
				if self.mass[int(pT)] == 0:
					masses = np.ones(len(self.IDs))*self.mass[int(pT)]
				else:
					masses = self.dataopen['PartType{}/Masses'.format(int(pT))][:]*self.massa_to_10msun/self.factorh

				if i == 0:
					self.masses = masses
				else:
					self.masses = np.append(self.masses, masses)
				i += 1
			return self.masses
		else:
			return self.masses

	def get_masscenter_temphalo(self, particles):
		"""Computes centre of mass for a list of given particle indices

		Parameters
		----------
		particles : int array or list
			indices of particles the centre of mass is to be calculated over

		Returns
		-------
			array
				coordinates of the centre of mass
		"""
		coords = self.get_coordinates()[particles]
		mass = self.get_masses()[particles]
		comtemp = 1./np.sum(mass)*np.sum((mass*coords.T).T, axis=0)
		tree = cKDTree(coords, boxsize=self.boxsize)
		particles2 = copy.deepcopy(particles)
		comnew = copy.deepcopy(comtemp)
		comtemp *= 2
		#Iterate until coordinates are within 1/20 the softening length of the simulation
		while (np.sum(comtemp - comnew)/3. > self.softeningLength/20.):
			dist, ind = tree.query([comnew], k=int(np.min([int(len(particles2)/2), 5000])))
			particles2 = particles[ind[0]]
			coords2 = coords[ind[0]]
			mass2 = mass[ind[0]]
			comtemp = copy.deepcopy(comnew)
			comnew = 1./np.sum(mass2)*np.sum((mass2*coords2.T).T, axis=0)
		return comnew

	def delete_used_indices(self, indices):
		"""Deletes particles from working memory

		Parameters
		----------
		indices : int array or list
			particle indices of the particles that are to be removed
		"""
		if isinstance(self.tree, cKDTree):
			return 0
		self.IDs = np.delete(self.IDs, indices)
		self.coordinates = np.delete(self.coordinates, indices, axis=0)
		self.velocities = np.delete(self.velocities, indices, axis=0)
		self.masses = np.delete(self.masses, indices)
		if 'H' in self.namePrefix:
			self.internalEnergy = np.delete(self.internalEnergy, indices)
			self.temperature = np.delete(self.temperature, indices)
			self.density = np.delete(self.density, indices)
		if 'S' in self.namePrefix:
			self.starFormationRate = np.delete(self.starFormationRate, indices)
			self.stellarFormationTime = np.delete(self.stellarFormationTime, indices)

	def get_temphalo(self, coord, radius = None, fixedRadius=np.logspace(-3, 0, 60), r200fac=1, partlim=20, 
		satellite=False, particledata=None, mass=False, initialise_profiles=True, use_existing_r200=False):
		"""
		Initialises a halo

		This function selects the appropriate particles, computes basic properties, and 
		bins the particles of different types according to their distance to the centre,
		necessary for computing profiles. All information is stored in the dictionary 'temphalo'.

		Parameters
		----------
		coord : float array
			centre of the halo
		radius : float, optional
			radius of the halo, needs to be set if no cKDTree is computed or if use_existing_r200=True
			(default is None)
		fixedRadius : float array
			bins for computing profiles (default is np.logspace(-3, 0, 60))
		r200fac : float, not in use
		partlim : int
			minimum number of particles for computing properties (default is 20)
		satellite : bool
			only compute virial properties if satellite = False (default is False)
		particledata : int array or list, optional
			particle indices to use for the computation of properties
		mass : bool, not in use
		initialise_profiles : bool
			if True, will initialise the bins for making profiles (default is True)
		use_existing_r200 : bool
			if True, will not compute virial properties and set 'radius' as R200
		"""

		# if not isinstance(self.tree, cKDTree) and particledata is None:
		# 	sys.exit("Error: no KDTree present")

		#Calling masses, because it is very expensive to do this multiple times
		massa = self.get_masses()

		c = constant(redshift=self.redshift)
		c.change_constants(self.redshift)

		if self.physical:
			comoving_rhocrit200 = deltaVir*c.rhocrit_Ms_Mpci3
		elif self.snapshottype == 'gadget':
			comoving_rhocrit200 = deltaVir*c.rhocrit_Ms_Mpci3*h/(h*(1+self.redshift))**3
		else:
			comoving_rhocrit200 = deltaVir*c.rhocrit_Ms_Mpci3_com

		
		#Finding the middle of bins for profiles
		self.temphalo['BinMiddleRadius'] = fixedRadius
		self.temphalo['MaxRadIndex'] = len(fixedRadius) - 1#np.abs(fixedRadius - r200fac*radius).argmin()
		if self.debug:
			print("maximum radius: ", fixedRadius[self.temphalo['MaxRadIndex']], self.temphalo['MaxRadIndex'])
		self.temphalo['Radius'] = np.zeros(len(fixedRadius))
		self.temphalo['Radius'][0] = 0.5 * fixedRadius[0]
		for i in range(1, len(self.temphalo['BinMiddleRadius'])):
			self.temphalo['Radius'][i] = fixedRadius[i-1] + (fixedRadius[i] - fixedRadius[i-1])/2.

		#Finding indices within the give search radius
		if self.debug:
			start_time = time.time()
		if isinstance(self.tree, cKDTree):
			self.temphalo['indices'] = np.array(self.tree.query_ball_point(coord, r=self.temphalo['Radius'][-1])).astype(int)
				#r=np.min([r200fac*radius, 
				#self.temphalo['Radius'][-1]]))).astype(int)
			if particledata is not None:
				snapID = pd.DataFrame({'A': self.get_IDs()[self.temphalo['indices']]})
				self.temphalo['indices'] = self.temphalo['indices'][np.where(snapID['A'].isin(particledata))[0]]
		else:
			self.temphalo['indices'] = np.where(self.get_radius(point=coord) < radius)[0]
		if self.debug:
			print('%i particles'%len(self.temphalo['indices']))
			print("--- %s seconds ---" % (time.time() - start_time), 'indices computed')
		if (len(self.temphalo['indices']) < partlim) or (len(self.temphalo['indices']) <= 0):
			self.temphalo['Npart'] = 0
			return 0

		#Sorting indices based on distance to the center of the halo
		if self.debug:
			start_time = time.time()
		self.temphalo['distance'] = self.get_radius(point=coord, coords=self.get_coordinates()[self.temphalo['indices']])
		sortorder = np.argsort(self.temphalo['distance']).astype(int)

		self.temphalo['indices'] = self.temphalo['indices'][sortorder]
		self.temphalo['distance'] = self.temphalo['distance'][sortorder]

		if self.temphalo['distance'][0] == 0.0:
			self.temphalo['distance'][0] = 0.001*self.temphalo['distance'][1]

		self.temphalo['Coord'] = coord
		if self.debug:
			print("--- %s seconds ---" % (time.time() - start_time), 'distances computed')
			start_time = time.time()

		#Compute virial radius and mass 
		#and inner velocity (based on particles within 10% of R200 if there are more than 1000 within R200)
		velocity = self.get_velocities()[self.temphalo['indices']]

		#If you don't want to compute R200, and just use the VR input
		if use_existing_r200:
			self.temphalo['densityprofile'] = np.cumsum(massa[self.temphalo['indices']])/(4./3.*np.pi*self.temphalo['distance']**3)*1.e10
			self.temphalo['R200'] = radius
			self.temphalo['virialradiusindex'] = np.abs(self.temphalo['distance'] - radius).argmin()
			virialradiusindex = self.temphalo['virialradiusindex']
			indicestemp = self.temphalo['indices'][:virialradiusindex]
			self.temphalo['Npart'] = len(indicestemp)
			self.temphalo['M200'] = np.sum(massa[indicestemp])
			if self.temphalo['virialradiusindex'] > 1000:
				self.temphalo['Vel'] = np.average(velocity[:int(self.temphalo['virialradiusindex']*0.10)], axis=0)
			else:
				self.temphalo['Vel'] = np.average(velocity[:self.temphalo['virialradiusindex']], axis=0)	

		#If the halo is not a satellite, and there is no given particle data, compute R200 and M200
		elif (satellite==False) and (particledata is None):
			#Compute initial density profile
			self.temphalo['densityprofile'] = np.cumsum(massa[self.temphalo['indices']])/(4./3.*np.pi*self.temphalo['distance']**3)*1.e10

			virialradiusindex = np.where(self.temphalo['densityprofile'] <= comoving_rhocrit200)[0]

			if len(virialradiusindex) == 0:
				print("Something is wrong with this halo", self.temphalo['densityprofile'][-1]/comoving_rhocrit200, 
					self.temphalo['distance'][0], len(self.temphalo['indices']))
				self.temphalo['indices'] = []
				self.temphalo['Npart'] = 0
				return 0
			virialradiusindex = virialradiusindex[0]
			print(virialradiusindex, partlim, len(self.temphalo['indices']))
			if (len(self.temphalo['indices']) < partlim) or (virialradiusindex <= 1):
				self.temphalo['indices'] = []
				self.temphalo['Npart'] = 0
				return 0
			self.temphalo['virialradiusindex'] = virialradiusindex
			self.temphalo['R200'] = self.temphalo['distance'][virialradiusindex] 
			indicestemp = self.temphalo['indices'][:virialradiusindex]
			self.temphalo['Npart'] = len(indicestemp)
			#[np.where(self.temphalo['distance'] < self.temphalo['R200'])[0]]
			self.temphalo['M200'] = np.sum(massa[indicestemp])

			if self.temphalo['virialradiusindex'] > 1000:
				self.temphalo['Vel'] = np.average(velocity[:int(self.temphalo['virialradiusindex']*0.10)], axis=0)
			else:
				self.temphalo['Vel'] = np.average(velocity[:self.temphalo['virialradiusindex']], axis=0)	

		#If the halo is a satellite, or if you use the particle data as input, R200 and M200 won't be calculated (because it wouldn't mean anything!)
		else:
			self.temphalo['R200'] = -1
			self.temphalo['M200'] = -1
			self.temphalo['Vel'] = np.average(velocity[:1000], axis=0)
			if particledata is not None:
				virialradiusindex = len(self.temphalo['indices'])-1
				self.temphalo['virialradiusindex'] = virialradiusindex
				self.temphalo['R200'] = self.temphalo['distance'][virialradiusindex]
				self.temphalo['M200'] = np.sum(massa[self.temphalo['indices']])

		self.temphalo['exists'] = True
		if self.debug:
			print("--- %s seconds ---" % (time.time() - start_time), 'virial radius and mass found')
			start_time = time.time()
		

		#Initialising the particle profiles etc. Apart from 'Npart' and 'Fraction', this is all used at a later point
		if hasattr(self, "partTypeArray") == False:
			self.partTypeArray = np.zeros(0).astype(int)
			for i in range(len(self.readParticles)):
				self.partTypeArray = np.append(self.partTypeArray, np.ones(self.partOffset[i])*self.readParticles[i])
		partTAtemp = self.partTypeArray[self.temphalo['indices']]
		if self.debug:
			print("--- %s seconds ---" % (time.time() - start_time), 'doing something')
			start_time = time.time()
		if len(self.readParticles) > 1:
			massanu = self.get_masses()[self.temphalo['indices']]
			if (satellite==False) or (particledata is not None):
				indicesM200 = self.temphalo['indices'][:virialradiusindex]
			for i in range(len(self.readParticles)):
				self.temphalo[self.namePrefix[i]+'particles'] = np.zeros(len(self.temphalo['indices']))
				self.temphalo[self.namePrefix[i]+'indices'] = np.where(partTAtemp == self.readParticles[i])[0]
				#np.where((self.temphalo['indices'] >= self.partOffset[i])&(self.temphalo['indices'] < self.partOffset[i+1]))[0]
				#self.temphalo[self.namePrefix[i]+'particles'][indicestemp] = 1

				if initialise_profiles:
					self.temphalo[self.namePrefix[i]+'indicesdict'] = {}
					self.temphalo[self.namePrefix[i]+'indicesdictCu'] = {}

				if (satellite==False) or (particledata is not None):
					self.temphalo[self.namePrefix[i]+'indicesM200'] = np.where(self.partTypeArray[indicesM200] == self.readParticles[i])[0]
					#np.where((indicesM200 >= self.partOffset[i])&(indicesM200 < self.partOffset[i+1]))[0]
					self.temphalo['Npart'+self.namePrefix[i]] = len(self.temphalo[self.namePrefix[i]+'indicesM200'])
					self.temphalo[self.namePrefix[i]+'Fraction'] = np.sum(self.get_masses()[indicesM200[self.temphalo[self.namePrefix[i]+'indicesM200']]])/np.sum(self.get_masses()[indicesM200])
				else:
					self.temphalo[self.namePrefix[i]+'Fraction'] = -1
		if self.debug:
			print("--- %s seconds ---" % (time.time() - start_time), 'indicesM200 etc')

		#Saving particles per shell that can be used by other functions:
		#	- get_angular_momentum_radius()
		#	- get_temphalo_profiles()
		if self.debug:
			start_time = time.time()
		if initialise_profiles:
			self.temphalo['indicesdictCu'] = {}
			self.temphalo['indicesdict'] = {}
			if len(self.readParticles) > 1:
				for i_pT in range(len(self.readParticles)):
					self.temphalo[self.namePrefix[i_pT]+'indicesdict'] = {}
					self.temphalo[self.namePrefix[i_pT]+'indicesdictCu'] = {}
			for i in range(len(self.temphalo['Radius'])):
				self.temphalo['indicesdictCu'][i] = np.zeros(0).astype(int)
				self.temphalo['indicesdict'][i] = np.zeros(0).astype(int)
				if len(self.readParticles) > 1:
					for i_pT in range(len(self.readParticles)):
						self.temphalo[self.namePrefix[i_pT]+'indicesdict'][i] = np.zeros(0).astype(int)
						self.temphalo[self.namePrefix[i_pT]+'indicesdictCu'][i] = np.zeros(0).astype(int)


			for i in range(0, self.temphalo['MaxRadIndex']+1):
				
				self.temphalo['indicesdictCu'][i] = np.where(self.temphalo['distance'] <= self.temphalo['BinMiddleRadius'][i])[0]
				if len(self.readParticles) > 1:
					for i_pT in range(len(self.readParticles)):
						self.temphalo[self.namePrefix[i_pT]+'indicesdictCu'][i] = (self.temphalo['indicesdictCu'][i]
							[np.where(partTAtemp[self.temphalo['indicesdictCu'][i]] == self.readParticles[i_pT])[0]])
				
				if i == 0:
					temp = np.where(self.temphalo['distance'] <= self.temphalo['Radius'][0])[0]
				else:
					temp = np.where((self.temphalo['distance'] > self.temphalo['Radius'][i-1]) & 
						(self.temphalo['distance'] <= self.temphalo['Radius'][i]))[0]

				self.temphalo['indicesdict'][i] = temp
				if len(self.readParticles) > 1:
					for i_pT in range(len(self.readParticles)):
						self.temphalo[self.namePrefix[i_pT]+'indicesdict'][i] = (self.temphalo['indicesdict'][i]
							[np.where(partTAtemp[self.temphalo['indicesdict'][i]] == self.readParticles[i_pT])[0]])
		

		if self.debug:
			print("--- %s seconds ---" % (time.time() - start_time), 'CU etc.')

	def get_number_of_particles(self):
		"""Returns total number of particles
		"""
		return len(self.get_IDs())

	def get_time(self):
		"""Returns the a-time of the snapshot
		"""
		return self.time

	def get_boxsize(self):
		"""Returns the length of a side of the simulation box
		"""
		return self.boxsize

	def get_radius(self, point=np.array([0, 0, 0]), coords = np.zeros((0, 3))):
		"""Computes the radii of given or all particles to 'point'

		Parameters
		----------
		point : np.array(3)
			the point of reference
		coords : np.array((x, 3)), optional
			list of coordinates to compute radii for, if not given, all particles are computed

		Returns
		-------
		array
			an array of radii to 'point'
		"""
		if len(coords) == 0 :
			coords = self.get_coordinates()
		coords = (coords - point)
		coords = np.where(np.abs(coords) > 0.5*self.boxsize, coords - np.sign(coords)*self.boxsize, coords)
		return np.sqrt((coords[:, 0])**2 + (coords[:, 1])**2 + (coords[:, 2])**2)

	def get_average_velocity(self):
		"""Computes average velocity of all particles
		"""
		vel = self.get_velocities()
		return np.sqrt(vel[:, 0]**2 + vel[:, 1]**2 + vel[:, 2]**2)

	def get_radialvelocity(self, coord, IDs=None, indices=None):
		"""Computes radial velocities of each (given) particle for a given 
		point of reference 'coord'

		Parameters
		----------
		coord : np.array(3)
			the point of reference
		IDs : int array, optional
			list of particle IDs
		indices : int array, optional
			list of particle indices
		if IDs and indices are not given, all particles are used

		Returns
		-------
		array
			radial velocities of all input particles
		"""
		if indices is not None:
			coords = (self.get_coordinates()[indices]-coord)*Mpc_to_km
			velocity = self.get_velocities()[indices]
			r = np.sqrt((coords[:, 0])**2 + (coords[:, 1])**2 + (coords[:, 2])**2)
		elif IDs is not None:
			self.useIDs = True
			indices = self.get_indices(IDs = IDs)
			coords = (self.get_coordinates()[indices]-coord)*Mpc_to_km
			velocity = self.get_velocities()[indices]
			r = self.get_radius(point = coord)[indices]*Mpc_to_km
		else:
			coords = (self.get_coordinates() - coord)*Mpc_to_km
			velocity = self.get_velocities()
			r = self.get_radius(point = coord)*Mpc_to_km
		vel = (velocity[:, 0]*(coords[:, 0]) + velocity[:, 1]*(coords[:, 1])  + 
			velocity[:, 2]*(coords[:, 2])) 
		vel_rad = np.zeros(len(velocity[:,0]))
		vel_rad[np.where(r > 0)] = vel[np.where(r > 0)]/r[np.where(r > 0)]
		return vel_rad

	def get_angular_momentum_from_coords(self, coords, radius):
		"""
		WARNING: NOT PROPERLY TESTED!
		Computes the angular momentum 'vector length' for all particles
		within radius 'radius' surrounding the point of reference 'coords'

		Parameters
		----------
		coord : np.array(3)
			the point of reference
		radius : float
			all particles within 'radius' are used for the computation

		Returns
		-------
		array
			angular momentum vector length		
		"""
		coord = self.get_coordinates() - coords
		rad = self.get_radius(point=coords)
		indices = np.where(rad <= radius)[0]

		xr, xteta, xphi, vr, vteta, vphi = self.get_spherical_coord_velocity(coords, indices)
		Jx = np.sum((xteta*vphi - xphi*vteta)*self.get_masses()[indices])
		Jy = np.sum((xphi*vr - xr*vphi)*self.get_masses()[indices])
		Jz = np.sum((xr*vteta - xteta*vr)*self.get_masses()[indices])
		return np.sqrt(Jx*Jx+Jy*Jy+Jz*Jz)

	def get_spherical_coords(self, coords, indices):
		"""
		WARNING: NOT PROPERLY TESTED!
		Computes the spherical coordinates of all particles 'indices' surrounding
		a point of reference 'coords'

		Parameters
		----------
		coord : np.array(3)
			the point of reference
		indices : int array or list
			particles to compute spherical coordinates for

		Returns
		-------
		float
			radial coordinate
		float
			theta coordinate
		float
			phi coordinate
		"""

		coord = self.get_coordinates()[indices, :] - coords
		xr = self.get_radius(point=coords)[indices]
		xteta = np.arctan(coord[:, 1]/coord[:, 0])
		xphi = np.arccos(coord[:, 2]/xr)
		xr[np.where(np.isnan(xr))] = 0
		xteta[np.where(np.isnan(xteta))] = 0
		xphi[np.where(np.isnan(xphi))] = 0
		return xr, xteta, xphi

	def get_spherical_coord_velocity(self, coords, indices):
		"""
		WARNING: NOT PROPERLY TESTED!
		Computes the spherical velocities of all particles 'indices' surrounding
		a point of reference 'coords'

		Parameters
		----------
		coord : np.array(3)
			the point of reference
		indices : int array or list
			particles to compute spherical coordinates for

		Returns
		-------
		float
			radial coordinate
		float
			theta coordinate
		float
			phi coordinate
		float
			radial velocity
		float
			theta velocity
		float
			phi velocity
		"""
		xr, xteta, xphi = self.get_spherical_coords(coords, indices)
		coord = self.get_coordinates()[indices, :] - coords
		vel = self.get_velocities()[indices, :]
		vr = self.get_radius(point=coords)[indices]
		coord *= Mpc_to_km
		vteta = (vel[:, 0]*coord[:, 1] - vel[:, 1]*coord[:, 0])/(coord[:, 0]**2+ coord[:, 1]**2)
		vphi = (coord[:, 2]*(coord[:, 0]*vel[:, 0] + coord[:, 1]*vel[:, 1]) -
			(coord[:, 0]**2 + coord[:, 1]**2)*vel[:, 2])/(xr*Mpc_to_km*np.sqrt(coord[:, 0]**2 + coord[:, 1]**2))
		vr[np.where(np.isnan(vr))] = 0
		vteta[np.where(np.isnan(vteta))] = 0
		vphi[np.where(np.isnan(vphi))] = 0
		return xr, xteta, xphi, vr, vteta, vphi

	def get_number_density_profile(self, coords, IDs, radius):
		"""Computes the number density profile

		Parameters
		----------
		coord : np.array(3)
			the point of reference
		IDs : int array or list
			particles to compute density profile over
		radius : array
			bins to compute number density profile for

		Returns
		-------
		array
			number density profile

		"""
		if not self.temphalo['exists']:
			indices = self.get_indices(coords=coords, IDs = IDs, radius=radius[-1])
			radcoord = self.get_radius(point=coords)[indices]
		else:
			indices = self.temphalo['indices']
			radcoord = self.temphalo['distance']
			radius = self.temphalo['Radius']
		n = np.zeros(len(radius)-1)
		H_per_particle = self.get_masses()[indices]*1e10*Msun_g/hydrogen_g
		for i in range(len(radius)-1):
			V = 4./3*np.pi*((radius[i+1]*Mpc_to_cm)**3 - (radius[i]*Mpc_to_cm)**3)
			temp = np.where((radcoord > radius[i]) & (radcoord <= radius[i+1]))[0]
			n[i] = np.sum(H_per_particle[temp])/V
		return n
	
	def get_virial_ratio(self, number):
		"""Computes the virial ratio for a random selection of 'number' particles within temphalo

		Parameters
		----------
		number : int
			number of particles used to compute the virial ratio

		Returns
		-------
		float
			virial ratio
		"""
		#Werkt niet als er verschillende massa's in een parType zitten!!!
		#Eerst moet get_temphalo_profiles gerund worden.
		if self.temphalo['virialradiusindex'] < number:
			indices = self.temphalo['indices'][:self.temphalo['virialradiusindex']]
		else:
			indices = np.random.choice(self.temphalo['indices'], size=number, replace=False)#[np.random.randint(self.temphalo['virialradiusindex'], size=amount)]
		if self.partType == 1:
			DMmass = self.mass[1]
			#Hmass = self.mass[0]
			#fM = DMmass/Hmass
			DMindices = indices
			#Hindices = np.where(self.get_masses()[indices]==Hmass)[0]
			DMparts = len(DMindices)
			#Hparts = len(Hindices)
			if number < self.temphalo['virialradiusindex']:
				DMparttot = self.temphalo['virialradiusindex']
				#Hparttot = np.sum(self.temphalo['Hparticles'][:self.temphalo['virialradiusindex']])
				DMmass = DMparttot/DMparts * DMmass
				#Hmass = Hparttot/Hparts * Hmass

			indices = DMindices
			coords = self.get_coordinates()[DMindices]

			#mi = np.ones(len(indices))
			#mi[DMindices] = DMmass
			#if Hparts > 0:
			#	mi[Hindices] = Hmass

			U = 0

			lengte = len(indices) - 1

			#print("--- %s seconds ---" % (time.time() - start_time), 'making arrays', len(indices))
			#start_time = time.time()
			for i in range(len(indices)):
				coordstemp = np.delete(coords, i, axis=0)
				#mij = np.delete(mi, i)*mi[i]
				mij = DMmass*DMmass
				rij = self.get_radius(point = coords[i], coords = coordstemp)
				rij[rij==0] = self.softeningLength
				U += np.sum(mij/rij)

			U *= G_Mpc_km2_Msi_si2 * 0.5 * 1.e10

			indices = self.temphalo['indices'][:self.temphalo['virialradiusindex']]
			velocities = self.get_velocities()[indices]
			velocities -= self.temphalo['Vel']

			K = np.sum(0.5*self.mass[1]*np.sum(velocities*velocities, axis=1))# + np.sum(self.get_internalEnergy()[indices])*self.mass[0]

			self.temphalo['Virial_ratio'] = 2*K/U
			return self.temphalo['Virial_ratio']

		else:
			DMmass = self.mass[1]
			#Hmass = self.mass[0]
			#fM = DMmass/Hmass
			DMindices = indices[np.where(self.get_masses()[indices]==DMmass)[0]]
			#Hindices = np.where(self.get_masses()[indices]==Hmass)[0]
			DMparts = len(DMindices)
			#Hparts = len(Hindices)
			if number < self.temphalo['virialradiusindex']:
				DMparttot = np.sum(self.temphalo['DMparticles'][:self.temphalo['virialradiusindex']])
				#Hparttot = np.sum(self.temphalo['Hparticles'][:self.temphalo['virialradiusindex']])
				DMmass = DMparttot/DMparts * DMmass
				#Hmass = Hparttot/Hparts * Hmass

			indices = DMindices
			coords = self.get_coordinates()[DMindices]

			#mi = np.ones(len(indices))
			#mi[DMindices] = DMmass
			#if Hparts > 0:
			#	mi[Hindices] = Hmass

			U = 0

			lengte = len(indices) - 1

			#print("--- %s seconds ---" % (time.time() - start_time), 'making arrays', len(indices))
			#start_time = time.time()
			for i in range(len(indices)):
				coordstemp = np.delete(coords, i, axis=0)
				#mij = np.delete(mi, i)*mi[i]
				mij = DMmass*DMmass
				rij = self.get_radius(point = coords[i], coords = coordstemp)
				rij[rij==0] = self.softeningLength
				U += np.sum(mij/rij)

			U *= G_Mpc_km2_Msi_si2 * 0.5 * 1.e10

			indices = self.temphalo['indices'][:self.temphalo['virialradiusindex']]
			booldm = self.temphalo['DMparticles'][:self.temphalo['virialradiusindex']].astype(bool)
			indices = indices[booldm]
			velocities = self.get_velocities()[indices]
			velocities -= self.temphalo['Vel']

			K = np.sum(0.5*self.mass[1]*np.sum(velocities*velocities, axis=1))# + np.sum(self.get_internalEnergy()[indices])*self.mass[0]

			self.temphalo['Virial_ratio'] = 2*K/U
			return self.temphalo['Virial_ratio']

	# def get_vmax(self):

	def get_temphalo_profiles(self):
		""" 
		Computes density, mass, radial velocity, temperature, and stellar age profiles
		"""
		indices = self.temphalo['indices']
		radcoord = self.temphalo['distance']
		radius = self.temphalo['Radius']
		self.temphalo['profile_density'] = np.zeros(len(radius))
		self.temphalo['profile_volume'] = np.zeros(len(radius))
		self.temphalo['profile_npart'] = np.zeros(len(radius)).astype(int)
		self.temphalo['profile_vrad'] = np.zeros(len(radius))
		#self.temphalo['profile_v'] = np.zeros(len(radius))
		#self.temphalo['profile_Hv'] = np.zeros(len(radius))
		#self.temphalo['profile_DMv'] = np.zeros(len(radius))
		self.temphalo['profile_mass'] = np.zeros(len(radius))
		if 'H' in self.namePrefix:
			self.temphalo['profile_temperature'] = np.zeros(len(radius))
		if 'S' in self.namePrefix:
			self.temphalo['profile_Sage'] = np.zeros(len(radius))

		if len(self.readParticles) > 1:
			for i_pT in range(len(self.readParticles)):
				self.temphalo['profile_'+self.namePrefix[i_pT]+'density'] = np.zeros(len(radius))
				self.temphalo['profile_'+self.namePrefix[i_pT]+'npart'] = np.zeros(len(radius)).astype(int)
				self.temphalo['profile_'+self.namePrefix[i_pT]+'vrad'] = np.zeros(len(radius))
				self.temphalo['profile_'+self.namePrefix[i_pT]+'mass'] = np.zeros(len(radius))


		c = constant()
		c.change_constants(self.redshift)
		
		coords = (self.get_coordinates()[indices] - self.temphalo['Coord'])
		coords = np.where(np.abs(coords) > 0.5*self.boxsize, coords - np.sign(coords)*self.boxsize, coords)
		if self.physical:
			coords = coords
		elif self.snapshottype == 'gadget':
			coords = coords/h/(1. + self.redshift)
		elif self.snapshottype == 'swift':
			coords = coords/(1. + self.redshift)
		dist = np.sqrt(np.sum(coords*coords, axis=1))

		velocity = self.get_velocities()[indices]
		velocity -= self.temphalo['Vel']
		if self.physical:
			velocity = velocity + c.H * coords
		elif self.snapshottype in ['gadget', 'swift']:
			velocity = velocity/(1. + self.redshift) + c.H * coords

		if 'H' in self.namePrefix:
			ie = self.get_temperature()[indices]
		if 'S' in self.namePrefix:
			sft = self.stellarFormationTime[indices]

		# self.temphalo['VelDM'] = np.average(velocity[:100][self.temphalo['DMparticles'][:100].astype(bool)], axis=0)
		# self.temphalo['VelH'] = np.average(velocity[:100][self.temphalo['Hparticles'][:100].astype(bool)], axis=0)

		
		M_per_particle = self.get_masses()[indices]*1e10
		vel = (velocity[:, 0]*(coords[:, 0])*Mpc_to_km + velocity[:, 1]*(coords[:, 1])*Mpc_to_km  + 
			velocity[:, 2]*(coords[:, 2])*Mpc_to_km) 

		vr = np.zeros(len(velocity[:,0]))
		if len(np.where(dist>0)[0]) == 0:
			return 0
		vr[np.where(dist > 0)] = vel[dist > 0]/(dist[dist > 0]*Mpc_to_km)


		for i in range(self.temphalo['MaxRadIndex']+1):
			temp = self.temphalo['indicesdict'][i]
			tempCu = self.temphalo['indicesdictCu'][i]

			if i == 0:
				self.temphalo['profile_volume'][i] = 4./3*np.pi*(radius[i])**3
			else:
				self.temphalo['profile_volume'][i] = 4./3*np.pi*((radius[i])**3 - (radius[i-1])**3)


			self.temphalo['profile_npart'][i] = len(temp)
			if len(temp) > 0:
				self.temphalo['profile_vrad'][i] = np.sum(vr[temp])/len(temp)
			masstemp = np.sum(M_per_particle[temp])
			self.temphalo['profile_density'][i] = masstemp/self.temphalo['profile_volume'][i]

			if len(tempCu) > 0:
				#self.temphalo['profile_v'][i] = np.sum(velocity[tempCu])/len(tempCu)
				self.temphalo['profile_mass'][i] = np.sum(M_per_particle[tempCu])


			if len(self.readParticles) > 1:
				for i_pT in range(len(self.readParticles)):
					tempP = self.temphalo[self.namePrefix[i_pT]+'indicesdict'][i]
					tempCuP = self.temphalo[self.namePrefix[i_pT] + 'indicesdictCu'][i]

					self.temphalo['profile_'+self.namePrefix[i_pT]+'npart'][i] = len(tempP)

					if len(tempP) > 0:
						self.temphalo['profile_'+self.namePrefix[i_pT]+'vrad'][i] = np.average(vr[tempP])
						masstemp = np.sum(M_per_particle[tempP])
						self.temphalo['profile_'+self.namePrefix[i_pT]+'density'][i] = masstemp/self.temphalo['profile_volume'][i]
						if self.namePrefix[i_pT] == 'H':
							self.temphalo['profile_temperature'][i] = np.average(ie[tempP])
						if self.namePrefix[i_pT] == 'S':
							self.temphalo['profile_Sage'][i] = np.average(sft[tempP])
					if len(tempCuP) > 0:
						self.temphalo['profile_'+self.namePrefix[i_pT]+'mass'][i] = np.sum(M_per_particle[tempCuP])

	def makeCoordTree(self):
		"""Computes cKDTree of all particle positions and stores it in self.tree
		"""
		print('Constructing cKDTree...')
		self.tree = cKDTree(self.get_coordinates(), boxsize=self.boxsize)
		print('Finished constructing cKDTree.')
	
	def get_specific_angular_momentum_radius(self, coords, radius):
		"""Computes the specific angular momentum profile of temphalo

		Parameters
		----------
		coord : np.array(3)
			the point of reference
		radius : array
			bins to compute number density profile for

		"""
		if not self.temphalo['exists']:
			sys.exit("Can only use this function if get_temphalo() is invoked")

		# Getting right value for h, to account for the Hubble flow
		c = constant()
		c.change_constants(self.redshift)
		
		indices =self.temphalo['indices']

		coords = (self.get_coordinates()[indices] - self.temphalo['Coord'])
		coords = np.where(np.abs(coords) > 0.5*self.boxsize, coords - np.sign(coords)*self.boxsize, coords)
		if self.physical:
			coords = coords
		elif self.snapshottype == 'gadget':
			coords = coords/h/(1. + self.redshift)
		elif self.snapshottype == 'swift':
			coords = coords/(1. + self.redshift)
		dist = np.sqrt(np.sum(coords*coords, axis=1))

		velocity = self.get_velocities()[indices]
		velocity -= self.temphalo['Vel']
		if self.physical:
			velocity = velocity + c.H * coords
		elif self.snapshottype in ['gadget', 'swift']:
			velocity = velocity/(1. + self.redshift) + c.H * coords

		self.temphalo['AngularMomentum'] = np.zeros_like(radius)
		JperRP = {}
		if len(self.readParticles) > 1:
			for i_pT in range(len(self.readParticles)):
				self.temphalo['AngularMomentum'+self.namePrefix[i_pT]] = np.zeros_like(radius)
				
		Jx = []
		Jy = []
		Jz = []
		JperR = np.zeros(len(radius))

		for i in range(self.temphalo['MaxRadIndex']+1):
			temp = self.temphalo['indicesdictCu'][i]
			if len(temp) == 0:
				continue
			
			coord = coords[temp]
			vel = velocity[temp]

			if len(temp) == 0:
				continue

			Jx = np.sum(coord[:, 1]*vel[:, 2] - coord[:, 2]*vel[:, 1])
			Jy = np.sum(coord[:, 2]*vel[:, 0] - coord[:, 0]*vel[:, 2])
			Jz = np.sum(coord[:, 0]*vel[:, 1] - coord[:, 1]*vel[:, 0])

			self.temphalo['AngularMomentum'][i] = np.sqrt(Jx*Jx+Jy*Jy+Jz*Jz)/len(temp)
			
			if len(self.readParticles) > 1:
				for i_pT in range(len(self.readParticles)):
					temp2 = self.temphalo[self.namePrefix[i_pT]+'indicesdictCu'][i]
					if len(temp2) == 0:
						continue
					coord2 = coord[temp2]
					vel2 = vel[temp2]
					
					Jx = np.sum(coord2[:, 1]*vel2[:, 2] - coord2[:, 2]*vel2[:, 1])
					Jy = np.sum(coord2[:, 2]*vel2[:, 0] - coord2[:, 0]*vel2[:, 2])
					Jz = np.sum(coord2[:, 0]*vel2[:, 1] - coord2[:, 1]*vel2[:, 0])
					self.temphalo['AngularMomentum'+self.namePrefix[i_pT]][i] = np.sqrt(Jx*Jx+Jy*Jy+Jz*Jz)/len(temp2)

	def get_Vmax_Rmax(self):
		"""Computes Vmax and Rmax of temphalo
		"""
		if not self.temphalo['exists']:
			sys.exit("Can only use this get_Vmax_Rmax() if get_temphalo() is invoked")
		#Get Vmax using particles
		massa = self.get_masses()
		Mtot = np.sum(massa[self.temphalo["indices"][:self.temphalo["virialradiusindex"]+1]])*1e10
		npart = len(self.temphalo["indices"])
		vmax = 0
		rmax=0
		enclpartmass = np.zeros(self.temphalo["virialradiusindex"]+1)
		for i in range(self.temphalo["virialradiusindex"]):

			enclpartmass[i] = massa[self.temphalo["indices"][i]]*1e10 + enclpartmass[i-1]
			vc = np.sqrt(G_Mpc_km2_Msi_si2*enclpartmass[i]/self.temphalo['distance'][i])

			if((vc>vmax) & (enclpartmass[i]>Mtot/np.sqrt(npart))):
				vmax=vc 
				rmax=self.temphalo['distance'][i]

		self.temphalo["Vmax_part"] = vmax
		self.temphalo["Rmax_part"] = rmax


		# sel = enclpartmass>Mtot/np.sqrt(npart) 
		# f = get_natural_cubic_spline_model(self.temphalo['distance'][:self.temphalo["virialradiusindex"]+1][sel]
		# 	,enclpartmass[sel],minval =self.temphalo['distance'][0],maxval= self.temphalo['distance'][self.temphalo["virialradiusindex"]],n_knots=20)
		# mass_prof = f.predict(self.temphalo['distance'][:self.temphalo["virialradiusindex"]+1][sel])
		# vc = np.sqrt(G_Mpc_km2_Msi_si2*mass_prof[mass_prof>0]/self.temphalo['distance'][:self.temphalo["virialradiusindex"]+1][sel][mass_prof>0])
		# maxindx = np.argmax(vc)

		self.temphalo["Vmax_interp"] = 0#vc[maxindx]
		self.temphalo["Rmax_interp"] = 0#self.temphalo['distance'][:self.temphalo["virialradiusindex"]+1][sel][maxindx]

	def get_spin_parameter(self):
		"""Computes the Bullock spin parameter of temphalo
		"""
		if not self.temphalo['exists']:
			sys.exit("Can only use this get_spin_parameter() if get_temphalo() is invoked")

		indices =self.temphalo['indices'][:self.temphalo['virialradiusindex']]
		rad = self.temphalo['distance'][:self.temphalo['virialradiusindex']]
		R = self.temphalo['R200']
		M = self.temphalo['M200']

		# Getting right value for h, to account for the Hubble flow
		c = constant(redshift=self.redshift)
		c.change_constants(self.redshift)

		if M == 0:
			return 0.0

		self.temphalo['lambda'] = np.zeros_like(R)
		vel0 = self.temphalo['Vel']

		if len(self.readParticles) > 1:
			for i_pT in range(len(self.readParticles)):
				self.temphalo['lambda'+self.namePrefix[i_pT]] = np.zeros_like(R)

		
		coords = (self.get_coordinates()[indices] - self.temphalo['Coord'])
		coords = np.where(np.abs(coords) > 0.5*self.boxsize, coords - np.sign(coords)*self.boxsize, coords)
		if self.physical:
			coords = coords
		elif self.snapshottype == 'gadget':
			coords = coords/h/(1. + self.redshift)
		elif self.snapshottype == 'swift':
			coords = coords/(1. + self.redshift)
		dist = np.sqrt(np.sum(coords*coords, axis=1))

		velocity = self.get_velocities()[indices]
		velocity -= self.temphalo['Vel']
		if self.physical:
			velocity = velocity + c.H * coords
		elif self.snapshottype in ['gadget', 'swift']:
			velocity = velocity/(1. + self.redshift) + c.H * coords
			

		massa = self.get_masses()[indices]
		Jx = np.sum((coords[:, 1]*velocity[:, 2] - coords[:, 2]*velocity[:, 1])*massa)
		Jy = np.sum((coords[:, 2]*velocity[:, 0] - coords[:, 0]*velocity[:, 2])*massa)
		Jz = np.sum((coords[:, 0]*velocity[:, 1] - coords[:, 1]*velocity[:, 0])*massa)
		J = np.sqrt(Jx*Jx+Jy*Jy+Jz*Jz)

		V = np.sqrt(G_Mpc_km2_Msi_si2*M*1.e10/R)

		self.temphalo['lambda'] = J/(1.414213562*M*V*R)
		if len(self.readParticles) > 1:
			for i_pT in range(len(self.readParticles)):
				coord2 = coords[self.temphalo[self.namePrefix[i_pT]+'indicesM200']]
				vel2 = velocity[self.temphalo[self.namePrefix[i_pT]+'indicesM200']]
				massa2 = massa[self.temphalo[self.namePrefix[i_pT]+'indicesM200']]
				Jx = np.sum((coord2[:, 1]*vel2[:, 2] - coord2[:, 2]*vel2[:, 1])*massa2)
				Jy = np.sum((coord2[:, 2]*vel2[:, 0] - coord2[:, 0]*vel2[:, 2])*massa2)
				Jz = np.sum((coord2[:, 0]*vel2[:, 1] - coord2[:, 1]*vel2[:, 0])*massa2)
				self.temphalo['lambda'+self.namePrefix[i_pT]] = np.sqrt(Jx*Jx+Jy*Jy+Jz*Jz)/(1.414213562*M*V*R)

	def get_indices(self, coords = None, IDs = None, radius=None):
		"""
		Returns particle indices for within a given region or for a given list of IDs

		Parameters
		----------
		coords : np.array(3), optional
			point of reference (default is None)
		IDs : int array or list, optional
			list of particle IDs (default is None)
		radius : float, optional
			radius around coords (default is None)

		options: 
			1) coords + radius
			2) IDs
			3) None, but temphalo is initialised
		"""
		if IDs is not None:
			snapID = pd.DataFrame({'A': self.get_IDs()})
			indices = np.where(snapID['A'].isin(IDs))[0]
			if radius is not None:
				rad = self.get_radius(point=coords)[indices]
				indices = indices[np.where(rad < radius)[0]]
			return indices

		if self.temphalo['exists'] and (((coords is None) and (radius is None)) or (IDs is None)):
			return self.temphalo['indices'] #Only within R200
			
		if ((radius is not None) and (coords is None)) or ((coords is not None) and (radius is None)):
			sys.exit('Error: no radius or coordinates specified')

		if (radius is not None) and (coords is not None):
			if not isinstance(self.tree, cKDTree):
				rad = self.get_radius(point=coords)
				return np.where(rad < radius)[0]
			else:
				return self.tree.query_ball_point(coords, r=radius)

	def get_mass(self, coords=None, IDs=None, radius=None):
		"""Computes total mass of particles within a given region or for a list of IDs

		Parameters
		----------
		coords : np.array(3), optional
			the point of reference, radius needs to be set (default is None)
		IDs : int array or list, optional
			particles to compute mass for (default is None)
		radius : float, optional
			radius to compute mass for (default is None)
		"""
		indices = self.get_indices(coords=coords, IDs=IDs, radius=radius)
		return np.sum(self.get_masses()[indices])

	def get_IDs_within_radius(self, coords, radius):
		"""Returns IDs within a given radius

		Parameters
		----------
		coords : np.array(3)
			point of reference
		radius : float
			radius to find IDs 

		Returns
		-------
		array
			returns an array of IDs
		"""
		#coord = self.get_coordinates() - coords
		rad = self.get_radius(point=coords)
		return self.get_IDs()[np.where(rad <= radius)[0]]

	def get_indici_within_radius(self, coords, radius):
		"""Returns indices within a given radius

		Parameters
		----------
		coords : np.array(3)
			point of reference
		radius : float
			radius to find indices

		Returns
		-------
		array
			returns an array of indices
		"""
		#coord = self.get_coordinates() - coords
		rad = self.get_radius(point=coords)
		return np.where(rad <= radius)[0]
	
	def get_mass_within_radius(self, coords, radius):
		"""Computes mass within a given radius

		Parameters
		----------
		coords : np.array(3)
			point of reference
		radius : float
			radius to find mass

		Returns
		-------
		float
			mass within given radius
		"""
		if self.temphalo['exists']:
			return np.sum(self.get_masses()[self.temphalo['indices']])
		else:
			#coord = self.get_coordinates() - coords
			rad = self.get_radius(point=coords)
			indices = np.where(rad <= radius)[0]
			return np.sum(self.get_masses()[indices])

	def open_snap(self, snappath, nsnap, filenumber=None):
		""" Open snapshot

		Parameters
		----------
		snappath : str
			path to snapshot
		nsnap : int
			number of snapshot
		filenumber : int, optional
			if the snapshot is divided in multiple subsnapshots,
			this should be set (default is None)
		"""
		if (filenumber is None):# or (self.nfiles==1):
			if self.snapshottype == 'gadget':
				snapnaam = "snapshot_%03d.hdf5" %nsnap
			else:
				snapnaam = "snap_%04d.hdf5" %nsnap
			if os.path.isfile(snappath+snapnaam):
				dataopen = h5py.File(snappath+snapnaam, "r")
			else:
				print("Error: Could not open %s" %(snappath), snapnaam)
				sys.exit()
		else:
			if os.path.isfile(snappath+"snapshot_%03d.%i.hdf5" %(nsnap,filenumber+self.nfilestart)):
				dataopen = h5py.File(snappath+"snapshot_%03d.%i.hdf5" %(nsnap, filenumber+self.nfilestart), "r")
			else:
				sys.exit("Error: Could not open %ssnapshot_%03d.%i.hdf5" %(snappath, nsnap, filenumber+self.nfilestart))

		return dataopen
