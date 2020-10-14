import numpy as np
import sys
import os
import h5py
import pandas as pd
from constants import *
from scipy.spatial import cKDTree
import time
import copy


class Snapshots:
	def __init__(self, path, snaplist, partType=1, useIDs=True, conversions=[1, 1, 1, 1], softeningLength = 0.002, bigFile=False, physical_units=False):
		self.snapshot = {}
		for i in snaplist:
			self.snapshot[i] = Snapshot(path, i, partType=partType, useIDs=useIDs, conversions=conversions, softeningLength=softeningLength, bigFile=bigFile, 
				physical_units=physical_units)

class Snapshot:
	def __init__(self, path, nsnap, partType=1, useIDs=True, conversions=[1, 1, 1, 1], softeningLength = 0.002, bigFile=False, physical_units=False, read_only_coords=False):
		self.lengte_to_Mpc = conversions[0]
		self.snelheid_to_kms = conversions[1]
		self.dichtheid_to_Mpc3 = conversions[2]
		self.massa_to_10msun = conversions[3]
		self.softeningLength = softeningLength
		self.bigFile=bigFile
		self.physical_units = physical_units

		self.snappath = path
		self.nsnap = nsnap
		self.partType = partType
		self.useIDs = useIDs
		self.dataopen = self.open_snap(self.snappath, nsnap)
		Header = self.dataopen['Header']
		self.time = Header.attrs['Time']
		self.boxsize = Header.attrs['BoxSize']*self.lengte_to_Mpc
		self.npart = Header.attrs['NumPart_ThisFile'][:]
		self.redshift = Header.attrs['Redshift']
		self.tree = []
		self.mass = Header.attrs['MassTable'][:]*self.massa_to_10msun

		self.temphalo = {}
		self.temphalo['exists'] = False

		if not self.bigFile:
			if partType < 6:
				if not read_only_coords:
					self.velocities = self.dataopen['PartType{}/Velocities'.format(int(partType))][:,:]*self.snelheid_to_kms*np.sqrt(1./(1.+self.redshift))
					#self.densities = dataopen['PartType{}/Density'.format(int(partType))][:]
					self.IDs = self.dataopen['PartType{}/ParticleIDs'.format(int(partType))][:]
					self.masses = np.ones(len(self.IDs))*self.mass[self.partType]
					if partType == 5:
						self.masses = self.dataopen['PartType5/Masses'][:]*self.massa_to_10msun
					if partType == 0:
						self.internalEnergy = self.dataopen['PartType0/InternalEnergy'][:]*self.massa_to_10msun*self.snelheid_to_kms**2
						self.density = self.dataopen['PartType0/Density'][:]*self.dichtheid_to_Mpc3

				self.coordinates = self.dataopen['PartType{}/Coordinates'.format(int(self.partType))][:,:]*self.lengte_to_Mpc
			elif partType == 6:
				if not read_only_coords:
					self.mass = Header.attrs['MassTable'][:]*self.massa_to_10msun
					self.velocities = self.dataopen['PartType1/Velocities'][:,:]
					self.IDs = self.dataopen['PartType1/ParticleIDs'][:]
					self.masses = self.mass[1]*np.ones(len(self.IDs))
					self.velocities = np.append(self.velocities, self.dataopen['PartType5/Velocities'][:,:], axis=0)*self.snelheid_to_kms*np.sqrt(1./(1.+self.redshift))
					self.IDs = np.append(self.IDs, self.dataopen['PartType5/ParticleIDs'][:])
					self.masses = np.append(self.masses, self.dataopen['PartType5/Masses'][:]*self.massa_to_10msun)
				self.coordinates = self.dataopen['PartType1/Coordinates'][:,:]
				self.coordinates = np.append(self.coordinates, self.dataopen['PartType5/Coordinates'][:,:], axis=0)*self.lengte_to_Mpc
			elif partType == 7:
				if not read_only_coords:
					self.mass = Header.attrs['MassTable'][:]*self.massa_to_10msun
					self.velocities = self.dataopen['PartType1/Velocities'][:,:]
					self.IDs = self.dataopen['PartType1/ParticleIDs'][:]
					self.masses = self.mass[1]*np.ones(len(self.IDs))
					self.internalEnergy = np.zeros_like(self.masses)
					self.velocities = np.append(self.velocities, self.dataopen['PartType0/Velocities'][:,:], axis=0)*self.snelheid_to_kms*np.sqrt(1./(1.+self.redshift))
					self.IDs = np.append(self.IDs, self.dataopen['PartType0/ParticleIDs'][:])
					self.masses = np.append(self.masses, self.mass[0]*np.ones(len(self.dataopen['PartType0/ParticleIDs'][:])))
					self.internalEnergy = np.append(self.internalEnergy, self.dataopen['PartType0/InternalEnergy'][:])
					self.temperature = self.internalEnergy*80.8
				self.coordinates = self.dataopen['PartType1/Coordinates'][:,:]
				self.coordinates = np.append(self.coordinates, self.dataopen['PartType0/Coordinates'][:,:], axis=0)*self.lengte_to_Mpc

			self.dataopen.close()


	def open_property(self, prop):
		if self.partType < 6:
			return self.dataopen['PartType{}/'.format(int(self.partType))+prop][:]
		elif self.partType == 6:
			een = self.dataopen['PartType1/'+prop][:]
			return np.append(een, self.dataopen['PartType5/'+prop][:], axis=0)
		elif self.partType == 7:
			een = self.dataopen['PartType1/'+prop][:]
			return np.append(een, self.dataopen['PartType0/'+prop][:], axis=0)			
	
	def get_IDs(self):
		if self.bigFile:
			return self.open_property('ParticleIDs')
		else:
			return self.IDs

	def get_coordinates(self):
		if self.bigFile:
			return self.open_property('Coordinates')*self.lengte_to_Mpc
		else:
			return self.coordinates

	def get_velocities(self):
		if self.bigFile:
			return self.open_property('Velocities')*self.snelheid_to_kms
		else:
			return self.velocities

	def get_internalEnergy(self):
		if self.partType == 0 or self.partType == 7:
			return self.internalEnergy
		# elif self.partType == 7:
		# 	iE = np.zeros(np.sum(self.npart))
		# 	iE[self.npart[1]:] = self.dataopen['PartType0/InternalEnergy'][:]
		# 	return iE
		else:
			return []

	def get_temperature(self):
		if self.partType == 0 or self.partType == 7:
			return self.temperature
		else:
			return []

	def get_density(self):
		if self.partType == 0:
			return self.density
		elif self.partType == 7:
			den = np.zeros(np.sum(self.npart))
			den[self.npart[1]:] = self.dataopen['PartType0/Density'][:]
			return den
		else:
			return []

	def get_masses(self):
		if self.bigFile:
			if self.partType < 6:
				if self.mass[self.partType] == 0:
					return self.open_property('Masses')*self.massa_to_10msun
				else:
					return np.ones(self.get_number_of_particles())*self.mass[self.partType]
			else:
				if self.partType == 6:
					if self.mass[1] == 0:
						een = self.dataopen['PartType1/Masses'][:]*self.massa_to_10msun
					else:
						een = np.ones(self.get_number_of_particles())*self.mass[1]
					if self.mass[5] == 0:
						return np.append(een, self.dataopen['PartType5/Masses'][:]*self.massa_to_10msun)
					else:
						return np.append(een, np.ones(self.get_number_of_particles)*self.mass[5])
				if self.partType == 7:
					if self.mass[1] == 0:
						een = self.dataopen['PartType1/Masses'][:]*self.massa_to_10msun
					else:
						een = np.ones(self.get_number_of_particles())*self.mass[1]
					if self.mass[0] == 0:
						return np.append(een, self.dataopen['PartType0/Masses'][:]*self.massa_to_10msun)
					else:
						return np.append(een, np.ones(self.get_number_of_particles())*self.mass[0])
		else:
			return self.masses


	def get_masscenter_temphalo(self, particles):
		coords = self.get_coordinates()[particles]
		mass = self.get_masses()[particles]
		comtemp = 1./np.sum(mass)*np.sum((mass*coords.T).T, axis=0)
		tree = cKDTree(coords, boxsize=self.boxsize)
		particles2 = copy.deepcopy(particles)
		comnew = copy.deepcopy(comtemp)
		comtemp *= 2
		while (np.sum(comtemp - comnew)/3. > self.softeningLength/20.):
			print(comnew)
			dist, ind = tree.query([comnew], k=int(np.min([int(len(particles2)/2), 5000])))
			print(np.sum(dist[0])/len(dist[0]))
			particles2 = particles[ind[0]]
			coords2 = coords[ind[0]]
			print(np.sum(np.sqrt((coords2[:, 0]-comnew[0])**2 + (coords2[:, 1]-comnew[1])**2 + (coords2[:, 2]-comnew[2])**2))/len(dist[0]))
			mass2 = mass[ind[0]]
			comtemp = copy.deepcopy(comnew)
			comnew = 1./np.sum(mass2)*np.sum((mass2*coords2.T).T, axis=0)
			print(comnew)
		return comnew

	def get_temphalo(self, coord, radius, fixedRadius=np.logspace(-3, 0, 60), r200fac=1, partlim=200, satellite=False):
		if not isinstance(self.tree, cKDTree):
			sys.exit("Error: no KDTree present")
		massa = self.get_masses()
		c = constant(redshift=self.redshift)
		c.change_constants(self.redshift)
		comoving_rhocrit200 = deltaVir*c.rhocrit_Ms_Mpci3*h/(h*(1+self.redshift))**3

		self.temphalo['BinMiddleRadius'] = fixedRadius
		self.temphalo['MaxRadIndex'] = np.abs(fixedRadius - r200fac*radius).argmin()
		self.temphalo['Radius'] = np.logspace(np.log10(fixedRadius[0]) - 
			0.5*(np.log10(fixedRadius[-1])-np.log10(fixedRadius[0]))/len(fixedRadius), 
			np.log10(fixedRadius[-1]) - 0.5*(np.log10(fixedRadius[-1])-np.log10(fixedRadius[0]))/len(fixedRadius), len(fixedRadius))

		self.temphalo['indices'] = np.array(self.tree.query_ball_point(coord, r=np.min([r200fac*radius, self.temphalo['Radius'][-1]])))
		if len(self.temphalo['indices']) < partlim:
			self.temphalo['Npart'] = 0
			return 0
		self.temphalo['distance'] = self.get_radius(point=coord, coords=self.get_coordinates()[self.temphalo['indices']])
		sortorder = np.argsort(self.temphalo['distance']).astype(int)

		self.temphalo['indices'] = self.temphalo['indices'][sortorder]
		self.temphalo['distance'] = self.temphalo['distance'][sortorder]

		if self.temphalo['distance'][0] == 0.0:
			self.temphalo['distance'][0] = 0.001*self.temphalo['distance'][1]

		self.temphalo['Coord'] = coord

		#Compute initial density profile
		self.temphalo['densityprofile'] = np.cumsum(massa[self.temphalo['indices']])/(4./3.*np.pi*self.temphalo['distance']**3)*1.e10

		#Compute virial radius and mass
		if not satellite:
			virialradiusindex = np.where(self.temphalo['densityprofile'] <= comoving_rhocrit200)[0]

			if len(virialradiusindex) == 0:
				print("Something is wrong with this halo", self.temphalo['densityprofile'][-1]/comoving_rhocrit200, 
					self.temphalo['distance'][0], len(self.temphalo['indices']))
				self.temphalo['indices'] = []
				self.temphalo['Npart'] = 0
				return 0
			virialradiusindex = virialradiusindex[0]

			if virialradiusindex < partlim:
				self.temphalo['indices'] = []
				self.temphalo['Npart'] = 0
				return 0
			self.temphalo['virialradiusindex'] = virialradiusindex
			self.temphalo['R200'] = self.temphalo['distance'][virialradiusindex] 
			indicestemp = self.temphalo['indices'][:virialradiusindex]
			self.temphalo['Npart'] = len(indicestemp)
			#[np.where(self.temphalo['distance'] < self.temphalo['R200'])[0]]
			self.temphalo['M200'] = np.sum(massa[indicestemp])
		else:
			self.temphalo['R200'] = -1
			self.temphalo['M200'] = -1

		self.temphalo['exists'] = True

		if self.partType < 6:
			self.temphalo['indicesdictCu'] = {}
			self.temphalo['indicesdict'] = {}
			for i in range(len(self.temphalo['Radius'])):
				self.temphalo['indicesdictCu'][i] = np.zeros(0).astype(int)
				self.temphalo['indicesdict'][i] = np.zeros(0).astype(int)
			for i in range(0, self.temphalo['MaxRadIndex']+1):
				temp2 = np.where(self.temphalo['distance'] <= self.temphalo['BinMiddleRadius'][i])[0]
				self.temphalo['indicesdictCu'][i] = temp2
				if i == 0:
					temp = np.where(self.temphalo['distance'] <= self.temphalo['Radius'][0])[0]
				else:
					temp = np.where((self.temphalo['distance'] > self.temphalo['Radius'][i-1]) & (self.temphalo['distance'] <= self.temphalo['Radius'][i]))[0]
				self.temphalo['indicesdict'][i] = temp

		elif self.partType == 7:

			massanu = self.get_masses()[self.temphalo['indices']]

			self.temphalo['DMparticles'] = np.zeros(len(self.temphalo['indices']))
			self.temphalo['Hparticles'] = np.zeros(len(self.temphalo['indices']))

			self.temphalo['DMindices'] = np.where(massanu == self.mass[1])[0]
			self.temphalo['Hindices'] = np.where(massanu == self.mass[0])[0]

			self.temphalo['DMparticles'][self.temphalo['DMindices']] = 1
			self.temphalo['Hparticles'][self.temphalo['Hindices']] = 1

			if not satellite:
				self.temphalo['virialradiusindex'] = virialradiusindex
				self.temphalo['DMindicesM200'] = np.where(self.get_masses()[self.temphalo['indices'][:virialradiusindex]] == self.mass[1])[0]
				self.temphalo['HindicesM200'] = np.where(self.get_masses()[self.temphalo['indices'][:virialradiusindex]] == self.mass[0])[0]	
				self.temphalo['DMFraction'] = self.mass[1]*len(self.temphalo['DMindicesM200'])/(self.mass[1]*len(self.temphalo['DMindicesM200']) +
					self.mass[0]*len(self.temphalo['HindicesM200']))
			else:
				self.temphalo['DMFraction'] = -1
			
			#Saving particles per shell that can be used by other functions:
			#	- get_angular_momentum_radius()
			#	- get_temphalo_profiles()
			self.temphalo['Hindicesdict'] = {}
			self.temphalo['DMindicesdict'] = {}
			self.temphalo['HindicesdictCu'] = {}
			self.temphalo['DMindicesdictCu'] = {}
			self.temphalo['indicesdictCu'] = {}
			self.temphalo['indicesdict'] = {}
			for i in range(len(self.temphalo['Radius'])):
				self.temphalo['Hindicesdict'][i] = np.zeros(0).astype(int)
				self.temphalo['DMindicesdict'][i] = np.zeros(0).astype(int)
				self.temphalo['HindicesdictCu'][i] = np.zeros(0).astype(int)
				self.temphalo['DMindicesdictCu'][i] = np.zeros(0).astype(int)
				self.temphalo['indicesdictCu'][i] = np.zeros(0).astype(int)
				self.temphalo['indicesdict'][i] = np.zeros(0).astype(int)
			for i in range(0, self.temphalo['MaxRadIndex']+1):
				temp2 = np.where(self.temphalo['distance'] <= self.temphalo['BinMiddleRadius'][i])[0]
				self.temphalo['indicesdictCu'][i] = temp2
				self.temphalo['HindicesdictCu'][i] = np.where(self.temphalo['Hparticles'][temp2] != 0)[0]
				self.temphalo['DMindicesdictCu'][i] = np.where(self.temphalo['DMparticles'][temp2] != 0)[0]
				if i == 0:
					temp = np.where(self.temphalo['distance'] <= self.temphalo['Radius'][0])[0]
				else:
					temp = np.where((self.temphalo['distance'] > self.temphalo['Radius'][i-1]) & (self.temphalo['distance'] <= self.temphalo['Radius'][i]))[0]
				self.temphalo['Hindicesdict'][i] = np.where(self.temphalo['Hparticles'][temp] != 0)[0]
				self.temphalo['DMindicesdict'][i] = np.where(self.temphalo['DMparticles'][temp] != 0)[0]
				self.temphalo['indicesdict'][i] = temp


	def get_number_of_particles(self):
		return len(self.get_IDs())

	def get_time(self):
		return self.time

	def get_boxsize(self):
		return self.boxsize

	def get_radius(self, point=np.array([0, 0, 0]), coords = np.zeros((0, 3))):
		if len(coords) == 0 :
			coords = self.get_coordinates()
		coords = (coords - point)
		coords = np.where(np.abs(coords) > 0.5*self.boxsize, coords - np.sign(coords)*self.boxsize, coords)
		return np.sqrt((coords[:, 0])**2 + (coords[:, 1])**2 + (coords[:, 2])**2)

	def get_average_velocity(self):
		vel = self.get_velocities()
		return np.sqrt(vel[:, 0]**2 + vel[:, 1]**2 + vel[:, 2]**2)

	def get_radialvelocity(self, coord, IDs=[], indices=[]):
		if len(indices) > 0:
			start_time = time.time()
			coords = (self.get_coordinates()[indices]-coord)*Mpc_to_km
			velocity = self.get_velocities()[indices]
			r = np.sqrt((coords[:, 0])**2 + (coords[:, 1])**2 + (coords[:, 2])**2)
		elif len(IDs) > 0:
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
		coord = self.get_coordinates() - coords
		rad = self.get_radius(point=coords)
		indices = np.where(rad <= radius)[0]

		xr, xteta, xphi, vr, vteta, vphi = self.get_spherical_coord_velocity(coords, indices)
		Jx = np.sum((xteta*vphi - xphi*vteta)*self.get_masses()[indices])
		Jy = np.sum((xphi*vr - xr*vphi)*self.get_masses()[indices])
		Jz = np.sum((xr*vteta - xteta*vr)*self.get_masses()[indices])
		return np.sqrt(Jx*Jx+Jy*Jy+Jz*Jz)*self.mass[self.partType]

	def get_spherical_coords(self, coords, indices):
		coord = self.get_coordinates()[indices, :] - coords
		xr = self.get_radius(point=coords)[indices]
		xteta = np.arctan(coord[:, 1]/coord[:, 0])
		xphi = np.arccos(coord[:, 2]/xr)
		xr[np.where(np.isnan(xr))] = 0
		xteta[np.where(np.isnan(xteta))] = 0
		xphi[np.where(np.isnan(xphi))] = 0
		return xr, xteta, xphi

	def get_spherical_coord_velocity(self, coords, indices):
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
		if not self.temphalo['exists']:
			indices = self.get_indices(coords, IDs = [], radius=radius[-1])
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
	
	def get_virial_ratio(self, amount):
		#Werkt niet als er verschillende massa's in een parType zitten!!!
		#Eerst moet get_temphalo_profiles gerund worden.
		if self.temphalo['virialradiusindex'] < amount:
			indices = self.temphalo['indices'][:self.temphalo['virialradiusindex']]
		else:
			indices = np.random.choice(self.temphalo['indices'], size=amount, replace=False)#[np.random.randint(self.temphalo['virialradiusindex'], size=amount)]
		if self.partType == 1:
			DMmass = self.mass[1]
			#Hmass = self.mass[0]
			#fM = DMmass/Hmass
			DMindices = indices
			#Hindices = np.where(self.get_masses()[indices]==Hmass)[0]
			DMparts = len(DMindices)
			#Hparts = len(Hindices)
			if amount < self.temphalo['virialradiusindex']:
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

		elif self.partType == 7:
			DMmass = self.mass[1]
			#Hmass = self.mass[0]
			#fM = DMmass/Hmass
			DMindices = indices[np.where(self.get_masses()[indices]==DMmass)[0]]
			#Hindices = np.where(self.get_masses()[indices]==Hmass)[0]
			DMparts = len(DMindices)
			#Hparts = len(Hindices)
			if amount < self.temphalo['virialradiusindex']:
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


	def get_temphalo_profiles(self):
		if self.partType == 7:
			self.get_temphalo_profiles_pt7()
			return
		indices = self.temphalo['indices']
		radcoord = self.temphalo['distance']
		radius = self.temphalo['Radius']
		self.temphalo['profile_density'] = np.zeros(len(radius))
		self.temphalo['profile_volume'] = np.zeros(len(radius))
		self.temphalo['profile_npart'] = np.zeros(len(radius))
		self.temphalo['profile_vrad'] = np.zeros(len(radius))
		#self.temphalo['profile_v'] = np.zeros(len(radius))
		self.temphalo['profile_mass'] = np.zeros(len(radius))

		coords = (self.get_coordinates()[indices]-self.temphalo['Coord'])
		coords = np.where(np.abs(coords) > 0.5*self.boxsize, coords - np.sign(coords)*self.boxsize, coords)*Mpc_to_km
		velocity = self.get_velocities()[indices]

		if self.temphalo['virialradiusindex'] > 1000:
			self.temphalo['Vel'] = np.average(velocity[:np.max([int(self.temphalo['virialradiusindex']*0.10), 1000])], axis=0)
		else:
			self.temphalo['Vel'] = np.average(velocity[:self.temphalo['virialradiusindex']], axis=0)
		velocity -= self.temphalo['Vel']

		vel = (velocity[:, 0]*(coords[:, 0]) + velocity[:, 1]*(coords[:, 1])  + 
			velocity[:, 2]*(coords[:, 2])) 
		r = radcoord*Mpc_to_km
		vr = np.zeros(len(velocity[:,0]))
		vr[np.where(r > 0)] = vel[np.where(r > 0)]/r[np.where(r > 0)]
		M_per_particle = self.get_masses()[indices]*1e10

		
		if self.partType==0:
			self.temphalo['profile_temperature'] = np.zeros(len(radius)-1)
			ie = self.get_temperature()[indices]
		for i in range(self.temphalo['MaxRadIndex'] + 1):
			temp = self.temphalo['indicesdict'][i]
			tempCu = self.temphalo['indicesdictCu'][i]
			if i == 0:
				self.temphalo['profile_volume'][i] = 4./3*np.pi*(radius[i])**3
			else:
				self.temphalo['profile_volume'][i] = 4./3*np.pi*((radius[i])**3 - (radius[i-1])**3)
			if len(temp) > 0:
				self.temphalo['profile_npart'][i] = len(temp)
				self.temphalo['profile_vrad'][i] = np.sum(vr[temp])/len(temp)
				masstemp = np.sum(M_per_particle[temp])
				if self.partType==0 or self.partType==1:
					self.temphalo['profile_density'][i] =masstemp/self.temphalo['profile_volume'][i]
				if self.partType==0:
					self.temphalo['profile_temperature'][i] = np.sum(ie[temp])/len(temp)
			if len(tempCu) > 0:
				#self.temphalo['profile_v'][i] = np.sum(velocity[tempCu])/len(tempCu)
				self.temphalo['profile_mass'][i] = np.sum(M_per_particle[tempCu])			

	def get_temphalo_profiles_pt7(self):
		indices = self.temphalo['indices']
		radcoord = self.temphalo['distance']
		radius = self.temphalo['Radius']
		self.temphalo['profile_Hdensity'] = np.zeros(len(radius))
		self.temphalo['profile_DMdensity'] = np.zeros(len(radius))
		self.temphalo['profile_density'] = np.zeros(len(radius))
		self.temphalo['profile_volume'] = np.zeros(len(radius))
		self.temphalo['profile_npart'] = np.zeros(len(radius))
		self.temphalo['profile_Hnpart'] = np.zeros(len(radius))
		self.temphalo['profile_DMnpart'] = np.zeros(len(radius))
		self.temphalo['profile_vrad'] = np.zeros(len(radius))
		self.temphalo['profile_Hvrad'] = np.zeros(len(radius))
		self.temphalo['profile_DMvrad'] = np.zeros(len(radius))
		#self.temphalo['profile_v'] = np.zeros(len(radius))
		#self.temphalo['profile_Hv'] = np.zeros(len(radius))
		#self.temphalo['profile_DMv'] = np.zeros(len(radius))
		self.temphalo['profile_mass'] = np.zeros(len(radius))
		self.temphalo['profile_Hmass'] = np.zeros(len(radius))
		self.temphalo['profile_DMmass'] = np.zeros(len(radius))
		self.temphalo['profile_temperature'] = np.zeros(len(radius))
		ie = self.get_temperature()[indices]
		coords = (self.get_coordinates()[indices] - self.temphalo['Coord'])
		coords = np.where(np.abs(coords) > 0.5*self.boxsize, coords - np.sign(coords)*self.boxsize, coords)*Mpc_to_km
		velocity = self.get_velocities()[indices]
		
		if self.temphalo['virialradiusindex'] > 1000:
			self.temphalo['Vel'] = np.average(velocity[:int(self.temphalo['virialradiusindex']*0.10)], axis=0)
		else:
			self.temphalo['Vel'] = np.average(velocity[:self.temphalo['virialradiusindex']], axis=0)

		# self.temphalo['VelDM'] = np.average(velocity[:100][self.temphalo['DMparticles'][:100].astype(bool)], axis=0)
		# self.temphalo['VelH'] = np.average(velocity[:100][self.temphalo['Hparticles'][:100].astype(bool)], axis=0)

		velocity -= self.temphalo['Vel']
		M_per_particle = self.get_masses()[indices]*1e10
		vel = (velocity[:, 0]*(coords[:, 0]) + velocity[:, 1]*(coords[:, 1])  + 
			velocity[:, 2]*(coords[:, 2])) 
		r = radcoord*Mpc_to_km
		vr = np.zeros(len(velocity[:,0]))
		vr[np.where(r > 0)] = vel[r > 0]/r[r > 0]


		for i in range(self.temphalo['MaxRadIndex']+1):
			temp = self.temphalo['indicesdict'][i]
			tempCu = self.temphalo['indicesdictCu'][i]

			if i == 0:
				self.temphalo['profile_volume'][i] = 4./3*np.pi*(radius[i])**3
			else:
				self.temphalo['profile_volume'][i] = 4./3*np.pi*((radius[i])**3 - (radius[i-1])**3)
			# Htemp = temp[np.where(self.temphalo['Hparticles'][temp] != 0)[0]]
			# DMtemp =temp[np.where(self.temphalo['DMparticles'][temp] != 0)[0]]
			# Htemp = temp[np.where(np.in1d(temp.ravel(), self.temphalo['Hindices']))[0]]
			# DMtemp = temp[np.where(np.in1d(temp.ravel(), self.temphalo['DMindices']))[0]]
			Htemp = temp[self.temphalo['Hindicesdict'][i]]
			DMtemp = temp[self.temphalo['DMindicesdict'][i]]
			HtempCu = tempCu[self.temphalo['HindicesdictCu'][i]]
			DMtempCu = tempCu[self.temphalo['DMindicesdictCu'][i]]
			if len(temp)>0:
				self.temphalo['profile_npart'][i] = len(temp)
				self.temphalo['profile_Hnpart'][i] = len(Htemp)
				self.temphalo['profile_DMnpart'][i] = len(DMtemp)
				self.temphalo['profile_vrad'][i] = np.sum(vr[temp])/len(temp)
				masstemp = np.sum(M_per_particle[temp])
				self.temphalo['profile_density'][i] = masstemp/self.temphalo['profile_volume'][i]
			if len(tempCu) > 0:
				#self.temphalo['profile_v'][i] = np.sum(velocity[tempCu])/len(tempCu)
				self.temphalo['profile_mass'][i] = np.sum(M_per_particle[tempCu])

			if len(DMtemp) > 0:
				self.temphalo['profile_DMvrad'][i] = np.sum(vr[DMtemp])/len(DMtemp)
				DMmasstemp = np.sum(M_per_particle[DMtemp])
				self.temphalo['profile_DMdensity'][i] = DMmasstemp/self.temphalo['profile_volume'][i]
			if len(DMtempCu) > 0:
				#self.temphalo['profile_DMv'][i] = np.sum(velocity[DMtempCu])/len(DMtempCu)
				self.temphalo['profile_DMmass'][i] = np.sum(M_per_particle[DMtempCu])
			if len(Htemp) > 0:
				self.temphalo['profile_Hvrad'][i] = np.sum(vr[Htemp])/len(Htemp)
				Hmastemp = np.sum(M_per_particle[Htemp])
				self.temphalo['profile_temperature'][i] = np.sum(ie[Htemp])/len(Htemp)
				self.temphalo['profile_Hdensity'][i] = Hmastemp/self.temphalo['profile_volume'][i]
			if len(HtempCu) > 0:
				#self.temphalo['profile_Hv'][i] = np.sum(velocity[HtempCu])/len(HtempCu)
				self.temphalo['profile_Hmass'][i] = np.sum(M_per_particle[HtempCu])
		

	def get_density_profile(self, coords=[], radius=[]): #Msun /Mpc^3
		self.useIDs = False

		if not self.temphalo['exists']:
			indices = self.get_indices(coords, IDs = [], radius=radius[-1])
			radcoord = self.get_radius(point=coords)[indices]
		else:
			indices = self.temphalo['indices']
			radcoord = self.temphalo['distance']
			radius = self.temphalo['Radius']
		
		n = np.zeros(len(radius)-1)
		M_per_particle = self.get_masses()[indices]*1e10
		for i in range(len(radius)-1):
			V = 4./3*np.pi*((radius[i+1])**3 - (radius[i])**3)
			temp = np.where((radcoord > radius[i]) & (radcoord <= radius[i+1]))[0]
			n[i] = np.sum(M_per_particle[temp])/V
		return n

	def get_radialvelocity_profile(self, coords=[], radius=[]): #Msun /Mpc^3
		self.useIDs = False

		indices = self.get_indices(coords, IDs = [], radius=radius[-1])
		radcoord = self.get_radius(point=coords)[indices]
		velocity = self.get_velocities()[indices]
		coord = (self.get_coordinates()[indices] - coords)
		coord = np.where(np.abs(coord) > 0.5*self.boxsize, coord - np.sign(coord)*self.boxsize, coord)
		vel = (velocity[:, 0]*(coord[:, 0]) + velocity[:, 1]*(coord[:, 1])  + 
			velocity[:, 2]*(coord[:, 2])) 
		vr = np.zeros(len(radcoord))
		vr[np.where(radcoord > 0)] = vel[np.where(radcoord > 0)]/radcoord[np.where(radcoord > 0)]/Mpc_to_km
		
		vel_rad= np.zeros(len(radius)-1)		

		for i in range(len(radius)-1):
			temp = np.where((radcoord > radius[i]) & (radcoord <= radius[i+1]))[0]
			vel_rad[i] = np.sum(vr[temp])/len(temp)

		return vel_rad

	def makeCoordTree(self):
		print('Constructing cKDTree...')
		self.tree = cKDTree(self.get_coordinates(), boxsize=self.boxsize)
		print('Finished constructing cKDTree.')

	def get_virial_radius(self, coords):
		if self.temphalo['exists']:
			return self.temphalo['R200']
		c = constant(redshift = self.redshift)
		coords = coords%self.boxsize
		if not isinstance(self.tree, cKDTree):
			self.makeCoordTree()
		massa = self.get_masses()
		def give_density(rad):
			return np.sum(massa[self.tree.query_ball_point(coords, r=rad)])/(4./3.*np.pi*rad**3)*1e10
		
		def give_density2(rad):
			return self.get_mass_within_radius(coords, rad)/(4./3.*np.pi*rad**3)*1e10

		boundsnotfound = True
		radius = 0.1
		density = give_density(radius)
		if density == 0:
			return 0
		while boundsnotfound:
			times = density/(deltaVir*c.rhocrit_Ms_Mpci3)
			if np.abs(times - 1) < self.softeningLength:
				return radius
			if times > 1:
				radius *= ((times)**(1./2.))
			else:
				boundsnotfound = False
			density = give_density(radius)
		boundsnotfound = True
		radleft = 0.8*radius
		denleft = give_density(radleft)
		while boundsnotfound:
			times = denleft/(deltaVir*c.rhocrit_Ms_Mpci3)
			if np.abs(times - 1) < self.softeningLength:
				return radleft
			if denleft == 0:
				radleft *= 1.1
			elif times < 1:
				radleft *= ((times)**(1./2.))
			else:
				boundsnotfound = False
			denleft = give_density(radleft)
		boundsnotfound = True
		radright = radius
		denright = density
		radleft = self.softeningLength*10.
		denleft = give_density(radleft)
		softeningDen = np.abs(density - give_density(radius + self.softeningLength))
		while boundsnotfound:
			#print(radright, radleft, denright, denleft, softeningDen)
			if np.abs(denright - deltaVir*c.rhocrit_Ms_Mpci3) < softeningDen:
				return radright
			if np.abs(denleft - deltaVir*c.rhocrit_Ms_Mpci3) < softeningDen:
				return radleft
			radius = radleft + (radright-radleft)/2.
			density = give_density(radius)
			softeningDen = np.abs(density - give_density(radius + self.softeningLength))
			if np.abs(density - deltaVir*c.rhocrit_Ms_Mpci3) < softeningDen:
				return radius
			if density < deltaVir*c.rhocrit_Ms_Mpci3:
				radright = radius
				denright = density
			else:
				radleft = radius
				denleft = density

	def get_angular_momentum_spherical(self, coords, IDs, radius = False, overmass = False):
		# ix = np.in1d(self.IDs.ravel(), IDs)
		# indices = np.where(ix)[0]
		# indices = indices.astype(int)
		indices = self.get_indices(coords, IDs, radius=radius)
		xr, xteta, xphi, vr, vteta, vphi = self.get_spherical_coord_velocity(coords, indices)
		if overmass:
			Jr = np.sum(xteta*vphi - xphi*vteta)
			Jteta = np.sum(xphi*vr - xr*vphi)
			Jphi = np.sum(xr*vteta - xteta*vr)
			return np.sqrt(Jx*Jx+Jy*Jy+Jz*Jz)/len(indices)
		else:
			massa = self.get_masses()
			Jr = np.sum((xteta*vphi - xphi*vteta)*massa[indices])
			Jteta = np.sum((xphi*vr - xr*vphi)*massa[indices])
			Jphi = np.sum((xr*vteta - xteta*vr)*massa[indices])
			return np.pi*np.sqrt(Jr*Jr+Jteta*Jteta+Jphi*Jphi)

	def get_angular_momentum(self, coords, IDs, radius = False, overmass = False):
		# ix = np.in1d(self.IDs.ravel(), IDs)
		# indices = np.where(ix)[0]
		# indices = indices.astype(int)
		#indices = mss.match(IDs, self.IDs)
		if not self.temphalo['exists']:
			indices = self.get_indices(coords, IDs, radius=radius)
			vel0 = np.array([0, 0, 0])
		else:
			indices = self.temphalo['indices'] #Only within R200
			vel0 = self.temphalo['Vel']

		coord = self.get_coordinates()[indices, :] - coords
		coord = np.where(np.abs(coord) > 0.5*self.boxsize, coord - np.sign(coord)*self.boxsize, coord)
		vel = self.get_velocities()[indices, :] - vel0
		if overmass:
			Jx = np.sum(coord[:, 1]*vel[:, 2] - coord[:, 2]*vel[:, 1])
			Jy = np.sum(coord[:, 2]*vel[:, 0] - coord[:, 0]*vel[:, 2])
			Jz = np.sum(coord[:, 0]*vel[:, 1] - coord[:, 1]*vel[:, 0])
			return np.sqrt(Jx*Jx+Jy*Jy+Jz*Jz)/len(indices)
		else:
			massa = self.get_masses()
			Jx = np.sum((coord[:, 1]*vel[:, 2] - coord[:, 2]*vel[:, 1])*massa[indices])
			Jy = np.sum((coord[:, 2]*vel[:, 0] - coord[:, 0]*vel[:, 2])*massa[indices])
			Jz = np.sum((coord[:, 0]*vel[:, 1] - coord[:, 1]*vel[:, 0])*massa[indices])
			return np.sqrt(Jx*Jx+Jy*Jy+Jz*Jz)

	def get_angular_momentum_radius(self, coords, IDs, radius, overmass=False):
		if not self.temphalo['exists']:
			indices = self.get_indices(coords, IDs = IDs, radius=radius[-1])
			rad = self.get_radius(point=coords)[indices]
		else:
			indices = self.temphalo['indices']
			rad = self.temphalo['distance']
			radius = self.temphalo['BinMiddleRadius']
			coord1 = self.get_coordinates()[indices] - coords
			coord1 = np.where(np.abs(coord1) > 0.5*self.boxsize, coord1 - np.sign(coord1)*self.boxsize, coord1)
			vel1 = self.get_velocities()[indices] - self.temphalo['Vel']
			self.temphalo['AngularMomentum'] = np.zeros_like(radius)
			if self.partType == 7:
				self.temphalo['AngularMomentumH'] = np.zeros_like(radius)
				self.temphalo['AngularMomentumDM'] = np.zeros_like(radius)
				JperRH = np.zeros(len(radius))
				jperRDM = np.zeros(len(radius))

		Jx = []
		Jy = []
		Jz = []
		JperR = np.zeros(len(radius))
		indicesold = []

		if not overmass:
			massa = self.get_masses()
		for i in range(self.temphalo['MaxRadIndex']+1):
			if isinstance(self.tree, cKDTree) and self.temphalo['exists']==False:
				indicesnew = self.get_indices(coords, IDs, radius=radius[i])
				coord = self.get_coordinates()[indicesnew, :] - coords
				vel = self.get_velocities()[indicesnew, :]
			else:
				temp = self.temphalo['indicesdictCu'][i]
				if len(temp) == 0:
					JperR[i] = 0.
					continue
				indicesnew = indices[temp]
				coord = coord1[temp]
				vel = vel1[temp]

			if len(indicesnew) == 0:
				JperR[i] = 0.
				continue
			#indicesnew = np.delete(indicesnew, np.where(np.in1d(indicesnew, indicesold))[0])

			if overmass:
				# Jx = np.append(Jx, coord[:, 1]*vel[:, 2] - coord[:, 2]*vel[:, 1])
				# Jy = np.append(Jy, coord[:, 2]*vel[:, 0] - coord[:, 0]*vel[:, 2])
				# Jz = np.append(Jz, coord[:, 0]*vel[:, 1] - coord[:, 1]*vel[:, 0])
				# JperR[i] = np.sqrt(np.sum(Jx)*np.sum(Jx)+np.sum(Jy)*np.sum(Jy)+np.sum(Jz)*np.sum(Jz))/len(Jx)
				Jx = np.sum(coord[:, 1]*vel[:, 2] - coord[:, 2]*vel[:, 1])
				Jy = np.sum(coord[:, 2]*vel[:, 0] - coord[:, 0]*vel[:, 2])
				Jz = np.sum(coord[:, 0]*vel[:, 1] - coord[:, 1]*vel[:, 0])
				JperR[i] = np.sqrt(Jx*Jx+Jy*Jy+Jz*Jz)/len(indicesnew)
				if self.temphalo['exists']:
					self.temphalo['AngularMomentum'][i] = JperR[i]
					if self.partType==7:
						Htemp = self.temphalo['HindicesdictCu'][i]
						DMtemp = self.temphalo['DMindicesdictCu'][i]
						if len(Htemp) > 0:
							coordH = coord[Htemp]
							velH = vel[Htemp]
							Jx = np.sum(coordH[:, 1]*velH[:, 2] - coordH[:, 2]*velH[:, 1])
							Jy = np.sum(coordH[:, 2]*velH[:, 0] - coordH[:, 0]*velH[:, 2])
							Jz = np.sum(coordH[:, 0]*velH[:, 1] - coordH[:, 1]*velH[:, 0])
							self.temphalo['AngularMomentumH'][i] = np.sqrt(Jx*Jx+Jy*Jy+Jz*Jz)/len(Htemp)
						if len(DMtemp) > 0:
							coordDM = coord[DMtemp]
							velDM = vel[DMtemp]
							Jx = np.sum(coordDM[:, 1]*velDM[:, 2] - coordDM[:, 2]*velDM[:, 1])
							Jy = np.sum(coordDM[:, 2]*velDM[:, 0] - coordDM[:, 0]*velDM[:, 2])
							Jz = np.sum(coordDM[:, 0]*velDM[:, 1] - coordDM[:, 1]*velDM[:, 0])
							self.temphalo['AngularMomentumDM'][i] = np.sqrt(Jx*Jx+Jy*Jy+Jz*Jz)/len(DMtemp)					
			else:
				Jx = np.append(Jx, (coord[:, 1]*vel[:, 2] - coord[:, 2]*vel[:, 1])*massa[indicesnew])
				Jy = np.append(Jy, (coord[:, 2]*vel[:, 0] - coord[:, 0]*vel[:, 2])*massa[indicesnew])
				Jz = np.append(Jz, (coord[:, 0]*vel[:, 1] - coord[:, 1]*vel[:, 0])*massa[indicesnew])
				JperR[i] = np.sqrt(np.sum(Jx)*np.sum(Jx) + np.sum(Jy)*np.sum(Jy) + np.sum(Jz)*np.sum(Jz))
			indicesold = indicesnew
		if self.temphalo['exists']:
			return
		return JperR
	
	def get_spin_parameter(self, coords, IDs, radius = False, M=False):
		if self.temphalo['exists']:
			indices =self.temphalo['indices'][:self.temphalo['virialradiusindex']]
			R = self.temphalo['R200']
			M = self.temphalo['M200']
			self.temphalo['lambda'] = np.zeros_like(R)
			vel0 = self.temphalo['Vel']
			if self.partType==7:
				self.temphalo['lambdaH'] = np.zeros_like(R)
				self.temphalo['lambdaDM'] = np.zeros_like(R)
				Htemp = self.temphalo['HindicesM200']
				DMtemp =self.temphalo['DMindicesM200']
		elif radius:
			R = radius
			indices = self.get_indices(coords, IDs, radius=radius)
			vel0 = np.array([0, 0, 0])
			if len(indices) == 0:
				#print("Error: no particles found")
				return 0.0
		else:
			R = np.max(self.get_radius(point=coords)[indices])
			indices = self.get_indices(coords, IDs, radius=radius)
			vel0 = np.array([0, 0, 0])
			if len(indices) == 0:
				#print("Error: no particles found")
				return 0.0

		coord = self.get_coordinates()[indices, :] - coords
		coord = np.where(np.abs(coord) > 0.5*self.boxsize, coord - np.sign(coord)*self.boxsize, coord)
		vel = self.get_velocities()[indices, :] - vel0
		massa = self.get_masses()[indices]
		Jx = np.sum((coord[:, 1]*vel[:, 2] - coord[:, 2]*vel[:, 1])*massa)
		Jy = np.sum((coord[:, 2]*vel[:, 0] - coord[:, 0]*vel[:, 2])*massa)
		Jz = np.sum((coord[:, 0]*vel[:, 1] - coord[:, 1]*vel[:, 0])*massa)
		J = np.sqrt(Jx*Jx+Jy*Jy+Jz*Jz)
		if not M:
			M = np.sum(massa)
		if M == 0:
			return 0.0
		V = np.sqrt(G_Mpc_km2_Msi_si2*M*1.e10/R)
		if self.temphalo['exists']:
			self.temphalo['lambda'] = J/(1.414213562*M*V*R)
			if self.partType==7:
				coordH = coord[Htemp]
				velH = vel[Htemp]
				massaH = massa[Htemp]
				Jx = np.sum((coordH[:, 1]*velH[:, 2] - coordH[:, 2]*velH[:, 1])*massaH)
				Jy = np.sum((coordH[:, 2]*velH[:, 0] - coordH[:, 0]*velH[:, 2])*massaH)
				Jz = np.sum((coordH[:, 0]*velH[:, 1] - coordH[:, 1]*velH[:, 0])*massaH)
				self.temphalo['lambdaH'] = np.sqrt(Jx*Jx+Jy*Jy+Jz*Jz)/(1.414213562*M*V*R)
				coordDM = coord[DMtemp]
				velDM = vel[DMtemp]
				massaDM = massa[DMtemp]
				Jx = np.sum((coordDM[:, 1]*velDM[:, 2] - coordDM[:, 2]*velDM[:, 1])*massaDM)
				Jy = np.sum((coordDM[:, 2]*velDM[:, 0] - coordDM[:, 0]*velDM[:, 2])*massaDM)
				Jz = np.sum((coordDM[:, 0]*velDM[:, 1] - coordDM[:, 1]*velDM[:, 0])*massaDM)
				self.temphalo['lambdaDM']  = np.sqrt(Jx*Jx+Jy*Jy+Jz*Jz)/(1.414213562*M*V*R)	
			return
		return J/(1.414213562*M*V*R)

	def get_indices(self, coords = [], IDs = [], radius=False):
		if self.temphalo['exists']:
			return self.temphalo['indices'] #Only within R200
			
		if not self.useIDs and not radius:
			sys.exit('Error: no radius specified')
		if (not self.useIDs) or (len(IDs) == 0):
			if len(coords) == 0:
				sys.exit('Error: no coordinates specified')
			if not isinstance(self.tree, cKDTree):
				rad = self.get_radius(point=coords)
				return np.where(rad < radius)[0]
			else:
				return self.tree.query_ball_point(coords, r=radius)
		snapID = pd.DataFrame({'A': self.get_IDs()})
		indices = np.where(snapID['A'].isin(IDs))[0]
		if radius:
			rad = self.get_radius(point=coords)[indices]
			indices = indices[np.where(rad < radius)[0]]
		return indices	

	def get_mass(self, coords, IDs, radius=False):
		indices = self.get_indices(coords, IDs, radius=radius)
		return np.sum(self.get_masses()[indices])

	def get_midpoint1(self, IDs, bound=False):
		snapID = pd.DataFrame({'A': self.get_IDs()})
		indices = np.where(snapID['A'].isin(IDs))[0]
		if len(indices) == 0:
			sys.exit("There should be particles, but there aren't???")
		coords = self.get_coordinates()[indices]
		midpoint = np.array([np.sum(coords[:, 0]/len(coords[:, 0])), 
			np.sum(coords[:, 1]/len(coords[:, 0])), 
			np.sum(coords[:, 2]/len(coords[:, 0]))])
		radius = self.get_radius(point=midpoint, coords=coords)
		avdev = np.sum(radius)/len(radius)
		if bound:
			raddel = np.where(radius > 2*avdev)[0]
			print(len(raddel))
			indices = np.delete(indices, raddel)
			coords = self.get_coordinates()[indices]
			midpoint = np.array([np.sum(coords[:, 0]/len(coords[:, 0])), 
				np.sum(coords[:, 1]/len(coords[:, 0])), 
				np.sum(coords[:, 2]/len(coords[:, 0]))])		
		return midpoint

	def get_midpoint(self, IDs, midpoint_Part1=[0, 0, 0]):
		#if np.max(IDs) > np.max(self.IDs) or np.min(IDs) < np.min(self.IDs):
		#	print('Error: given IDs out of range!')
		#indices = np.abs(np.subtract.outer(self.IDs, IDs)).argmin(0)
		self.useIDs = True
		indices = self.get_indices([0], IDs, radius=False)
		indices = indices.astype(int)
		coords = self.get_coordinates()[indices]
		midpoint = np.array([np.sum(coords[:, 0]/len(coords[:, 0])), 
			np.sum(coords[:, 1]/len(coords[:, 0])), 
			np.sum(coords[:, 2]/len(coords[:, 0]))])
		radius = self.get_radius(point=midpoint, coords=coords)
		maxx = np.max(np.abs(coords[:, 0] - midpoint[0]))
		maxy = np.max(np.abs(coords[:, 1] - midpoint[1]))
		maxz = np.max(np.abs(coords[:, 2] - midpoint[2]))
		if self.partType == 5:
			print("Masses: ", self.get_masses()[indices][0])
			print("Distance: ", np.min(self.get_radius(point=midpoint_Part1, coords=coords)))
		return midpoint, np.max(radius), maxx, maxy, maxz

	def get_IDs_within_radius(self, coords, radius):
		#coord = self.get_coordinates() - coords
		rad = self.get_radius(point=coords)
		return self.get_IDs()[np.where(rad <= radius)[0]]

	def get_indici_within_radius(self, coords, radius):
		#coord = self.get_coordinates() - coords
		rad = self.get_radius(point=coords)
		return np.where(rad <= radius)[0]
	
	def get_mass_within_radius(self, coords, radius):
		if self.temphalo['exists']:
			return np.sum(self.get_masses()[self.temphalo['indices']])
		else:
			#coord = self.get_coordinates() - coords
			rad = self.get_radius(point=coords)
			indices = np.where(rad <= radius)[0]
			return np.sum(self.get_masses()[indices])

	def open_snap(self, snappath, nsnap):
		#if nsnap < 10:
		if os.path.isfile(snappath+"snapshot_%03d.hdf5" %nsnap):
			dataopen = h5py.File(snappath+"snapshot_%03d.hdf5" %nsnap, "r")
		else:
			print("Error: Could not open %s snapshot_%03d.hdf5" %(snappath, nsnap))
		return dataopen
