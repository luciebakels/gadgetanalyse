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

class OrbitTree:
	"""
	Class to compute orbital properties and categorise (sub)haloes

	This class can read in the organised trees that are output by the OrbitInfo class. 
	Other functionalities are the classification of orbital histories, the identification
	of third body interactions, and some optional fixes of the trees, which might not be
	necessary anymore for future VELOCIraptor versions. This class also allows for easy
	overwriting and adding of additional fields to the trees and combining information from
	SHARK and VELOCIraptor outputs.

	Attributes
	----------
	path : str
		path to orbit tree file directory
	zstart : float
		highest redshift in tree files
	filenames :	list of str
		filenames in directory
	file_halo :	dict
		not sure...
	d_hostinfo : dict
		dictionary containing header information of host haloes
	d_mhtree : dict
		dictionary containing tree information of host haloes
	d_nbtree : dict
		dictionary containing tree information of neighbour (sub)haloes
	d_nbtree_merger : dict
		dictionary containing tree information of merged haloes
	haloprop : dict
	d_categorise : dict
		dictionary of categorised neighbour (sub)haloes
	only_add_existing_hosts : bool
		a flag when set only information will be added to hosts that are
		already loaded, otherwise all hosts will be read

	Functions
	---------
	convert_to_Mpc(in_Mpc = 1)
		Converting all length units in the neighbour trees
	convert_to_Mpc_host(self, in_Mpc = 1)
		Converthing all length units in the host trees
	doealles(firstsnap = 101, read_velocities=False, read_positions=False, include_merged=False)
		Compute all necessary steps for my analysis

	--finding all three-body interactions--
	find_and_write_encounters(firstsnap=101, boxsize=105)
		Compute all encounters between (sub)haloes and write them to file
	find_interaction_subhalo(host, boxize=105)
		Computes all encounters between (sub)haloes
	npart_R200()
		Replacing M200 with 6dFOF mass and computing related R200

	--fixing problems in the trees--
	fix_formation_times(self)
		Using interpolated M200 to find accurate formation times
	peak_fix(nbtree_indices_label='snap(R200)', datasets=['Vmax', 'Rmax', 'npart'])
		Finding peak and infall values of accreted systems
	find_crossing(nbtree, mhtree, R200type='R200_inter')
		Find the correct infall snapshot, closest distance and if there is a crossing
	
	--orbital properties--
	find_numorbits_afterinfall()
		Find the number of orbits that take place after infall
	categorise(R200type='R200_inter')
		Categorise (sub)haloes according to their orbital history
	smooth_R200(self)
		Interpolate M200 (removing 'satellite' instances where M200 is incorrect) and compute
		relating R200. Save these fields to M200_inter and R200_inter
	select_hosts(massrange=None, z025range=None, z050range=None, z075range=None)
		Select host haloes within specified mass or formation time range

	--combining information from other codes (Shark, VR)--
	matchSharkData(d_output=None, sharkpath='/home/luciebakels/DMO11/Shark/189/', nfolders=128, datasets=[],
		hd_path ='/home/luciebakels/DMO11/Velcopy/', hd_name= '11DM.snapshot_189.quantities.hdf5', 
		snapshot=189, totnumsnap=189, boxsize=105, old_shark_version=False)
		Reading Shark values and writing these to the correct haloes
	read_VelData(datasets=['cNFW'], velpath='/home/luciebakels/DMO11/VELz0/', snapshot=189)
		Add extra field to trees from the VELOCIraptor catalogue
	read_HaloData(datasets=['R_HalfMass'], hd_path ='/home/luciebakels/DMO11/Velcopy/11DM.', snapshot=189,
		totzstart=20, totnumsnap=189, boxsize=105)
		Add extra field to trees from my catalogue 'snapshot_xxx.quantities.hdf5'

	--read and close files--
	readFile(filename, readtype='r')
		Open specified file
	closeFile(filename)
		Close specified file
	check_if_exists(dictionary, filename, datasets=None)
		Check if given datasets exist in file
	readHostInfo()
		Read all global information of host haloes
	readHostHalo(datasets=None)
		Read the tree information of host haloes
	readNeighbourTree(datasets=None)
		Read the tree information of (sub)haloes
	readMergedTree(datasets=None)
		Read the tree information of merged haloes
	readHostInfo_onefile(filename)
		Read global host halo information for specified file
	readHostHalo_onefile(filename, datasets=None, closefile=True, only_add_existing_hosts=False)
		Read the tree information of host haloes for specified file
	readNeighbourTree_onefile(filename, datasets=None, closefile=True, only_add_existing_hosts=False)
		Read the tree information of (sub)haloes for specified file
	readMergedTree_onefile(filename, datasets=None, closefile=True, only_add_existing_hosts=False)
		Read the tree information of merged haloes for specified file
	readAllData(datasets=None, datasets_host=None)
		Read all tree data

	--rewrite datasets--
	rewriteHostHalo(datasets=None, rewrite=True)
		Rewrite or add information to the trees of host haloes
	rewriteNeighbourTree(datasets=None, rewrite=True)
		Rewrite or add information to the trees of (sub)haloes
	rewriteMergedTree(datasets=None, rewrite=True)
		Rewrite or add information to the trees of merged haloes
	rewriteHostHalo_onefile(filename, datasets=None, rewrite=True)
		Rewrite or add information to the trees of host haloes for specified file
	rewriteNeighbourTree_onefile(filename, datasets=None, rewrite=True)
		Rewrite or add information to the trees of (sub)haloes for specified file
	rewriteMergedTree_onefile(filename, datasets=None, rewrite=True)
		Rewrite or add information to the trees of merged haloes for specified file
	
	"""
	def __init__(self, path, zstart=20):
		"""
		Parameters
		----------
		path : str
			path to orbit tree file directory
		zstart : float
			highest redshift in tree files (default is 20)
		"""
		self.path = path
		self.zstart = zstart
		self.filenames = []
		self.file_halo = {}
		self.d_hostinfo = {}
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
		"""Converting all length units in the neighbour trees

		Parameters
		----------
		in_Mpc : float
			conversion factor (default is 1)
		"""
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
		"""Converthing all length units in the host trees

		Parameters
		----------
		in_Mpc : float
			conversion factor (default is 1)
		"""
		self.readHostHalo(datasets=['R200', 'Coord'])
		for host in self.d_mhtree.keys():
			for i in range(len(self.d_mhtree[host]['Coord'])):
				waar = np.where(self.d_mhtree[host]['Coord'][i]>0)[0]
				self.d_mhtree[host]['Coord'][i, waar] = self.d_mhtree[host]['Coord'][i, waar]*in_Mpc
			waar = np.where(self.d_mhtree[host]['R200']>0)[0]
			self.d_mhtree[host]['R200'][waar] = self.d_mhtree[host]['R200'][waar]*in_Mpc

		self.rewriteHostHalo(datasets=['Distance', 'R200', 'X', 'Y', 'Z'])

	def doealles(self, firstsnap = 101, read_velocities=False, read_positions=False, include_merged=False):
		"""Compute all necessary steps for my analysis

		- This function reads in all hosts and removes 'problematic' hosts (i.e. hosts that VR identifies
		  as substructure at z=0, or hosts that have a first appearance later than snapshot 'firstsnap')
		- It then reads the tree information of these hosts and neighbouring (sub)haloes.
		- It fixes R200 and M200 values of host haloes, since at time of writing, VELOCIraptor 
		  computes incorrect values for R200 and M200 when haloes are identified to be substructure. It
		  also recomputes the correct formation times of host haloes, and recomputes the infall snapshots 
		  of subhaloes, storing this in 'snap(R200)'. 
		- Using the corrected infall times, the number of orbits after infall are computed for each halo.
		- Lastly, categorisation of haloes by their orbital histories is then done using the corrected
		  information computed before in this function.
		The flag 'only_add_existing_hosts' is set so that when reading in more datasets, only hosts 
		are used that are not problematic.

		Parameters
		----------
		firstsnap : int
			host haloes need to have their first appearance before this snapshot for them to be considered
		read_velocities : bool
			a flag when set allows for ONLY velocity data to be read in
		read_positions : bool
			a flag when set allows for ONLY positions to be read in
		include_merged : bool
			a flag when set allows for merged trees to be read

		Returns
		-------
		None

		"""
		print("Find good haloes...")
		self.readHostInfo()
		self.readHostHalo(datasets=['M200', 'R200', 'hostHaloIndex'])
		treelength = np.zeros(len(self.d_mhtree.keys()))
		
		self.readNeighbourTree(datasets=['R200'])

		#Eliminating hosts that are identified as substructure by VELOCIraptor or that have a 
		#first appearance later than snapshot 'firstsnap'.
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
		"""Compute all encounters between (sub)haloes and write them to file

		An encounters is defined to be when a lower mass halo enters the 'virial radius' of a larger halo.
		These masses are the 6DFOF particle mass of the haloes and the 'virial radius' are R200 computed when 
		assuming the 6DFOF particle mass is M200.

		Parameters
		----------
		firstsnap : int
			host haloes need to have their first appearance before this snapshot for them to be considered
		read_velocities : bool
			a flag when set allows for ONLY velocity data to be read in
		read_positions : bool
			a flag when set allows for ONLY positions to be read in
		include_merged : bool
			a flag when set allows for merged trees to be read

		Returns
		-------
		None
		"""
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
		"""Replacing M200 with 6dFOF mass and computing related R200.
		
		These values are stored in the d_nbtree dictionary with keys:
		'M200_npart' and 'R200_npart'
		"""
		hosts = np.array(list(self.d_mhtree.keys()))
		snapshots = np.arange(len(self.d_mhtree[hosts[0]]['M200']))
		rhocrit = np.zeros(len(self.d_mhtree[hosts[0]]['M200']))
		redshifts = ha.snapshot_to_redshift(snapshots, zstart=self.zstart, numsnap=len(self.d_mhtree[hosts[0]]['M200'])-1)
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
		"""Computes all encounters between (sub)haloes

		An encounters is defined to be when a lower mass halo enters the 'virial radius' of a larger halo.
		These masses are the 6DFOF particle mass of the haloes and the 'virial radius' are R200 computed when 
		assuming the 6DFOF particle mass is M200.
		Each encounter is stored in the d_nbtree dictionary under the key: 'interaction2'

		Parameters
		----------
		host : int
			key of the host within the d_nbtree and d_mhtree dictionaries
		boxsize : float
			the length of a side of the simulation box (default is 105)

		Returns
		-------
		None
		"""
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
		"""Using interpolated M200 to find accurate formation times

		This function computes the quarter-mass, half-mass, and three-quarter-mass formation times
		of each host halo.
		"""
		hosts = np.array(list(self.d_mhtree.keys()))
		z = ha.snapshot_to_redshift(np.arange(len(self.d_mhtree[hosts[0]]['M200_inter'])), zstart=self.zstart, numsnap=len(self.d_mhtree[hosts[0]]['M200_inter'])-1)
		for i in range(len(hosts)):
			host = hosts[i]
			ww = np.where(self.d_mhtree[host]['M200_inter'] > 0)[0]
			temp = interp1d(self.d_mhtree[host]['M200_inter'][ww], z[ww])
			at0 = self.d_mhtree[host]['M200_inter'][-1]
			self.d_hostinfo['z$_{0.50}$'][i] = temp(0.5*at0)
			self.d_hostinfo['z$_{0.75}$'][i] = temp(0.75*at0)
			if 0.25*at0 > np.min(self.d_mhtree[host]['M200_inter'][ww]):
				self.d_hostinfo['z$_{0.25}$'][i] = temp(0.25*at0)

	def peak_fix(self, nbtree_indices_label='snap(R200)', datasets=['Vmax', 'Rmax', 'npart']):
		"""Finding peak and infall values of accreted systems using correct infall snapshot

		Parameters
		----------
		nbtree_indices_label : str
			key name of the d_nbtree dictionary, pointing to the field that gives the desired
			infall snapshots (default is 'snap(R200)')
		datasets : list of str
			keys of the datasets the peak and infall values are to be calculated for

		Returns
		-------
		None
		"""
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
		"""Find the correct infall snapshot, closest distance, and flag if there is a crossing

		Parameters
		----------
		nbtree : dict
			neighbour tree of a specific host (d_nbtree[host] or d_nbtree_merger[host])
		nbtree : dict
			host tree of a specific host (d_mhtree[host])
		R200type : str
			key of boundary within to consider haloes to be substructure (default is 'R200_inter')
		"""
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
		"""Find the number of orbits that take place after infall
		"""
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
		"""Categorise (sub)haloes according to their orbital history

		- infalling subhaloes: orbits==0 and subhalo==True
		- orbital satellites: orbits_after_infall>0 and subhalo==True
		- backsplash haloes: notcrossed==False and subhalo==False
		- orbital halo: (orbits>0 and notcrossed==True) or (orbits_after_infall==0.5 and orbits>0))
		- pristine haloes: preprocessed==False and notcrossed==True and orbits==0 and subhalo==False
		- secondary backsplash: preprocessed==True and notcrossed==True and orbits==0 and subhalo_other==False
		- secondary subhalo: preprocessed==True and notcrossed==True and orbits==0 and subhalo_other==True

		- other: haloes outside of 4xr200 or that had their first appearance within their host halo
		"""

		# find_closest_approach(self.d_nbtree, self.d_mhtree, physical=False,zstart=20)
		for host in self.d_nbtree.keys():
			nbtree = self.d_nbtree[host]
			mhtree = self.d_mhtree[host]

			#Did the halo complete an orbit around its primary halo?
			orbits = nbtree['orbits']
			#Did the halo complete a pericentric passage?
			nperi = nbtree['N_peri']
			#Did the (sub)halo complete orbits after first infall onto the primary?
			orbits_ai = nbtree['orbits_after']
			#Was the halo preprocessed?
			preprocessed = nbtree['Preprocessed']
			#Has the halo crossed R200 of its primary halo?
			notcrossed = nbtree['no_crossing_1.0']
			#Is the halo substructure of its primary halo?
			subhalo = np.where(nbtree['Distance'][:, -1] - mhtree[R200type][-1]<0, True, False)
			#Is the halo substructure of a halo other than its primary?
			subander = np.where((nbtree['hostHaloIndex'][:, -1]>-1)&(nbtree['hostHaloIndex'][:, -1]!=mhtree['HaloIndex'][-1]), True, False)
			#First appearance within R200 of its primary halo?
			borninside = nbtree['born_within_1.0']
			#Distance to its primary at z=0
			afstandnu = nbtree['Distance'][:, -1]/mhtree[R200type][-1]

			#mindist = np.zeros(len(nbtree['Distance']))
			self.find_crossing(nbtree, mhtree)
			mindist_r200 = nbtree['Closest_R200']
			#ca = nbtree['closest_approach']/mhtree[R200type][-1]

			self.d_categorise[host] = {}
			self.d_categorise[host]['infalling subhaloes'] = np.where((subhalo==True)&(borninside==False)&(nperi==0)&(orbits_ai==0))[0]#&(ca>=0.9*afstandnu))[0]
			self.d_categorise[host]['orbital subhaloes'] = np.where((subhalo==True)&(borninside==False)&((orbits_ai>0)))[0]#|(ca<0.9*afstandnu)))[0]

			self.d_categorise[host]['backsplash haloes'] = np.where((mindist_r200<=4)&(borninside==False)&(notcrossed==False)&(subhalo==False))[0]
			self.d_categorise[host]['orbital haloes ($>$r$_{200}$)'] = np.where((mindist_r200<=4)&(borninside==False)&(subhalo==False)&(notcrossed==True)&(nperi>0))[0]
			self.d_categorise[host]['orbital haloes ($<$r$_{200}$)'] = np.where((subhalo==True)&(borninside==False)&(nperi>0)&(orbits_ai==0))[0]#&(ca>=0.9*afstandnu))[0]

			self.d_categorise[host]['pristine haloes'] = np.where((subhalo==False)&(mindist_r200<=4)&(borninside==False)&
				(preprocessed==-1)&(notcrossed==True)&(nperi==0))[0]

			self.d_categorise[host]['secondary backsplash'] = np.where((subander==False)&(subhalo==False)&(mindist_r200<=4)&(borninside==False)&
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
		"""
		Interpolate M200 (removing VELOCIraptor 'satellite' instances where M200 is incorrect) and compute
		relating R200. Save these fields to M200_inter and R200_inter
		"""
		hosts = np.array(list(self.d_mhtree.keys()))
		snapshots = np.arange(len(self.d_mhtree[hosts[0]]['M200']))
		rhocrit = np.zeros(len(self.d_mhtree[hosts[0]]['M200']))
		redshifts = ha.snapshot_to_redshift(snapshots, zstart=self.zstart, numsnap=len(self.d_mhtree[hosts[0]]['M200'])-1)
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
		"""Select host haloes within specified mass or formation time range

		This function needs to be run before reading in any other data, and will automatically
		restrict only host haloes to be read in if they meet the specified conditions.

		Parameters
		----------
		massrange : array(2) or list(2), optional
			the minimum and maximum value of masses required for host haloes (default is None)
		z025range : array(2) or list(2), optional
			the minimum and maximum value of quarter-mass formation times required for host haloes 
			(default is None)
		z050range : array(2) or list(2), optional
			the minimum and maximum value of half-mass formation times required for host haloes 
			(default is None)
		z075range : array(2) or list(2), optional
			the minimum and maximum value of three-quarter mass formation times required for host 
			haloes (default is None)

		Returns
		-------
		None
		"""
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
		hd_path ='/home/luciebakels/DMO11/Velcopy/', hd_name= '11DM.snapshot_189.quantities.hdf5', snapshot=189, 
		totnumsnap=189, boxsize=105, old_shark_version=False):
		"""Reading Shark values and writing these to the correct haloes in the trees

		Combined VELOCIraptor catalogues and TreeFrog outputs are needed in order to
		use the older shark version. The newer version would only need the VR catalogue,
		but I have not yet implemented this (but it should be quite easy). Now you need
		to have the converted VR + TF catalogue that is made using the haloanalyse module.

		Parameters
		----------
		d_output : dict, optional
			dictionary in the shape of d_categorise (default is None)
		sharkpath : str
			path to directory of shark subvolume directories
		nfolders : int
			number of shark subvolume directories (default is 128)
		dataset : list or array of str, optional
			SHARK datasets to be read in
		hd_path : str
			path to HaloData directory
		hd_name : str
			name of HaloData file containing the IDs and Head and Tail values of haloes
		snapshot : int
			number of the snapshot
		totnumsnap : int
			total number of snapshots in the simulation
		boxsize : float
			length of a side of the simulation box
		old_shark_version : bool
			if the shark version is old, it requires the Tail to match up with the SHARK halo 
			instead of the VR haloIDs

		Returns
		-------
		None
		"""

		if d_output is None:
			d_output = self.d_categorise

		for key in ['id_subhalo_tree', 'id_halo_tree', 'mvir_hosthalo', 'type', 'mvir_subhalo', 'mgas_bulge', 'mgas_disk', 'mhot', 'mstars_bulge', 'mstars_disk']:
			datasets.append(key)
		
		self.readNeighbourTree(datasets=['HaloIndex'])
		
		#Reading in HaloData file that has the combined VR+TF catalogue
		keys = (['infalling subhaloes', 'orbital subhaloes', 'backsplash haloes', 'pristine haloes', 'secondary backsplash', 
			'secondary subhaloes', 'orbital haloes ($<$r$_{200}$)', 'orbital haloes ($>$r$_{200}$)'])
		start_time = time.time()

		hd = ha.HaloData(hd_path, hd_name, snapshot=snapshot, totzstart=self.zstart, totnumsnap=totnumsnap, boxsize=boxsize)
		hd.readSharkData(sharkpath=sharkpath, nfolders=nfolders, datasets=datasets)
		if old_shark_version:
			hd.readData(datasets=['Tail'])
		else: #TODO: make it read in VR catalogue instead
			hd.readData(datasets=['HaloID'])

		#Combining all baryonic matter in the galaxy
		allH = (hd.sharkdata['mgas_bulge'] + hd.sharkdata['mgas_disk'] + 
			hd.sharkdata['mstars_bulge'] + hd.sharkdata['mstars_disk'])

		#shark halo ids and fix shark id issues of old version
		shark_haloes_match = hd.sharkdata['id_subhalo_tree']
		if old_shark_version:
			replace = np.where(shark_haloes_match > (hd.snapshot+1)*hd.THIDVAL)[0]
			shark_haloes_match[replace] = shark_haloes_match[replace]%(1000*hd.THIDVAL)

		allindices = np.zeros(0).astype(int)

		hosts = np.array(list(self.d_nbtree.keys()))
		print("--- %s seconds ---" % (time.time() - start_time), 'read data')

		start_time = time.time()
		for host in hosts:
			for key in keys:
				allindices = np.append(allindices, self.d_nbtree[host]['HaloIndex'][d_output[host][key], -1])
		
		#Read appropriate halo IDs and link to SHARK halo IDs
		if old_shark_version:
			allindices = hd.hp['Tail'][allindices] + 1
		else:
			allindices = hd.hp['HaloID'][allindices]
		waarmatch = np.where(np.in1d(shark_haloes_match, allindices))[0]
		shark_haloes_match = shark_haloes_match[waarmatch]
		print("--- %s seconds ---" % (time.time() - start_time), 'found matching indices')

		start_time = time.time()
		for host in hosts:
			#List IDs of all subhaloes in the neighbourtree beloning to 'host'
			subhaloids = np.zeros(0).astype(int)
			welke = np.zeros(0).astype(int)
			for key in keys: #TODO: optimise this by allocating the array first, very easy...
				subhaloids = np.append(subhaloids, self.d_nbtree[host]['HaloIndex'][d_output[host][key], -1])
				welke = np.append(welke, d_output[host][key])
			haloid = self.d_mhtree[host]['HaloIndex'][-1]

			#Allocate memory
			self.d_nbtree[host]['SharkDM'] = np.zeros(len(self.d_nbtree[host]['HaloIndex']))
			self.d_nbtree[host]['SharkH'] = np.zeros(len(self.d_nbtree[host]['HaloIndex']))
			for ds in datasets:
				self.d_nbtree[host][ds] = np.zeros(len(self.d_nbtree[host]['HaloIndex']))
				if ds == 'type':
					self.d_nbtree[host][ds] = np.ones(len(self.d_nbtree[host]['HaloIndex']))*-1
				elif ds not in ['mvir_subhalo', 'mvir_hosthalo', 'position_x', 'position_y', 'position_z']:
					self.d_nbtree[host][ds+'_main'] = np.zeros(len(self.d_nbtree[host]['HaloIndex']))
			self.d_nbtree[host]['Ngalaxy'] = np.zeros(len(self.d_nbtree[host]['HaloIndex']))

			#Find halo IDs
			if old_shark_version:
				my_haloes_match = hd.hp['Tail'][subhaloids] + 1
				replace = np.where(my_haloes_match == 0)[0]
				my_haloes_match[replace] = subhaloids[replace] + hd.snapshot*hd.THIDVAL + 1
			else:
				my_haloes_match = hd.hp['HaloID'][subhaloids]

			shark_found = np.zeros(0).astype(int)
			my_found = np.zeros(0).astype(int)
			
			#Find haloes in the SHARK catalogue
			h_in_shark = np.where(np.in1d(my_haloes_match, shark_haloes_match))[0]
			temparange = np.where(np.in1d(shark_haloes_match, my_haloes_match))[0]
			tempid = np.array(shark_haloes_match[temparange])
			
			#Write the SHARK properties to the neighbour tree
			for halo in h_in_shark:
				waar = np.where(my_haloes_match[halo] == tempid)[0]
				if len(waar) == 0:
					continue
				allgalaxies = waarmatch[temparange[waar]]

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

			#Do the same for the host tree
			if old_shark_version:
				my_haloes_match = hd.hp['Tail'][subhaloids] + 1
			else:
				my_haloes_match = hd.hp['HaloID'][subhaloids]

			if my_haloes_match == 0:
				my_haloes_match = haloid + hd.snapshot*hd.THIDVAL + 1

			h_in_shark = np.where(my_haloes_match == hd.sharkdata['id_subhalo_tree'])[0]
			if len(h_in_shark) == 0:
				continue
			self.d_mhtree[host]['SharkDM'] = hd.sharkdata['mvir_hosthalo'][h_in_shark][0]
			self.d_mhtree[host]['SharkH'] = np.sum(allH[h_in_shark]) + hd.sharkdata['mhot'][h_in_shark][0]
			self.d_mhtree[host]['Ngalaxy'] = len(h_in_shark)
			for ds in datasets:
				self.d_mhtree[host][ds] = np.sum(hd.sharkdata[ds][h_in_shark])
		print("--- %s seconds ---" % (time.time() - start_time), 'finished matching shark')

	def readHostInfo(self):
		"""Read all information of host haloes at z=0
		"""
		for filename in self.filenames:
			self.readHostInfo_onefile(filename)

	def readHostHalo(self, datasets=None):
		"""Read the tree information of host haloes
		"""
		for filename in self.filenames:
			self.readHostHalo_onefile(filename, datasets=datasets, only_add_existing_hosts=self.only_add_existing_hosts)

	def readNeighbourTree(self, datasets=None):
		"""Read the tree information of (sub)haloes
		"""
		for filename in self.filenames:
			self.readNeighbourTree_onefile(filename, datasets=datasets, only_add_existing_hosts=self.only_add_existing_hosts)

	def readMergedTree(self, datasets=None):
		"""Read the tree information of merged haloes
		"""
		for filename in self.filenames:
			self.readMergedTree_onefile(filename, datasets=datasets, only_add_existing_hosts=self.only_add_existing_hosts)

	def rewriteHostHalo(self, datasets=None, rewrite=True):
		"""Rewrite or add information to the trees of host haloes
		"""
		for filename in self.filenames:
			self.rewriteHostHalo_onefile(filename, datasets=datasets, rewrite=rewrite)

	def rewriteNeighbourTree(self, datasets=None, rewrite=True):
		"""Rewrite or add information to the trees of (sub)haloes
		"""
		for filename in self.filenames:
			self.rewriteNeighbourTree_onefile(filename, datasets=datasets, rewrite=rewrite)

	def rewriteMergedTree(self, datasets=None, rewrite=True):
		"""Rewrite or add information to the trees of merged haloes
		"""
		for filename in self.filenames:
			self.rewriteMergedTree_onefile(filename, datasets=datasets, rewrite=rewrite)

	def readFile(self, filename, readtype='r'):
		"""Open specified file
		"""
		if filename not in self.haloprop.keys():
			self.haloprop[filename] = h5py.File(self.path+filename, readtype)

	def closeFile(self, filename):
		"""Close specified file
		"""
		if filename in self.haloprop.keys():
			self.haloprop[filename].close()
			del self.haloprop[filename]

	def readHostInfo_onefile(self, filename):
		"""Read host halo information for specified file
		"""
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

	def check_if_exists(self, dictionary, filename, datasets=None):
		"""Check if given datasets exist in file
		"""
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
		"""Read the tree information of host haloes for specified file
		"""
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
		"""Read the tree information of host haloes for specified file
		"""
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
		"""Read the tree information of (sub)haloes for specified file
		"""
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
		"""Read the tree information of merged haloes for specified file
		"""
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
		"""Rewrite or add information to the trees of (sub)haloes for specified file
		"""
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
		"""Rewrite or add information to the trees of merged haloes for specified file
		"""
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
		"""Add extra field to trees from the VELOCIraptor catalogue
		"""
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
		"""Read all tree data
		"""
		for filename in self.filenames:
			self.readFile(filename)

			self.readHostInfo(filename, closefile=False)

			self.readMergedSort(filename, closefile=False)

			self.readHostInfo_onefile(filename, datasets=datasets_host, closefile=False)

			self.readNeighbourTree_onefile(filename, datasets=datasets, closefile=False)

			self.readMergedTree_onefile(filename, datasets=datasets, closefile=False)

			self.closeFile(filename)

#Selecting haloes and subhaloes according to their properties
def find_outputtemp_massfractions(d_output, d_nbtree, d_mhtree, massfracmin, massfracmax, keys=None, atmax=False,
	nparttype='npart', preprocessed=False, unbound = False, atinfall=False, nofrac=False):
	"""Selecting haloes and subhaloes according to their subhalo to host halo fraction

	Parameters
	----------
	d_output : dict
		dictionary in the shape of OrbitTree.d_categorise
	d_nbtree : dict
		dictionary containing the neighbour trees
	d_mhtree : dict
		dictionary containing the host halo trees
	massfracmin : float
		minimum subhalo to host halo mass
	massfracmax : float
		maximum subhalo to host halo mass
	keys : list, optional
		keys of d_output[hosts].keys() to be considered. 
		If not set, all keys will be considered
		(default is None)
	atmax : bool
		a flag if set, the massfraction is computed at maximum instead of at z=0
		(default is False)
	atinfall : bool
		a flag if set, the massfraction is computed at infall instead of at z=0
		(default is False)
	nparttype : str
		the name of the field to compute the mass fractions for
		(default is 'npart')
	preprocessed : bool
		a flag if set, only preprocessed haloes are considered
		(default is False)
	unbound : bool
		a flag if set, only haloes with positive orbital energy are considered
		(default is False)
	nofrac : bool
		a flag if set, subhalo masses instead of fractions are used

	Returns
	-------
	dict
		dictionary with the new categorised output file
	float
		average massfraction
	array
		list of hosts, starting at 0 
	int
		number of hosts
	"""

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
	nparttype='npart', minhosts=10, preprocessed=False, bound=False, groupinfall=False, minfrac_limit=True):
	"""Selecting host haloes according to their mass

	Parameters
	----------
	d_output : dict
		dictionary in the shape of OrbitTree.d_categorise
	d_nbtree : dict
		dictionary containing the neighbour trees
	d_mhtree : dict
		dictionary containing the host halo trees
	min_particle_limit : int
		minimum number of particles for a halo to be considered (default is 50)
	massmin : float
		minimum host halo mass
	massmax : float
		maximum host halo mass
	keys : list, optional
		keys of d_output[hosts].keys() to be considered. 
		If not set, all keys will be considered
		(default is None)
	nparttype : str
		the name of the field to compute the mass fractions for
		(default is 'npart')
	minhosts : int
		the minimum amount of hosts within the specified massrange (default is 10)
	preprocessed : bool
		a flag if set, only preprocessed haloes are considered
		(default is False)
	bound : bool
		a flag if set, only haloes with negative orbital energy are considered
		(default is False)
	groupinfall : bool
		a flag if set, only specified group infall haloes are considered
		(default is False)

	Returns
	-------
	dict
		dictionary with the new categorised output file
	float
		median mass
	array
		list of hosts indices 
	"""

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
	"""Returns selection of d_categorise of neighbours that have treelengths > 8
	"""
	d_output = {}
	hosts = np.array(list(ot.d_mhtree.keys()))
	for host in hosts:
		d_output[host] = {}
		for key in ot.d_categorise[hosts[0]]:
			d_output[host][key] = {}

	redshift_arr = ha.snapshot_to_redshift(np.arange(len(ot.d_mhtree[host]['M200'])), 
		numsnap=len(ot.d_mhtree[host]['M200'])-1)
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
	"""Returns selection of d_categorise of neighbours that have a minimum particle number 'minpart'
	and a minimum snapshot length 'min_snapshots'.

	Parameters
	----------
	ot : object
		OrbitTree class object
	minpart : int
		minimum number of particles within a (sub)halo (default is 50)
	nparttype : str
		type of particles to count over (default is 'npart')
	atinfall : bool
		a flag when set, considers particle numbers at infall
	min_snapshots : int
		minimum treelength

	Returns
	-------
	dict
		dictionary with the new categorised output file
	"""
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

def select_by_shark_stellarmass(ot, d_output_in=None, minmass=1e8, maxmass=None):
	"""Returns selection of d_categorise of neighbours that have SHARK galaxies within 
	a specified massrange.

	Parameters
	----------
	ot : object
		OrbitTree class object
	d_output_in : dict, optional
		d_categorise like object, if None, uses ot.d_categorise (default is None)
	minmass : int
		minimum mass of galaxy in solar masses (default is 1e8)
	maxmass : int, optional
		maximum mass of galaxy in solar masses (default is None)

	Returns
	-------
	dict
		dictionary with the new categorised output file
	"""
	if d_output_in is None:
		d_output_in = {}
		for host in ot.hosts:
			d_output_in[host] = {}
			for key in ot.d_categorise[host]:
				d_output_in[host][key] = np.zeros(0).astype(int)
	d_output = {}
	for host in d_output_in.keys():
		d_output[host] = {}
		for key in d_output_in[host].keys():
			waar1 = np.where(ot.d_nbtree[host]['mstars_bulge_main'][d_output_in[host][key]] + 
				ot.d_nbtree[host]['mstars_disk_main'][d_output_in[host][key]] > minmass)[0]
			if len(waar1) == 0:
				d_output[host][key] = np.zeros(len(waar1)).astype(int)
				continue
			if maxmass is not None:
				waar1 = waar1[np.where(ot.d_nbtree[host]['mstars_bulge_main'][d_output_in[host][key][waar1]] + 
					ot.d_nbtree[host]['mstars_disk_main'][d_output_in[host][key][waar1]] < maxmass)[0]]
			d_output[host][key] = d_output_in[host][key][waar1].astype(int)

	return d_output
	
def select_by_npart_merged(ot, minpart=50, nparttype='npart', min_snapshots=8):
	"""Returns selection of d_categorise of merged neighbours that have a minimum particle number 'minpart'
	and a minimum snapshot length 'min_snapshots'.

	Parameters
	----------
	ot : object
		OrbitTree class object
	minpart : int
		minimum number of particles within a (sub)halo (default is 50)
	nparttype : str
		type of particles to count over (default is 'npart')
	min_snapshots : int
		minimum treelength

	Returns
	-------
	dict
		dictionary with the new categorised output file
	"""

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

#Profiles
def read_profiles(path = '/home/luciebakels/DMO11/Profiles/', 
	datasets=['Radius', 'HaloIndex', 'Mass_profile', 'Density', 'MassTable']):
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

#Virial properties
def R200mean(r, R200crit, firstterm, c):
	#firstterm = (3*M200crit*rhom/(800*np.pi*(np.log(1+cNFW)-cNFW/(1+cNFW))))
	return firstterm*(np.log(1 + r * c / R200crit) - r * c / (R200crit + r * c))**(1/3) - r

def R200crit_from_R200mean(r, R200mean, firstterm, rhoc):
	M200 = 800./3.*np.pi*rhoc*r**3
	c = gdp.cnfw(M200)
	temp = R200mean * c / r
	return firstterm*((np.log(1+c) - c/(1+c)) / (np.log(1 + temp) - temp / (1 + temp)))**(1./3.) - r

def computeR200mean(ot):
	z = ha.snapshot_to_redshift_2048(np.arange(190))
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

#Vmax Rmax computations
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

