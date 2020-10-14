import os
import sys
import numpy as np
import velociraptor_python_tools as vpt
import h5py
from constants import *
from haloanalyse import *

#Version-1.3


class OrbWeaverData:
	def __init__(self, filenamelist, entrytypes=[0, 1, -1, 99, -99], merged=False):
		self.orbitdata = {}
		self.filenamelist = filenamelist
		self.indices = None
		self.entrytypes = entrytypes
		self.merged = merged

	def setIndices(self, sort=True):
		datasets = ['entrytype']
		if self.merged == False:
			datasets.append('MergedFlag')
		orbitdata = self.readOrbitData_private(datasets=datasets)
		orbitdata['entrytype'] = orbitdata['entrytype'].astype(int)
		
		if (99 in self.entrytypes) or (-99 in self.entrytypes):
			self.indices99 = np.zeros(0).astype(int)

		self.indices = np.zeros(0).astype(int)
		# if self.merged:
		# 	self.indices = np.where(orbitdata['MergedFlag'] == False)[0]
		for i in self.entrytypes:
			if np.abs(i) == 99:
				self.indices99 = np.append(self.indices99, np.where(orbitdata['entrytype'] == i)[0])
			else:
				self.indices = np.append(self.indices, np.where(orbitdata['entrytype'] == i)[0])
		if sort:
			self.indices = np.sort(self.indices)
			if (99 in self.entrytypes) or (-99 in self.entrytypes):
				self.indices99 = np.sort(self.indices99)

		self.orbitdata['entrytype'] = orbitdata['entrytype'][self.indices].astype(int)
		if (99 in self.entrytypes) or (-99 in self.entrytypes):
			self.orbitdata['entrytype99'] = orbitdata['entrytype'][self.indices99].astype(int)

		del orbitdata

	def readOrbitData(self, desiredfields=['HaloID_orig', 'Mass', 'OrbitID','OrbitedHaloRootProgen_orig',
		'numorbits', 'Mass_host', 'VXrel', 'VYrel', 'VZrel', 'closestapproach','HaloRootProgen_orig',
		'Xrel', 'Yrel', 'Zrel', 'Radius_host', 'scalefactor', 'orbitecc_ratio', 'MergedFlag'], overwrite=False, sort=True):

		datasets = desiredfields.copy()
		if overwrite == False:
			for ds in self.orbitdata.keys():
				if ds in datasets:
					datasets.remove(ds)

		if len(datasets) == 0:
			return 0

		if (self.indices is None) and (overwrite == False):
			self.setIndices(sort=sort)

		for ds in datasets:
			orbitdata = self.readOrbitData_private(datasets=[ds])
			if ds == 'orbitecc_ratio':
				if (99 in self.entrytypes) or (-99 in self.entrytypes):
					self.orbitdata[ds] = orbitdata[ds][self.indices99]
			elif ds == 'entrytype':
				self.orbitdata[ds] = orbitdata[ds][self.indices].astype(int)
			else:	
				self.orbitdata[ds] = orbitdata[ds][self.indices]
				if (ds in ['OrbitID', 'scalefactor']) and ((99 in self.entrytypes) or (-99 in self.entrytypes)):
					self.orbitdata[ds+'99'] = orbitdata[ds][self.indices99]

			orbitdata = []

	def readOrbitData_private(self, datasets=[]):

		"""
		Function to read in the data from the .orbweaver.orbitdata.hdf files
		Parameters
		----------
		filenamelist : str
			The file containing the basenames for the orbweaver catalogue (can use the file generated from the creation of the (preprocessed) orbit catalogue)
		iFileno : bool, optional
			If True, the file number where the each of the entries came from is to be outputted.
		desiredfields : list, optional
			A list containing the desired fields to put returned, please see the FieldsDescriptions.md for the list of fields availible. If not supplied then all fields are returned
		Returns
		-------
		orbitdata : dict
			Dictionary of the fields to be outputted, where each field is a ndarray .
		"""
		filenamelist = self.filenamelist
		#First see if the file exits
		if(os.path.isfile(filenamelist)==False):
			raise IOError("The filelist",filenamelist,"does not exist")

		filelist = open(filenamelist,"r")

		#Try and read the first line as int
		try:
			numfiles = int(filelist.readline())
		except ValueError:
			raise IOError("The first line of the filelist (which says the number of files), cannot be interpreted as a integer")

		numentries = np.zeros(numfiles,dtype=np.uint64)
		maxorbitIDs = np.zeros(numfiles,dtype=np.uint64)
		prevmaxorbitID = np.uint64(0)
		filenames = [""]*numfiles
		orbitdatatypes = {}
		orbitdatakeys = None

		#Loop through the filelist and read the header of each file
		for i in range(numfiles):

			#Extract the filename
			filename = filelist.readline().strip("\n")
			filename+=".orbweaver.orbitdata.hdf"

			if(os.path.isfile(filename)==False):
				raise IOError("The file",filename,"does not exits")

			#Open up the file
			hdffile = h5py.File(filename,"r")

			#Read the header information
			numentries[i] =  np.uint64(hdffile.attrs["Number_of_entries"][...])
			maxorbitIDs[i] = prevmaxorbitID
			prevmaxorbitID += np.uint64(hdffile.attrs["Max_orbitID"][...])

			#If the first file then file then find the dataset names and their datatypes
			if(i==0):
				if(len(datasets)>0):
					orbitdatakeys = datasets
				else:
					orbitdatakeys = list(hdffile.keys())

				for key in orbitdatakeys:
					orbitdatatypes[key] = hdffile[key].dtype

			hdffile.close()

			#Add this filename to the filename list
			filenames[i] = filename

		#Now can initilize the array to contain the data
		totnumentries = np.sum(numentries)
		orbitdata = {key:np.zeros(totnumentries,dtype = orbitdatatypes[key]) for key in orbitdatakeys}

		ioffset = np.uint64(0)

		#Now read in the data
		for i in range(numfiles):

			filename=filenames[i]

			#print("Reading orbitdata from",filename)

			#Open up the file
			hdffile = h5py.File(filename,"r")

			#Get the start and end index
			startindex = ioffset
			endindex = np.uint64(ioffset+numentries[i])

			#Read the datasets
			for key in orbitdatakeys:
				orbitdata[key][startindex:endindex] = np.asarray(hdffile[key],dtype=orbitdatatypes[key])

			if("OrbitID" in orbitdatakeys):
				#Lets offset the orbitID to make it unique across all the data
				orbitdata["OrbitID"][startindex:endindex]+=maxorbitIDs[i]

			hdffile.close()

			ioffset+=numentries[i]

		return orbitdata

# welke = np.where((orbitdata['MergedFlag']==False)&(orbitdata['entrytype']==0)&(orbitdata['scalefactor']==1.0)&#)[0]#&
# 	#(orbitdata['Mass_host']<1.1e13)&(orbitdata['Mass_host']>=1e13))[0]#)[0]#&
# 	(orbitdata['Mass_host']>=0.99*np.max(orbitdata['Mass_host'])))[0]

# if rootTailhostIDs is None:
# 	hosts = np.array(list(set(orbitdata['OrbitedHaloRootProgen_orig'][welke])))
# 	print(len(hosts))
# else:
# 	hosts = rootTailhostIDs
# hosts = np.array([35000000013844,  5000000000155, 39000000241851, 84000003812713,
#        24000000020767, 24000000035554, 13000000004340, 21000000023022,
#        15000000004386, 19000000017367, 11000000002269])

class OrbitInfo:
	def __init__(self, orbweaverdata, hosts=None, minhaloes=None, THIDVAL=1000000000000, physical=True, max_distance=None, max_times_r200=None,
		newversion=True, npart_list=None, last_snap=200, skeleton_only=True):
		self.ow = orbweaverdata
		self._hosts_init = hosts
		self._hosts = np.copy(np.array(self._hosts_init).astype(int))
		self._minhaloes = minhaloes
		self._output = {}
		self._indices = None
		self._calculate_main_properties = False
		self._max_distance = max_distance
		self._max_times_r200 = max_times_r200
		self.level1 = False
		self.level2 = False
		self.velIDs = None
		self.THIDVAL = THIDVAL
		self.physical = physical
		self.newversion = newversion
		self.npart_list = npart_list
		self.last_snap = last_snap
		self.skeleton_only=skeleton_only

		self.ow.readOrbitData(desiredfields=['HaloID_orig', 'Mass', 'OrbitID','OrbitedHaloRootProgen_orig',
			'numorbits', 'Mass_host', 'VXrel', 'VYrel', 'VZrel', 'closestapproach','HaloRootProgen_orig',
			'Xrel', 'Yrel', 'Zrel', 'Radius_host', 'scalefactor', 'MergedFlag', 'closestapproachscalefactor', 
			'OrbitedHaloRootDescen_orig', 'HaloRootDescen_orig'], overwrite=False, sort=True)

		# if (self._hosts is not None) and (self._minhaloes is not None):
		# 	niet = np.zeros(0).astype(int)
		# 	self._indices = np.where((self.ow.orbitdata['entrytype']==0)&(self.ow.orbitdata['scalefactor']==1.0))[0]
		# 	j = -1
		# 	for i in self._hosts:
		# 		j += 1
		# 		if len(np.where(self.ow.orbitdata['OrbitedHaloRootProgen_orig'][self._indices] == i)[0]) < self._minhaloes:
		# 			niet = np.append(niet, j)

		# 	self._hosts = np.delete(self._hosts, niet)

		self.computeIndices()

	def computeIndices(self):
		if self._hosts is not None:
			self._indices = np.zeros(0).astype(int)
			self._host_per_index = np.zeros(0).astype(int)
			for i in self._hosts:
				newindicestemp = np.where((self.ow.orbitdata['entrytype']==0)&
					(self.ow.orbitdata['OrbitedHaloRootProgen_orig']==i)&(self.ow.orbitdata['scalefactor']==1.0))[0]
				self._indices = np.append(self._indices, newindicestemp)

			self._host_per_index = self.ow.orbitdata['OrbitedHaloRootProgen_orig'][self._indices]
		
		if self.skeleton_only == False:
			self.cut_maximum_distance()
		else:
			self.calculate_main_properties()

	def computeMergedIndices(self):
		if self._hosts is not None:
			self._merged_indices = np.zeros(0).astype(int)
			self._host_per_merged_index = np.zeros(0).astype(int)
			for i in self._hosts:
				newindicestemp = np.where((self.ow.orbitdata['entrytype']==0)&
					(self.ow.orbitdata['OrbitedHaloRootProgen_orig']==i)&(self.ow.orbitdata['MergedFlag']==True))[0]
				self._merged_indices = np.append(self._merged_indices, newindicestemp)

			self._host_per_merged_index = self.ow.orbitdata['OrbitedHaloRootProgen_orig'][self._merged_indices]
			self._merged_vel_indices  = np.array(self.ow.orbitdata['HaloID_orig'][self._merged_indices]).astype(int)

	@property
	def indices(self):
		return self._indices

	@property
	def vel_indices(self):
		if hasattr(self, "_vel_indices"):
			return self._vel_indices
		elif self.indices is not None:
			self.ow.readOrbitData(desiredfields=['HaloID_orig'])
			self._vel_indices = np.array(self.ow.orbitdata['HaloID_orig'][self.indices]).astype(int)%self.THIDVAL - 1
			return self._vel_indices

	@property
	def vel_host_per_index(self):
		if hasattr(self, "_vel_host_per_index"):
			return self._vel_host_per_index
		elif self.indices is not None:
			self.ow.readOrbitData(desiredfields=['HaloID_orig'])
			self._vel_host_per_index = np.array(self.ow.orbitdata['OrbitedHaloRootDescen_orig'][self.indices]).astype(int)%self.THIDVAL - 1
			return self._vel_host_per_index

	@property
	def merged_indices(self):
		if hasattr(self, "_merged_indices"):
			return self._merged_indices
		else:
			self.computeMergedIndices()
			return self._merged_indices
	
	@property
	def merged_vel_indices(self):
		if hasattr(self, "_merged_indices"):
			return self._merged_vel_indices
		else:
			self.computeMergedIndices()
			return self._merged_vel_indices

	@property
	def host_per_merged_index(self):
		if hasattr(self, "_host_per_merged_index"):
			return self._host_per_merged_index
		else:
			self.computeMergedIndices()
			return self._host_per_merged_index

	@property
	def velIDs(self):
		if hasattr(self, "_velIDs"):
			return self._velIDs

	@property
	def hosts(self):
		return self._hosts

	@property
	def host_per_index(self):
		if hasattr(self, "_host_per_index"):
			return self._host_per_index

	@property
	def hosts_init(self):
		return self._hosts_init

	@property
	def minhaloes(self):
		return self._minhaloes

	@property
	def distance(self):
		if hasattr(self, "_distance"):
			return self._distance
	
	@property
	def max_distance(self):
		if hasattr(self, "_max_distance"):
			return self._max_distance

	@property
	def max_times_r200(self):
		if hasattr(self, "_max_distance"):
			return self._max_times_r200

	@property
	def velocity(self):
		if hasattr(self, "_velocity"):
			return self._velocity

	@property
	def vrad(self):
		if hasattr(self, "_vrad"):
			return self._vrad

	@property
	def r200_host(self):
		return self.ow.orbitdata['Radius_host'][self.indices]

	@property
	def dist_r200(self):
		if hasattr(self, "_dist_r200"):
			return self._dist_r200

	@property
	def eccentricity(self):
		if hasattr(self, "_eccentricity"):
			return self._eccentricity

	@property
	def orbits(self):
		return self.ow.orbitdata['numorbits'][self.indices]

	@property
	def ca_r200(self):
		if hasattr(self, "_ca_r200"):
			return self._ca_r200

	@property
	def infall(self):
		if hasattr(self, "_infall"):
			return self._infall

	@property
	def other(self):
		if hasattr(self, "_other"):
			return self._other
	
	@property
	def cat1a(self):
		if hasattr(self, "_cat1a"):
			return self._cat1a

	@property
	def cat1b(self):
		if hasattr(self, "_cat1b"):
			return self._cat1b

	@property
	def cat2a(self):
		if hasattr(self, "_cat2a"):
			return self._cat2a

	@property
	def cat2b(self):
		if hasattr(self, "_cat2b"):
			return self._cat2b

	@property
	def other(self):
		if hasattr(self, "_other"):
			return self._other

	@property
	def infall(self):
		if hasattr(self, "_infall"):
			return self._infall

	@property
	def infallexsat(self):
		if hasattr(self, "_infallexsat"):
			return self._infallexsat

	@property
	def cat3_cat1a(self):
		if hasattr(self, "_cat3_cat1a"):
			return self._cat3_cat1a 

	@property
	def cat3_cat1b(self):
		if hasattr(self, "_cat3_cat1b"):
			return self._cat3_cat1b

	@property
	def cat3_cat2a_pre(self):
		if hasattr(self, "_cat3_cat2a_pre"):
			return self._cat3_cat2a_pre 

	@property
	def cat3_cat2a_post(self):
		if hasattr(self, "_cat3_cat2a_post"):
			return self._cat3_cat2a_post 

	@property
	def cat3_cat2b(self):
		if hasattr(self, "_cat3_cat2b"):
			return self._cat3_cat2b 

	@property
	def output_merged(self):
		if hasattr(self, "_output_merged"):
			return self._output_merged
		self._output_merged = {}
		self._output_merged['alloi'] = self.merged_indices
		self._output_merged['allvel'] = self.merged_vel_indices%self.THIDVAL - 1
		self._output_merged['allsnap'] = (self.merged_vel_indices/self.THIDVAL).astype(int)
		self._output_merged['orbits'] = self.ow.orbitdata['numorbits'][self.merged_indices]
		self._output_merged['scalefactor_merged'] = self.ow.orbitdata['scalefactor'][self.merged_indices]
		self._output_merged['host'] = self.host_per_merged_index

		return self._output_merged

	@property
	def output_notmerged(self):
		if hasattr(self, "_output_notmerged"):
			return self._output_notmerged
		self._output_notmerged = {}
		self._output_notmerged['alloi'] = self.indices
		self._output_notmerged['allvel'] = self.vel_indices
		self._output_notmerged['host'] = self.host_per_index

		return self._output_notmerged

	@property
	def output_level1(self):
		if self.level1 == False:
			self.categorise_level1()

		#if not hasattr(self, "_output_level1"):
		self._output_level1 = {}
		self._output_level1['alloi'] = self.indices
		self._output_level1['allvel'] = self.vel_indices
		self._output_level1['other'] = self.other
		self._output_level1['first infall satellites'] = self.cat1a
		self._output_level1['orbital satellites'] = self.cat1b
		self._output_level1['ex-satellite'] = self.cat2a
		self._output_level1['orbital halo'] = self.cat2b
		self._output_level1['first infall'] = self.infall[self.infallexsat==False]
		self._output_level1['preprocessed infall'] = self.infall[self.infallexsat]
		if len(self.hosts) > 1:
			self._output_level1['host'] = self.host_per_index

		return self._output_level1



	@property
	def output_level2(self):
		if self.level2 == False:
			self.categorise_level2()

		if not hasattr(self, "_output_level2"):
			self._output_level2 = {}
			self._output_level2['first infall satellites'] = self.cat3_cat1a
			self._output_level2['orbital satellites'] = self.cat3_cat1b
			self._output_level2['ex-satellite'] = {}
			self._output_level2['ex-satellite']['preprocessed'] = self.cat3_cat2a_pre
			self._output_level2['ex-satellite']['postprocessed'] = self.cat3_cat2a_post
			self._output_level2['orbital halo'] = self.cat3_cat2b

		return self._output_level2

	@max_distance.setter
	def max_distance(self, max_distance):
		if self._max_distance == max_distance:
			return 0

		elif (self._max_distance is None) or (self._max_distance > max_distance):
			self._max_distance = max_distance
			self.cut_maximum_distance()
		elif self._max_distance < max_distance:
			self._max_distance = max_distance
			self.computeIndices()

		if self.level2 == True:
			self.level1 = False
			self.level2 = False
			self.categorise_level2()
		elif self.level1 == True:
			self.categorise_level1()

	@merged_indices.setter
	def merged_indices(self, merged_indices):
		del self._output_merged
		self._merged_indices = merged_indices

		self._host_per_merged_index = self.ow.orbitdata['OrbitedHaloRootProgen_orig'][self._merged_indices]
		self._merged_vel_indices  = np.array(self.ow.orbitdata['HaloID_orig'][self._merged_indices]).astype(int)

	@indices.setter
	def indices(self, indices):
		self.level1 = False
		self.level2 = False

		self._indices = indices
		self._host_per_index = self.ow.orbitdata['OrbitedHaloRootProgen_orig'][indices]
		if hasattr(self, "_vel_indices"):
			del self._vel_indices
		if hasattr(self, "_calculate_main_properties"):
			self.calculate_main_properties()
		if (hasattr(self, "reset_hosts")==False) or (self.reset_hosts == True):
			self.reset_indices = False
			self.hosts = np.array(list(set(self.ow.orbitdata['OrbitedHaloRootProgen_orig'][indices])))

		self.reset_hosts = True

	@hosts_init.setter
	def hosts_init(self, hosts):
		if self.hosts_init is None:
			self._hosts_init = hosts
	
	@hosts.setter
	def hosts(self, hosts):
		self.hosts_init = np.copy(hosts)
		self._hosts = hosts

		if (hasattr(self, "reset_indices")==False) or (self.reset_indices == True):
			if self.minhaloes != None and self.minhaloes > 0:
				niet = np.zeros(0).astype(int)
				indices = np.where((self.ow.orbitdata['entrytype']==0)&(self.ow.orbitdata['scalefactor']==1.0))[0]
				j = -1
				for i in self.hosts:
					j += 1
					if len(np.where(self.ow.orbitdata['OrbitedHaloRootProgen_orig'][indices] == i)[0]) < 10000:
						niet = np.append(niet, j)

				self._hosts = np.delete(self.hosts, niet)

			indices = np.zeros(0).astype(int)
			for i in self.hosts:	
				indices = np.append(indices, np.where((self.ow.orbitdata['entrytype']==0)&
					(self.ow.orbitdata['OrbitedHaloRootProgen_orig']==i)&(self.ow.orbitdata['scalefactor']==1.0))[0])

			self.reset_hosts = False
			self.indices = indices

		self.reset_indices = True

	@minhaloes.setter
	def minhaloes(self, value):
		if self.hosts_init is None:
			self._minhaloes = value
		elif value != self.minhaloes:
			self._minhaloes = value
			self.hosts = np.copy(np.array(self.hosts_init).astype(int))

	@velIDs.setter
	def velIDs(self, value):
		self._velIDs = value

	def find_vel_indices(self, ow_indices):
		self.ow.readOrbitData(desiredfields=['HaloID_orig'])
		vel_indices = np.array(self.ow.orbitdata['HaloID_orig'][ow_indices]).astype(int)%self.THIDVAL - 1
		return vel_indices

	def output_level1_host(self, host, overwrite_all=False, level2 = False):
		output_level1_host_temp = {}

		output_temp = self.output_level1
		if level2:
			output_level2_host_temp = {}
			output2_temp = self.output_level2

		if len(self.hosts) <= 1:
			return output_temp

		waar = np.where(output_temp['host'] == host)[0]
		if len(waar) == 0:
			if level2:
				return output_level1_host_temp, None, output_level2_host_temp
			return output_level1_host_temp, None
		r200 = self.r200_host[waar[0]]
		for key in output_temp.keys():
			if key in ['allvel', 'alloi']:
				continue
			if overwrite_all:
				booltemp = np.in1d(waar, output_temp[key])
				output_level1_host_temp[key] = np.where(booltemp)[0]
				if level2:
					booltemp = np.in1d(output_temp[key], waar)
					if isinstance(output2_temp[key], dict):
						for key2 in output2_temp[key].keys():
							if key not in output_level2_host_temp.keys():
								output_level2_host_temp[key] = {}
							output_level2_host_temp[key][key2] = output2_temp[key][key2][booltemp]
					else:
						output_level2_host_temp[key] = output2_temp[key][booltemp]
			else:
				booltemp = np.in1d(output_temp[key], waar)
				output_level1_host_temp[key] = output_temp[key][booltemp]
				if level2:
					if key not in output2_temp.keys():
						continue
					if isinstance(output2_temp[key], dict):
						for key2 in output2_temp[key].keys():
							if key not in output_level2_host_temp.keys():
								output_level2_host_temp[key] = {}
							output_level2_host_temp[key][key2] = output2_temp[key][key2][booltemp]
					else:
						output_level2_host_temp[key] = output2_temp[key][booltemp]

		if overwrite_all:
			output_level1_host_temp['allvel'] = output_temp['allvel'][waar]
			output_level1_host_temp['alloi'] = output_temp['alloi'][waar]
		else:
			output_level1_host_temp['allvel'] = output_temp['allvel']
			output_level1_host_temp['alloi'] = output_temp['alloi']

		if level2:
			return output_level1_host_temp, r200, output_level2_host_temp
		return output_level1_host_temp, r200

	def output_level1_subselection(self, indices, output_temp = None, overwrite_all=False, vel=False):
		output_level1_temp = {}
		if output_temp is None:
			output_temp = self.output_level1

		waar = np.where(np.in1d(self.vel_indices, indices))[0]
		for key in output_temp.keys():
			if key in ['allvel', 'alloi']:
				continue
			if overwrite_all:
				output_level1_temp[key] = np.where(np.in1d(waar, output_temp[key]))[0]
			else:
				output_level1_temp[key] = waar[np.in1d(waar, output_temp[key])]

		if overwrite_all:
			output_level1_host_temp['allvel'] = output_temp['allvel'][waar]
			output_level1_host_temp['alloi'] = output_temp['alloi'][waar]
		else:
			output_level1_host_temp['allvel'] = output_temp['allvel']
			output_level1_host_temp['alloi'] = output_temp['alloi']

		return output_level1_temp


	def set_velIDs(velpath='/mnt/sshfs/pleiades/MagnusData/9p/parent/Hydro/nonrad/VELOCIraptorDM_nieuw/snapshot_200/snapshot_200'):
		self.velpath = velpath
		vel = vpt.ReadPropertyFile(velpath, ibinary=2, 
			desiredfields=['ID'])[0]
		self._velIDs = vel['ID']

	def calculate_main_properties(self):
		if self.indices is not None:
			self._calculate_main_properties = True
			self._distance = np.sqrt(self.ow.orbitdata['Xrel'][self.indices]**2 + 
				self.ow.orbitdata['Yrel'][self.indices]**2 + self.ow.orbitdata['Zrel'][self.indices]**2)
			self._velocity = np.array([self.ow.orbitdata['VXrel'][self.indices], 
				self.ow.orbitdata['VYrel'][self.indices], self.ow.orbitdata['VZrel'][self.indices]])*-1 + h*100*self.distance
			self._vrad = (self.velocity[0, :]*(self.ow.orbitdata['Xrel'][self.indices]*-1)*Mpc_to_km + 
				self.velocity[1, :]*(self.ow.orbitdata['Yrel'][self.indices]*-1)*Mpc_to_km  + 
				self.velocity[2, :]*(self.ow.orbitdata['Zrel'][self.indices]*-1)*Mpc_to_km)/(self.distance*Mpc_to_km)
			self._dist_r200 = self.distance/self.r200_host
			self._ca_r200 = self.ow.orbitdata['closestapproach'][self.indices]/self.r200_host
			#self.find_first_eccentricity()
			if self.newversion:
				self._ca = self.ow.orbitdata['closestapproach'][self.indices] / self.ow.orbitdata['closestapproachscalefactor'][self.indices]
				self._ca_r200 /= self.ow.orbitdata['closestapproachscalefactor'][self.indices]

	def cut_maximum_distance(self):
		if self.newversion:
			self.cut_maximum_distance_ca()
			return 0
		if ((self.max_distance is not None) or (self.max_times_r200 is not None)) and (self.indices is not None):
			self.calculate_main_properties()

			if self.max_times_r200 is not None:
				wel = np.where(self._dist_r200 <= self.max_times_r200)[0]
			else:	
				wel = np.where(self.distance <= self.max_distance)[0]
			self.reset_hosts = False
			self.indices = self.indices[wel]

	def cut_maximum_distance_ca(self):
		if ((self.max_distance is not None) or (self.max_times_r200 is not None)) and (self.indices is not None):
			self.calculate_main_properties()

			if self.max_times_r200 is not None:
				wel = np.where(self.ca_r200 < self.max_times_r200)[0]
			else:
				wel = np.where(self.ow.orbitdata['closestapproach'][self.indices] <= self.max_distance)[0]
			self.reset_hosts = False
			self.indices = self.indices[wel]

	def find_first_eccentricity(self):
		waar = np.where(self.orbits >= 1)[0]
		self._eccentricity = np.zeros(len(self.indices))
		for i in waar:
			temp = np.where(self.ow.orbitdata['OrbitID'][self.indices[i]] == self.ow.orbitdata['OrbitID99'])[0]
			if len(temp) < 2:
				self._eccentricity[i] = -1
				continue
			i_ecc = (temp[np.argsort(self.ow.orbitdata['scalefactor99'][temp])])[1] #1 because measured at 0.5*2
			self._eccentricity[i] = self.ow.orbitdata['orbitecc_ratio'][i_ecc]

	def find_first_eccentricity_merged(self):
		self.output_merged
		waar = np.where(self.ow.orbitdata['numorbits'][self.output_merged['alloi']] >= 1)[0]
		self.output_merged['eccentricity'] = np.zeros(len(self.output_merged['alloi']))
		for i in waar:
			temp = np.where(self.ow.orbitdata['OrbitID'][self.output_merged['alloi'][i]] == self.ow.orbitdata['OrbitID99'])[0]
			if len(temp) < 2:
				self.output_merged['eccentricity'][i] = -1
				continue
			i_ecc = (temp[np.argsort(self.ow.orbitdata['scalefactor99'][temp])])[1] #1 because measured at 0.5*2
			self.output_merged['eccentricity'][i] = self.ow.orbitdata['orbitecc_ratio'][i_ecc]

	def categorise_level1(self, no_merger_with_self=True, ht=None, mass_dependent=True):

		"""	
		cat1a: first infalling satellites
		cat1b: orbital satellites

		cat2a: ex-satellites
		cat2b: orbital haloes

		"""
		if (ht is not None) and (hasattr(self, "ht") == False):
			self.ht = ht
		elif hasattr(self, "ht"):
			ht = self.ht


		if self._calculate_main_properties == False:
			self.calculate_main_properties()

		self._cat1a = np.where((self.dist_r200 <= 1)&((self.orbits==0)))[0]
		self._cat1b = np.where((self.dist_r200 <= 1)&((self.orbits!=0)))[0]
		exsats = np.where((self.dist_r200>1)&((self.orbits!=0)))[0]

		virradcross = {}
		#Finding all haloes that have ever crossed the virial halo of host i
		for i in self.hosts:
			virradcross[i] = np.where((np.abs(self.ow.orbitdata['entrytype']) == 1)&
				(self.ow.orbitdata['OrbitedHaloRootProgen_orig']==i))[0]
		exsatscross = np.array([False]*len(exsats)).astype(bool)
		j = -1
		for i in exsats:
			j += 1
			if len(np.where(self.ow.orbitdata['OrbitID'][self.indices[i]] == 
				self.ow.orbitdata['OrbitID'][virradcross[self.ow.orbitdata['OrbitedHaloRootProgen_orig'][self.indices[i]]]])[0]) > 0:
				exsatscross[j] = True

		self._cat2a = exsats[exsatscross]
		self._cat2b = exsats[exsatscross==False]
		if (no_merger_with_self == True) and (ht is not None):
			self.ow.readOrbitData(desiredfields=['OrbitedHaloID_orig'])
		
		infall = np.where((self.ca_r200 > 1)&(self.vrad<0)&(self.orbits<=0.5))[0]

		infallexsat = np.array([False]*len(infall)).astype(bool)

		#Finding all crossings
		entry0_excl = np.where(np.abs(self.ow.orbitdata['entrytype']) == 1)[0]
		
		#Finding mergers with self
		if (no_merger_with_self == True):
			#All rootdescendants of the crossing haloes
			roottails_exl = self.ow.orbitdata['HaloRootDescen_orig'][entry0_excl]
			#All rootdescendants of the hosts at crossing points
			roottails_host_exl = self.ow.orbitdata['OrbitedHaloRootDescen_orig'][entry0_excl]
			if mass_dependent and (self.npart_list is not None):
				#Crossing haloes
				## 1) Find all halo ids
				## 2) Find all haloes that have a RootDescen that still exists at z=0
				## 3) Find the number of particles within the halo at z=0
				npart_exl = np.zeros_like(roottails_exl)
				haloids = roottails_exl%self.THIDVAL - 1
				snaps = (roottails_exl/self.THIDVAL).astype(int)
				waartemp = np.where(snaps==self.last_snap)[0]
				npart_exl[waartemp] = self.npart_list[haloids[waartemp]]
				#Same for their hosts
				npart_host_exl = np.zeros_like(roottails_host_exl)
				haloids = roottails_host_exl%self.THIDVAL - 1
				snaps = (roottails_host_exl/self.THIDVAL).astype(int)
				waartemp = np.where(snaps==self.last_snap)[0]
				npart_host_exl[waartemp] = self.npart_list[haloids[waartemp]]					

			#finding all the instances when the orbit was actually orbiting a part of its past self
			if mass_dependent:
				#If the mass of the subhalo at z=0 is larger than the mass of the host, the subhalo is skipped
				temp = np.where((npart_host_exl.astype(int)-npart_exl.astype(int))<=0)[0]
				# temp = np.where(((roottails_exl.astype(int) - roottails_host_exl.astype(int))==0) | 
				# 	((npart_host_exl.astype(int)-npart_exl.astype(int))<=0))[0]
				print(len(temp), 'out of', len(entry0_excl), 'orbited something equal or smaller than themselves.')
			else:
				#If the rootdescen of the host and the halo it orbited is the same, the halo is skipped
				temp = np.where((roottails_exl.astype(int) - roottails_host_exl.astype(int))==0)[0]

				print(len(temp), 'out of', len(entry0_excl), 'orbited part of themselves.')
				
			entry0_excl = np.delete(entry0_excl, temp)

		#waar = np.where(self.ow.orbitdata['OrbitedHaloRootProgen_orig'][self.indices[infall]] == i)[0]
		infallexsat = np.in1d(self.ow.orbitdata['HaloRootProgen_orig'][self.indices[infall]],
			self.ow.orbitdata['HaloRootProgen_orig'][entry0_excl])

		self._infall = infall
		self._infallexsat = infallexsat
		self._other = np.arange(len(self.indices)).astype(int)
		self._other = np.delete(self.other, 
			np.append(self.cat1a, np.append(self.cat1b, np.append(self.cat2a, np.append(self.cat2b, infall)))))

		self.level1 = True


	def categorise_level2(self, no_merger_with_self=True, ht = None, mass_dependent=True):
		if (ht is not None) and (hasattr(self, "ht") == False):
			self.ht = ht
		elif hasattr(self, "ht"):
			ht = self.ht

		if self.level1 == False:
			self.categorise_level1(no_merger_with_self=no_merger_with_self, ht=ht, mass_dependent=mass_dependent)

		self._cat3_cat1a = np.array([False]*len(self.cat1a)).astype(bool)
		self._cat3_cat1b = np.array([False]*len(self.cat1b)).astype(bool)
		self._cat3_cat2a_pre = np.array([False]*len(self.cat2a)).astype(bool)
		self._cat3_cat2a_post = np.array([False]*len(self.cat2a)).astype(bool)
		self._cat3_cat2b = np.array([False]*len(self.cat2b)).astype(bool)
		
		if (no_merger_with_self == True) and (ht is not None) and (self.newversion == False):
			self.ow.readOrbitData(desiredfields=['OrbitedHaloID_orig'])

		#Everywhere it crossed the virial radius of something (including its own host)
		if self.newversion:
			virradcross_excl = np.where(np.abs(self.ow.orbitdata['entrytype']) == 1)[0]
		else:
			virradcross_excl = np.where((np.abs(self.ow.orbitdata['entrytype']) == 1)&
				(self.ow.orbitdata['HaloID_orig']!=0))[0]

		if (no_merger_with_self == True):
			if self.newversion:
				roottails_exl = self.ow.orbitdata['HaloRootDescen_orig'][virradcross_excl]
				roottails_host_exl = self.ow.orbitdata['OrbitedHaloRootDescen_orig'][virradcross_excl]
				if mass_dependent and (self.npart_list is not None):
					npart_exl = np.zeros_like(roottails_exl)
					haloids = roottails_exl%self.THIDVAL - 1
					snaps = (roottails_exl/self.THIDVAL).astype(int)
					waartemp = np.where(snaps==self.last_snap)[0]
					npart_exl[waartemp] = self.npart_list[haloids[waartemp]]

					npart_host_exl = np.zeros_like(roottails_host_exl)
					haloids = roottails_host_exl%self.THIDVAL - 1
					snaps = (roottails_host_exl/self.THIDVAL).astype(int)
					waartemp = np.where(snaps==self.last_snap)[0]
					npart_host_exl[waartemp] = self.npart_list[haloids[waartemp]]	

			elif (ht is not None):
				haloids, snaps = ht.key_halo(self.ow.orbitdata['HaloID_orig'][virradcross_excl])
				haloids -= 1
				roottails_exl = np.zeros_like(haloids)
				snapunique = np.unique(snaps)
				for snap in snapunique:
					if snap == 0:
						continue
					temp = np.where(snaps==snap)[0]
					roottails_exl[temp] = ht.halotree[snap].hp['RootHead'][haloids[temp]]
				if mass_dependent:
					haloids, snaps = ht.key_halo(roottails_exl)
					npart_exl = np.zeros_like(haloids)
					snapunique = np.unique(snap)
					for snap in snapunique:
						if snap == 0:
							continue
						temp = np.where(snaps==snap)[0]
						npart_exl[temp] = ht.halotree[snap].hp['Npart'][haloids[temp]]

				haloids, snaps = ht.key_halo(self.ow.orbitdata['OrbitedHaloID_orig'][virradcross_excl])
				haloids -= 1
				roottails_host_exl = np.zeros_like(haloids)
				snapunique = np.unique(snaps)
				for snap in snapunique:
					if snap == 0:
						continue
					temp = np.where(snaps==snap)[0]
					roottails_host_exl[temp] = ht.halotree[snap].hp['RootHead'][haloids[temp]]
				if mass_dependent:
					haloids, snaps = ht.key_halo(roottails_host_exl)
					npart_host_exl = np.zeros_like(haloids)
					snapunique = np.unique(snap)
					for snap in snapunique:
						if snap == 0:
							continue
						temp = np.where(snaps==snap)[0]
						npart_host_exl[temp] = ht.halotree[snap].hp['Npart'][haloids[temp]]

			if mass_dependent:
				temp = np.where(((roottails_exl.astype(int) - roottails_host_exl.astype(int))==0) | ((npart_host_exl.astype(int)-npart_exl.astype(int))<=0))[0]
			else:		
				temp = np.where((roottails_exl.astype(int) - roottails_host_exl.astype(int))==0)[0]

			print(len(temp), 'out of', len(virradcross_excl), 'orbited part of themselves.')
			#Deleting the orbits around themselves
			virradcross_excl = np.delete(virradcross_excl, temp)

		for i in self.hosts:
			#Getting the orbits that are not around host i
			virradcross_excl_host = virradcross_excl[self.ow.orbitdata['OrbitedHaloRootProgen_orig'][virradcross_excl]!=i]

			#Which ones crossed the virial radius of host i?
			virradcross = np.where((np.abs(self.ow.orbitdata['entrytype']) == 1)&
				(self.ow.orbitdata['OrbitedHaloRootProgen_orig']==i))[0]

			#Which first infall satellites have been preprocessed before entering host i?
			waar = np.where(self.ow.orbitdata['OrbitedHaloRootProgen_orig'][self.indices[self.cat1a]] == i)[0]
			self._cat3_cat1a[waar] = np.in1d(self.ow.orbitdata['HaloRootProgen_orig'][self.indices[self.cat1a][waar]], 
				self.ow.orbitdata['HaloRootProgen_orig'][virradcross_excl_host])

			#Which orbital satellites have been preprocessed before the first time entering host i?
			waar = np.where(self.ow.orbitdata['OrbitedHaloRootProgen_orig'][self.indices[self.cat1b]] == i)[0]
			for j in waar:
				temp = np.where(self.ow.orbitdata['OrbitID'][self.indices[self.cat1b][j]] == 
					self.ow.orbitdata['OrbitID'][virradcross])[0]
				if len(temp) == 0:
					continue
				sfmin = np.min(self.ow.orbitdata['scalefactor'][virradcross[temp]])
				if len(np.where((self.ow.orbitdata['HaloRootProgen_orig'][virradcross_excl_host] == 
					self.ow.orbitdata['HaloRootProgen_orig'][self.indices[self.cat1b][j]])&
					(self.ow.orbitdata['scalefactor'][virradcross_excl_host] < sfmin))[0]) > 0:
					self._cat3_cat1b[j] = True

			#Which ex-satellites are pre- and post processed?
			waar = np.where(self.ow.orbitdata['OrbitedHaloRootProgen_orig'][self.indices[self.cat2a]] == i)[0]
			for j in waar:
				temp = np.where(self.ow.orbitdata['OrbitID'][self.indices[self.cat2a][j]] == 
					self.ow.orbitdata['OrbitID'][virradcross])[0]
				if len(temp) == 0:
					continue
				sfmax = np.max(self.ow.orbitdata['scalefactor'][virradcross[temp]])
				sfmin = np.min(self.ow.orbitdata['scalefactor'][virradcross[temp]])
				if len(np.where((self.ow.orbitdata['HaloRootProgen_orig'][virradcross_excl_host] == 
					self.ow.orbitdata['HaloRootProgen_orig'][self.indices[self.cat2a][j]])&
					(self.ow.orbitdata['scalefactor'][virradcross_excl_host] < sfmin))[0]) > 0:
					self._cat3_cat2a_pre[j] = True
				if len(np.where((self.ow.orbitdata['HaloRootProgen_orig'][virradcross_excl_host] == 
					self.ow.orbitdata['HaloRootProgen_orig'][self.indices[self.cat2a][j]])&
					(self.ow.orbitdata['scalefactor'][virradcross_excl_host] > sfmax))[0]) > 0:
					self._cat3_cat2a_post[j] = True

			waar = np.where(self.ow.orbitdata['OrbitedHaloRootProgen_orig'][self.indices[self.cat2b]] == i)[0]
			self._cat3_cat2b[waar] = np.in1d(self.ow.orbitdata['HaloRootProgen_orig'][self.indices[self.cat2b][waar]],
				self.ow.orbitdata['HaloRootProgen_orig'][virradcross_excl_host])

			self.level2 = True

def select_hosts(hd, radius=4, minmass=0, maxmass=None):
	hd.readData(datasets=['M200', 'RootTail', 'Coord', 'R200'])
	hd.makeHaloCoordTree()
	
	possible_hosts = np.where(hd.hp['M200'] >= minmass)[0]
	sortorder = np.argsort(hd.hp['M200'])[::-1]
	mask = np.ones(len(hd.hp['M200'])).astype(bool)
	mask[hd.hp['M200'] < minmass] = False
	for i in range(len(possible_hosts)):
		haloes = np.array(hd.halocoordtree.query_ball_point(hd.hp['Coord'][sortorder[i]], r = radius*hd.hp['R200'][sortorder[i]])).astype(int)
		haloes = haloes[np.where(hd.hp['M200'][haloes] < hd.hp['M200'][sortorder[i]])[0]]
		mask[haloes] = False

	if maxmass is not None:
		mask[hd.hp['M200'] >= maxmass] = False

	return hd.hp['RootTail'][mask]+1, np.where(mask)[0]