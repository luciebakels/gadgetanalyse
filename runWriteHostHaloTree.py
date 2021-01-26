import numpy as np
import argparse
import sys
import os
from scipy.spatial import cKDTree
import h5py
import re
from constants import *
from snapshot import *
import haloanalyse as ha
import orbweaverdata as ow
from writeHostHaloTrees import *

class Unbuffered(object):
	"""Copied from stackoverflow: flushes prints on HPC facilities
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

class Params:
	"""Class to read and store input parameters

	Attributes
	----------
	paths : dict
		dictionary containing the OrbWeaver, VELOCIraptor, and output path
	runparams : dict
		dictionary containing run parameters

	"""
	def __init__(self, configfile):
		"""
		Parameters
		----------
		configfile : str
			location of the configuration file
		"""
		self.paths = {}
		self.runparams = {}
		self.runparams['Maxmass'] = None
		self.paths['Output_name'] = 'HaloTrees_'
		with open(configfile,"r") as f:

			for line in f:

				if line[0]=="#": 
					continue

				line = line.replace(" ","")

				line = line.strip()

				if not line:
					continue

				line = line.split("=")

				if line[0] in ['Orbweaver_path', 'Velcopy_path', 'TreeFrog_path', 'Output_path', 'Output_name']:
					self.paths[line[0]] = line[1]

				elif line[0] in ['Snapend', 'Massbins', 'Minneighbours']:
					self.runparams[line[0]] = int(line[1])

				elif line[0] in ['Boxsize', 'Zstart', 'Minmass', 'Maxmass']:
					self.runparams[line[0]] = float(line[1])

				elif line[0] in ['VELOCIraptor']:
					self.runparams[line[0]] = bool(line[1])
					print(self.runparams['VELOCIraptor'])

#Reading command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-c",action="store",dest="configfile",help="Configuration file (hostHaloTree.cfg)",required=True)
opt = parser.parse_args()

#Reading the passed configuration file
param = Params(opt.configfile)
orbpath = param.paths['Orbweaver_path']
htpath = param.paths['Velcopy_path']
tfpath = param.paths['TreeFrog_path']
outpath = param.paths['Output_path']
outname = param.paths['Output_name']
boxsize = param.runparams['Boxsize']
snapend = param.runparams['Snapend']
zstart = param.runparams['Zstart']
massbins = param.runparams['Massbins']
minmass = param.runparams['Minmass']
maxmass = param.runparams['Maxmass']
min_neighbours = param.runparams['Minneighbours']

print("Read orbitdata")
owd = ow.OrbWeaverData(orbpath)
owd.readOrbitData()

print("Read halo data")
ht = ha.HaloTree(htpath, new_format=True, physical = False, boxsize=boxsize, zstart=zstart, snapend=snapend, 
	VELOCIraptor=param.runparams['VELOCIraptor'], TreeFrog_path=tfpath)
ht.readData(datasets=['RootHead', 'RootTail', 'Tail', 'Head', 'hostHaloIndex', 'hostHaloID', 'M200', 'npart', 
	'Vel', 'R200', 'Vmax', 'Rmax', 'Coord', 'Efrac', 'cNFW'])
for i in ht.halotree.keys():
	ht.halotree[i].hp['redshift'] = [ht.halotree[i].redshift]
	if ((ht.halotree[i].hp['hostHaloIndex'] is None) or 
		(len(ht.halotree[i].hp['hostHaloIndex']) != len(ht.halotree[i].hp['hostHaloID']))):
		print("Make hostHaloIndices for snapshot", i)
		ht.halotree[i].hp['hostHaloIndex'] = np.copy(ht.halotree[i].hp['hostHaloID'])
		fixSatelliteProblems(ht.halotree[i].hp, boxsize=boxsize)
		ht.halotree[i].addData(datasets=['hostHaloIndex'])

print("Selecting isolated hosts")
hosts_all, mainhaloes_all = ow.select_hosts(ht.halotree[ht.snapend], minmass=minmass, maxmass=maxmass, radius = 8)
masses_all = ht.halotree[ht.snapend].hp['M200'][mainhaloes_all]
mass_bins = np.logspace(np.log10(np.min(masses_all)), np.log10(np.max(masses_all+0.1)), massbins)
print("Selected hosts: ", len(hosts_all), ", divided in: ", len(mass_bins), " bins.")
k = -1
for i in range(len(mass_bins) - 1):
	k += 1
	if k >= len(mass_bins)-1:
		break
	waar_temp = np.where((masses_all >= mass_bins[k]) & (masses_all < mass_bins[k+1]))[0]
	if len(waar_temp) == 0:
		continue
	j = 0
	while (len(waar_temp) == 1) and (k+1+j <= len(mass_bins)):
		j += 1
		waar_temp = np.where((masses_all >= mass_bins[k]) & (masses_all < mass_bins[k+1+j]))[0]
	k += j
	print("Processing hosts between", mass_bins[k-j], "and", mass_bins[k+1], ":", len(waar_temp))
	hosts_temp = np.array(hosts_all[waar_temp])
	mainhaloes_temp = np.array(mainhaloes_all[waar_temp])
	oi = ow.OrbitInfo(owd, hosts=hosts_temp, physical=False, max_times_r200=4, 
		npart_list=ht.halotree[ht.snapend].hp['npart']*ht.halotree[ht.snapend].hp['Efrac'], 
		skeleton_only=True)
	#Efrac_cut(oi, ht.halotree[ht.snapend].hp, Efrac_limit=0.8)

	#oi.categorise_level1(no_merger_with_self=True, mass_dependent=True)
	mh_tree, nbtree, nbtree_merger, hosts, mainhaloes = find_all_values_crossing(ht, oi, 
		hosts_temp, mainhaloes_temp, min_neighbours=min_neighbours)
	print("Writing %i out of %i files"%(k+1, len(mass_bins)-1))
	writeDataToHDF5(outpath, outname+'%i.hdf5'%i, ht, hosts, mainhaloes, mh_tree, 
		oi.output_notmerged, oi.output_merged, nbtree, nbtree_merger)