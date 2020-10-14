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
	def __init__(self, configfile):
		self.paths = {}
		self.runparams = {}

		with open(configfile,"r") as f:

			for line in f:

				if line[0]=="#": 
					continue

				line = line.replace(" ","")

				line = line.strip()

				if not line:
					continue

				line = line.split("=")

				if line[0] in ['Orbweaver_path', 'Velcopy_path', 'Output_path']:
					self.paths[line[0]] = line[1]

				elif line[0] in ['Snapend', 'Massbins']:
					self.runparams[line[0]] = int(line[1])

				elif line[0] in ['Boxsize', 'Zstart', 'Minmass']:
					self.runparams[line[0]] = float(line[1])


parser = argparse.ArgumentParser()
parser.add_argument("-c",action="store",dest="configfile",help="Configuration file (hostHaloTree.cfg)",required=True)
opt = parser.parse_args()
param = Params(opt.configfile)

orbpath = param.paths['Orbweaver_path']
htpath = param.paths['Velcopy_path']
outpath = param.paths['Output_path']
boxsize = param.runparams['Boxsize']
snapend = param.runparams['Snapend']
zstart = param.runparams['Zstart']
massbins = param.runparams['Massbins']
minmass = param.runparams['Minmass']

print("Read orbitdata")
owd = ow.OrbWeaverData(orbpath)
owd.readOrbitData()

print("Read halo data")
ht = ha.HaloTree(htpath, new_format=True, physical = True, boxsize=boxsize, zstart=zstart, snapend=snapend)
ht.readData(datasets=['RootHead', 'RootTail', 'Tail', 'Head', 'hostHaloIndex', 'hostHaloID', 'M200', 'npart', 'Vel', 
	'R200', 'Vmax', 'Rmax', 'Coord', 'Efrac'])
for i in ht.halotree.keys():
	ht.halotree[i].hp['redshift'] = [ht.halotree[i].redshift]
	if len(ht.halotree[i].hp['hostHaloIndex']) != len(ht.halotree[i].hp['hostHaloID']):
		print("Make hostHaloIndices for snapshot", i)
		ht.halotree[i].hp['hostHaloIndex'] = np.copy(ht.halotree[i].hp['hostHaloID'])
		fixSatelliteProblems(ht.halotree[i], hp, boxsize=boxsize)
		ht.halotree[i].addData(datasets=['hostHaloIndex'])

print("Selecting isolated hosts")
hosts_all, mainhaloes_all = ow.select_hosts(ht.halotree[ht.snapend], minmass=minmass, radius = 8)
masses_all = ht.halotree[i].hp['M200'][mainhaloes_all]
mass_bins = np.logspace(np.log10(np.min(masses_all)), np.log10(np.max(masses_all)*1.1), massbins)

for i in range(len(mass_bins) - 1):
	waar_temp = np.where((masses_all >= mass_bins[i]) & (masses_all < mass_bins[i+1]))[0]
	if len(waar_temp) == 0:
		continue
	print("Processing hosts between", mass_bins[i], "and", mass_bins[i+1], ":", len(waar_temp))
	hosts_temp = np.array(hosts_all[waar_temp])
	mainhaloes_temp = np.array(mainhaloes_all[waar_temp])
	oi = ow.OrbitInfo(owd, hosts=hosts_temp, physical=True, max_times_r200=4, npart_list=ht.halotree[ht.snapend].hp['npart'])
	Efrac_cut(oi, ht.halotree[ht.snapend].hp, Efrac_limit=0.8)

	oi.categorise_level1(no_merger_with_self=True, mass_dependent=True)
	d_output, mh_tree, nbtree, nbtree_merger, mask_vel_indices, hosts, mainhaloes = find_all_values_crossing(ht, oi, hosts_temp, mainhaloes_temp)
	print("Writing %i out of %i files"%(i+1, len(mass_bins-1)))
	writeDataToHDF5(outpath, 'HaloTrees_%i.hdf5'%i, ht, hosts, mainhaloes, mh_tree, d_output, oi.output_merged, nbtree, nbtree_merger, mask_vel_indices=mask_vel_indices)