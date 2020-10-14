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

				if line[0] in ['Velcopy_path']:
					self.paths[line[0]] = line[1]

				elif line[0] in ['Snapend']:
					self.runparams[line[0]] = int(line[1])

				elif line[0] in ['Boxsize', 'Zstart']:
					self.runparams[line[0]] = float(line[1])


parser = argparse.ArgumentParser()
parser.add_argument("-c",action="store",dest="configfile",help="Configuration file (hostHaloTree.cfg)",required=True)
parser.add_argument("-s",action="store",dest="snapshot",help="Number of snapshot to analyse",required=True, type=int)
opt = parser.parse_args()
param = Params(opt.configfile)

hdpath = param.paths['Velcopy_path']
boxsize = param.runparams['Boxsize']
snapend = param.runparams['Snapend']
zstart = param.runparams['Zstart']

print("Read halo data")
hd = ha.HaloData(hdpath, 'snapshot_%03d.quantities.hdf5'%opt.snapshot, 
	snapshot=opt.snapshot, new=True, TEMPORALHALOIDVAL=1000000000000, 
	boxsize=boxsize, extra=True, physical=False, totzstart=zstart, totnumsnap=snapend)

hd.readData(datasets=['hostHaloIndex', 'hostHaloID', 'M200', 'npart', 'R200', 'Coord'])
if (hd.hp['hostHaloIndex'] is None) or (len(hd.hp['hostHaloIndex']) != len(hd.hp['hostHaloID'])):
	print("Make hostHaloIndices for snapshot", opt.snapshot)
	hd.hp['hostHaloIndex'] = np.copy(hd.hp['hostHaloID'])
	fixSatelliteProblems(hd.hp, boxsize=boxsize)
	hd.rewriteData(datasets=['hostHaloIndex'])

print("Finished")