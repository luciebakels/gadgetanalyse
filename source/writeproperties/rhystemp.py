import numpy as np
from writeproperties import *
import argparse
import re


class Params:
	def __init__(self, configfile):
		self.paths = {}
		self.runparams = {}
		self.parttypes = {}
		self.rad = {}
		self.boxsize = None
		self.halolistpath = None

		with open(configfile,"r") as f:

			for line in f:

				if line[0]=="#": 
					continue

				line = line.replace(" ","")

				line = line.strip()

				if not line:
					continue

				line = line.split("=")

				if line[0] in ['snappath', 'velpath', 'treepath', 'outputpath']:
					self.paths[line[0]] = line[1]

				elif line[0] in ['BoxSize', 'boxsize', 'Boxsize', 'boxSize']:
					self.boxsize = float(line[1])

				elif line[0] in ['SO', 'TreeData', 'Quantities', 'Profiles', 'KDTree', 'VELconvert', 'Snapshot', 'VELcopy']:
					if line[1] in  ['0', 'No', 'NO', 'n', 'N', 'False', 'F']:
						self.runparams[line[0]] = False
					elif line[1] in ['1', 'Yes', 'YES', 'y', 'Y', 'True', 'T']:
						self.runparams[line[0]] = True
					else:
						sys.exit("For", line[0], ", ", line[1], "is not good.")

				elif line[0] in ['NOutput', 'Nfiles', 'Nfilestart']:
					self.runparams[line[0]] = int(line[1])

				elif line[0] == 'ParticleDataType':
					self.runparams[line[0]] = line[1]

				elif line[0] == 'AllPartTypes':
					if line[1] in  ['0', 'No', 'NO', 'n', 'N', 'False', 'F']:
						self.runparams[line[0]] = False
					elif line[1] in ['1', 'Yes', 'YES', 'y', 'Y', 'True', 'T']:
						self.runparams[line[0]] = True
					else:
						sys.exit("For", line[0], ", ", line[1], "is not good.")

				elif line[0] in (['PartType0', 'PartType1',
					'PartType2', 'PartType3', 'PartType4', 'PartType5']):
					self.parttypes[line[0]] = line[1]

				elif line[0] in ['MaxRad', 'Rfrac']:
					self.rad[line[0]] = float(line[1])
				
				elif line[0] == 'Rchoice':
					self.rad[line[0]] = line[1]

				elif line[0] == 'HaloListPath':
					self.halolistpath =line[1]

		#if self.runparams['Profiles']:
		self.rad['profile'] = np.logspace(-3, np.log10(1.5), 60)#np.concatenate([np.logspace(-3, np.log10(0.5), 60), np.arange(0.6, self.rad['MaxRad'], 0.1)])

		self.d_partType = {}
		self.d_partType['particle_type'] = []
		self.d_partType['particle_number'] = np.array([]).astype(int)
		for key in self.parttypes.keys():
			if key == 'AllPartTypes':
				continue
			if self.parttypes[key] == 'None':
				continue
			self.d_partType['particle_number'] = np.append(self.d_partType['particle_number'], int(re.findall(r"\d", key)[0]))
			temp = ''
			if self.parttypes[key] == 'DM':
				temp = 'DM'
			elif self.parttypes[key] == 'Gas':
				temp = 'H'
			elif self.parttypes[key] == 'Star':
				temp = 'S'
			else:
				sys.exit("The particle type given is not correct: choose DM, Gas, or Star, not ", self.parttypes[key])
			self.d_partType['particle_type'].append(temp)

parser = argparse.ArgumentParser()
parser.add_argument("-c",action="store",dest="configfile",help="Configuration file (orbweaver.cfg)",required=True)
parser.add_argument("-s",action="store",dest="snapshot",help="Number of snapshot to analyse",required=True, type=int)
parser.add_argument("-q",action="store",dest="Nfiles",help="Number of subfiles in the snapshot to look over",required=False, type=int)
parser.add_argument("-n",action="store",dest="Nfilestart",help="Number of starting subfile number",required=False, type=int)
opt = parser.parse_args()
param = Params(opt.configfile)

halolist = None
if param.halolistpath is not None:
	halolist = np.loadtxt(param.halolistpath)
	halolist = halolist.astype(int)

if opt.Nfiles is not None:
	param.runparams['Nfiles'] = opt.Nfiles
if opt.Nfilestart is not None:
	param.runparams['Nfilestart'] = opt.Nfilestart

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


d_snap['File'] = Snapshot(param.paths['snappath'], opt.snapshot, d_partType = param.d_partType, 
	read_only_header=False, nfiles=None, nfilestart=None)

iparttypes = 0
if len(param.d_partType['particle_type']) >1:
	iparttypes = 1

partdata = ReadParticleDataFile(param.paths['velpath'] + 
	'/snapshot_%03d/snapshot_%03d' %(opt.snapshot, opt.snapshot), 
	ibinary=2, iparttypes=iparttypes, unbound=True)

indices = d_snap['File'].get_indices(IDs = partdata['Particle_IDs'][160])

coordinates = d_snap['File'].coordinates[indices]