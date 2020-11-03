import numpy as np
from writeproperties import *
import argparse
import re

velcopy_list = (['Mass_200crit', 'R_200crit', 'R_HalfMass', 'R_size', 'Efrac', 'Ekin', 'Epot', 'ID', 'Rmax', 'Vmax', 'SO_Mass_500_rhocrit',
	'SO_R_500_rhocrit', 'hostHaloID', 'Xcminpot', 'Ycminpot', 'Zcminpot', 'Xc', 'Yc', 'Zc', 'VXc', 'VYc', 'VZc', 'npart', 
	'VXcminpot', 'VYcminpot', 'VZcminpot', 's', 'q', 'Krot', 'lambda_B', 'Lx', 'Ly', 'Lz'])
velcopy_list_gas = (['Mass_200crit_gas', 'SO_Mass_gas_500_rhocrit', 'n_gas', 'Efrac_gas', 'Krot_gas', 'q_gas', 's_gas', 'Lx_gas', 'Ly_gas', 'Lz_gas'])
debug = False

class Params:
	def __init__(self, configfile):
		self.paths = {}
		self.runparams = {}
		self.parttypes = {}
		self.rad = {}
		self.boxsize = None
		self.halolistpath = None

		self.runparams['Physical'] = False

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

				elif line[0] in ['SO', 'TreeData', 'Quantities', 'Profiles', 'KDTree', 'Snapshot', 'Physical']:
					if line[1] in  ['0', 'No', 'NO', 'n', 'N', 'False', 'F']:
						self.runparams[line[0]] = False
					elif line[1] in ['1', 'Yes', 'YES', 'y', 'Y', 'True', 'T']:
						self.runparams[line[0]] = True
					else:
						sys.exit("For", line[0], ", ", line[1], "is not good.")
				elif line[0] == 'SnapshotType':
					self.runparams[line[0]] = line[1]

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
		self.rad['profile'] = np.logspace(-3, np.log10(1), 61)#np.concatenate([np.logspace(-3, np.log10(0.5), 60), np.arange(0.6, self.rad['MaxRad'], 0.1)])

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
parser.add_argument("-o",action="store",dest="Output",help="Output path",required=False, type=str)
parser.add_argument("-f",action="store",dest="Haloes",help="Halo ID file",required=False, type=str)
opt = parser.parse_args()
param = Params(opt.configfile)

if opt.Haloes is not None:
    param.halolistpath = opt.Haloes
halolist = None
if param.halolistpath is not None:
	halolist = np.loadtxt(param.halolistpath)
	halolist = halolist.astype(int)

if opt.Nfiles is not None:
	param.runparams['Nfiles'] = opt.Nfiles
if opt.Nfilestart is not None:
	param.runparams['Nfilestart'] = opt.Nfilestart
if opt.Output is not None:
    param.paths['outputpath'] = opt.Output
print(param.paths['outputpath'],halolist[1])

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

if 'H' in param.d_partType['particle_type']:
	for key in velcopy_list_gas:
		velcopy_list.append(key)


print("Reading VELOCIraptor catalog")

selected_files = None

catalog, haloes, atime = vpt.ReadPropertyFile(param.paths['velpath'] + 
	'/snapshot_%03d/snapshot_%03d' %(opt.snapshot,opt.snapshot), desiredfields=velcopy_list,
	selected_files=selected_files, halolist=halolist)

print("Opening snapshot_%03d" %(opt.snapshot+1))
d_snap = {}
d_snap['snapshot'] = opt.snapshot+1
read_only_header = False
if param.runparams['VELconvert']:
	read_only_header = True
else:
	read_only_header = False
	param.runparams['Snapshot'] = True

if param.runparams['Snapshot']:
	if 'Nfiles' in param.runparams:
		nfiles = param.runparams['Nfiles']
	else:
		nfiles = None
	if 'Nfilestart' in param.runparams:
		nfilestart = param.runparams['Nfilestart']
	else:
		nfilestart = None
	print(nfiles)
	#nfiles = None
	d_snap['File'] = Snapshot(param.paths['snappath'], opt.snapshot+1, d_partType = param.d_partType, 
		read_only_header=read_only_header, nfiles=nfiles, nfilestart=nfilestart, physical=param.runparams['Physical'],
		snapshottype=param.runparams['SnapshotType'])
	d_snap['redshift'] = d_snap['File'].redshift
	print('atime:', atime, 1/(1.+d_snap['redshift']))
	if round(atime, 3) != round(1./(1.+d_snap['redshift']), 3):
		print('atime:', atime, 1/(1.+d_snap['redshift']))
		sys.exit("Incorrect snapshot - VR catalogue combination")
	boxsize = d_snap['File'].boxsize
	print('Boxsize: ', boxsize)
else:
	d_snap['redshift'] = 1./atime - 1.
	d_snap['File'] = {}
	if param.boxsize is None:
		sys.exit('No boxsize present in .cfg file')
	else:
		boxsize = param.boxsize

partdata = None
if (param.runparams['VELconvert'] == False) & (param.runparams['KDTree'] == True):
	print("Building coordinate tree...")
	d_snap['File'].makeCoordTree()
if param.runparams['ParticleDataType'] == 'FOF' and len(catalog['Mass_200crit'])>0:
	print("Reading particle data")
	iparttypes = 0
	if len(param.d_partType['particle_type']) >1:
		iparttypes = 1
	partdata = ReadParticleDataFile(param.paths['velpath'] + 
		'/snapshot_%03d/snapshot_%03d' %(opt.snapshot, opt.snapshot), 
		iparttypes=iparttypes, unbound=True, selected_files=selected_files)
elif param.runparams['ParticleDataType'] == 'Bound' and len(catalog['Mass_200crit'])>0:
	print("Reading particle data")
	iparttypes = 0
	if len(param.d_partType['particle_type']) >1:
		iparttypes = 1
	partdata = ReadParticleDataFile(param.paths['velpath'] + 
		'/snapshot_%03d/snapshot_%03d' %(opt.snapshot, opt.snapshot), 
		iparttypes=iparttypes, unbound=False, selected_files=selected_files)
if param.runparams['TreeData']:
	print("Reading walkable tree")
	tree, numsnaps = ReadWalkableHDFTree(param.paths['treepath'], False)
	catalog['Head'] = tree[opt.snapshot]['Head']
	catalog['Tail'] = tree[opt.snapshot]['Tail']
	catalog['RootHead'] = tree[opt.snapshot]['RootHead']
	catalog['RootTail'] = tree[opt.snapshot]['RootTail']

Nhalo = int(haloes)
if halolist is not None:
	Nhalo = len(halolist)
haloproperties = findHaloPropertiesInSnap_nieuw(catalog, d_snap, d_runparams = param.runparams, halolist=halolist,
	partdata=partdata, d_radius=param.rad, d_partType=param.d_partType, Nhalo=Nhalo, startHalo=0, 
	boxsize=boxsize, debug=debug)


print("Writing data...")

quantityname = 'snapshot_%03d.quantities.hdf5' %opt.snapshot
profilename = 'snapshot_%03d.profiles.hdf5' %opt.snapshot

if 'Nfilestart' in param.runparams:
	quantityname = 'snapshot_%03d.%i.quantities.hdf5' %(opt.snapshot, param.runparams['Nfilestart'])
	profilename = 'snapshot_%03d.%i.profiles.hdf5' %(opt.snapshot, param.runparams['Nfilestart'])

if param.runparams['Quantities'] or param.runparams['VELconvert']:
	writeDataToHDF5quantities(param.paths['outputpath'], quantityname, 
		haloproperties, overwrite=False, convertVel = param.runparams['VELconvert'])

if param.runparams['Profiles'] and (param.runparams['VELconvert']==False):
	writeDataToHDF5profiles(param.paths['outputpath'], profilename, haloproperties, overwrite=True)

print("Finished")
