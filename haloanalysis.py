import numpy as np
from writeproperties import *

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


filenames = np.genfromtxt(sys.argv[1], dtype=None)
indirsnap = filenames[0].decode('utf-8')
indirvel = filenames[1].decode('utf-8')
outdir = filenames[2].decode('utf-8')
snapstart = int(filenames[3].decode('utf-8'))
snapend = int(filenames[4].decode('utf-8'))
softeningLength = float(filenames[5].decode('utf-8'))
Hydro = int(filenames[6].decode('utf-8'))
Cycle = int(filenames[7].decode('utf-8'))
r200fac = float(filenames[8].decode('utf-8'))
tree = int(filenames[9].decode('utf-8'))
particledata = int(filenames[10].decode('utf-8'))
fixingstuff = int(filenames[11].decode('utf-8'))
savePartData = int(filenames[12].decode('utf-8'))
TEMPORALHALOIDVAL = 1000000000000
if Hydro > 0:
	Hydro = True
mass=False
if fixingstuff == 1:
	mass=True

partlim = 1

if tree: #Fixed radius, saving tails and heads
	print("Reading unified tree and halo catalog...")
	# atime,tree,numhalos,halodata,cosmodata,unitdata = vpt.ReadUnifiedTreeandHaloCatalog(indirvel, desiredfields=['R_200crit', 
	# 	'Xc', 'Yc', 'Zc', 'Xcmbp', 'Ycmbp', 'Zcmbp', 'Mass_tot', 'hostHaloID', 'ID', 'npart', 'Mass_FOF', 'Mass_200crit', 'Rmax'], 
	# 	icombinedfile=1,iverbose=1)
	halodata = vpt.ReadPropertyFile(indirvel, ibinary=2)[0]
	#print("Rewriting heads and tails...")
	#sortorder, newtail, newhead = rewriteHeadTails(halodata)
	for snap in range(snapstart, snapend+1):
		print("snapshot_%03d: Computing halo properties ..." %snap)
		haloes = len(halodata['R_200crit'])#[200-snap]

		#tails = halodata[snap]['Tail'][:]

		haloproperties = findHaloPropertiesInSnap_fromUnifiedTreeCatalog(halodata, indirsnap, snap, #[200-snap]
			Radius=r200fac, Hydro=Hydro, Nhalo=haloes, startHalo=0, softeningLength=softeningLength, mass=mass,
			sortorder=[], partlim=partlim)

		print("Writing data...")
		writeDataToHDF5(outdir, 'snapshot_%03d.info.hdf5' %snap, haloproperties, overwrite=False, savePartData=False)
	print("Finished")

else:
	for snap2 in range(snapstart, snapend+1):
		print("snapshot_%03d: Computing halo properties ..." %snap2)
		snap = snap2
		# if fixingstuff == 1 and Hydro==True:
		# 	d_snap = {}
		# 	d_snap['snapshot'] = snap
		# 	d_snap['File'] = Snapshot(indirsnap, snap, useIDs=False, partType=7, softeningLength=softeningLength)
		# 	hp = readHDF5Data(outdir, 'snapshot_%03d.info.hdf5' %snap, Hydro=Hydro)
		# 	getRidOfBadHaloes(hp)
		# 	everythingOutside(hp, d_snap)
		# 	writeDataToHDF5(outdir, 'snapshot_%03d.info.hdf5' %snap, hp, overwrite=True)
		# 	continue
		print("Reading " + indirvel + "/snapshot_%03d/snapshot_%03d" %(snap,snap))
		catalog, numtot, atime = vpt.ReadPropertyFile(indirvel + '/snapshot_%03d/snapshot_%03d' %(snap,snap), ibinary=2)
		partdata = []
		if particledata==1:
			partdata = vpt.ReadParticleDataFile(indirvel + '/snapshot_%03d/snapshot_%03d' %(snap, snap), ibinary=2)
		if Cycle > 0:
			for i in range(200):
				if catalog['npart'][i*Cycle] < 100:
					break
				if i > len(catalog['R_200crit']):
					break
				haloes = Cycle
				hp = findHaloPropertiesInSnap(catalog, indirsnap, snap, particledata=partdata, 
					Hydro=True, Nhalo=haloes, startHalo=i*Cycle, softeningLength=softeningLength, 
					r200fac=r200fac, mass=mass, partlim=partlim)
				writeDataToHDF5(outdir, 'snapshot_%03d.info.hdf5' %snap, hp, overwrite = False)
		else:
			haloes = len(catalog['R_200crit'])
			hp = findHaloPropertiesInSnap(catalog, indirsnap, snap, particledata=partdata, 
				Hydro=Hydro, Nhalo=haloes, startHalo=0, r200fac=r200fac, mass=mass, 
				savePartData=savePartData, partlim=partlim)
			print("Writing data...")
			writeDataToHDF5(outdir, 'snapshot_%03d.info.hdf5' %snap, hp, overwrite=False, savePartData=savePartData)
			print("Finished!")