import numpy as np
from writeproperties import *
import argparse
import re
import time


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


parser = argparse.ArgumentParser()
parser.add_argument("-p",action="store",dest="inputfolder",help="Path to folder containing all subfiles",required=True)
parser.add_argument("-o",action="store",dest="outputpath",help="Outputpath",required=False)
parser.add_argument("-s",action="store",dest="snapshot",help="Snapshot",required=False, type=int)
opt = parser.parse_args()

if opt.outputpath is None:
	opt.outputpath = opt.inputfolder + '/All.'


def read_profiles(path = '/home/luciebakels/DMO11/Profiles/'):
	filenames = []
	for filename in os.listdir(path):
		if filename.endswith(".hdf5") == False:
			continue
		filenames.append(filename)


	profiles = {}

	for filename in filenames:
		print(filename)
		haloprop = h5py.File(path+filename, 'r')
		if 'HaloID' in profiles:
			start_time = time.time()
			waarvervang = np.where(profiles['HaloID'] == -1)[0]
			print("--- %s seconds ---" % (time.time() - start_time), 'HaloID')
			start_time = time.time()
			npart = haloprop['Npart_profile'.encode('utf-8')][:]
			print("--- %s seconds ---" % (time.time() - start_time), 'npart')
			start_time = time.time()
			npartprevious = profiles['Npart_profile']
			print("--- %s seconds ---" % (time.time() - start_time), 'npartprevious')
			start_time = time.time()
			nietnul = np.where(npart+npartprevious>0)[0]
			npartnietnul = npart[nietnul] + npartprevious[nietnul]
			print("--- %s seconds ---" % (time.time() - start_time), 'nietnul')
		for key in haloprop.id:
			start_time = time.time()
			print(key)
			if key in ['AngularMomentum', 'Velrad']:
				temp = haloprop[key][:].astype(float32)
			else:
				temp = haloprop[key][:]
			if key.decode('utf-8') not in profiles:
				profiles[key.decode('utf-8')] = temp
			elif key.decode('utf-8') in ['Radius', 'MassTable']:
				continue
			elif key.decode('utf-8') in ['HaloID', 'HaloIndex']:
				profiles[key.decode('utf-8')][waarvervang] = temp[waarvervang]
			elif key.decode('utf-8') in ['AngularMomentum', 'Velrad']:
				stukje = temp[nietnul]*npart[nietnul]
				stukje += profiles[key.decode('utf-8')][nietnul]*npartprevious[nietnul]
				profiles[key.decode('utf-8')][nietnul] = (stukje)/npartnietnul
			else:
				profiles[key.decode('utf-8')] += haloprop[key]
			print("--- %s seconds ---" % (time.time() - start_time), 'key')
		haloprop.close()
	return profiles

profiles = read_profiles(path=opt.inputfolder)

profilename = 'snapshot_%03d.profiles.hdf5' %opt.snapshot
writeDataToHDF5profiles(opt.outputpath, profilename, profiles, overwrite=False)