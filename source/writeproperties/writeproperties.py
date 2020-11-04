import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import fsolve, brentq
from scipy.interpolate import interp1d
import scipy.integrate
import sys
import os
import velociraptor_python_tools as vpt
from scipy.spatial import cKDTree
import h5py
import re
from constants import *
from snapshot import *
import copy
import itertools

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

def getHaloCoord(catalog, halo, z=0, snapshottype='GADGET', physical=False): #Mpc/h
	coords = np.zeros(3)
	if (('Xcminpot' not in catalog.keys())):# or
		# (np.abs(catalog['Xcminpot'][halo])>0.1) or
		# (np.abs(catalog['Ycminpot'][halo])>0.1) or
		# (np.abs(catalog['Zcminpot'][halo])>0.1)):
		return getHaloCoordCOM(catalog, halo, z=z, snapshottype=snapshottype, physical=physical)
	if physical:
		coords[0] = (catalog['Xcminpot'][halo])
		coords[1] = (catalog['Ycminpot'][halo])
		coords[2] = (catalog['Zcminpot'][halo])
	elif snapshottype in ['GADGET', 'Gadget', 'gadget']:
		coords[0] = (catalog['Xcminpot'][halo])*h*(1+z)
		coords[1] = (catalog['Ycminpot'][halo])*h*(1+z)
		coords[2] = (catalog['Zcminpot'][halo])*h*(1+z)
	elif snapshottype in ['SWIFT', 'Swift', 'swift']:
		coords[0] = (catalog['Xcminpot'][halo])*(1+z)
		coords[1] = (catalog['Ycminpot'][halo])*(1+z)
		coords[2] = (catalog['Zcminpot'][halo])*(1+z)
	else:
		print('Snapshottype not set')
	return coords

def getHaloRadius(catalog, halo, z=0, rtype='R_200crit', snapshottype='GADGET', physical=False): #Mpc/h
	if physical:
		return catalog[rtype][halo]
	elif snapshottype in ['GADGET', 'Gadget', 'gadget']:
		return catalog[rtype][halo]*h*(1+z)
	elif snapshottype in ['SWIFT', 'Swift', 'swift']:
		return catalog[rtype][halo]*(1+z)

def getHaloCoordCOM(catalog, halo, z=0, snapshottype='GADGET', physical=False): #Mpc/h
	coords = np.zeros(3)
	if physical:
		coords[0] = catalog['Xc'][halo]
		coords[1] = catalog['Yc'][halo]
		coords[2] = catalog['Zc'][halo]
	elif snapshottype in ['GADGET', 'Gadget', 'gadget']:
		coords[0] = catalog['Xc'][halo]*h*(1+z)
		coords[1] = catalog['Yc'][halo]*h*(1+z)
		coords[2] = catalog['Zc'][halo]*h*(1+z)
	elif snapshottype in ['SWIFT', 'Swift', 'swift']:
		coords[0] = catalog['Xc'][halo]*(1+z)
		coords[1] = catalog['Yc'][halo]*(1+z)
		coords[2] = catalog['Zc'][halo]*(1+z)
	return coords

def readHaloFile(halofile):
	atime,tree,numhalos,halodata,cosmodata,unitdata = vpt.ReadUnifiedTreeandHaloCatalog(halofile, desiredfields=[], icombinedfile=1,iverbose=0)
	return atime,tree,numhalos,halodata,cosmodata,unitdata

def findSurroundingHaloProperties(hp, halolist, d_snap, boxsize=32.):
	coords = hp['Coord']
	halotree = cKDTree(coords, boxsize=boxsize)
	for k in halolist:
		if hp['R200'][k] == -1:
			continue
		halostring = hp['HaloIndex'][k]

		length_of_neighbours = len(np.array(halotree.query_ball_point([hp['Coord'][k]], r=hp['R200'][k]*5)[0]))
		distance, indices = halotree.query([hp['Coord'][k]], k=length_of_neighbours)
		indices = np.array(indices[0])[1:]
		distance = np.array(distance[0])[1:]
		hp['Neighbours'][halostring] = hp['HaloIndex'][indices]
		hp['Neighbour_distance'][halostring] = distance
		hp['Neighbour_Velrad'][halostring] = np.zeros(len(distance))
		j=0
		for i in indices:
			partindices = hp['Partindices'][hp['HaloIndex'][i]]
			hp['Neighbour_Velrad'][halostring][j] = np.sum(d_snap['File'].get_radialvelocity(hp['Coord'][k], indices=partindices))/len(partindices)
			j+=1

def fixSatelliteProblems(hp, TEMPORALHALOIDVAL=1000000000000, boxsize=32):
	welke = np.where(hp['Coord'][:, 0] >= 0)[0]
	halotree = cKDTree(hp['Coord'][welke], boxsize=boxsize)

	toolarge = welke[np.where(hp['R200'][welke] > hp['R200'][np.argmax(hp['n_part'])]*1.2)[0]]
	#print(i, toolarge)
	if len(toolarge) != 0:
		for tl in toolarge:
			hp['M200'][tl] = -1
			hp['R200'][tl] = -1
			hp['hostHaloIndex'][hp['HaloIndex'][tl]==hp['hostHaloIndex']] = -2

	for halo in welke:#range(len(hp['M200'])):
		if hp['M200'][halo] == -1:
			continue
		buren = np.array(halotree.query_ball_point(hp['Coord'][halo], r = 2*hp['R200'][halo]))
		if len(buren) <= 1:
			continue
		buren = buren[hp['R200'][buren] != -1]
		if len(buren) == 0:
			continue
		i_largest = np.argmax(hp['n_part'][buren])
		index_largest = buren[i_largest]
		buren = np.delete(buren,i_largest)

		coords = hp['Coord'][buren] - hp['Coord'][index_largest]
		coords = np.where(np.abs(coords) > 0.5*boxsize, coords - coords/np.abs(coords)*boxsize, coords)
		rad = np.sqrt(np.sum(coords*coords, axis=1))
		burentemp = np.where(hp['R200'][buren]-rad+hp['R200'][index_largest] > 0)[0]
		if len(burentemp) == 0:
			continue
		buren = buren[burentemp]

		hp['hostHaloIndex'][buren] = index_largest
		hp['M200'][buren] = -1
		hp['R200'][buren] = -1

def findSubHaloFraction(hp, catalog):
	if len(hp['hostHaloIndex']) < 10:
		hp['Msub'] = np.zeros(len(hp['M200']))
		return 0
	i_hostH = np.where(hp['hostHaloIndex'] > -1)[0]
	hp['Msub'] = np.zeros(len(hp['M200']))
	for i in i_hostH:
		isattemp = np.where(hp['HaloID'][i] == catalog['ID'])[0]
		hp['Msub'][hp['hostHaloIndex'][i]] += catalog['Mass_FOF'][isattemp]

def buildHaloDictionary(Hydro=None, partType=None, multiple=None):
	if ('DM' in partType) or ('H' in partType) or ('S' in partType):
		return buildHaloDictionary_nieuw(partType=partType, multiple=multiple)
	haloproperties = {}
	if partType is None:
		if Hydro is None:
			sys.exit("buildHaloDictionary should have an entry for either Hydro or partType")
	if partType is not None:
		if partType in [0, 2, 3, 4, 5]:
			sys.exit("Bestaat nog niet voor partType = %i" %partType)
		elif partType == 7:
			Hydro = True
		elif partType == 8:
			Hydro = True
	haloarray = (['HaloIndex', 'HaloID', 'Coord', 'R200', 'M200', 'redshift', 'snapshot', 'lambda', 'Density', 'Npart', 'Vmax', 'Rmax',
			'AngularMomentum', 'Npart_profile', 'Radius', 'Velrad', 'Vel', 'Mass_profile', 'Partindices', 'n_part', 'MaxRadIndex', 
			'Virial_ratio', 'COM_offset', 'Msub', 'CrossTime', 'hostHaloIndex', 'MassTable'])
	if Hydro:
		haloarray.extend(['lambdaDM', 'lambdaH', 'DensityDM', 'DensityH', 
			'NpartH_profile', 'DMFraction', 'DMFraction_profile', 'HFraction', 'HFraction_profile', 'MassH_profile', 'MassDM_profile', 
			'VelradDM', 'VelradH', 'Temperature', 'AngularMomentumDM', 'AngularMomentumH'])
	if partType == 8:
		haloarray.extend(['lambdaS', 'DensityS',
			'NpartS_profile', 'SFraction', 'SFraction_profile', 'MassS_profile',
			'VelradB', 'VelradS', 'AgeS', 'AngularMomentumS'])

	for key in haloarray:
		if (multiple is not None) and (key=='Partindices'):
			haloproperties[key] = {}
		else:
			haloproperties[key] = np.zeros(0)
	return haloproperties


def allocateSizes(key, lengte):
	if key in ['R200', 'M200', 'redshift', 'lambda', 'Vmax', 'Rmax', 'Vmax_part', 'Rmax_part', 'Vmax_interp', 'Rmax_interp',
			'Virial_ratio', 'COM_offset', 'Msub', 'CrossTime', 'lambdaDM', 'lambdaH', 
			'DMFraction', 'HFraction', 'lambdaS', 'SFraction']:
		return np.ones(lengte[0])*-1

	if key in ['HaloIndex', 'HaloID', 'snapshot', 'Npart', 'NpartDM', 'NpartH','NpartS', 
			'n_part', 'MaxRadIndex', 'hostHaloIndex', 'Tail', 'Head', 
			'RootHead', 'RootTail']:
		return np.ones(lengte[0]).astype(int)*-1

	elif key in ['Coord', 'Vel']:
		return np.ones((lengte[0], 3))*-1

	elif key in ['Density', 'AngularMomentum', 'Velrad', 'Mass_profile',
			'DensityDM', 'DensityH', 'DMFraction_profile', 'HFraction_profile', 'MassH_profile', 'MassDM_profile', 
			'VelradDM', 'VelradH', 'Temperature', 'AngularMomentumDM', 'AngularMomentumH', 'lambdaS', 'DensityS',
			'SFraction_profile', 'MassS_profile','VelradB', 'VelradS', 'AgeS', 'AngularMomentumS']:
		return np.zeros((lengte[0], lengte[1]))

	elif key in ['Npart_profile', 'NpartDM_profile', 'NpartH_profile', 'NpartS_profile']:
		return np.zeros((lengte[0], lengte[1])).astype(int)

def buildHaloDictionary_nieuw(partType=None, multiple=None):
	haloproperties = {}
	if partType is None:
		sys.exit("buildHaloDictionary should have an entry for partType")

	haloarray = (['HaloIndex', 'HaloID', 'Coord', 'R200', 'M200', 'redshift', 'snapshot', 'lambda', 'Density', 'Npart', 'Vmax', 'Rmax',
			'AngularMomentum', 'Npart_profile', 'Radius', 'Velrad', 'Vel', 'Mass_profile', 'Partindices', 'n_part', 'MaxRadIndex', 
			'Virial_ratio', 'COM_offset', 'Msub', 'CrossTime', 'hostHaloIndex', 'MassTable', 'Tail', 'Head', 'Vmax_part', 'Rmax_part', 
			'Vmax_interp', 'Rmax_interp', 'RootHead', 'RootTail'])
	if 'H' in partType:
		haloarray.extend(['lambdaDM', 'lambdaH', 'DensityDM', 'DensityH', 'NpartDM_profile','NpartH', 'NpartDM',
			'NpartH_profile', 'DMFraction', 'DMFraction_profile', 'HFraction', 'HFraction_profile', 'MassH_profile', 'MassDM_profile', 
			'VelradDM', 'VelradH', 'Temperature', 'AngularMomentumDM', 'AngularMomentumH'])
	if 'S' in partType:
		haloarray.extend(['lambdaS', 'DensityS', 'NpartS',
			'NpartS_profile', 'SFraction', 'SFraction_profile', 'MassS_profile',
			'VelradB', 'VelradS', 'AgeS', 'AngularMomentumS'])

	for key in haloarray:
		if (multiple is not None) and (key=='Partindices'):
			haloproperties[key] = {}
		elif multiple is not None:
			haloproperties[key] = allocateSizes(key, multiple)
		else:
			haloproperties[key] = None
	return haloproperties

def quantity_keys():
	return (['HaloIndex', 'HaloID', 'Coord', 'R200', 'M200', 'redshift', 'snapshot', 'lambda', 'Npart', 'NpartDM', 
		'NpartH', 'NpartS', 'Vel', 'n_part', 'Tail', 'Head', 'RootHead', 'RootTail',
		'Virial_ratio', 'COM_offset', 'Msub', 'CrossTime', 'hostHaloIndex', 'MassTable', 'lambdaDM', 'lambdaH', 
		'lambdaS', 'DMFraction', 'HFraction', 'SFraction',
		'Vmax_part', 'Rmax_part', 'Vmax_interp', 'Rmax_interp'])
def profile_keys():
	return (['HaloIndex', 'HaloID', 'AngularMomentum', 'Npart_profile', 'Radius', 'Velrad', 'MassTable',
		'Mass_profile', 'MaxRadIndex', 'Density', 'DensityDM', 'DensityH', 'NpartH_profile', 'DMFraction_profile', 
		'HFraction_profile', 'MassH_profile', 'MassDM_profile', 'VelradDM', 'VelradH', 'Temperature', 
		'AngularMomentumDM', 'AngularMomentumH', 'NpartS_profile', 'SFraction_profile', 'MassS_profile',
		'VelradB', 'VelradS', 'AgeS', 'AngularMomentumS'])
def convertVel_keys():
	return (['HaloIndex', 'HaloID', 'Npart', 'NpartDM', 'NpartH', 'NpartS', 'n_part', 'Vel', 'Coord', 'R200', 'M200', 
		'Tail', 'Head', 'RootHead', 'RootTail', 'redshift', 'snapshot', 'hostHaloIndex'])


def findHaloPropertiesInSnap_nieuw(catalog, d_snap, Nhalo=100, halolist=None,
	startHalo=0, d_radius=None, d_partType = None, d_runparams=None,
	partdata = None, TEMPORALHALOIDVAL=1000000000000, boxsize=None, debug=False):
	#Keeping all VELOCIraptor haloes, but saving 'wrong' haloes as HaloIndex = -1

	if d_runparams['VELconvert'] == False:
		boxsize = d_snap['File'].boxsize
	partType = d_partType['particle_type']
	print("Computing properties for %i haloes in snapshot %i" %(Nhalo, d_snap['snapshot']))

	if 'profile' in d_radius.keys():
		ylen = len(d_radius['profile'])
	else:
		ylen = 0

	haloproperties = buildHaloDictionary(partType=partType, multiple=[Nhalo, ylen])

	if len(catalog['Mass_200crit']) == 0:
		return haloproperties

	# if (d_runparams['VELconvert'] == False):
	# 	sortorder = np.argsort(catalog['Mass_tot'][:])[::-1]
	# 	sortorderinvert = np.argsort(sortorder)

	# 	for key in catalog.keys():
	# 		catalog[key][:] = catalog[key][sortorder]
	# else:
	#sortorder = np.arange(len(catalog['Mass_tot'])).astype(int)

	# if partdata is not None:
	# 	for key in partdata.keys():
	# 		partdata[key][:] = partdata[key][sortorder]
	if halolist is None:
		haloindices = np.arange(startHalo, startHalo+Nhalo).astype(int)
		use_existing_r200 = False
	else:
		haloindices = (halolist%TEMPORALHALOIDVAL - 1).astype(int)
		use_existing_r200 = False
	
	halo_i = -1
	for halo in haloindices:
		halo_i += 1
		#if halolist is not None:
		#	print('Computing properties for halo %i'%halo)
		if halo%10000==0:
			print('Computing properties for halo %i-%i' %(halo, halo+10000))
		if halo > len(catalog['Xc'])-1:
			print("Nhalo > N(velociraptor haloes)")
			break

		halopropertiestemp = {}
		coords = getHaloCoord(catalog, halo, z=d_snap['redshift'], snapshottype=d_runparams['SnapshotType'], 
			physical=d_runparams['Physical'])

		coords = coords%boxsize
		radhier = getHaloRadius(catalog, halo, z=d_snap['redshift'], 
			rtype = d_radius['Rchoice'], snapshottype=d_runparams['SnapshotType'], 
			physical=d_runparams['Physical'])
		satellite = False

		#Trusting VELOCIraptor not to falsely identify haloes as satellites
		if (halolist is None) and (catalog['hostHaloID'][halo] != -1):
			satellite = True
			hostHaloIDtemp = np.where(catalog['hostHaloID'][halo]==catalog['ID'])[0]
			if len(hostHaloIDtemp) == 0:
				hostHaloIDtemp = -2
			else:
				hostHaloIDtemp = hostHaloIDtemp[0]
		else:
			hostHaloIDtemp = -1

		#All happens here
		if debug:
			start_time = time.time()
			print('M200: ', catalog['Mass_200crit'][halo])
			print('R200: ', catalog['R_200crit'][halo])
			print('ID: ', catalog['ID'][halo])
		if d_runparams['VELconvert']:
			if d_runparams['ParticleDataType'] != 'None':
				halopropertiestemp = copyVELOCIraptor(catalog, halo, coords, redshift = d_snap['redshift'],
					partType=partType, particledata=partdata['Particle_Types'], d_partType=d_partType)
			else:
				halopropertiestemp = copyVELOCIraptor(catalog, halo, coords, redshift = d_snap['redshift'],
					partType=partType)
			halopropertiestemp['hostHaloIndex'] = hostHaloIDtemp
		elif d_runparams['ParticleDataType'] == 'None':
			#print("Halo", halo)
			halopropertiestemp = findHaloProperties(d_snap, halo, coords, d_radius, 
				partType=partType, satellite=satellite, rad = radhier, partlim=0, use_existing_r200=use_existing_r200,
				profiles=d_runparams['Profiles'], quantities=d_runparams['Quantities'], debug=debug)
		else:
			#print("Halo", halo,len(partdata['Particle_IDs'][sortorder[halo]]))
			halopropertiestemp = findHaloProperties(d_snap, halo, coords, d_radius, 
				partType=partType, satellite=satellite, rad = radhier, partlim=0, use_existing_r200=use_existing_r200,
				profiles=d_runparams['Profiles'], quantities=d_runparams['Quantities'], debug=debug, 
				particledata=partdata['Particle_IDs'][halo])
		if halopropertiestemp is None:
			if debug:
				print("De halo is leeg???")
			continue
		if debug:
			print("--- %s seconds ---" % (time.time() - start_time), 'halopropertiestemp computed')
			start_time = time.time()
		if d_runparams['TreeData']:
			halopropertiestemp['Tail'] = catalog['Tail'][halo]-1
			halopropertiestemp['Head'] = catalog['Head'][halo]-1
			halopropertiestemp['RootTail'] = catalog['RootTail'][halo]-1
			halopropertiestemp['RootHead'] = catalog['RootHead'][halo]-1
		if d_runparams['VELconvert'] == False:
			if halopropertiestemp is None:
				halopropertiestemp = buildHaloDictionary(partType=partType)
				halopropertiestemp['HaloID'] = catalog['ID'][halo]
				halopropertiestemp['HaloIndex'] = -1
				halopropertiestemp['COM_offset'] = -1
				halopropertiestemp['CrossTime'] = -1
				halopropertiestemp['Coord'] = coords
			else:
				if satellite:
					halopropertiestemp['Npart'] = catalog['npart'][halo]
				
				halopropertiestemp['n_part'] = catalog['npart'][halo]
				halopropertiestemp['HaloID'] = catalog['ID'][halo]
				halopropertiestemp['hostHaloIndex'] = hostHaloIDtemp
				if not satellite:
					afstandtemp = coords - getHaloCoordCOM(catalog, halo, z=d_snap['redshift'], snapshottype=d_runparams['SnapshotType'], physical=d_runparams['Physical'])
					rhier = np.where(np.abs(afstandtemp)>0.5*boxsize, np.abs(afstandtemp) - boxsize, afstandtemp)
					halopropertiestemp['COM_offset'] = np.sqrt(np.sum(rhier**2))/halopropertiestemp['R200']
					halopropertiestemp['CrossTime'] = (2.*halopropertiestemp['R200']*Mpc_to_km /
						np.sqrt(G_Mpc_km2_Msi_si2*halopropertiestemp['M200']*1e10/
							halopropertiestemp['R200']))*s_to_yr/1.e6
				else:
					halopropertiestemp['COM_offset'] = -1
					halopropertiestemp['CrossTime'] = -1

		for key in haloproperties.keys():
			doorgaan = False
			if (d_runparams['Profiles'] == True) and (key in profile_keys()):
				doorgaan = True
			if (d_runparams['Quantities'] == True) and (key in quantity_keys()):
				doorgaan = True
			if (d_runparams['VELconvert'] == True) and (key in convertVel_keys()):
				doorgaan = True

			if doorgaan == False:
				continue
			if key in ['Radius', 'MassTable', 'snapshot', 'redshift']:
				continue
			elif key == 'Neighbours' or key == 'Neighbour_distance' or key == 'Neighbour_Velrad':
				continue
			if (halopropertiestemp['HaloIndex'] == -1) and (key != 'HaloID'):
				continue
			if halopropertiestemp[key] is None:
				continue
			elif key=='Partindices':
				haloproperties[key][halopropertiestemp['HaloIndex']] = halopropertiestemp[key][:]
			else:
				haloproperties[key][halo_i] = halopropertiestemp[key]
		if debug:
			print("--- %s seconds ---" % (time.time() - start_time), 'haloproperties updated')
	if 'profile' in d_radius.keys():
		haloproperties['Radius'] = d_radius['profile']
	haloproperties['redshift'] = np.array([d_snap['redshift']])
	haloproperties['snapshot'] = np.array([d_snap['snapshot']])
	
	j = 0
	if d_runparams['VELconvert'] == False:
		haloproperties['MassTable'] = d_snap['File'].mass
		for i in d_snap['File'].readParticles:
			if haloproperties['MassTable'][i] == 0 and d_snap['File'].npart[i] != 0:
				waar = np.where(d_snap['File'].partTypeArray == i)[0][0]
				haloproperties['MassTable'][i] = d_snap['File'].masses[waar]
				j += 1
	
	if d_runparams['TreeData']:
		haloproperties['Tail'] = haloproperties['Tail'].astype(int)
		haloproperties['Head'] = haloproperties['Head'].astype(int)
		haloproperties['RootTail'] = haloproperties['RootTail'].astype(int)
		haloproperties['RootHead'] = haloproperties['RootHead'].astype(int)

	if (len(haloproperties['Coord']) > 0) and (halolist is None):
		if d_runparams['Quantities'] or d_runparams['VELconvert']:
			print("Reassigning satellite haloes")
			fixSatelliteProblems(haloproperties, boxsize=boxsize)

	return haloproperties


def findHaloPropertiesInSnap(catalog, snappath, snapshot, partType=8, Nhalo=100, 
	startHalo=0, softeningLength=0.002, Radius=1., partlim=200, sortorder=None,
	boxsize=32, TEMPORALHALOIDVAL=1000000000000, particledata=None, mass=False):
	
	print("Computing properties for %i haloes in snapshot %i" %(Nhalo, snapshot))
	haloproperties = buildHaloDictionary(partType=partType, multiple=True)
	if len(catalog['Mass_tot']) == 0:
		return haloproperties

	if sortorder is None:
		sortorder = np.argsort(catalog['Mass_tot'][:])[::-1]
		sortorderinvert = np.argsort(sortorder)
	else:
		sortorderinvert = np.argsort(sortorder)
	d_snap = {}
	d_snap['snapshot'] = snapshot
	limiet = 0

	d_snap['File'] = Snapshot(snappath, snapshot, useIDs=False, partType=partType, softeningLength=softeningLength)
	d_snap['File'].makeCoordTree()

	for key in catalog.keys():
		catalog[key][:] = catalog[key][sortorder]

	for halo in range(startHalo, startHalo+Nhalo):
		#start_time = time.time()
		#print(halo)
		#print(catalog['npart'][halo])
		if halo%1000==0:
			print('Computing properties for halo %i-%i' %(halo, halo+1000))
		if halo > len(catalog['Xc'])-1:
			print("Halo limit reached: nhalo = %i, hlim = %i" %(halo, limiet))
			print("Coordinates: ", coords)
			break
		if limiet > 500: #Only computing sats
			if catalog['hostHaloID'][halo] == -1:
				continue

		halopropertiestemp = {}
		coords = getHaloCoord(catalog, halo, z=d_snap['File'].redshift)

		coords = coords%boxsize
		radhier = getHaloRadius(catalog, halo, z=d_snap['File'].redshift)

		satellite = False
		if (catalog['npart'][halo] < 20) or (catalog['Mass_200crit'][halo]*h == 0):
			startHalo += 1
			# haloproperties['TreeBool'][halo] = 0
			continue

		#Checking for dissapeared host haloes
		if (catalog['hostHaloID'][halo] != -1) and len(haloproperties['HaloID'])>1:
			haloindextemp = np.where((haloproperties['HaloID']%TEMPORALHALOIDVAL)==catalog['hostHaloID'][halo]%TEMPORALHALOIDVAL)[0]
			if len(haloindextemp) == 0:
				hostHaloIDtemp = -1
				if catalog['npart'][halo] < partlim/2.:
					hostHaloIDtemp = -2
					satellite = True
			else:
				afstandtemp = (haloproperties['Coord'][haloindextemp[0]]-coords)
				afstandtemp = np.where(np.abs(afstandtemp)>0.5*boxsize, np.abs(afstandtemp) - boxsize, afstandtemp)
				afstandtemp = (np.sum(afstandtemp*afstandtemp))**0.5
				if afstandtemp < haloproperties['R200'][haloindextemp[0]]: # and catalog['npart'][halo] > 50:
					#print(afstandtemp, haloproperties['R200'][haloindextemp[0]], haloproperties['Coord'][haloindextemp[0]], coords)
					hostHaloIDtemp = haloindextemp[0]
					satellite = True
				else:
					#print(afstandtemp, haloproperties['R200'][haloindextemp[0]], haloproperties['Coord'][haloindextemp[0]], coords)
					hostHaloIDtemp = -1
		else:
			hostHaloIDtemp = -1

		#All happens here
		halopropertiestemp = findHaloProperties(d_snap, halo, coords, Radius, partType=partType,
			rad=radhier, mass=mass, satellite=satellite, partlim=partlim)
		#print("--- %s seconds ---" % (time.time() - start_time), 'halopropertiestemp computed')


		if halopropertiestemp is None:
			startHalo += 1
			limiet += 1
			# haloproperties['TreeBool'][halo] = 0
			continue
		if satellite == False and halopropertiestemp['Npart'] < partlim:
			startHalo += 1
			limiet += 1
			# haloproperties['TreeBool'][halo] = 0
			continue
		limiet = 0

		if satellite:
			halopropertiestemp['Npart'] = catalog['npart'][halo]
			
		#start_time = time.time()
		halopropertiestemp['n_part'] = catalog['npart'][halo]
		halopropertiestemp['HaloID'] = catalog['ID'][halo]
		halopropertiestemp['hostHaloIndex'] = hostHaloIDtemp
		if not satellite:
			afstandtemp = coords - getHaloCoord(catalog, halo, z=d_snap['File'].redshift)
			rhier = np.where(np.abs(afstandtemp)>0.5*boxsize, np.abs(afstandtemp) - boxsize, afstandtemp)
			halopropertiestemp['COM_offset'] = np.sqrt(np.sum(rhier**2))/halopropertiestemp['R200']
			halopropertiestemp['CrossTime'] = (2.*halopropertiestemp['R200']*Mpc_to_km /
				np.sqrt(G_Mpc_km2_Msi_si2*halopropertiestemp['M200']*1e10/halopropertiestemp['R200']))*s_to_yr/1.e6
		else:
			halopropertiestemp['COM_offset'] = -1
			halopropertiestemp['CrossTime'] = -1

		for key in haloproperties.keys():
			if key in ['TreeBool', 'Tail', 'Head', 'Radius', 'MassTable', 'snapshot', 'redshift']:
				continue
			elif key == 'Neighbours' or key == 'Neighbour_distance' or key == 'Neighbour_Velrad':
				continue
			elif key=='Partindices':
				haloproperties[key][halopropertiestemp['HaloIndex']] = halopropertiestemp[key][:]
			elif halo == startHalo:
				haloproperties[key] = [halopropertiestemp[key]]
			else:
				haloproperties[key] = np.concatenate((haloproperties[key], [halopropertiestemp[key]]))
		#print("--- %s seconds ---" % (time.time() - start_time), 'haloproperties updated')

	haloproperties['Radius'] = Radius
	haloproperties['redshift'] = np.array([d_snap['File'].redshift])
	haloproperties['snapshot'] = np.array([d_snap['snapshot']])
	haloproperties['MassTable'] = d_snap['File'].mass
	j = 0
	for i in d_snap['File'].readParticles:
		if haloproperties['MassTable'][i] == 0 and d_snap['File'].npart[i] != 0:
			waar = np.where(d_snap['File'].partTypeArray == i)[0][0]
			haloproperties['MassTable'][i] = d_snap['File'].masses[waar]
			j += 1
	
	findSubHaloFraction(haloproperties, catalog)
	print("Reassigning satellite haloes")
	if len(haloproperties['Coord']) > 0:
		if 'DMFraction' in haloproperties.keys():
			Hydro = True
		else:
			Hydro = False
		fixSatelliteProblems(haloproperties, Hydro = Hydro)
	#print("Computing subhalo fraction")
	print(haloproperties.keys())
	return haloproperties


def findHaloProperties(d_snap, halo, Coord, fixedRadius, r200fac = 8, partType=None, rad=None, satellite=False, 
	partlim=200, profiles=False, quantities=True, particledata=None, debug=False, use_existing_r200=False):
	haloproperties = buildHaloDictionary(partType=partType)
	if isinstance(fixedRadius, dict):
		if 'profile' in fixedRadius.keys():
			radprofile = fixedRadius['profile']
			radfrac = fixedRadius['Rfrac']
		else:
			radfrac = fixedRadius['Rfrac']
	else:
		radprofile = fixedRadius
		radfrac = r200fac

	snap = d_snap['File']
	haloproperties['HaloIndex'] = halo
	haloproperties['HaloID'] = halo#catalog['ID'][halo]
	snap.debug = debug

	coord = Coord
	if debug:
		start_time = time.time()
	if rad is None:
		rad = fixedRadius[-1]
	snap.get_temphalo(coord, rad, r200fac=radfrac, fixedRadius=radprofile, satellite=satellite, 
		particledata=particledata, partlim=partlim, initialise_profiles=profiles, use_existing_r200=use_existing_r200)

	if len(snap.temphalo['indices']) < partlim or len(snap.temphalo['indices'])<=1:
		if debug:
			print('Halo has %i particles, and is thus too small' %len(snap.temphalo['indices']))
		return None
	if debug:
		print("--- %s seconds ---" % (time.time() - start_time), 'halo initiated', snap.temphalo['R200'])	
	
	if profiles:
		if debug:
			start_time = time.time()
		snap.get_temphalo_profiles()
		snap.get_specific_angular_momentum_radius(coord, radius=snap.temphalo['Radius'])
		haloproperties['AngularMomentum'] = snap.temphalo['AngularMomentum']
		haloproperties['Density'] = snap.temphalo['profile_density']
		haloproperties['Velrad'] = snap.temphalo['profile_vrad']
		haloproperties['Npart_profile'] = snap.temphalo['profile_npart']
		haloproperties['Mass_profile'] = snap.temphalo['profile_mass']
		haloproperties['MaxRadIndex'] = snap.temphalo['MaxRadIndex']
		if debug:
			print("--- %s seconds ---" % (time.time() - start_time), 'halo profiles calculated')

	haloproperties['Coord'] = snap.temphalo['Coord']
	#Virial radius and mass
	R200 = snap.temphalo['R200']
	haloproperties['M200']= snap.temphalo['M200']
	haloproperties['R200'] = R200
	#Assigning halo properties
	
	if quantities:
		if debug:
			start_time = time.time()
		if (satellite == False) or (particledata is not None):
			snap.get_spin_parameter(coord)
			haloproperties['lambda'] = snap.temphalo['lambda']
		haloproperties['lambda'] = snap.temphalo['lambda']

		snap.get_Vmax_Rmax()
		haloproperties['Vmax_part'] = snap.temphalo['Vmax_part']
		haloproperties['Rmax_part'] = snap.temphalo['Rmax_part']
		haloproperties['Vmax_interp'] = snap.temphalo['Vmax_interp']
		haloproperties['Rmax_interp'] = snap.temphalo['Rmax_interp']
		if debug:
			print("--- %s seconds ---" % (time.time() - start_time), 'lambda calculated')	


	haloproperties['Vel'] = snap.temphalo['Vel']
	haloproperties['Partindices'] = snap.temphalo['indices']
	haloproperties['Npart'] = len(haloproperties['Partindices'])

	# if satellite == False:
	# 	haloproperties['Virial_ratio'] = snap.get_virial_ratio(1000)
	# else:
	# 	haloproperties['Virial_ratio'] = -1
	if debug:
		start_time = time.time()
	if len(snap.readParticles) > 1:
		nietnulhier=np.where(haloproperties['Mass_profile']!=0)
		for i_pT in range(len(snap.readParticles)):	
			if quantities:	
				if (satellite == False) or (particledata is not None):
					haloproperties['lambda'+snap.namePrefix[i_pT]] = snap.temphalo['lambda'+snap.namePrefix[i_pT]]
				else:
					haloproperties['lambda'+snap.namePrefix[i_pT]] = -1
				haloproperties['Npart'+snap.namePrefix[i_pT]] = snap.temphalo['Npart'+snap.namePrefix[i_pT]]
				haloproperties[snap.namePrefix[i_pT]+'Fraction'] = snap.temphalo[snap.namePrefix[i_pT]+'Fraction']

			if profiles:
				haloproperties['AngularMomentum'+snap.namePrefix[i_pT]] = snap.temphalo['AngularMomentum'+snap.namePrefix[i_pT]]
				haloproperties['Density'+snap.namePrefix[i_pT]] = snap.temphalo['profile_'+snap.namePrefix[i_pT]+'density']

				haloproperties['Npart'+snap.namePrefix[i_pT]+'_profile'] = snap.temphalo['profile_'+snap.namePrefix[i_pT]+'npart']
				haloproperties['Velrad'+snap.namePrefix[i_pT]] = snap.temphalo['profile_'+snap.namePrefix[i_pT]+'vrad']
				haloproperties['Mass'+snap.namePrefix[i_pT]+'_profile'] = snap.temphalo['profile_'+snap.namePrefix[i_pT]+'mass']		

				if snap.readParticles[i_pT] == 0:
					haloproperties['Temperature'] = snap.temphalo['profile_temperature']
				elif snap.readParticles[i_pT] == 5:
					haloproperties['AgeS'] = snap.temphalo['profile_Sage']

				haloproperties[snap.namePrefix[i_pT]+'Fraction_profile'] = np.zeros_like(haloproperties['Mass_profile'])
				haloproperties[snap.namePrefix[i_pT]+'Fraction_profile'][nietnulhier] = haloproperties['Mass'+snap.namePrefix[i_pT]+'_profile'][nietnulhier]/haloproperties['Mass_profile'][nietnulhier]
	if debug:
		print("--- %s seconds ---" % (time.time() - start_time), 'particle types done')
	if particledata is not None:
		if debug:
			start_time = time.time()
		snap.delete_used_indices(snap.temphalo['indices'])
		if debug:
			print("--- %s seconds ---" % (time.time() - start_time), 'Deleted particles')

	return haloproperties


def copyVELOCIraptor(catalog, halo, Coord, redshift, d_partType=None, partType=None, particledata=None):
	c = constant(redshift=redshift)
	c.change_constants(redshift)
	comoving_rhocrit200 = deltaVir*c.rhocrit_Ms_Mpci3*h/(h*(1+redshift))**3

	haloproperties = buildHaloDictionary(partType=partType)

	haloproperties['HaloIndex'] = halo
	haloproperties['HaloID'] = catalog['ID'][halo]
	haloproperties['n_part'] = catalog['npart'][halo]
	haloproperties['Coord'] = Coord
	#Virial radius and mass
	haloproperties['M200'] = catalog['Mass_200crit'][halo]*h
	haloproperties['R200'] = (haloproperties['M200']*1.e10/(comoving_rhocrit200 * 4./3. * np.pi))**(1./3.)

	#Assigning halo properties
	haloproperties['Vel'] = np.array([catalog['VXc'][halo], catalog['VYc'][halo], catalog['VZc'][halo]])*(1+redshift)
	haloproperties['Npart'] = catalog['npart'][halo]

	if (particledata is not None) and (len(d_partType['particle_type']) > 1):
		allpart = len(particledata[halo])
		for i_pT in range(len(d_partType['particle_type'])):
			if allpart == 0:
				haloproperties['Npart'+d_partType['particle_type'][i_pT]] = 0
			else:
				haloproperties['Npart'+d_partType['particle_type'][i_pT]] = len(np.where(particledata[halo] == d_partType['particle_number'][i_pT])[0])
				#print(d_partType['particle_type'][i_pT], d_partType['particle_number'][i_pT], haloproperties['Npart'+d_partType['particle_type'][i_pT]])
	return haloproperties

def everythingOutside(haloproperties, d_snap):
	allpin = np.zeros(0)
	iets=0
	allpinBool = np.array([True]*np.sum(d_snap['File'].npart))
	for i in haloproperties['HaloIndex']:
		allpinBool[haloproperties['Partindices'][i]] = False
	outsideIndices = np.where(allpinBool)[0]
	insideIndices = np.where(allpinBool==False)[0]
	outsideIndicesDM = outsideIndices[np.where(outsideIndices < d_snap['File'].npart[0])[0]]
	outsideIndicesH = outsideIndices[np.where(outsideIndices >= d_snap['File'].npart[0])[0]]
	insideIndicesDM = insideIndices[np.where(insideIndices < d_snap['File'].npart[0])[0]]
	insideIndicesH = insideIndices[np.where(insideIndices >= d_snap['File'].npart[0])[0]]
	dmmass = d_snap['File'].get_masses()[0]
	hmass = d_snap['File'].get_masses()[-1]
	haloproperties['Outside_fdm_temp_DMpart_Hpart_dmmass_hmass'] = np.array([len(outsideIndicesDM)*dmmass/(len(outsideIndicesDM)*dmmass+len(outsideIndicesH)*hmass), 
		np.sum(d_snap['File'].get_temperature()[outsideIndicesH])/len(outsideIndicesH), len(outsideIndicesDM), len(outsideIndicesH), dmmass, hmass])
	haloproperties['Inside_fdm_temp_DMpart_Hpart_dmmass_hmass'] = np.array([len(insideIndicesDM)*dmmass/(len(insideIndicesDM)*dmmass+len(insideIndicesH)*hmass), 
		np.sum(d_snap['File'].get_temperature()[insideIndicesH])/len(insideIndicesH), len(insideIndicesDM), len(insideIndicesH), dmmass, hmass])

def writeDataToHDF5quantities(path, name, haloproperties, overwrite=False, savePartData=False, 
	convertVel=False, copyVel=False):
	existing = False
	if overwrite==False and os.path.isfile(path + name):
		haloprop = h5py.File(path + name, 'r+')
		existing = True
		HaloIndices = haloprop['HaloIndex'][:]
		overlap = np.where(np.in1d(haloproperties['HaloIndex'], HaloIndices))[0]
		nonoverlap = np.delete(haloproperties['HaloIndex'][:], overlap)
		nonoverlapindex = np.delete(np.arange(0, len(haloproperties['HaloIndex']), 1).astype(int), overlap)
		nonoverlaplist = ['haloIndex_%05d' %i for i in nonoverlap]
	else:
		haloprop = h5py.File(path+name, 'w')


	for key in haloproperties.keys():
		if (copyVel==False) and (convertVel==False) and (key not in quantity_keys()):
			continue
		if (copyVel==False) and convertVel and (key not in convertVel_keys()):
			continue
		if isinstance(haloproperties[key], dict):
			if not savePartData:
				if key == 'DMpartIDs' or key == 'HpartIDs' or key=='Partindices':
					continue
			if existing:
				temp = haloprop[key]
			else:
				temp = haloprop.create_group(key)
			for key2 in haloproperties[key].keys():
				if haloproperties[key][key2] is None:
					print(key)
					continue
				key2string = 'haloIndex_%05d' %key2
				if existing:
					if len(np.where(np.in1d(key2string, nonoverlaplist))[0]) > 0:
						temp.create_dataset(key2string, data = np.array(haloproperties[key][key2]))
				else:
					temp.create_dataset(key2string, data = np.array(haloproperties[key][key2]))
		else:
			if haloproperties[key] is None:
				print(key)
				continue
			if existing:
				if key == 'Radius' or key == 'MassTable' or key == 'snapshot' or key == 'redshift':
					continue
				data = haloprop[key][:]
				for i in nonoverlapindex:
					data = np.concatenate((data, [haloproperties[key][i]]))
				del haloprop[key]
				haloprop.create_dataset(key, data = data)
			else:
				haloprop.create_dataset(key, data = np.array(haloproperties[key]))
	haloprop.close()

def writeDataToHDF5profiles(path, name, haloproperties, overwrite=False, savePartData=False):
	existing = False
	if overwrite==False and os.path.isfile(path + name):
		haloprop = h5py.File(path + name, 'r+')
		existing = True
		HaloIndices = haloprop['HaloIndex'][:]
		overlap = np.where(np.in1d(haloproperties['HaloIndex'], HaloIndices))[0]
		nonoverlap = np.delete(haloproperties['HaloIndex'][:], overlap)
		nonoverlapindex = np.delete(np.arange(0, len(haloproperties['HaloIndex']), 1).astype(int), overlap)
		nonoverlaplist = ['haloIndex_%05d' %i for i in nonoverlap]
	else:
		haloprop = h5py.File(path+name, 'w')


	for key in haloproperties.keys():
		if key not in profile_keys():
			continue
		if isinstance(haloproperties[key], dict):
			if not savePartData:
				if key == 'DMpartIDs' or key == 'HpartIDs' or key=='Partindices':
					continue
			if existing:
				temp = haloprop[key]
			else:
				temp = haloprop.create_group(key)
			for key2 in haloproperties[key].keys():
				if haloproperties[key][key2] is None:
					print(key)
					continue
				key2string = 'haloIndex_%05d' %key2
				if existing:
					if len(np.where(np.in1d(key2string, nonoverlaplist))[0]) > 0:
						temp.create_dataset(key2string, data = np.array(haloproperties[key][key2]))
				else:
					temp.create_dataset(key2string, data = np.array(haloproperties[key][key2]))
		else:
			if haloproperties[key] is None:
				print(key)
				continue
			if existing:
				if key == 'Radius' or key == 'MassTable' or key == 'snapshot' or key == 'redshift':
					continue
				data = haloprop[key][:]
				for i in nonoverlapindex:
					data = np.concatenate((data, [haloproperties[key][i]]))
				del haloprop[key]
				haloprop.create_dataset(key, data = data)
			else:
				haloprop.create_dataset(key, data = np.array(haloproperties[key]))
	haloprop.close()

def writeDataToHDF5(path, name, haloproperties, overwrite=False, savePartData=False):
	existing = False
	if overwrite==False and os.path.isfile(path + name):
		haloprop = h5py.File(path +name, 'r+')
		existing = True
		HaloIndices = haloprop['HaloIndex'][:]
		overlap = np.where(np.in1d(haloproperties['HaloIndex'], HaloIndices))[0]
		nonoverlap = np.delete(haloproperties['HaloIndex'][:], overlap)
		nonoverlapindex = np.delete(np.arange(0, len(haloproperties['HaloIndex']), 1).astype(int), overlap)
		nonoverlaplist = ['haloIndex_%05d' %i for i in nonoverlap]
	else:
		haloprop = h5py.File(path+name, 'w')


	for key in haloproperties.keys():
		if isinstance(haloproperties[key], dict):
			if not savePartData:
				if key == 'DMpartIDs' or key == 'HpartIDs' or key=='Partindices':
					continue
			if existing:
				temp = haloprop[key]
			else:
				temp = haloprop.create_group(key)
			for key2 in haloproperties[key].keys():
				if haloproperties[key][key2] is None:
					print(key)
					continue
				key2string = 'haloIndex_%05d' %key2
				if existing:
					if len(np.where(np.in1d(key2string, nonoverlaplist))[0]) > 0:
						temp.create_dataset(key2string, data = np.array(haloproperties[key][key2]))
				else:
					temp.create_dataset(key2string, data = np.array(haloproperties[key][key2]))
		else:
			if haloproperties[key] is None:
				print(key)
				continue
			if existing:
				if key == 'Radius' or key == 'MassTable' or key == 'snapshot' or key == 'redshift':
					continue
				data = haloprop[key][:]
				for i in nonoverlapindex:
					data = np.concatenate((data, [haloproperties[key][i]]))
				del haloprop[key]
				haloprop.create_dataset(key, data = data)
			else:
				haloprop.create_dataset(key, data = np.array(haloproperties[key]))
	haloprop.close()

def readHDF5Data(path, name, Hydro=True):
	existing = False
	if os.path.isfile(path + name):
		haloprop = h5py.File(path +name, 'r')
	else:
		sys.exit('Error: file '+path+name+' not found.')

	haloproperties = buildHaloDictionary(Hydro=Hydro, multiple=True)

	for key in haloprop.id:
		if isinstance(haloproperties[key.decode('utf-8')], dict):
			if isinstance(haloprop[key].id, h5py.h5d.DatasetID):
				continue
			temp = haloprop[key]
			for key2 in haloprop[key].id:
				haloindex = [int(s) for s in re.findall(r'\d+', key2.decode('utf-8'))][0]
				haloproperties[key.decode('utf-8')][haloindex] = temp[key2][:]
		else:
			haloproperties[key.decode('utf-8')] = haloprop[key][:]
	haloprop.close()
	return haloproperties

def readHDF5DataSets(path, name, datasets, Hydro=True):
	existing = False
	if os.path.isfile(path + name):
		haloprop = h5py.File(path +name, 'r')
	else:
		sys.exit('Error: file '+path+name+' not found.')

	haloproperties = buildHaloDictionary(Hydro=Hydro, multiple=True)

	for key in haloprop.id:
		if key.decode('utf-8') in datasets:
			if isinstance(haloproperties[key.decode('utf-8')], dict):
				if isinstance(haloprop[key].id, h5py.h5d.DatasetID):
					continue
				temp = haloprop[key]
				for key2 in haloprop[key].id:
					haloindex = [int(s) for s in re.findall(r'\d+', key2.decode('utf-8'))][0]
					haloproperties[key.decode('utf-8')][haloindex] = temp[key2][:]
			else:
				haloproperties[key.decode('utf-8')] = haloprop[key][:]
	haloprop.close()
	return haloproperties

def getRidOfBadHaloes(hp):
	c = constant()
	c.change_constants(hp['redshift'])
	wrong = np.where(4./3*np.pi*hp['R200']**3*200*c.rhocrit_Ms_Mpci3/(h**2*(1+hp['redshift'])**3) > 1.2*hp['M200']*1e10)[0]
	wrong = np.append(wrong, np.where(4./3*np.pi*hp['R200']**3*200*c.rhocrit_Ms_Mpci3/(h**2*(1+hp['redshift'])**3) < 0.8*hp['M200']*1e10)[0])
	wronghi = hp['HaloIndex'][wrong]
	print(len(wronghi))
	for i in hp.keys():
		if i == 'Inside_fdm_temp_DMpart_Hpart_dmmass_hmass' or i == 'Outside_fdm_temp_DMpart_Hpart_dmmass_hmass':
			continue
		if isinstance(hp[i], dict):
			for j in wronghi:
				hp[i].pop(j, None)
		else:
			hp[i] = np.delete(hp[i], wrong)


def rewriteHeadTails(halodata, snapmin=0, snapmax=200, TEMPORALHALOIDVAL=1000000000000):
	sortorder = {}
	sortorderinvert = {}
	newtail = {}
	newhead = {}
	for snap in range(snapmin, snapmax+1):
		sortorder[snap] = np.argsort(halodata[snap]['Mass_tot'][:])[::-1]
		sortorderinvert[snap] = np.argsort(sortorder[snap]) #Orders it to point to the right position in the ID list
	for snap in range(snapmin, snapmax+1):
		oldhead = halodata[snap]['Head'][sortorder[snap]]
		oldtail = halodata[snap]['Tail'][sortorder[snap]]
		newtail[snap] = np.zeros(len(oldtail))
		newhead[snap] = np.zeros(len(oldhead))

		tempsnaps = (oldtail/TEMPORALHALOIDVAL).astype(int)
		if len(tempsnaps) == 0:
			continue

		for i in range(min(tempsnaps), min(snap-1, max(tempsnaps))+1):
			loctemp = np.where(tempsnaps == i)[0]
			if len(loctemp) == 0:
				continue
			prevhalotemp = (oldtail[loctemp]%TEMPORALHALOIDVAL - 1).astype(int)
			newtail[snap][loctemp] = (sortorderinvert[i][prevhalotemp]%TEMPORALHALOIDVAL).astype(int) + i*TEMPORALHALOIDVAL

		tempsnaps = (oldhead/TEMPORALHALOIDVAL).astype(int)
		for i in range(max(min(tempsnaps), snap+1), max(tempsnaps)+1):
			loctemp = np.where(tempsnaps == i)[0]
			if len(loctemp) == 0:
				continue
			prevhalotemp = (oldhead[loctemp]%TEMPORALHALOIDVAL - 1).astype(int)
			newhead[snap][loctemp] = sortorderinvert[i][prevhalotemp] + i*TEMPORALHALOIDVAL
		newtail[snap] = newtail[snap].astype(int)
		newhead[snap] = newhead[snap].astype(int)
	return sortorder, newtail, newhead


def ReadParticleTypes(basefilename,iseparatesubfiles=0,iverbose=0, unbound=True):
    """
    VELOCIraptor/STF catalog_group and catalog_parttypes in hdf5

    Note that a file will indicate how many files the total output has been split into

    """
    inompi=True
    if (iverbose): print("reading particle data",basefilename)
    gfilename=basefilename+".catalog_groups"
    tfilename=basefilename+".catalog_parttypes"
    utfilename=tfilename+".unbound"
    #check for file existence
    if (os.path.isfile(gfilename)==True):
        numfiles=0
    else:
        gfilename+=".0"
        tfilename+=".0"
        utfilename+=".0"
        inompi=False
        if (os.path.isfile(gfilename)==False):
            print("file not found")
            return []
    byteoffset=0

    #load header information from file to get total number of groups
    #hdf

    gfile = h5py.File(gfilename, 'r')
    filenum=int(gfile["File_id"][0])
    numfiles=int(gfile["Num_of_files"][0])
    numhalos=np.uint64(gfile["Num_of_groups"][0])
    numtothalos=np.uint64(gfile["Total_num_of_groups"][0])
    gfile.close()

    particledata=dict()
    particledata['Npart']=np.zeros(numtothalos,dtype=np.uint64)
    particledata['Particle_Types']=[[] for i in range(numtothalos)]

    #now for all files
    counter=np.uint64(0)
    subfilenames=[""]
    if (iseparatesubfiles==1): subfilenames=["",".sublevels"]
    for ifile in range(numfiles):
        for subname in subfilenames:
            bfname=basefilename+subname
            gfilename=bfname+".catalog_groups"
            tfilename=bfname+".catalog_parttypes"
            utfilename=tfilename+".unbound"
            if (inompi==False):
                gfilename+="."+str(ifile)
                tfilename+="."+str(ifile)
                utfilename+="."+str(ifile)
            if (iverbose) : print("reading",bfname,ifile)

            gfile = h5py.File(gfilename, 'r')
            numhalos=np.uint64(gfile["Num_of_groups"][0])
            numingroup=np.uint64(gfile["Group_Size"])
            uoffset=np.uint64(gfile["Offset_unbound"])
            offset=np.uint64(gfile["Offset"])
            gfile.close()

            tfile = h5py.File(tfilename, 'r')
            utfile = h5py.File(utfilename, 'r')
            tdata=np.uint16(tfile["Particle_types"])
            utdata=np.uint16(utfile["Particle_types"])
            npart=len(tdata)
            unpart=len(utdata)
            tfile.close()
            utfile.close()


            #now with data loaded, process it to produce data structure

            unumingroup=np.zeros(numhalos,dtype=np.uint64)
            for i in range(int(numhalos-1)):
                unumingroup[i]=(uoffset[i+1]-uoffset[i]);
            unumingroup[-1]=(unpart-uoffset[-1])

            if unbound:
                particledata['Npart'][counter:counter+numhalos]=numingroup
            else:
                particledata['Npart'][counter:counter+numhalos] = numingroup-unumingroup

            for i in range(numhalos):
                if unbound:
                    particledata['Particle_Types'][int(i+counter)]=np.zeros(numingroup[i],dtype=np.int64)
                    particledata['Particle_Types'][int(i+counter)][:int(numingroup[i]-unumingroup[i])]=tdata[offset[i]:offset[i]+numingroup[i]-unumingroup[i]]
                    particledata['Particle_Types'][int(i+counter)][int(numingroup[i]-unumingroup[i]):numingroup[i]]=utdata[uoffset[i]:uoffset[i]+unumingroup[i]]
                else:
                    particledata['Particle_Types'][int(i+counter)]=np.zeros(numingroup[i]-unumingroup[i],dtype=np.int64)
                    particledata['Particle_Types'][int(i+counter)][:int(numingroup[i]-unumingroup[i])]=tdata[offset[i]:offset[i]+numingroup[i]-unumingroup[i]]

            counter+=numhalos

    return particledata
    
def ReadParticleDataFile(basefilename,iseparatesubfiles=0,iparttypes=0,iverbose=1, binarydtype=np.int64, 
	unbound=True, selected_files=None, halolist=None, TEMPORALHALOIDVAL = 1000000000000):
    """
    VELOCIraptor/STF catalog_group, catalog_particles and catalog_parttypes in various formats

    Note that a file will indicate how many files the total output has been split into

    """
    inompi=True
    if (iverbose): print("reading particle data",basefilename)
    gfilename=basefilename+".catalog_groups"
    pfilename=basefilename+".catalog_particles"
    upfilename=pfilename+".unbound"
    tfilename=basefilename+".catalog_parttypes"
    utfilename=tfilename+".unbound"
    #check for file existence
    if (os.path.isfile(gfilename)==True):
        numfiles=0
    else:
        gfilename+=".0"
        pfilename+=".0"
        upfilename+=".0"
        tfilename+=".0"
        utfilename+=".0"
        inompi=False
        if (os.path.isfile(gfilename)==False):
            print("file not found")
            return []
    byteoffset=0

    #If a list of haloes is given, we only want to read in the haloes (memory efficient)
    if halolist is not None:
        haloindices = (halolist%TEMPORALHALOIDVAL - 1).astype(int)

    #load header information from file to get total number of groups
    gfile = h5py.File(gfilename, 'r')
    numfiles=int(gfile["Num_of_files"][0])
    numtothalos=np.uint64(gfile["Total_num_of_groups"][0])
    gfile.close()

    if selected_files is not None:
        numtothalos = np.uint64(0)
        numfiles = len(selected_files)
        for ifile in selected_files:
            filename = basefilename+".catalog_groups"+"."+str(ifile)
            halofile = h5py.File(filename, 'r')
            numtothalos += np.uint64(halofile["Num_of_groups"][0])
            halofile.close()

    if halolist is not None:
        numtothalos = len(haloindices)        

    particledata=dict()
    particledata['Npart']=np.zeros(numtothalos,dtype=np.uint64)
    particledata['Npart_unbound']=np.zeros(numtothalos,dtype=np.uint64)
    particledata['Particle_IDs']=[[] for i in range(numtothalos)]
    if (iparttypes==1):
        particledata['Particle_Types']=[[] for i in range(numtothalos)]

    #now for all files
    counter=np.uint64(0)
    if halolist is not None:
        noffset = np.uint64(0)
    subfilenames=[""]
    if (iseparatesubfiles==1): subfilenames=["",".sublevels"]
    for ifile in range(numfiles):
        if selected_files is not None:
            ifile_temp = selected_files[ifile]
        else:
            ifile_temp = ifile
        for subname in subfilenames:
            bfname=basefilename+subname
            gfilename=bfname+".catalog_groups"
            pfilename=bfname+".catalog_particles"
            upfilename=pfilename+".unbound"
            tfilename=bfname+".catalog_parttypes"
            utfilename=tfilename+".unbound"
            if (inompi==False):
                gfilename+="."+str(ifile_temp)
                pfilename+="."+str(ifile_temp)
                upfilename+="."+str(ifile_temp)
                tfilename+="."+str(ifile_temp)
                utfilename+="."+str(ifile_temp)
            if (iverbose) : print("reading",bfname,ifile_temp)

            gfile = h5py.File(gfilename, 'r')
            numhalos=np.uint64(gfile["Num_of_groups"][0])
            # if halolist is not None:
            # 	ww = haloindices[np.where((haloindices >= noffset)&(haloindices < noffset+numhalos))[0]] - noffset
            # 	noffset += numhalos
            # 	numhalos = len(ww)
            # else:
            ww = np.arange(0, numhalos, 1).astype(int)
            numingroup=np.uint64(gfile["Group_Size"])[ww]
            offset=np.uint64(gfile["Offset"])[ww]
            uoffset=np.uint64(gfile["Offset_unbound"])[ww]
            gfile.close()
            pfile = h5py.File(pfilename, 'r')
            upfile = h5py.File(upfilename, 'r')
            piddata=np.int64(pfile["Particle_IDs"])
            upiddata=np.int64(upfile["Particle_IDs"])
            npart=len(piddata)
            unpart=len(upiddata)

            pfile.close()
            upfile.close()
            if (iparttypes==1):
                tfile = h5py.File(tfilename, 'r')
                utfile = h5py.File(utfilename, 'r')
                tdata=np.uint16(tfile["Particle_types"])
                utdata=np.uint16(utfile["Particle_types"])
                tfile.close()
                utfile.close()


            #now with data loaded, process it to produce data structure
            unumingroup=np.zeros(numhalos,dtype=np.uint64)
            for i in range(int(numhalos-1)):
                unumingroup[i]=(uoffset[i+1]-uoffset[i]);
            unumingroup[-1]=(unpart-uoffset[-1])

            if unbound:
                particledata['Npart'][int(counter):int(counter+numhalos)]=numingroup
            else:
                particledata['Npart'][int(counter):int(counter+numhalos)] = numingroup-unumingroup

            particledata['Npart_unbound'][int(counter):int(counter+numhalos)]=unumingroup
            for i in range(numhalos):
                if unbound:
                    particledata['Particle_IDs'][int(i+counter)]=np.zeros(numingroup[i],dtype=np.int64)
                    particledata['Particle_IDs'][int(i+counter)][:int(numingroup[i]-unumingroup[i])]=piddata[offset[i]:offset[i]+numingroup[i]-unumingroup[i]]
                    particledata['Particle_IDs'][int(i+counter)][int(numingroup[i]-unumingroup[i]):numingroup[i]]=upiddata[uoffset[i]:uoffset[i]+unumingroup[i]]
                    if (iparttypes==1):
                        particledata['Particle_Types'][int(i+counter)]=np.zeros(numingroup[i],dtype=np.int64)
                        particledata['Particle_Types'][int(i+counter)][:int(numingroup[i]-unumingroup[i])]=tdata[offset[i]:offset[i]+numingroup[i]-unumingroup[i]]
                        particledata['Particle_Types'][int(i+counter)][int(numingroup[i]-unumingroup[i]):numingroup[i]]=utdata[uoffset[i]:uoffset[i]+unumingroup[i]]
                else:
                    particledata['Particle_IDs'][int(i+counter)]=np.zeros(numingroup[i]-unumingroup[i],dtype=np.int64)
                    particledata['Particle_IDs'][int(i+counter)][:int(numingroup[i]-unumingroup[i])]=piddata[offset[i]:offset[i]+numingroup[i]-unumingroup[i]]
                    if (iparttypes==1):
                        particledata['Particle_Types'][int(i+counter)]=np.zeros(numingroup[i]-unumingroup[i],dtype=np.int64)
                        particledata['Particle_Types'][int(i+counter)][:int(numingroup[i]-unumingroup[i])]=tdata[offset[i]:offset[i]+numingroup[i]-unumingroup[i]]

            counter+=numhalos

    return particledata

def ReadWalkableHDFTree(fname, iverbose=True):
    """
    Reads a simple walkable hdf tree file.
    Assumes the input has
    ["RootHead", "RootHeadSnap", "Head", "HeadSnap", "Tail", "TailSnap", "RootTail", "RootTailSnap", "ID", "Num_progen"]
    along with attributes per snap of the scale factor (eventually must generalize to time as well )
    should also have a header gropu with attributes like number of snapshots.
    Returns the halos IDs with walkable tree data, number of snaps, and the number of snapshots searched.
    """
    hdffile = h5py.File(fname, 'r')
    numsnaps = hdffile['Header'].attrs["NSnaps"]
    #nsnapsearch = ["Header/TreeBuilder"].attrs["Temporal_linking_length"]
    if (iverbose):
        print("number of snaps", numsnaps)
    halodata = [dict() for i in range(numsnaps)]
    for i in range(numsnaps):
        # note that I normally have information in reverse order so that might be something in the units
        if (iverbose):
            print("snap ", i)
        for key in hdffile['Snapshots']['Snap_%03d' % i].keys():
            halodata[i][key] = np.array(
                hdffile['Snapshots']['Snap_%03d' % i][key])
    hdffile.close()
    # , nsnapsearch
    return halodata, numsnaps
