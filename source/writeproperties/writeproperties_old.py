import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import fsolve, brentq
from scipy.interpolate import interp1d
import scipy.integrate
import sys
import os
import writeproperties.velociraptor_python_tools as vpt
from scipy.spatial import cKDTree
import h5py
import re
from constants import *
from snapshot import *
import copy
import itertools

def getHaloCoord(catalog, halo, z=0): #Mpc/h
	coords = np.zeros(3)
	coords[0] = (catalog['Xcmbp'][halo]+catalog['Xc'][halo])*h*(1+z)
	coords[1] = (catalog['Ycmbp'][halo]+catalog['Yc'][halo])*h*(1+z)
	coords[2] = (catalog['Zcmbp'][halo]+catalog['Zc'][halo])*h*(1+z)
	return coords
def getHaloRadius(catalog, halo, z=0): #Mpc/h
	return catalog['R_200crit'][halo]*h*(1+z)

def getHaloCoordCOM(catalog, halo, z=0): #Mpc/h
	coords = np.zeros(3)
	coords[0] = catalog['Xc'][halo]*h*(1+z)
	coords[1] = catalog['Yc'][halo]*h*(1+z)
	coords[2] = catalog['Zc'][halo]*h*(1+z)
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

def fixSatelliteProblems(hp, Hydro=False, TEMPORALHALOIDVAL=1000000000000):
	halotree = cKDTree(hp['Coord'], boxsize=32)

	toolarge = np.where(hp['R200'] > hp['R200'][0]*1.2)[0]
	#print(i, toolarge)
	if len(toolarge) != 0:
		for tl in toolarge:
			hp['M200'][tl] = -1
			hp['R200'][tl] = -1
			if Hydro:
				hp['DMFraction'][tl] = -1
			hp['hostHaloIndex'][hp['HaloIndex'][tl]==hp['hostHaloIndex']] = -2
	for halo in range(len(hp['M200'])):
		if hp['M200'][halo] == -1:
			continue
		buren = np.array(halotree.query_ball_point(hp['Coord'][halo], r = 2*hp['R200'][halo]))
		if len(buren) <= 1:
			continue
		buren = buren[hp['R200'][buren] != -1]

		i_largest = np.argmax(hp['n_part'][buren])
		index_largest = buren[i_largest]
		buren = np.delete(buren,i_largest)

		coords = hp['Coord'][buren] - hp['Coord'][index_largest]
		coords = np.where(np.abs(coords) > 0.5*32, coords - coords/np.abs(coords)*32, coords)
		rad = np.sqrt(np.sum(coords*coords, axis=1))
		burentemp = np.where(hp['R200'][buren]-rad+hp['R200'][index_largest] > 0)[0]
		if len(burentemp) == 0:
			continue
		buren = buren[burentemp]

		hp['hostHaloIndex'][buren] = index_largest
		hp['M200'][buren] = -1
		hp['R200'][buren] = -1
		if Hydro:
			hp['DMFraction'][buren] = -1

def findSubHaloFraction(hp, catalog):
	if len(hp['hostHaloIndex']) < 10:
		hp['Msub'] = np.zeros(len(hp['M200']))
		return 0
	i_hostH = np.where(hp['hostHaloIndex'] > -1)[0]
	hp['Msub'] = np.zeros(len(hp['M200']))
	for i in i_hostH:
		isattemp = np.where(hp['HaloID'][i] == catalog['ID'])[0]
		hp['Msub'][hp['hostHaloIndex'][i]] += catalog['Mass_FOF'][isattemp]

def buildHaloDictionary(Hydro=False, multiple=False, mainBranch=False):
	haloproperties = {}
	if Hydro:
		haloarray = (['HaloIndex', 'HaloID', 'Coord', 'R200', 'M200', 'redshift', 'snapshot', 
			'lambda', 'lambdaDM', 'lambdaH', 'DensityDM', 'DensityH', 'Npart', 
			'DMpartIDs', 'Partindices', 'HpartIDs', 'NpartDM_profile', 'Npart_profile', 'DMFraction', 'DMFraction_profile', 'MassH_profile',
			'VelradDM', 'Velrad', 'VelradH', 'Vel', 'Temperature', 'Mass_profile', 'MassDM_profile', 'COM_offset',
			'AngularMomentumDM', 'AngularMomentumH', 'AngularMomentum', 'Radius', 'MaxRadIndex', 'hostHaloIndex', 'n_part', 'Msub', 'CrossTime',
			'Virial_ratio'])
	else:
		haloarray = (['HaloIndex', 'HaloID', 'Coord', 'R200', 'M200', 'redshift', 'snapshot', 'DMpartIDs', 'lambda', 'Density', 'Npart', 
			'AngularMomentum', 'Npart_profile', 'Radius', 'Velrad', 'Vel', 'MassDM_profile', 'Partindices', 'n_part', 'MaxRadIndex', 
			'Virial_ratio', 'COM_offset', 'Msub', 'CrossTime', 'hostHaloIndex'])
	if mainBranch:
		haloarray.append('Head')
		haloarray.append('Tail')
		haloarray.append('TreeBool')
	for key in haloarray:
		if multiple and (key == 'DMpartIDs' or key == 'HpartIDs' or key=='Partindices' or key=='Neighbours' or key=='Neighbour_distance' or key=='Neighbour_Velrad'):
			haloproperties[key] = {}
		else:
			haloproperties[key] = np.zeros(0)
	return haloproperties

def findHaloPropertiesInSnap(catalog, snappath, snapshot, particledata=[], Hydro=False, Nhalo=100, 
	startHalo=0, softeningLength=0.002, r200fac=1., mass=False, partlim=200, savePartData=False):
	print("Computing properties for %i haloes in snapshot %i" %(Nhalo, snapshot))
	haloproperties = buildHaloDictionary(Hydro = Hydro, multiple=True)
	if len(catalog['Mass_tot']) == 0:
		return haloproperties
	sortorder = np.argsort(catalog['Mass_tot'][:])
	d_snap = {}
	d_snap['snapshot'] = snapshot
	limiet = 0
	if Hydro:
		d_snap['File'] = Snapshot(snappath, snapshot, useIDs=False, partType=7, softeningLength=softeningLength)
		d_snap['File'].makeCoordTree()
	else:
		d_snap['File'] = Snapshot(snappath, snapshot, useIDs=False, partType=1, softeningLength=softeningLength)
		d_snap['File'].makeCoordTree()
	for key in catalog.keys():
		catalog[key][:] = catalog[key][sortorder]
		catalog[key][:] = catalog[key][::-1]
	if len(particledata) > 0:
		mass=True
		for key in particledata.keys():
			particledata[key][:] = np.array(particledata[key])[sortorder]
			particledata[key][:] = particledata[key][::-1]
	for halo in range(startHalo, startHalo+Nhalo):
		masshier = False
		if mass:
			masshier=catalog['Mass_200crit'][halo]*h
			if masshier <= 0.000001:
				startHalo += 1
				limiet += 1
				continue
		if halo > len(catalog['Xc'])-1 or limiet > 500:
			print("Halo limit reached: nhalo = %i, hlim = %i" %(halo, limiet))
			print("Coordinates: ", coords)
			break
		halopropertiestemp = {}
		coords = getHaloCoord(catalog, halo, z=d_snap['File'].redshift)
		if coords[0] < 0 or coords[1] < 0 or coords[2] < 0 or coords[0]%32 < 0.5 or coords[1]%32 < 0.5 or coords[2]%32 < 0.5:
			startHalo += 1
			continue
		coords = coords%32
		radhier = getHaloRadius(catalog, halo, z=d_snap['File'].redshift)
		if ((coords[0] < 2.*radhier) or (coords[1] < 2.*radhier) or (coords[2] < 2.*radhier)
			or (np.abs(coords[0]-32) < 2.*radhier) or (np.abs(coords[1]-32) < 2.*radhier)
			or (np.abs(coords[2]-32) < 2.*radhier)):
			startHalo += 1
			continue
		if (catalog['hostHaloID'][halo] != -1) or (catalog['npart'][halo] < 20) or (catalog['Mass_200crit'][halo]*h == 0):# or ((catalog['Mass_200crit'][halo]/4.e3)**(1./3) < catalog['R_200crit'][halo]):
			startHalo += 1
			continue
		halopropertiestemp = findHaloProperties(halo, catalog, d_snap, particledata=particledata, Hydro=Hydro, r200fac=r200fac, mass=masshier, partlim=partlim)
		if len(halopropertiestemp) == 0:
			startHalo += 1
			limiet += 1
			continue
		if halopropertiestemp['Npart'] < partlim:
			startHalo += 1
			limiet += 1
			continue
		limiet = 0
		for key in haloproperties.keys():
			if key == 'Neighbours' or key == 'Neighbour_distance' or key == 'Neighbour_Velrad':
				iets = 1
			elif key == 'DMpartIDs' or key == 'HpartIDs' or key=='Partindices':
				if key=='Partindices':
					haloproperties[key][halopropertiestemp['HaloIndex']] = halopropertiestemp[key][:]
				elif savePartData:
					haloproperties[key][halopropertiestemp['HaloIndex']] = halopropertiestemp[key][:]
			elif halo == startHalo:
				haloproperties[key] = [halopropertiestemp[key]]
			else:
				haloproperties[key] = np.concatenate((haloproperties[key], [halopropertiestemp[key]]))
	#if startHalo + Nhalo >= len(catalog['npart']) and len(haloproperties['Npart']) > partlim:
	print("Compute temperatures outside haloes...")
	everythingOutside(haloproperties, d_snap)
	print("finding surrounding haloes...")
	if len(haloproperties['M200']) > 100:
		findSurroundingHaloProperties(haloproperties, np.arange(0, 100, 1).astype(int), d_snap)
	return haloproperties

def findHaloPropertiesInSnap_fromUnifiedTreeCatalog(catalog, snappath, snapshot, Hydro=False, Nhalo=100, 
	startHalo=0, softeningLength=0.002, Radius=1., partlim=200, savePartData=False, sortorder=[],
	boxsize=32, TEMPORALHALOIDVAL=1000000000000):
	
	print("Computing properties for %i haloes in snapshot %i" %(Nhalo, snapshot))
	haloproperties = buildHaloDictionary(Hydro = Hydro, multiple=True, mainBranch=True)
	if len(catalog['Mass_tot']) == 0:
		return haloproperties

	if len(sortorder)==0:
		sortorder = np.argsort(catalog['Mass_tot'][:])[::-1]
		sortorderinvert = np.argsort(sortorder)
	else:
		sortorderinvert = np.argsort(sortorder)
	d_snap = {}
	d_snap['snapshot'] = snapshot
	limiet = 0
	if Hydro:
		d_snap['File'] = Snapshot(snappath, snapshot, useIDs=False, partType=7, softeningLength=softeningLength)
		d_snap['File'].makeCoordTree()
	else:
		d_snap['File'] = Snapshot(snappath, snapshot, useIDs=False, partType=1, softeningLength=softeningLength)
		d_snap['File'].makeCoordTree()
	for key in catalog.keys():
		catalog[key][:] = catalog[key][sortorder]
		#catalog[key][:] = catalog[key][::-1]

	#haloproperties['TreeBool'] = np.ones(len(tails), dtype=int)

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
				# haloproperties['TreeBool'][halo] = 0
				continue

		halopropertiestemp = {}
		coords = getHaloCoord(catalog, halo, z=d_snap['File'].redshift)

		coords = coords%32
		radhier = getHaloRadius(catalog, halo, z=d_snap['File'].redshift)

		satellite = False
		if (catalog['npart'][halo] < 20) or (catalog['Mass_200crit'][halo]*h == 0):
			startHalo += 1
			# haloproperties['TreeBool'][halo] = 0
			continue

		if (catalog['hostHaloID'][halo] != -1) and len(haloproperties['HaloID'])>1:
			haloindextemp = np.where((haloproperties['HaloID']%TEMPORALHALOIDVAL)==catalog['hostHaloID'][halo]%TEMPORALHALOIDVAL)[0]
			if len(haloindextemp) == 0:
				hostHaloIDtemp = -1
				if catalog['npart'][halo] < 50:
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
		halopropertiestemp = findHaloPropertiesFixedRadius(d_snap, halo, coords, np.logspace(-3, 0, 60)*Radius, 
			Hydro=Hydro, rad=radhier, mass=False, satellite=satellite, mainBranch=True)
		#print("--- %s seconds ---" % (time.time() - start_time), 'halopropertiestemp computed')


		if len(halopropertiestemp) == 0:
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
			afstandtemp = coords - getHaloCoordCOM(catalog, halo, z=d_snap['File'].redshift)
			rhier = np.where(np.abs(afstandtemp)>0.5*boxsize, np.abs(afstandtemp) - boxsize, afstandtemp)
			halopropertiestemp['COM_offset'] = np.sqrt(np.sum(rhier**2))/halopropertiestemp['R200']
			halopropertiestemp['CrossTime'] = (2.*halopropertiestemp['R200']*Mpc_to_km /
				np.sqrt(G_Mpc_km2_Msi_si2*halopropertiestemp['M200']*1e10/halopropertiestemp['R200']))*s_to_yr/1.e6
		else:
			halopropertiestemp['COM_offset'] = -1
			halopropertiestemp['CrossTime'] = -1

		for key in haloproperties.keys():
			if key == 'TreeBool' or key == 'Tail' or key == 'Head' or key == 'Radius':
				continue
			elif key == 'Neighbours' or key == 'Neighbour_distance' or key == 'Neighbour_Velrad':
				continue
			elif key == 'DMpartIDs' or key == 'HpartIDs' or key=='Partindices':
				if key=='Partindices':
					haloproperties[key][halopropertiestemp['HaloIndex']] = halopropertiestemp[key][:]
				elif savePartData:
					haloproperties[key][halopropertiestemp['HaloIndex']] = halopropertiestemp[key][:]
			elif halo == startHalo:
				haloproperties[key] = [halopropertiestemp[key]]
			else:
				#print(key)
				haloproperties[key] = np.concatenate((haloproperties[key], [halopropertiestemp[key]]))
		#print("--- %s seconds ---" % (time.time() - start_time), 'haloproperties updated')
	# print("Correcting hostHaloIndices...")
	# for i in range(len(haloproperties['HaloID'])):
	# 	if (haloproperties['hostHaloIndex'][i] != -1):
	# 		haloindextemp = np.where((haloproperties['HaloID']%TEMPORALHALOIDVAL)==haloproperties['hostHaloIndex'][i]%TEMPORALHALOIDVAL)[0]
	# 		if len(haloindextemp) == 0:
	# 			haloproperties['hostHaloIndex'][i] = -2
	# 		else:
	# 			haloproperties['hostHaloIndex'][i] = haloindextemp[0]
	# 	else:
	# 		haloproperties['hostHaloIndex'][i] = -1
	# haloproperties['Tail'] = tails
	# haloproperties['Head'] = heads
	haloproperties['Radius'] = np.logspace(-3, 0, 60)*Radius
	
	findSubHaloFraction(haloproperties, catalog)
	print("Reassigning satellite haloes")
	if len(haloproperties['Coord']) > 0:
		fixSatelliteProblems(haloproperties, Hydro=Hydro)
	#print("Computing subhalo fraction")
	

	return haloproperties

def findHaloPropertiesFixedRadius(d_snap, halo, Coord, Radius, Hydro=False, rad=False, mass=False, satellite=False, partlim=200, mainBranch=False):
	haloproperties = buildHaloDictionary(Hydro=Hydro, mainBranch=mainBranch)
	snap = d_snap['File']
	haloproperties['HaloIndex'] = halo
	haloproperties['HaloID'] = halo#catalog['ID'][halo]
	haloproperties['redshift'] = snap.redshift
	haloproperties['snapshot'] = d_snap['snapshot']
	coord = Coord

	#start_time = time.time()
	snap.get_temphalo(coord, rad, r200fac=4, fixedRadius=Radius, satellite=satellite, partlim=partlim)
	#print("--- %s seconds ---" % (time.time() - start_time), 'halo initiated', snap.temphalo['R200'])	

	if len(snap.temphalo['indices']) < partlim:
		return []

	#start_time = time.time()
	snap.get_temphalo_profiles()
	#print("--- %s seconds ---" % (time.time() - start_time), 'halo profiles calculated')	

	haloproperties['Coord'] = snap.temphalo['Coord']
	#Virial radius and mass
	R200 = snap.temphalo['R200']
	haloproperties['M200']= snap.temphalo['M200']
	haloproperties['R200'] = R200
	#Assigning halo properties
	if not Hydro:
		snap.get_angular_momentum_radius(coord, IDs=[], radius=haloproperties['Radius'], overmass=True)
		if satellite == False:
			snap.get_spin_parameter(coord, IDs = [], radius = R200, M=haloproperties['M200'])
			haloproperties['lambda'] = snap.temphalo['lambda']
		haloproperties['AngularMomentum'] = snap.temphalo['AngularMomentum']
		haloproperties['Density'] = snap.temphalo['profile_density']
		haloproperties['lambda'] = snap.temphalo['lambda']
		haloproperties['Velrad'] = snap.temphalo['profile_vrad']
		haloproperties['Vel'] = snap.temphalo['Vel']
		haloproperties['Npart_profile'] = snap.temphalo['profile_npart']
		haloproperties['Mass_profile'] = snap.temphalo['profile_mass']
		haloproperties['Partindices'] = snap.temphalo['indices']
		haloproperties['Npart'] = len(haloproperties['Partindices'])
		haloproperties['MaxRadIndex'] = snap.temphalo['MaxRadIndex']
		if satellite == False:
			haloproperties['Virial_ratio'] = snap.get_virial_ratio(1000)
		else:
			haloproperties['Virial_ratio'] = -1
	else:
		snap.get_angular_momentum_radius(coord, IDs=[], radius=haloproperties['Radius'], overmass=True)
		if satellite == False:
			snap.get_spin_parameter(coord, IDs = [], radius = R200, M=haloproperties['M200'])
			haloproperties['lambda'] = snap.temphalo['lambda']
			haloproperties['lambdaH'] = snap.temphalo['lambdaH']
			haloproperties['lambdaDM'] = snap.temphalo['lambdaDM']
		else:
			haloproperties['lambda'] = -1
			haloproperties['lambdaH'] = -1
			haloproperties['lambdaDM'] = -1
		haloproperties['AngularMomentumH'] = snap.temphalo['AngularMomentumH']
		haloproperties['AngularMomentum'] = snap.temphalo['AngularMomentum']
		haloproperties['AngularMomentumDM'] =  snap.temphalo['AngularMomentumDM']
		haloproperties['DensityH'] = snap.temphalo['profile_Hdensity']
		haloproperties['DensityDM'] = snap.temphalo['profile_DMdensity']
		haloproperties['Temperature'] = snap.temphalo['profile_temperature']
		haloproperties['Npart'] = snap.temphalo['Npart']
		haloproperties['Npart_profile'] = snap.temphalo['profile_npart']
		haloproperties['NpartDM_profile'] = snap.temphalo['profile_DMnpart']
		haloproperties['Velrad'] = snap.temphalo['profile_vrad']
		haloproperties['VelradH'] = snap.temphalo['profile_Hvrad']
		haloproperties['VelradDM'] = snap.temphalo['profile_DMvrad']
		haloproperties['Vel'] = snap.temphalo['Vel']
		haloproperties['MassH_profile'] = snap.temphalo['profile_Hmass']
		haloproperties['Mass_profile'] = snap.temphalo['profile_mass']
		haloproperties['MassDM_profile'] = snap.temphalo['profile_DMmass']
		nietnulhier=np.where(haloproperties['Mass_profile']!=0)
		haloproperties['DMFraction_profile'] = np.zeros_like(haloproperties['Mass_profile'])
		haloproperties['DMFraction_profile'][nietnulhier] = haloproperties['MassDM_profile'][nietnulhier]/haloproperties['Mass_profile'][nietnulhier]
		haloproperties['DMFraction'] = snap.temphalo['DMFraction']
		haloproperties['MaxRadIndex'] = snap.temphalo['MaxRadIndex']
		if satellite == False:
			#start_time = time.time()
			haloproperties['Virial_ratio'] = snap.get_virial_ratio(1000)
			#print(haloproperties['Virial_ratio'])
			#haloproperties['Virial_ratio_10000'] = snap.get_virial_ratio(10000)
			#print("--- %s seconds ---" % (time.time() - start_time), 'Virial ratio')
		else:
			haloproperties['Virial_ratio'] = -1
			#haloproperties['Virial_ratio_10000'] = -1

	return haloproperties

def findHaloProperties(halo, catalog, d_snap, particledata=[], Hydro=False, r200fac=1, mass=False, partlim=200):
	if halo%1000==0:
		print('Computing properties for halo %i' %halo)
	haloproperties = buildHaloDictionary(Hydro=Hydro)
	snap = d_snap['File']
	haloproperties['HaloIndex'] = halo
	haloproperties['HaloID'] = catalog['ID'][halo]
	haloproperties['redshift'] = snap.redshift
	haloproperties['snapshot'] = d_snap['snapshot']
	coord = getHaloCoord(catalog, halo, z=haloproperties['redshift'])
	if len(particledata) > 0:
		particleIDs_data = particledata['Particle_IDs'][halo]
	else:
		particleIDs_data = []
	snap.get_temphalo(coord, getHaloRadius(catalog, halo, z=haloproperties['redshift']), particleIDs_data=particleIDs_data, 
		r200fac=r200fac, mass=mass, partlim=partlim)
	if len(snap.temphalo['indices']) < partlim:
		return []
	snap.get_temphalo_profiles()
	haloproperties['Coord'] = snap.temphalo['Coord']
	#Virial radius and mass
	R200 = snap.temphalo['R200']
	haloproperties['M200']= snap.temphalo['M200']
	haloproperties['R200'] = R200
	haloproperties['Radius'] = snap.temphalo['Radius']
	#Assigning halo properties
	if not Hydro:
		snap.get_angular_momentum_radius(coord, IDs=[], radius=haloproperties['Radius'], overmass=True)
		snap.get_spin_parameter(coord, IDs = [], radius = R200, M=haloproperties['M200'])
		haloproperties['AngularMomentum'] = snap.temphalo['AngularMomentum']
		haloproperties['Density'] = snap.temphalo['profile_density']
		haloproperties['lambda'] = snap.temphalo['lambda']
		haloproperties['Velrad'] = snap.temphalo['profile_vrad']
		haloproperties['Vel'] = snap.temphalo['profile_v']
		haloproperties['Npart_profile'] = snap.temphalo['profile_npart']
		haloproperties['Mass_profile'] = snap.temphalo['profile_mass']
		haloproperties['DMpartIDs'] = snap.temphalo['partIDs']
		haloproperties['Partindices'] = snap.temphalo['indices']
		haloproperties['Npart'] = len(haloproperties['Partindices'])
	else:
		snap.get_angular_momentum_radius(coord, IDs=[], radius=haloproperties['Radius'], overmass=True)
		snap.get_spin_parameter(coord, IDs = [], radius = R200, M=haloproperties['M200'])
		haloproperties['AngularMomentumH'] = snap.temphalo['AngularMomentumH']
		haloproperties['AngularMomentum'] = snap.temphalo['AngularMomentum']
		haloproperties['AngularMomentumDM'] =  snap.temphalo['AngularMomentumDM']
		haloproperties['lambda'] = snap.temphalo['lambda']
		haloproperties['DensityH'] = snap.temphalo['profile_Hdensity']
		haloproperties['DensityDM'] = snap.temphalo['profile_DMdensity']
		haloproperties['Temperature'] = snap.temphalo['profile_temperature']
		haloproperties['lambdaH'] = snap.temphalo['lambdaH']
		haloproperties['lambdaDM'] = snap.temphalo['lambdaDM']
		haloproperties['Npart_profile'] = snap.temphalo['profile_npart']
		haloproperties['NpartDM_profile'] = snap.temphalo['profile_DMnpart']
		haloproperties['Velrad'] = snap.temphalo['profile_vrad']
		haloproperties['VelradH'] = snap.temphalo['profile_Hvrad']
		haloproperties['VelradDM'] = snap.temphalo['profile_DMvrad']
		haloproperties['Vel'] = snap.temphalo['profile_v']
		haloproperties['MassH_profile'] = snap.temphalo['profile_Hmass']
		haloproperties['Mass_profile'] = snap.temphalo['profile_mass']
		haloproperties['MassDM_profile'] = snap.temphalo['profile_DMmass']
		nietnulhier=np.where(haloproperties['Mass_profile']!=0)
		haloproperties['DMFraction_profile'] = np.zeros_like(haloproperties['Mass_profile'])
		haloproperties['DMFraction_profile'][nietnulhier] = haloproperties['MassDM_profile'][nietnulhier]/haloproperties['Mass_profile'][nietnulhier]
		if r200fac != 1:
			r200index = np.abs(snap.temphalo['Radius']-snap.temphalo['R200']).argmin()
			haloproperties['DMFraction'] = haloproperties['DMFraction_profile'][r200index]
		else:
			haloproperties['DMFraction'] = haloproperties['DMFraction_profile'][-1]
		haloproperties['HpartIDs'] = snap.temphalo['HpartIDs']
		haloproperties['DMpartIDs'] = snap.temphalo['DMpartIDs']
		haloproperties['Partindices'] = snap.temphalo['indices']
		haloproperties['Npart'] = len(snap.temphalo['indices'])
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


def writeDataToHDF5(path, name, haloproperties, overwrite=False, savePartData=True):
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
				key2string = 'haloIndex_%05d' %key2
				if existing:
					if len(np.where(np.in1d(key2string, nonoverlaplist))[0]) > 0:
						temp.create_dataset(key2string, data = np.array(haloproperties[key][key2]))
				else:
					temp.create_dataset(key2string, data = np.array(haloproperties[key][key2]))
		else:
			if existing:
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
