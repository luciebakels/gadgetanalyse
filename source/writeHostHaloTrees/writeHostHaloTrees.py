#!/usr/bin/env python
import numpy as np
import sys
import os
from scipy.spatial import cKDTree
import h5py
import re
from constants import *
from snapshot import *
import haloanalyse as ha
import orbweaverdata as ow

def find_all_values_crossing(ht, oi, hosts, mainhaloes, times_r200=1, 
	datasets=['Vmax', 'Rmax', 'Vel', 'Coord', 'M200', 'redshift', 'snapshot', 'npart', 'Efrac'], add_label='(R200)', 
	no_crossing_peak=True, min_neighbours=50):

	d_output = {}
	r200 = {}
	mask_vel_indices = {}
	mh_tree = {}
	nbtree = {}
	nbtree_merger = {}
	datasets_nbtree = list(np.copy(datasets))
	d_out_merged = oi.output_merged
	if 'Coord' not in datasets:
		datasets_nbtree.append('Coord')
	if 'R200' not in datasets:
		datasets_nbtree.append('R200')
	if 'hostHaloIndex' not in datasets:
		datasets_nbtree.append('hostHaloIndex')

	oi.find_first_eccentricity()
	oi.find_first_eccentricity_merged()

	i = 0
	for ii in range(len(hosts)):
		#Find mh_tree and nbtree
		waarnm = np.where(oi.output_notmerged['host'] == hosts[i])[0]
		nbidnm = ht.halo_key(oi.output_notmerged['allvel'][waarnm], ht.snapend)
		mh_tree[i], nbtree[i] = ht.neighbourtreemaker2(ht.halo_key(mainhaloes[i], ht.snapend), 
			nbidnm, datasets=datasets_nbtree)
		waar = np.where(oi.output_merged['host'] == hosts[i])[0]
		nbid = oi.output_merged['allvel'][waar] + oi.output_merged['allsnap'][waar]*ht.THIDVAL
		mh_tree_temp, nbtree_merger[i] = ht.neighbourtreemaker2(ht.halo_key(mainhaloes[i], ht.snapend),
			nbid, datasets=datasets_nbtree)

		mh_tree[i]['npart_bound'] = mh_tree[i]['npart'] * mh_tree[i]['Efrac']
		nbtree[i]['npart_bound'] = nbtree[i]['npart'] * nbtree[i]['Efrac']
		nbtree_merger[i]['npart_bound'] = nbtree_merger[i]['npart'] * nbtree_merger[i]['Efrac']
		
		# start_time = time.time()
		nbtree[i]['oi_indices'] = waarnm
		nbtree_merger[i]['oi_indices'] = waar
		find_first_crossing(mh_tree[i], nbtree[i], zstart=ht.zstart, numsnap=ht.snapend, 
			times_r200=times_r200, overwrite=False)
		find_first_crossing(mh_tree[i], nbtree_merger[i], zstart=ht.zstart, numsnap=ht.snapend, 
			times_r200=times_r200, overwrite=False)
		#mask_vel_indices[i] = find_mask_vel_indices(d_output[i], mh_tree[i], nbtree[i])
		delete_no_crossings = np.where(nbtree_merger[i]['no_crossing_%1.1f'%times_r200])
		if len(delete_no_crossings) > 0:
			for kk in nbtree_merger[i].keys():
				nbtree_merger[i][kk] = np.delete(nbtree_merger[i][kk], delete_no_crossings, axis=0)
		find_merging_redshift(oi, nbtree_merger[i])
		# print("--- %s seconds ---" % (time.time() - start_time), 'find_first_crossing')

		# start_time = time.time()
		find_formation_time_nbtree(nbtree[i], zstart=ht.zstart, numsnap=ht.snapend, mass_frac=[0.75, 0.5, 0.25], masstype='npart_bound')
		find_formation_time_nbtree(nbtree_merger[i], zstart=ht.zstart, numsnap=ht.snapend, mass_frac=[0.75, 0.5, 0.25], masstype='npart_bound')
		find_formation_time_mhtree(mh_tree[i], zstart=ht.zstart, numsnap=ht.snapend, mass_frac=[0.75, 0.5, 0.25], masstype='npart_bound')
		# print("--- %s seconds ---" % (time.time() - start_time), 'find_formation_time')

		if no_crossing_peak:
			ncp = 'no_crossing_%1.1f'%times_r200
		else:
			ncp = None

		# start_time = time.time()
		find_values_crossing(nbtree[i], mh_tree=mh_tree[i], datasets=datasets, add_label=add_label, masstype='npart_bound',
			nbtree_indices_label='%1.1fxR200'%times_r200, no_crossing_peak=ncp)
		find_values_crossing(nbtree_merger[i], mh_tree=mh_tree[i], datasets=datasets, add_label=add_label, masstype='npart_bound',
			nbtree_indices_label='%1.1fxR200'%times_r200, no_crossing_peak=ncp)
		# print("--- %s seconds ---" % (time.time() - start_time), 'find_values_crossing')

		# start_time = time.time()
		find_z_crossings(oi, ht, mh_tree[i], nbtree[i], times_r200=times_r200, overwrite=True)
		# print("--- %s seconds ---" % (time.time() - start_time), 'find_z_crossings')
		# start_time = time.time()
		find_z_crossings_merger(ht, oi, mh_tree[i], nbtree_merger[i], times_r200=1, overwrite=False)
		# print("--- %s seconds ---" % (time.time() - start_time), 'find_z_crossings_merger')

		find_merging_distance(nbtree_merger[i], mh_tree[i])

		# start_time = time.time()
		# waar = nbtree[i]['oi_indices'][np.where(oi.ow.orbitdata['numorbits'][d_output['alloi'][nbtree[i]['oi_indices']]] >= 1)[0]]
		# nbtree[i]['eccentricity'] = np.ones(len(nbtree[i]['npart']))*-1
		# for j in waar:
		# 	temp = np.where(oi.ow.orbitdata['OrbitID'][d_output[i]['alloi'][j]] == oi.ow.orbitdata['OrbitID99'])[0]
		# 	if len(temp) < 2:
		# 		continue
		# 	i_ecc = (temp[np.argsort(oi.ow.orbitdata['scalefactor99'][temp])])[1] #1 because measured at 0.5*2
		# 	nbtree[i]['eccentricity'][j] = oi.ow.orbitdata['orbitecc_ratio'][i_ecc]
		nbtree[i]['eccentricity'] = oi.eccentricity[nbtree[i]['oi_indices']]
		nbtree_merger[i]['eccentricity'] = oi.output_merged['eccentricity'][nbtree_merger[i]['oi_indices']]
		# print("--- %s seconds ---" % (time.time() - start_time), 'eccentricity')

		nbtree[i]['orbits'] = oi.ow.orbitdata['numorbits'][oi.output_notmerged['alloi'][nbtree[i]['oi_indices']]]
		nbtree_merger[i]['orbits'] = oi.ow.orbitdata['numorbits'][oi.output_merged['alloi'][nbtree_merger[i]['oi_indices']]]

		find_preprocess(ht, nbtree[i], mh_tree[i], merged=False, min_neighbours=min_neighbours)
		find_preprocess(ht, nbtree_merger[i], mh_tree[i], merged=True, min_neighbours=min_neighbours)

		print('Done processing halo', mainhaloes[i])
		i += 1
	
	# mvi = np.zeros(0).astype(int)
	# for i in mask_vel_indices.keys():
	# 	mvi = np.append(mvi, mask_vel_indices[i])

	return mh_tree, nbtree, nbtree_merger, hosts, mainhaloes #np.sort(np.unique(mvi)), d_output, 

def Efrac_cut(oi, hp, Efrac_limit=0.8):
	waar = np.where(hp['Efrac'][oi.vel_indices] > Efrac_limit)[0]
	oi.indices = oi.indices[waar]

def snapshot_to_redshift(snapshot, zstart=30, numsnap=200):
	tstart = (1/(1+zstart))
	tdelta = (1/tstart)**(1/numsnap)
	if isinstance(snapshot, np.ndarray):
		antwoord = np.zeros(len(snapshot))
		antwoord[snapshot!=numsnap] = 1./(tstart*tdelta**snapshot[snapshot!=numsnap]) -1
		return antwoord
	if snapshot == numsnap:
		return 0
	return 1/(tstart*tdelta**snapshot) - 1

def redshift_to_snapshot(redshift, zstart=30, numsnap=200):
	tstart = (1/(1+zstart))
	tdelta = (1/tstart)**(1/numsnap)
	if isinstance(redshift, np.ndarray):
		antwoord = np.zeros(len(redshift))
		antwoord = np.log10(1./(redshift + 1.)/tstart)/np.log10(tdelta)
		return np.rint(antwoord).astype(int)
	return np.int(np.rint(np.log10(1./(redshift + 1.)/tstart)/np.log10(tdelta)))

def find_treelengths(nbtree):
	if 'Treelength' in nbtree.keys():
		return 0
	nbtree['TreeLength'] = np.zeros(len(nbtree['HaloIndex']))
	for i in range(len(nbtree['HaloIndex'])):
		nbtree['TreeLength'][i] = len(np.where(nbtree['HaloIndex'][i, :]>0)[0])


def find_first_crossing(mh_tree, nbtree, zstart=30, numsnap=200, times_r200=1, new_format=True, overwrite=False):
	if new_format:
		if (overwrite==False) and ('%1.1fxR200'%times_r200 in nbtree.keys()):
			return 0
		nbtree['born_within_%1.1f'%times_r200] = np.array([False]*len(nbtree['Distance'])).astype(bool)
		nbtree['no_crossing_%1.1f'%times_r200] = np.array([False]*len(nbtree['Distance'])).astype(bool)
		nbtree['%1.1fxR200'%times_r200] = np.ones(len(nbtree['Distance'])).astype(int)*(len(mh_tree['R200']) -1)
		for i in range(len(nbtree['Distance'])):
			waar = np.where((nbtree['Distance'][i, :] > 0) & (mh_tree['R200'] > 0))[0]
			temp = np.where((nbtree['Distance'][i, waar] - times_r200*mh_tree['R200'][waar]) < 0)[0]
			if len(temp) > 0:
				if temp[0] == 0:
					nbtree['born_within_%1.1f'%times_r200][i] = True
					nbtree['%1.1fxR200'%times_r200][i] = waar[temp[0]]
				else:
					nbtree['%1.1fxR200'%times_r200][i] = waar[temp[0]-1]
			else:
				nbtree['no_crossing_%1.1f'%times_r200][i] = True
		return 0

	for nb in nbtree.keys():
		waar = np.where((nbtree[nb]['Distance'] > 0) & (mh_tree['R200'] > 0))[0]
		temp = np.where((nbtree[nb]['Distance'][waar] - times_r200*mh_tree['R200'][waar]) < 0)[0]
		if len(temp) == 0:
			nbtree[nb]['FirstCrossing'] = 0
			nbtree[nb]['FirstCrossingSnapshot'] = 200
		else:
			nbtree[nb]['FirstCrossingSnapshot'] = waar[temp[0]]
			nbtree[nb]['FirstCrossing'] = snapshot_to_redshift(waar[temp[0]], zstart=zstart, numsnap=numsnap)

def find_z_crossings(oi, ht, mh_tree, nbtree, times_r200=1, overwrite=False):
	if 'oi_indices' not in nbtree.keys():
		print("oi_indices need to exist in nbtree.keys()")
		return 0

	alloi = oi.output_notmerged['alloi'][nbtree['oi_indices']]
	orbitid = oi.ow.orbitdata['OrbitID'][alloi]
	hostid = mh_tree['HaloIndex'][-1] + 1 + ht.THIDVAL*ht.snapend

	ii_1 = np.where((oi.ow.orbitdata['entrytype'] == 1*times_r200)&
		(oi.ow.orbitdata['OrbitedHaloRootDescen_orig'] == hostid)&
		(np.in1d(oi.ow.orbitdata['OrbitID'], orbitid)))[0]
	ii_m1 = np.where((oi.ow.orbitdata['entrytype'] == -1*times_r200)&
		(oi.ow.orbitdata['OrbitedHaloRootDescen_orig'] == hostid)&
		(np.in1d(oi.ow.orbitdata['OrbitID'], orbitid)))[0]
	if (not 'TimeInside' in nbtree.keys()) or (overwrite==True):
		nbtree['TimeInside'] = np.zeros(len(nbtree['npart']))
		nbtree['TimeOutside'] = np.zeros(len(nbtree['npart']))
		nbtree['Nout'] = np.ones(len(nbtree['npart']))*-1
		nbtree['Nin'] = np.ones(len(nbtree['npart']))*-1
		nbtree['CrossingIn'] = np.zeros_like(nbtree['npart'])
		nbtree['CrossingOut'] = np.zeros_like(nbtree['npart'])
		nbtree['Inside'] = np.zeros_like(nbtree['npart'])
		nbtree['TimeAcc'] = np.zeros(len(nbtree['npart']))
	

	for i in range(len(orbitid)):
		crossing_i = ii_1[np.in1d(oi.ow.orbitdata['OrbitID'][ii_1], orbitid[i])]
		z_crossing = np.array(1./oi.ow.orbitdata['scalefactor'][crossing_i] - 1.)
		nbtree['CrossingIn'][i, redshift_to_snapshot(z_crossing, zstart=ht.zstart, numsnap=ht.snapend)] += 1
		nbtree['Nin'][i] = len(crossing_i)

		crossing_mi = ii_m1[np.in1d(oi.ow.orbitdata['OrbitID'][ii_m1], orbitid[i])]
		z_mcrossing = np.array(1./oi.ow.orbitdata['scalefactor'][crossing_mi] - 1.)
		nbtree['CrossingOut'][i, redshift_to_snapshot(z_mcrossing, zstart=ht.zstart, numsnap=ht.snapend)] += 1
		nbtree['Nout'][i] = len(crossing_mi)

		z_crossing = np.sort(z_crossing)[::-1]
		if len(z_crossing) == 0:
			continue
		z_mcrossing = np.sort(z_mcrossing)[::-1]
		if (len(z_mcrossing) > 0) and (z_crossing[0] < z_mcrossing[0]):
			continue
		snapstemp = redshift_to_snapshot(z_crossing, zstart=ht.zstart, numsnap=ht.snapend)
		snapstempm = redshift_to_snapshot(z_mcrossing, zstart=ht.zstart, numsnap=ht.snapend)
		nbtree['TimeAcc'][i] = ht.timeDifference(z_crossing[0], 0, H0=h*100, Om0=Om0)
		if (len(z_mcrossing) == 0) or (len(z_crossing) - len(z_mcrossing) < 0) or (len(z_crossing) - len(z_mcrossing) > 1):
			nbtree['TimeInside'][i] = nbtree['TimeAcc'][i]
			nbtree['Inside'][i, snapstemp[0]:ht.snapend+1] = 1
			continue

		if len(z_crossing) - len(z_mcrossing) == 1:
			z_mcrossing = np.append(z_mcrossing, 0)

		# timearr = timeDifference(ht, z_crossing, z_mcrossing, H0=h*100, Om0=Om0)
		# nbtree['TimeInside'][i] = np.sum(timearr)
		for j in range(len(z_crossing)):
			if z_mcrossing[j] == 0:
				nbtree['TimeInside'][i] += ht.timeDifference(z_crossing[j], 0, H0=h*100, Om0=Om0)
				nbtree['Inside'][i, snapstemp[j]:ht.snapend+1] = 1
			else:
				nbtree['TimeInside'][i] += ht.timeDifference(z_crossing[j], z_mcrossing[j], H0=h*100, Om0=Om0)
				nbtree['Inside'][i, snapstemp[j]:snapstempm[j]] = 1	

	nbtree['TimeInside'] *=  s_to_yr / 1.e9
	nbtree['TimeAcc'] *= s_to_yr / 1.e9
	nbtree['TimeOutside'] = nbtree['TimeAcc'] - nbtree['TimeInside']

	nbtree['f_tin'] = np.ones(len(nbtree['npart']))*-1
	waar = np.where((nbtree['TimeInside'] > 0)&(nbtree['TimeOutside']>=0))[0]
	nbtree['f_tin'][waar] = nbtree['TimeInside'][waar]/(nbtree['TimeInside'][waar] + nbtree['TimeOutside'][waar])

def find_z_crossings_merger(ht, oi, mh_tree, nbtree_merger, times_r200=1, overwrite=False):
	if 'oi_indices' not in nbtree_merger.keys():
		print("oi_indices need to exist in nbtree_merger.keys()")
		return 0
	if 'redshift(merged)' not in nbtree_merger.keys():
		print("redshift(merged) is not computed")
		return 0

	alloi = oi.output_merged['alloi'][nbtree_merger['oi_indices']]
	orbitid = oi.ow.orbitdata['OrbitID'][alloi]
	hostid = mh_tree['HaloIndex'][-1] + 1 + ht.THIDVAL*ht.snapend

	ii_1 = np.where((oi.ow.orbitdata['entrytype'] == 1*times_r200)&(oi.ow.orbitdata['OrbitedHaloRootDescen_orig'] == hostid)&
		(np.in1d(oi.ow.orbitdata['OrbitID'], orbitid)))[0]
	ii_m1 = np.where((oi.ow.orbitdata['entrytype'] == -1*times_r200)&(oi.ow.orbitdata['OrbitedHaloRootDescen_orig'] == hostid)&
		(np.in1d(oi.ow.orbitdata['OrbitID'], orbitid)))[0]

	if (not 'TimeInside' in nbtree_merger.keys()) or (overwrite==True):
		nbtree_merger['TimeInside'] = np.zeros(len(nbtree_merger['npart']))
		nbtree_merger['TimeOutside'] = np.zeros(len(nbtree_merger['npart']))
		nbtree_merger['Nout'] = np.ones(len(nbtree_merger['npart']))*-1
		nbtree_merger['Nin'] = np.ones(len(nbtree_merger['npart']))*-1
		nbtree_merger['CrossingIn'] = np.zeros_like(nbtree_merger['npart'])
		nbtree_merger['CrossingOut'] = np.zeros_like(nbtree_merger['npart'])
		nbtree_merger['Inside'] = np.zeros_like(nbtree_merger['npart'])
		nbtree_merger['TimeAcc'] = np.zeros(len(nbtree_merger['npart']))

	snapmerge = redshift_to_snapshot(nbtree_merger['redshift(merged)'], zstart=ht.zstart, numsnap=ht.snapend)

	for i in range(len(orbitid)):
		crossing_i = ii_1[np.in1d(oi.ow.orbitdata['OrbitID'][ii_1], orbitid[i])]
		z_crossing = np.array(1./oi.ow.orbitdata['scalefactor'][crossing_i] - 1.)
		nbtree_merger['CrossingIn'][i, redshift_to_snapshot(z_crossing, zstart=ht.zstart, numsnap=ht.snapend)] += 1
		nbtree_merger['Nin'][i] = len(crossing_i)

		crossing_mi = ii_m1[np.in1d(oi.ow.orbitdata['OrbitID'][ii_m1], orbitid[i])]
		z_mcrossing = np.array(1./oi.ow.orbitdata['scalefactor'][crossing_mi] - 1.)
		nbtree_merger['CrossingOut'][i, redshift_to_snapshot(z_mcrossing, zstart=ht.zstart, numsnap=ht.snapend)] += 1
		nbtree_merger['Nout'][i] = len(crossing_mi)

		z_crossing = np.sort(z_crossing)[::-1]
		if len(z_crossing) == 0:
			continue
		z_mcrossing = np.sort(z_mcrossing)[::-1]
		if (len(z_mcrossing) > 0) and (z_crossing[0] < z_mcrossing[0]):
			continue
		snapstemp = redshift_to_snapshot(z_crossing, zstart=ht.zstart, numsnap=ht.snapend)
		snapstempm = redshift_to_snapshot(z_mcrossing, zstart=ht.zstart, numsnap=ht.snapend)
		nbtree_merger['TimeAcc'][i] = ht.timeDifference(z_crossing[0], nbtree_merger['redshift(merged)'][i], H0=h*100, Om0=Om0)

		if len(z_mcrossing) == 0:
			nbtree_merger['TimeInside'][i] = nbtree_merger['TimeAcc'][i]
			nbtree_merger['Inside'][i, snapstemp[0]:snapmerge[i]+1] = 1
			continue

		for j in range(len(z_crossing)):
			if j > len(z_mcrossing) - 1:
				nbtree_merger['TimeInside'][i] += ht.timeDifference(z_crossing[j], nbtree_merger['redshift(merged)'][i], H0=h*100, Om0=Om0)
				nbtree_merger['Inside'][i, snapstemp[j]:snapmerge[i]+1] = 1
			else:
				nbtree_merger['TimeInside'][i] += ht.timeDifference(z_crossing[j], z_mcrossing[j], H0=h*100, Om0=Om0)
				nbtree_merger['Inside'][i, snapstemp[j]:snapstempm[j]] = 1

	nbtree_merger['TimeInside'] *=  s_to_yr / 1.e9
	nbtree_merger['TimeAcc'] *= s_to_yr / 1.e9
	nbtree_merger['TimeOutside'] = nbtree_merger['TimeAcc'] - nbtree_merger['TimeInside']

	nbtree_merger['f_tin'] = np.ones(len(nbtree_merger['npart']))*-1
	waar = np.where((nbtree_merger['TimeInside'] > 0)&(nbtree_merger['TimeOutside']>=0))[0]
	nbtree_merger['f_tin'][waar] = nbtree_merger['TimeInside'][waar]/(nbtree_merger['TimeInside'][waar] + nbtree_merger['TimeOutside'][waar])

def find_mask_vel_indices(d_output_vel, mh_tree, nbtree, zstart=30, numsnap=200, times_r200=1, min_treelength=0):
	find_first_crossing(mh_tree, nbtree, zstart=zstart, numsnap=numsnap, times_r200=times_r200)
	find_treelengths(nbtree)
	return d_output_vel['allvel'][np.where((nbtree['born_within_%1.1f'%times_r200]==False) &
		(nbtree['TreeLength']>=min_treelength))[0]]

def find_values_crossing(nbtree, nbtree_indices_label='1.0xR200', hp=None, mh_tree=None, 
	datasets=['Vmax', 'Rmax', 'M200', 'npart', 'redshift', 'Efrac', 'npart_bound'], masstype='npart_bound',
	add_label='(R200)', no_crossing_peak='no_crossing_1.0', overwrite=False):

	welke = np.where((nbtree['no_crossing_1.0'] == False) & (nbtree['born_within_1.0'] == False))[0]
	hp_indices = nbtree['HaloIndex'][welke, -1]
	nbtree_indices = nbtree['1.0xR200'][welke]
	datasetshier = np.copy(datasets)
	if 'VelRad' in nbtree.keys():
		datasetshier = np.append(datasetshier, 'VelRad')
	if 'Vcirc' in nbtree.keys():
		datasetshier = np.append(datasetshier, 'Vcirc')
	if 'Eorb' in nbtree.keys():
		datasetshier = np.append(datasetshier, 'Eorb')
	if 'SpecificEorb' in nbtree.keys():
		datasetshier = np.append(datasetshier, 'SpecificEorb')
	if 'SpecificAngularMomentum' in nbtree.keys():
		datasetshier = np.append(datasetshier, 'SpecificAngularMomentum')
	if 'Eta' in nbtree.keys():
		datasetshier = np.append(datasetshier, 'Eta')
	if 'Phi' in nbtree.keys():
		datasetshier = np.append(datasetshier, 'Phi')
	if 'Theta' in nbtree.keys():
		datasetshier = np.append(datasetshier, 'Theta')
	if masstype not in datasets:
		datasetshier = np.append(datasetshier, masstype)
	for ds in datasetshier:
		if ds in ['Coord', 'Vel']:
			continue
		if ds in ['redshift', 'snapshot']:
			if mh_tree is None:
				print('If you want to add a redshift, you need to give mh_tree')
				return 0
			if (hp is not None):
				if (ds+add_label not in hp.keys()) or (overwrite == True):
					hp[ds+add_label] = np.ones_like(hp['M200'])*-1
				hp[ds+add_label][hp_indices] = mh_tree[ds][nbtree_indices]
			if ds+add_label not in nbtree.keys():
				nbtree[ds+add_label] = np.ones(len(nbtree['HaloIndex']))*-1
			nbtree[ds+add_label][welke] = mh_tree[ds][nbtree_indices]
			continue

		if (hp is not None) and ((ds+add_label not in hp.keys()) or (overwrite == True)):
			hp[ds+add_label] = np.ones_like(hp[ds])*-1
			if ds not in ['VelRad', 'Eorb', 'SpecificEorb', 'SpecificAngularMomentum', 'Vcirc', 'Eta', 'Phi', 'Theta', 'Efrac']:
				hp[ds+'change'] = np.ones_like(hp[ds])*-1
				hp[ds+'$_{peak}$'] = np.ones_like(hp[ds])*-1
		if ds+add_label not in nbtree.keys():
			nbtree[ds+add_label] = np.ones(len(nbtree[ds]))*-1
			if ds not in ['VelRad', 'Eorb', 'SpecificEorb', 'SpecificAngularMomentum', 'Vcirc', 'Eta', 'Phi', 'Theta', 'Efrac']:			
				nbtree[ds+'change'] = np.ones(len(nbtree[ds]))*-1
				nbtree[ds+'$_{peak}$'] = np.ones(len(nbtree[ds]))*-1
			if (ds == masstype) and (mh_tree is not None):
				nbtree['Mnb/Mhost'+add_label] = np.ones(len(nbtree[ds]))*-1

		if (hp is not None):
			hp[ds+add_label][hp_indices] = nbtree[ds][welke, nbtree_indices]
		nbtree[ds+add_label][welke] = nbtree[ds][welke, nbtree_indices]
		if (ds == masstype) and (mh_tree is not None):
			nbtree['Mnb/Mhost'+add_label][welke] = nbtree[ds+add_label][welke]/mh_tree[ds][nbtree_indices]
		# if no_crossing_peak is not None:
		# 	no_crossing = np.where(nbtree[no_crossing_peak])[0]
		# 	hp[ds+add_label][hp_indices[no_crossing]] = np.amax(nbtree[ds][no_crossing, :], axis=0)
		if ds in ['VelRad', 'Eorb', 'SpecificEorb', 'SpecificAngularMomentum', 'Vcirc', 'Eta', 'Phi', 'Theta', 'Efrac']:
			continue
		for i in range(len(welke)):
			if (hp is not None):
				hp[ds+'$_{peak}$'][hp_indices[i]] = np.max(nbtree[ds][welke[i], :(nbtree_indices[i]+1)])
			nbtree[ds+'$_{peak}$'][welke[i]] = np.max(nbtree[ds][welke[i], :(nbtree_indices[i]+1)])
		if (hp is not None):
			hp[ds+'change'][hp_indices] = hp[ds][hp_indices]/hp[ds+add_label][hp_indices]
		nbtree[ds+'change'][welke] = nbtree[ds][welke, -1]/nbtree[ds+add_label][welke]

def find_merging_redshift(oi, nbtree_merger):
	if len(nbtree_merger['HaloIndex']) == len(oi.output_merged['scalefactor_merged']):
		nbtree_merger['redshift(merged)'] = 1./oi.output_merged['scalefactor_merged'] - 1
	else:
		if 'oi_indices' not in nbtree_merger.keys():
			print("You need to link the indices of the nbtree_merger to the oi.output_merged catalog")
			return 0
		else:
			nbtree_merger['redshift(merged)'] = 1./oi.output_merged['scalefactor_merged'][nbtree_merger['oi_indices']] - 1.

def find_formation_time_mhtree(mh_tree, zstart=30, numsnap=200, hp=None, mass_frac=[0.5], masstype='npart'):
	indices = mh_tree['HaloIndex'][-1]
	mah_temp = mh_tree[masstype][:]
	for mf in mass_frac:
		waar = np.where(mah_temp > mf * mah_temp[-1])[0]
		mh_tree['z$_{%1.2f}$'%mf] = snapshot_to_redshift(waar[0], zstart=zstart, numsnap=numsnap)

def find_formation_time_nbtree(nbtree, zstart=30, numsnap=200, hp=None, mass_frac=[0.5], masstype='npart'):
	indices = nbtree['HaloIndex'][:, -1]
	for mf in mass_frac:
		if (hp is not None) and ('z$_{%1.2f}$'%mf not in hp.keys()):
			hp['z$_{%1.2f}$'%mf] = np.zeros(len(hp['M200']))
		if 'z$_{%1.2f}$'%mf not in nbtree.keys():
			nbtree['z$_{%1.2f}$'%mf] = np.zeros(len(nbtree[masstype]))

	for i in range(len(indices)):
		mah_temp = nbtree[masstype][i, :]
		for mf in mass_frac:
			waar = np.where(mah_temp > mf * mah_temp[-1])[0]
			if hp is not None:
				hp['z$_{%1.2f}$'%mf][indices[i]] = snapshot_to_redshift(waar[0], zstart=zstart, numsnap=numsnap)
			nbtree['z$_{%1.2f}$'%mf][i] = snapshot_to_redshift(waar[0], zstart=zstart, numsnap=numsnap)

def writeDataToHDF5(path, name, ht, hosts, mainhaloes, mh_tree, d_output, d_output_merger, nbtree, nbtree_merger, mask_vel_indices=None):
	haloprop = h5py.File(path+name, 'w')

	masses = ht.halotree[ht.snapend].hp['M200'][mainhaloes]

	header = haloprop.create_group("Header")
	header.attrs.__setitem__('NumHosts', len(hosts))
	header.attrs.__setitem__('MassRange', np.array([np.min(masses), np.max(masses)]))

	mass_frac = [0.75, 0.5, 0.25]
	datatemp = {}
	for mf in mass_frac:
		datatemp['z$_{%1.2f}$'%mf] = np.zeros(len(hosts))
		i = -1
		for key in mh_tree.keys():
			i += 1
			datatemp['z$_{%1.2f}$'%mf][i] = mh_tree[key]['z$_{%1.2f}$'%mf]
		haloprop.create_dataset('z$_{%1.2f}$'%mf, data=datatemp['z$_{%1.2f}$'%mf])

	haloprop.create_dataset('M200', data=np.array(masses))
	haloprop.create_dataset('HostsRootProgen', data = np.array(hosts))
	haloprop.create_dataset('HostsRootDescen', data = np.array(mainhaloes))
	if mask_vel_indices is not None:
		haloprop.create_dataset('MaskVelIndices', data = np.array(mask_vel_indices))
	outnbm = haloprop.create_group('MergedSort')
	for key in d_output_merger.keys():
		outnbm.create_dataset(str(key), data = np.array(d_output_merger[key]))
	outnbnm = haloprop.create_group('NotMergedSort')
	for key in d_output.keys():
		outnbnm.create_dataset(str(key), data = np.array(d_output[key]))
	for key in mh_tree.keys():
		hp2 = haloprop.create_group(str(key))
		mh = hp2.create_group('MainHaloTree')
		nb = hp2.create_group('NeighbourTree')
		nbm = hp2.create_group('MergedTree')
		for key2 in mh_tree[key].keys():
			if key2 in ['z$_{%1.2f}$'%mass_frac[0], 'z$_{%1.2f}$'%mass_frac[1], 'z$_{%1.2f}$'%mass_frac[2]]:
				continue
			mh.create_dataset(key2.replace("/", ""), data = np.array(mh_tree[key][key2]))
		for key2 in nbtree[key].keys():
			nb.create_dataset(key2.replace("/", ""), data = np.array(nbtree[key][key2]))
		for key2 in nbtree_merger[key].keys():
			nbm.create_dataset(key2.replace("/", ""), data = np.array(nbtree_merger[key][key2]))
	haloprop.close()

def readDataHDF5(path, name, datasets=None, datasets_host=None, read_only_header=False, read_only_hostinfo=False):
	haloprop = h5py.File(path+name, 'r')
	
	outnbm = haloprop['MergedSort'.encode('utf-8')]
	out_merged = {}
	for key in outnbm.id:
		out_merged[key.decode('utf-8')] = outnbm[key][:]

	out_notmerged = {}
#	if 'NotMergedSort'.encode('utf-8') in haloprop.id:
	# outnbnm = haloprop['NotMergedSort'.encode('utf-8')]
	# for key in outnbnm.id:
	# 	out_notmerged[key.decode('utf-8')] = outnbnm[key][:]
	
	Header = haloprop['Header']
	massrange = Header.attrs['MassRange'][:]
	numhosts = Header.attrs['NumHosts']
	if read_only_header:
		return numhosts, massrange

	d_mh_tree = {}
	d_nbtree = {}
	d_nbtree_merger = {}
	d_hostinfo = {}
	d_hostinfo['massrange'] = massrange
	d_hostinfo['numhosts'] = numhosts
	mvi = 0
	if datasets_host is None:
		datasets_host = datasets
	for key in haloprop.id:
		if isinstance(haloprop[key].id, h5py.h5g.GroupID):
			if read_only_hostinfo:
				continue
			if key.decode('utf_8') in ['Header', 'MergedSort', 'NotMergedSort']:
				continue
			hp2 = haloprop[key]
			mh = hp2['MainHaloTree'.encode('utf-8')]
			nb = hp2['NeighbourTree'.encode('utf-8')]
			nbm = hp2['MergedTree'.encode('utf-8')]
			d_mh_tree[int(key.decode('utf-8'))] = {}
			d_nbtree[int(key.decode('utf-8'))] = {}
			d_nbtree_merger[int(key.decode('utf-8'))] = {}

			for key2 in mh.id:
				if datasets_host is None:
					d_mh_tree[int(key.decode('utf-8'))][key2.decode('utf-8')] = mh[key2][:]
				elif key2.decode('utf-8') in datasets_host:
					d_mh_tree[int(key.decode('utf-8'))][key2.decode('utf-8')] = mh[key2][:]
			for key2 in nb.id:
				if datasets is None:
					d_nbtree[int(key.decode('utf-8'))][key2.decode('utf-8')] = nb[key2][:]
				elif key2.decode('utf-8') in datasets:
					d_nbtree[int(key.decode('utf-8'))][key2.decode('utf-8')] = nb[key2][:]
			for key2 in nbm.id:
				if datasets is None:
					d_nbtree_merger[int(key.decode('utf-8'))][key2.decode('utf-8')] = nbm[key2][:]
				elif key2.decode('utf-8') in datasets:
					d_nbtree_merger[int(key.decode('utf-8'))][key2.decode('utf-8')] = nbm[key2][:]

		elif isinstance(haloprop[key].id, h5py.h5d.DatasetID):
			if key.decode('utf-8') == 'MaskVelIndices':
				if read_only_hostinfo:
					continue
				mvi = haloprop['MaskVelIndices'.encode('utf-8')][:]
			else:
				d_hostinfo[key.decode('utf-8')] = haloprop[key][:]

	haloprop.close()

	if read_only_hostinfo:
		return d_hostinfo
	return d_hostinfo, out_notmerged, out_merged, d_mh_tree, d_nbtree, d_nbtree_merger, mvi

def find_merging_distance(nbtree_merger, mh_tree):
	nbtree_merger['Distance(merged)'] = np.zeros(len(nbtree_merger['Distance']))
	nbtree_merger['Distance(merged)/R200'] = np.zeros(len(nbtree_merger['Distance']))
	for i in range(len(nbtree_merger['Distance(merged)'])):
		waar = np.where(nbtree_merger['Distance'][i, :]>=0)[0]
		nbtree_merger['Distance(merged)/R200'][i] = nbtree_merger['Distance'][i, waar[-1]]/mh_tree['R200'][waar[-1]]
		nbtree_merger['Distance(merged)'][i] = nbtree_merger['Distance'][i, waar[-1]]

def find_time_since_first_crossing(d_output_vel, ht, snapshot=200, mainhalo=None, mh_tree=None, nbtree=None, 
	keys=['orbital satellites', 'first infall satellites', 'ex-satellite']):
	if mh_tree is None:
		mh_tree, nbtree = ht.neighbourtreemaker2(mainhalo, ht.halo_key(d_output_vel['allvel'], snapshot), datasets = 
			['R200', 'Coord', 'redshift'])
	find_first_crossing(mh_tree, nbtree, zstart=self.zstart, numsnap=ht.snapend, times_r200=1, overwrite=False)
	if 'time_since_acc' not in ht.halotree[snapshot].hp.keys():
		ht.halotree[snapshot].hp['time_since_acc'] = np.zeros(len(ht.halotree[snapshot].hp['M200']))
	if 'z$_{acc}$' not in ht.halotree[snapshot].hp.keys():
		ht.halotree[snapshot].hp['z$_{acc}$'] = np.zeros(len(ht.halotree[snapshot].hp['M200']))
	if 'CrossingTime' not in ht.halotree[snapshot].hp.keys():
		ht.halotree[snapshot].hp['CrossingTime'] = np.ones(len(ht.halotree[snapshot].hp['M200']))*-1
		waar = np.where(ht.halotree[snapshot].hp['M200'] > 0.000001)[0]
		ht.halotree[snapshot].hp['CrossingTime'][waar] = (2.*ht.halotree[snapshot].hp['R200'][waar]*Mpc_to_km /
						np.sqrt(G_Mpc_km2_Msi_si2*ht.halotree[snapshot].hp['M200'][waar]*1e10/
						ht.halotree[snapshot].hp['R200'][waar]))*s_to_yr/1.e9
	if 'tacc_cross' not in ht.halotree[snapshot].hp.keys():
		ht.halotree[snapshot].hp['tacc_cross'] = np.ones(len(ht.halotree[snapshot].hp['M200']))*-1

	for key in keys:
		for halo in range(len(d_output_vel[key])):
			fc_i = nbtree['1.0xR200'][d_output_vel[key][halo]]
			zacc = mh_tree['redshift'][fc_i]
			znu = 0
			times = ht.timeDifference(zacc, znu, H0=h*100, Om0=Om0)*s_to_yr / 1.e9
			ht.halotree[snapshot].hp['time_since_acc'][d_output_vel['allvel'][d_output_vel[key][halo]]] = times
			ht.halotree[snapshot].hp['z$_{acc}$'][d_output_vel['allvel'][d_output_vel[key][halo]]] = zacc
		
		ht.halotree[snapshot].hp['tacc_cross'][d_output_vel['allvel'][d_output_vel[key]]] = (
			ht.halotree[snapshot].hp['time_since_acc'][d_output_vel['allvel'][d_output_vel[key]]] / 
			ht.halotree[snapshot].hp['CrossingTime'][mainhalo])

def find_preprocess(ht, nbtree, mh_tree, merged=False, find_infalling_groups=True, min_neighbours=50):
	"""
	This function adds a field to nbtree, describing if the neighbour halo was preprocessed,
	postprocessed or did not have any interaction before the encounter with the main host.

	To be preprocessed or postprocessed, the particle number at encounter AND at z=0 of the host must be larger.
	If the preprocess host has merged with the main host, the preprocess host mass at encounter must be larger
	than the mass of the neighbour, but it also needs to be larger than the mass of the neighbour at z=0.
	Preprocessing must happen BEFORE first infall
	Postprocessing must happen AFTER first infall

	In case of merged=True, all neighbours are assumed to have merged with the host. The preprocessed host mass
	at encounter must be larger than the mass of the neighbour, but it also needs to be larger than the neighbours'
	mass at its first infall. A postprocessed host only has to be larger than the neighbour at time of encounter.

	It saves the haloIndex of the host it last encountered.
	"""
	if 'hostHaloIndex' not in nbtree.keys():
		print("You need to build your neighbour trees with hostHaloIndexs")
		return 0
	if 'snapshot(R200)' not in nbtree.keys():
		print("You need to run find_crossing_values() to get snapshot(R200)")
		return 0

	hostids = mh_tree['HaloIndex']%ht.THIDVAL# + np.arange(ht.snapend+1)*ht.THIDVAL
	hostids[mh_tree['HaloIndex']==-1] = -1

	#Find all interactions that were not with the main host (For all times!!!)
	matrix_temp = nbtree['hostHaloIndex']%ht.THIDVAL - hostids[None, :]

	i_halonb, snapnb = np.where((nbtree['hostHaloIndex'] > 0) & (matrix_temp != 0))
	#poss_prehost_index = nbtree['hostHaloIndex'][i_halonb, snapnb]# + snapnb*ht.THIDVAL
	neighbour_index = nbtree['HaloIndex'][i_halonb, snapnb]

	if merged:
		npart_halonb_end = nbtree['npart(R200)'][i_halonb]*nbtree['Efrac(R200)'][i_halonb]
	else:
		npart_halonb_end = nbtree['npart'][i_halonb, -1]*nbtree['Efrac'][i_halonb, -1]
	snap_halonb_fi = nbtree['snapshot(R200)'][i_halonb]
	snap_halonb_fi[snap_halonb_fi < 0] = ht.snapend + 1
	npart_halonb = nbtree['npart'][i_halonb, snapnb]*nbtree['Efrac'][i_halonb, snapnb]

	ht.readData(datasets=['RootHead'])

	poss_prehost_roothead = np.zeros(len(i_halonb)).astype(int)
	halonb_roothead = np.zeros(len(i_halonb)).astype(int)
	poss_prehost_index = (ht.key_halo(poss_prehost)[0]).astype(int)
	poss_prehost_npart = np.zeros(len(i_halonb)).astype(int)
	snaps = np.unique(snapnb)
	for snap in snaps:
		#Find possible preprocessed host roothead for every host halo index per snapshot
		waar = np.where(snapnb == snap)[0]
		poss_prehost_roothead[waar] = ht.halotree[snap].hp['RootHead'][poss_prehost_index[waar]]
		if np.max(neighbour_index[waar] >= len(ht.halotree[snap].hp['RootHead'])):
			print(snap, len(ht.halotree[snap].hp['RootHead']), neighbour_index[waar], snapnb[waar])
			continue
		#Every roothead of the neighbour
		halonb_roothead[waar] = ht.halotree[snap].hp['RootHead'][neighbour_index[waar]]
		#Amount of bound particles within the possible preprocessed host at the snapshot of interaction
		poss_prehost_npart[waar] = ht.halotree[snap].hp['npart'][poss_prehost_index[waar]]*ht.halotree[snap].hp['Efrac'][poss_prehost_index[waar]]

	if merged:
		#Find out if the interaction took place before or after the first time the nb crossed R200 of its host
		#Find out if the host had more bound particle than the nb and for the preprocessed also more than at the time of accretion with main host
		i_preprocessed = np.where((snap_halonb_fi >= snapnb)&(poss_prehost_npart > npart_halonb)&(poss_prehost_npart > npart_halonb_end))[0]
		i_postprocessed = np.where((snap_halonb_fi < snapnb)&(poss_prehost_npart > npart_halonb))[0]
	else:
		#Find indices of the roothead
		poss_prehost_head_index, poss_prehost_head_snap = ht.key_halo(poss_prehost_roothead)
		#Amount of bound particles
		poss_prehost_head_npart = np.zeros(len(poss_prehost_head_index))
		snaps = np.unique(poss_prehost_head_snap)
		waar200 = np.zeros(0)
		for snap in snaps:
			waar = np.where(poss_prehost_head_snap == snap)[0]
			poss_prehost_head_npart[waar] = ht.halotree[snap].hp['npart'][poss_prehost_head_index[waar]]*ht.halotree[snap].hp['Efrac'][poss_prehost_head_index[waar]]
			if snap == ht.snapend:
				waar200 = np.copy(waar)
		#Check if the prehost is merged with the mainhost, if so, use the particle mass of the prehost at interaction
		if len(waar200) > 0:
			merged_with_mainhost = waar200[np.where(poss_prehost_head_index[waar200] == mh_tree['HaloIndex'][-1])[0]]
			poss_prehost_head_npart[merged_with_mainhost] = poss_prehost_npart[merged_with_mainhost]
			npart_halonb_end[merged_with_mainhost] = nbtree['npart(R200)'][i_halonb[merged_with_mainhost]]*nbtree['Efrac(R200)'][i_halonb[merged_with_mainhost]]
		else:
			merged_with_mainhost = np.zeros(0)

		#Look for things that don't have a prehost roothead the same as the nb
		#Interactions that happened before first infall (or after in postprocessed
		#Interactions where the host was larger at time of interaction
		#Hosts that are larger at z=0
		i_preprocessed = np.where(#(np.in1d(poss_prehost_roothead, halonb_roothead)==False)&
			(snap_halonb_fi >= snapnb)&(npart_halonb < poss_prehost_npart)&
			(npart_halonb_end < poss_prehost_head_npart))[0]
		i_postprocessed = np.where(#(np.in1d(poss_prehost_roothead, halonb_roothead)==False)&
			(snap_halonb_fi < snapnb)&(npart_halonb < poss_prehost_npart))[0]

		waarmerged = i_preprocessed[np.in1d(i_preprocessed, merged_with_mainhost)]
		poss_prehost_roothead[waarmerged] = poss_prehost_index[waarmerged] + snapnb[waarmerged]*ht.THIDVAL
		waarmerged = i_postprocessed[np.in1d(i_postprocessed, merged_with_mainhost)]
		poss_prehost_roothead[waarmerged] = poss_prehost_index[waarmerged] + snapnb[waarmerged]*ht.THIDVAL

	nbtree['Preprocessed'] = np.ones(len(nbtree['npart'])).astype(int)*-1
	nbtree['NPreHosts'] = np.ones(len(nbtree['npart'])).astype(int)*-1
	nbtree['Postprocessed'] = np.ones(len(nbtree['npart'])).astype(int)*-1
	nbtree['NPostHosts'] = np.ones(len(nbtree['npart'])).astype(int)*-1
	nbtree['LengthPreprocessed'] = np.ones(len(nbtree['npart'])).astype(int)*-1
	nbtree['LengthPostprocessed'] = np.ones(len(nbtree['npart'])).astype(int)*-1

	nbtree['Preprocessed'][i_halonb[i_preprocessed]] = poss_prehost_roothead[i_preprocessed]
	i_htemp, n_temp = np.unique(i_halonb[i_preprocessed], return_counts=True)
	for i in i_htemp:
		waar = i_preprocessed[np.where(i_halonb[i_preprocessed] == i)[0]]
		nbtree['NPreHosts'][i] = len(np.unique(poss_prehost_roothead[waar]))
	nbtree['LengthPreprocessed'][i_htemp] = n_temp
	nbtree['Postprocessed'][i_halonb[i_postprocessed]] = poss_prehost_roothead[i_postprocessed]
	i_htemp, n_temp = np.unique(i_halonb[i_postprocessed], return_counts=True)
	for i in i_htemp:
		waar = i_postprocessed[np.where(i_halonb[i_postprocessed] == i)[0]]
		nbtree['NPostHosts'][i] = len(np.unique(poss_prehost_roothead[waar]))
	nbtree['LengthPostprocessed'][i_htemp] = n_temp


	if find_infalling_groups == False:
		return 0

	"""
	The second part uses the poss_prehost_roothead to find large infalling haloes and to order the satellites falling into the main host 
	as a group
	"""

	# group_host, n_sats = np.unique(poss_prehost_roothead[i_preprocessed], return_counts=True)
	# waar = np.where((n_sats > min_neighbours)&((group_host/ht.THIDVAL).astype(int)>0))[0]
	# if len(waar) == 0:
	# 	return 0
	# mhtreetemp, nbtreetemp = ht.neighbourtreemaker2(mh_tree['HaloIndex'][-1]+ht.snapend*ht.THIDVAL,
	# 	group_host[waar], datasets=['Coord', 'R200'])

	# find_first_crossing(mhtreetemp, nbtreetemp, times_r200=1)
	# crossings = np.where((nbtreetemp['no_crossing_1.0'] == False) & (nbtreetemp['born_within_1.0'] == False))[0]
	# waar = waar[crossings]

	# if merged:
	# 	addlab = 'Merged'
	# else:
	# 	addlab = ''

	# mh_tree['InfallingGroup'+addlab] = group_host[waar]
	# mh_tree['NsatsGroup'+addlab] = n_sats[waar]

	# datasets = ['Vmax', 'Rmax', 'Vel', 'Coord', 'M200', 'R200', 'redshift', 'snapshot', 'npart', 'Efrac']
	# mhtreetemp, nbtreetemp = ht.neighbourtreemaker2(mh_tree['HaloIndex'][-1]+ht.snapend*ht.THIDVAL,
	# 	group_host[waar], datasets=datasets)
	# mhtreetemp['npart_bound'] = mhtreetemp['Efrac']*mhtreetemp['npart']
	# nbtreetemp['npart_bound'] = nbtreetemp['Efrac']*nbtreetemp['npart']
	# find_first_crossing(mhtreetemp, nbtreetemp, times_r200=1)
	# find_values_crossing(nbtreetemp, mh_tree=mhtreetemp, datasets=datasets, add_label='(R200)', masstype='npart_bound',
	# 	nbtree_indices_label='1.0xR200')

	# mh_tree['GroupDistance'+addlab] = nbtreetemp['Distance']
	# mh_tree['GroupX'+addlab] = nbtreetemp['X']
	# mh_tree['GroupY'+addlab] = nbtreetemp['Y']
	# mh_tree['GroupZ'+addlab] = nbtreetemp['Z']
	# mh_tree['VelRad'+addlab] = nbtreetemp['VelRad']
	# mh_tree['GroupR200'+addlab] = nbtreetemp['R200']
	# mh_tree['GroupM200'+addlab] = nbtreetemp['M200']
	# mh_tree['GroupVmax(R200)'+addlab] = nbtreetemp['Vmax(R200)']
	# mh_tree['GroupRmax(R200)'+addlab] = nbtreetemp['Rmax(R200)']
	# mh_tree['GroupRedshift(R200)'+addlab] = nbtreetemp['redshift(R200)']
	# mh_tree['GroupEta(R200)'+addlab] = nbtreetemp['Eta(R200)']
	# mh_tree['GroupPhi(R200)'+addlab] = nbtreetemp['Phi(R200)']
	# mh_tree['GroupTheta(R200)'+addlab] = nbtreetemp['Theta(R200)']
	# mh_tree['GroupVelRad(R200)'+addlab] = nbtreetemp['VelRad(R200)']
	# mh_tree['GroupEfrac(R200)'+addlab] = nbtreetemp['Efrac(R200)']
	# mh_tree['GroupMnbMhost(R200)'+addlab] = nbtreetemp['Mnb/Mhost(R200)']


def fixSatelliteProblems(hp, TEMPORALHALOIDVAL=1000000000000, boxsize=32):
	hp['Coord'] = hp['Coord']%boxsize
	if len(hp['Coord']) == 0:
		return 0
	halotree = cKDTree(hp['Coord'], boxsize=boxsize)

	# toolarge = np.where(hp['R200'] > hp['R200'][np.argmax(hp['npart'])]*1.2)[0]
	# #print(i, toolarge)
	# if len(toolarge) != 0:
	# 	for tl in toolarge:
	# 		hp['hostHaloIndex'][hp['HaloIndex'][tl]==hp['hostHaloIndex']] = -2

	for halo in range(len(hp['M200'])):
		if hp['M200'][halo] == -1:
			continue
		buren = np.array(halotree.query_ball_point(hp['Coord'][halo], r = 2*hp['R200'][halo]))
		if len(buren) <= 1:
			continue
		buren = buren[hp['R200'][buren] != -1]
		if len(buren) == 0:
			continue
		i_largest = np.argmax(hp['npart'][buren])
		index_largest = halo
		buren = np.delete(buren,i_largest)

		coords = hp['Coord'][buren] - hp['Coord'][index_largest]
		coords = np.where(np.abs(coords) > 0.5*boxsize, coords - coords/np.abs(coords)*boxsize, coords)
		rad = np.sqrt(np.sum(coords*coords, axis=1))
		burentemp = np.where(hp['R200'][buren]-rad+hp['R200'][index_largest] > 0)[0]
		if len(burentemp) == 0:
			continue
		buren = buren[burentemp]
		hp['hostHaloIndex'][buren] = index_largest
