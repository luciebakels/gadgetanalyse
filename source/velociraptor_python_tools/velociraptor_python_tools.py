import sys,os,os.path,string,time,re,struct
import math,operator
#from pylab import *
import numpy as np
import h5py #import hdf5 interface
import pandas as pd
from copy import deepcopy
#from collections import deque
#import itertools
#from sklearn.neighbors import NearestNeighbors
import scipy.interpolate as scipyinterp
import scipy.spatial as spatial
#import multiprocessing as mp
#from collections import deque

import cython
from cython.parallel import prange, parallel

#would be good to compile these routines with cython
#try to speed up search
#cimport numpy as np

"""
Copyright (c) 2015 Pascal Jahan Elahi

Routines for reading velociraptor output

"""

"""
    IO Routines
"""
def ReadPropertyFile(basefilename,iverbose=1, desiredfields=[], selected_files=None):
    """
    VELOCIraptor/STF files in various formats
    for example ascii format contains
    a header with
        filenumber number_of_files
        numhalos_in_file nnumhalos_in_total
    followed by a header listing the information contain. An example would be
        ID(1) ID_mbp(2) hostHaloID(3) numSubStruct(4) npart(5) Mvir(6) Xc(7) Yc(8) Zc(9) Xcmbp(10) Ycmbp(11) Zcmbp(12) VXc(13) VYc(14) VZc(15) VXcmbp(16) VYcmbp(17) VZcmbp(18) Mass_tot(19) Mass_FOF(20) Mass_200mean(21) Mass_200crit(22) Mass_BN97(23) Efrac(24) Rvir(25) R_size(26) R_200mean(27) R_200crit(28) R_BN97(29) R_HalfMass(30) Rmax(31) Vmax(32) sigV(33) veldisp_xx(34) veldisp_xy(35) veldisp_xz(36) veldisp_yx(37) veldisp_yy(38) veldisp_yz(39) veldisp_zx(40) veldisp_zy(41) veldisp_zz(42) lambda_B(43) Lx(44) Ly(45) Lz(46) q(47) s(48) eig_xx(49) eig_xy(50) eig_xz(51) eig_yx(52) eig_yy(53) eig_yz(54) eig_zx(55) eig_zy(56) eig_zz(57) cNFW(58) Krot(59) Ekin(60) Epot(61) n_gas(62) M_gas(63) Xc_gas(64) Yc_gas(65) Zc_gas(66) VXc_gas(67) VYc_gas(68) VZc_gas(69) Efrac_gas(70) R_HalfMass_gas(71) veldisp_xx_gas(72) veldisp_xy_gas(73) veldisp_xz_gas(74) veldisp_yx_gas(75) veldisp_yy_gas(76) veldisp_yz_gas(77) veldisp_zx_gas(78) veldisp_zy_gas(79) veldisp_zz_gas(80) Lx_gas(81) Ly_gas(82) Lz_gas(83) q_gas(84) s_gas(85) eig_xx_gas(86) eig_xy_gas(87) eig_xz_gas(88) eig_yx_gas(89) eig_yy_gas(90) eig_yz_gas(91) eig_zx_gas(92) eig_zy_gas(93) eig_zz_gas(94) Krot_gas(95) T_gas(96) Zmet_gas(97) SFR_gas(98) n_star(99) M_star(100) Xc_star(101) Yc_star(102) Zc_star(103) VXc_star(104) VYc_star(105) VZc_star(106) Efrac_star(107) R_HalfMass_star(108) veldisp_xx_star(109) veldisp_xy_star(110) veldisp_xz_star(111) veldisp_yx_star(112) veldisp_yy_star(113) veldisp_yz_star(114) veldisp_zx_star(115) veldisp_zy_star(116) veldisp_zz_star(117) Lx_star(118) Ly_star(119) Lz_star(120) q_star(121) s_star(122) eig_xx_star(123) eig_xy_star(124) eig_xz_star(125) eig_yx_star(126) eig_yy_star(127) eig_yz_star(128) eig_zx_star(129) eig_zy_star(130) eig_zz_star(131) Krot_star(132) tage_star(133) Zmet_star(134)

    then followed by data

    Note that a file will indicate how many files the total output has been split into

    Not all fields need be read in. If only want specific fields, can pass a string of desired fields like
    ['ID', 'Mass_FOF', 'Krot']
    #todo still need checks to see if fields not present and if so, not to include them or handle the error
    """
    #this variable is the size of the char array in binary formated data that stores the field names
    CHARSIZE=40

    start = time.clock()
    inompi=True
    if (iverbose): print("reading properties file",basefilename)
    filename=basefilename+".properties"
    #load header
    if (os.path.isfile(filename)==True):
        numfiles=0
    else:
        filename=basefilename+".properties"+".0"
        if selected_files is not None:
            filename = basefilename + ".properties.%i" %selected_files[0]
        inompi=False
        if (os.path.isfile(filename)==False):
            print("file not found")
            return []
    byteoffset=0
    #used to store fields, their type, etc
    fieldnames=[]
    fieldtype=[]
    fieldindex=[]

    #load hdf file
    halofile = h5py.File(filename, 'r')
    filenum=int(halofile["File_id"][0])
    numfiles=int(halofile["Num_of_files"][0])
    numhalos=np.uint64(halofile["Num_of_groups"][0])
    numtothalos=np.uint64(halofile["Total_num_of_groups"][0])
    atime=np.float(halofile.attrs["Time"]) 
    fieldnames=[str(n) for n in halofile.keys()]
    #clean of header info
    fieldnames.remove("File_id")
    fieldnames.remove("Num_of_files")
    fieldnames.remove("Num_of_groups")
    if 'Total_num_of_groups' in fieldnames:
        fieldnames.remove("Total_num_of_groups")
    if 'Configuration' in fieldnames:
        fieldnames.remove("Configuration")
    if 'SimulationInfo' in fieldnames:
        fieldnames.remove("SimulationInfo")
    if 'UnitInfo' in fieldnames:
        fieldnames.remove("UnitInfo")
    fieldtype=[halofile[fieldname].dtype for fieldname in fieldnames]
    #if the desiredfields argument is passed only these fieds are loaded
    if (len(desiredfields)>0):
        if (iverbose):print("Loading subset of all fields in property file ", len(desiredfields), " instead of ", len(fieldnames))
        fieldnames=desiredfields
        fieldtype=[halofile[fieldname].dtype for fieldname in fieldnames]
    halofile.close()

    #allocate memory that will store the halo dictionary
    if selected_files is not None:
        numtothalos = np.uint64(0)
        numfiles = len(selected_files)
        for ifile in selected_files:
            filename = basefilename+".properties"+"."+str(ifile)
            halofile = h5py.File(filename, 'r')
            numtothalos += np.uint64(halofile["Num_of_groups"][0])
            halofile.close()
    
    catalog={fieldnames[i]:np.zeros(numtothalos,dtype=fieldtype[i]) for i in range(len(fieldnames))}
    
    noffset=np.uint64(0)
    for ifile in range(numfiles):
        if (inompi==True): filename=basefilename+".properties"
        elif selected_files is not None: filename=basefilename+".properties"+"."+str(selected_files[ifile])
        else: filename=basefilename+".properties"+"."+str(ifile)
        if (iverbose) : print("reading ",filename)

        #here convert the hdf information into a numpy array
        halofile = h5py.File(filename, 'r')
        numhalos=np.uint64(halofile["Num_of_groups"][0])
        if (numhalos>0):htemp=[np.array(halofile[catvalue]) for catvalue in fieldnames]
        halofile.close()
        #numhalos=len(htemp[0])
        for i in range(len(fieldnames)):
            catvalue=fieldnames[i]
            if (numhalos>0): catalog[catvalue][noffset:noffset+numhalos]=htemp[i]
        noffset+=numhalos

    if (iverbose): print("done reading properties file ",time.clock()-start)
    return catalog,numtothalos,atime

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
    
def ReadParticleDataFile(basefilename,ibinary=0,iseparatesubfiles=0,iparttypes=0,iverbose=0, binarydtype=np.int64, unbound=True):
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

    #load header information from file to get total number of groups
    #ascii
    if (ibinary==0):
        gfile = open(gfilename, 'r')
        [filenum,numfiles]=gfile.readline().split()
        filenum=int(filenum);numfiles=int(numfiles)
        [numhalos, numtothalos]= gfile.readline().split()
        numhalos=np.uint64(numhalos);numtothalos=np.uint64(numtothalos)
    #binary
    elif (ibinary==1):
        gfile = open(gfilename, 'rb')
        [filenum,numfiles]=np.fromfile(gfile,dtype=np.int32,count=2)
        [numhalos,numtothalos]=np.fromfile(gfile,dtype=np.uint64,count=2)
    #hdf
    elif (ibinary==2):
        gfile = h5py.File(gfilename, 'r')
        filenum=int(gfile["File_id"][0])
        numfiles=int(gfile["Num_of_files"][0])
        numhalos=np.uint64(gfile["Num_of_groups"][0])
        numtothalos=np.uint64(gfile["Total_num_of_groups"][0])
    gfile.close()

    particledata=dict()
    particledata['Npart']=np.zeros(numtothalos,dtype=np.uint64)
    particledata['Npart_unbound']=np.zeros(numtothalos,dtype=np.uint64)
    particledata['Particle_IDs']=[[] for i in range(numtothalos)]
    if (iparttypes==1):
        particledata['Particle_Types']=[[] for i in range(numtothalos)]

    #now for all files
    counter=np.uint64(0)
    subfilenames=[""]
    if (iseparatesubfiles==1): subfilenames=["",".sublevels"]
    for ifile in range(numfiles):
        for subname in subfilenames:
            bfname=basefilename+subname
            gfilename=bfname+".catalog_groups"
            pfilename=bfname+".catalog_particles"
            upfilename=pfilename+".unbound"
            tfilename=bfname+".catalog_parttypes"
            utfilename=tfilename+".unbound"
            if (inompi==False):
                gfilename+="."+str(ifile)
                pfilename+="."+str(ifile)
                upfilename+="."+str(ifile)
                tfilename+="."+str(ifile)
                utfilename+="."+str(ifile)
            if (iverbose) : print("reading",bfname,ifile)

            #ascii
            if (ibinary==0):
                gfile = open(gfilename, 'r')
                #read header information
                gfile.readline()
                [numhalos,foo]= gfile.readline().split()
                numhalos=np.uint64(numhalos)
                gfile.close()
                #load data
                gdata=np.loadtxt(gfilename,skiprows=2,dtype=np.uint64)
                numingroup=gdata[:numhalos]
                offset=gdata[int(numhalos):int(2*numhalos)]
                uoffset=gdata[int(2*numhalos):int(3*numhalos)]
                #particle id data
                pfile=open(pfilename, 'r')
                pfile.readline()
                [npart,foo]= pfile.readline().split()
                npart=np.uint64(npart)
                pfile.close()
                piddata=np.loadtxt(pfilename,skiprows=2,dtype=np.int64)
                upfile= open(upfilename, 'r')
                upfile.readline()
                [unpart,foo]= upfile.readline().split()
                unpart=np.uint64(unpart)
                upfile.close()
                upiddata=np.loadtxt(upfilename,skiprows=2,dtype=np.int64)
                if (iparttypes==1):
                    #particle id data
                    tfile= open(tfilename, 'r')
                    tfile.readline()
                    [npart,foo]= tfile.readline().split()
                    tfile.close()
                    tdata=np.loadtxt(tfilename,skiprows=2,dtype=np.uint16)
                    utfile= open(utfilename, 'r')
                    utfile.readline()
                    [unpart,foo]= utfile.readline().split()
                    utfile.close()
                    utdata=np.loadtxt(utfilename,skiprows=2,dtype=np.uint16)
            #binary
            elif (ibinary==1):
                gfile = open(gfilename, 'rb')
                np.fromfile(gfile,dtype=np.int32,count=2)
                [numhalos,foo]=np.fromfile(gfile,dtype=np.uint64,count=2)
                #need to generalise to
                numingroup=np.fromfile(gfile,dtype=binarydtype ,count=numhalos)
                offset=np.fromfile(gfile,dtype=binarydtype,count=numhalos)
                uoffset=np.fromfile(gfile,dtype=binarydtype,count=numhalos)
                gfile.close()
                pfile = open(pfilename, 'rb')
                np.fromfile(pfile,dtype=np.int32,count=2)
                [npart,foo]=np.fromfile(pfile,dtype=np.uint64,count=2)
                piddata=np.fromfile(pfile,dtype=binarydtype ,count=npart)
                pfile.close()
                upfile = open(upfilename, 'rb')
                np.fromfile(upfile,dtype=np.int32,count=2)
                [unpart,foo]=np.fromfile(upfile,dtype=np.uint64,count=2)
                upiddata=np.fromfile(upfile,dtype=binarydtype ,count=unpart)
                upfile.close()
                if (iparttypes==1):
                    tfile = open(tfilename, 'rb')
                    np.fromfile(tfile,dtype=np.int32,count=2)
                    [npart,foo]=np.fromfile(tfile,dtype=np.uint16,count=2)
                    tdata=np.fromfile(tfile,dtype=binarydtype ,count=npart)
                    tfile.close()
                    utfile = open(utfilename, 'rb')
                    np.fromfile(utfile,dtype=np.int32,count=2)
                    [unpart,foo]=np.fromfile(utfile,dtype=np.uint16,count=2)
                    utdata=np.fromfile(utfile,dtype=binarydtype ,count=unpart)
                    utfile.close()
            #hdf
            elif (ibinary==2):
                gfile = h5py.File(gfilename, 'r')
                numhalos=np.uint64(gfile["Num_of_groups"][0])
                numingroup=np.uint64(gfile["Group_Size"])
                offset=np.uint64(gfile["Offset"])
                uoffset=np.uint64(gfile["Offset_unbound"])
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
                particledata['Npart'][counter:counter+numhalos]=numingroup
            else:
                particledata['Npart'][counter:counter+numhalos] = numingroup-unumingroup

            particledata['Npart_unbound'][counter:counter+numhalos]=unumingroup
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
                    particledata['Particle_IDs'][int(i+counter)]=np.zeros(numingroup[i],dtype=np.int64)
                    particledata['Particle_IDs'][int(i+counter)][:int(numingroup[i]-unumingroup[i])]=piddata[offset[i]:offset[i]+numingroup[i]-unumingroup[i]]
                    if (iparttypes==1):
                        particledata['Particle_Types'][int(i+counter)]=np.zeros(numingroup[i],dtype=np.int64)
                        particledata['Particle_Types'][int(i+counter)][:int(numingroup[i]-unumingroup[i])]=tdata[offset[i]:offset[i]+numingroup[i]-unumingroup[i]]

            counter+=numhalos

    return particledata

    
def ReadPropertyFileMultiWrapper(basefilename,index,halodata,numhalos,atime,ibinary=2,iseparatesubfiles=0,iverbose=0,desiredfields=[]):
    """
    Wrapper for multithreaded reading
    """
    #call read routine and store the data
    halodata[index],numhalos[index],atime[index]=ReadPropertyFile(basefilename,ibinary,iseparatesubfiles,iverbose,desiredfields)

def ReadPropertyFileMultiWrapperNamespace(index,basefilename,ns,ibinary=2,iseparatesubfiles=0,iverbose=0,desiredfields=[]):
    #call read routine and store the data
    ns.hdata[index],ns.ndata[index],ns.adata[index]=ReadPropertyFile(basefilename,ibinary,iseparatesubfiles,iverbose,desiredfields)
# def ReadHaloMergerTree(numsnaps,treefilename,ibinary=0,iverbose=0):
#     """
#     VELOCIraptor/STF merger tree in ascii format contains
#     a header with
#         number_of_snapshots
#         a description of how the tree was built
#         total number of halos across all snapshots

#     then followed by data
#     for each snapshot
#         snapshotvalue numhalos
#         haloid_1 numprogen_1
#         progenid_1
#         progenid_2
#         ...
#         progenid_numprogen_1
#         haloid_2 numprogen_2
#         .
#         .
#         .
#     one can also have an output format that has an additional field for each progenitor, the meritvalue

#     """
#     start = time.clock()
#     tree=[]
#     if (iverbose): print("reading Tree file",treefilename,os.path.isfile(treefilename))
#     if (os.path.isfile(treefilename)==False):
#         print("Error, file not found")
#         return tree
#     #if ascii format
#     if (ibinary==0):
#         treefile = open(treefilename, 'r')
#         numsnap=int(treefile.readline())
#         descrip=treefile.readline().strip()
#         tothalos=int(treefile.readline())
#         tree=[{"haloID": [], "Num_progen": [], "Progen": []} for i in range(numsnap)]
#         offset=0
#         totalnumprogen=0
#         for i in range(numsnap):
#             [snapval,numhalos]=treefile.readline().strip().split('\t')
#             snapval=int(snapval);numhalos=int(numhalos)
#             #if really verbose
#             if (iverbose==2): print(snapval,numhalos)
#             tree[i]["haloID"]=np.zeros(numhalos, dtype=np.int64)
#             tree[i]["Num_progen"]=np.zeros(numhalos, dtype=np.int32)
#             tree[i]["Progen"]=[[] for j in range(numhalos)]
#             for j in range(numhalos):
#                 [hid,nprog]=treefile.readline().strip().split('\t')
#                 hid=np.int64(hid);nprog=int(nprog)
#                 tree[i]["haloID"][j]=hid
#                 tree[i]["Num_progen"][j]=nprog
#                 totalnumprogen+=nprog
#                 if (nprog>0):
#                     tree[i]["Progen"][j]=np.zeros(nprog,dtype=np.int64)
#                     for k in range(nprog):
#                         tree[i]["Progen"][j][k]=np.int64(treefile.readline())

#     elif(ibinary==2):

#         tree=[{"haloID": [], "Num_progen": [], "Progen": []} for i in range(numsnaps)]
#         snaptreelist=open(treefilename,'r')
#         for snap in range(numsnaps):
#             snaptreename = snaptreelist.readline().strip()+".tree.hdf5"
#             if (iverbose): print("Reading",snaptreename)
#             treedata = h5py.File(snaptreename,"r")

#             tree[snap]["haloID"] = np.array(treedata["ID"])
#             tree[snap]["Num_progen"] = np.array(treedata["NumProgen"])

#             #See if the dataset exits
#             if("Progenitors" in treedata.keys()):

#                 #Find the indices to split the array
#                 split = np.zeros(len(tree[snap]["Num_progen"]),dtype=int)
#                 for i,numdesc in enumerate(tree[snap]["Num_progen"]):
#                     split[i] = split[i-1] + numdesc

#                 #Read in the progenitors, splitting them as reading them in
#                 tree[snap]["Progen"] = np.split(treedata["Progenitors"][:],split)
 
#         snaptreelist.close()
#     if (iverbose): print("done reading tree file ",time.clock()-start)
#     return tree


def ReadHaloMergerTree(treefilename,ibinary=0,iverbose=0,imerit=False,inpart=False):
    """
    VELOCIraptor/STF merger tree in ascii format contains
    a header with
        number_of_snapshots
        a description of how the tree was built
        total number of halos across all snapshots

    then followed by data
    for each snapshot
        snapshotvalue numhalos
        haloid_1 numprogen_1
        progenid_1
        progenid_2
        ...
        progenid_numprogen_1
        haloid_2 numprogen_2
        .
        .
        .
    one can also have an output format that has an additional field for each progenitor, the meritvalue

    """
    start = time.clock()
    tree=[]
    if (iverbose): print("reading Tree file",treefilename,os.path.isfile(treefilename))
    if (os.path.isfile(treefilename)==False):
        print("Error, file not found")
        return tree
    #if ascii format
    if (ibinary==0):
        treefile = open(treefilename, 'r')
        numsnap=int(treefile.readline())
        treefile.close()
    elif(ibinary==2):
        snaptreelist=open(treefilename,'r')
        numsnap = sum(1 for line in snaptreelist)
        snaptreelist.close()
    else:
        print("Unknown format, returning null")
        numsnap=0
        return tree

    tree=[{"haloID": [], "Num_progen": [], "Progen": []} for i in range(numsnap)]
    if (imerit):
        for i in range(numsnap):
            tree[i]['Merit']=[]
    if (inpart):
        for i in range(numsnap):
            tree[i]['Npart']=[]
            tree[i]['Npart_progen']=[]

    #if ascii format
    if (ibinary==0):
        treefile = open(treefilename, 'r')
        numsnap=int(treefile.readline())
        descrip=treefile.readline().strip()
        tothalos=int(treefile.readline())
        offset=0
        totalnumprogen=0
        for i in range(numsnap):
            [snapval,numhalos]=treefile.readline().strip().split('\t')
            snapval=int(snapval);numhalos=int(numhalos)
            #if really verbose
            if (iverbose==2): print(snapval,numhalos)
            tree[i]["haloID"]=np.zeros(numhalos, dtype=np.int64)
            tree[i]["Num_progen"]=np.zeros(numhalos, dtype=np.uint32)
            tree[i]["Progen"]=[[] for j in range(numhalos)]
            if (imerit): tree[i]["Merit"]=[[] for j in range(numhalos)]
            if (inpart):
                tree[i]["Npart"]=np.zeros(numhalos, dtype=np.uint32)
                tree[i]["Npart_progen"]=[[] for j in range(numhalos)]
            for j in range(numhalos):
                data=treefile.readline().strip().split('\t')
                hid=np.int64(data[0]);nprog=np.uint32(data[1])
                tree[i]["haloID"][j]=hid
                tree[i]["Num_progen"][j]=nprog
                if (inpart):tree[i]["Npart"][j]=np.uint32(data[2])
                totalnumprogen+=nprog
                if (nprog>0):
                    tree[i]["Progen"][j]=np.zeros(nprog,dtype=np.int64)
                    if (imerit): tree[i]["Merit"][j]=np.zeros(nprog,dtype=np.float32)
                    if (inpart): tree[i]["Npart_progen"][j]=np.zeros(nprog,dtype=np.uint32)
                    for k in range(nprog):
                        data=treefile.readline().strip().split(' ')
                        tree[i]["Progen"][j][k]=np.int64(data[0])
                        if (imerit):tree[i]["Merit"][j][k]=np.float32(data[1])
                        if (inpart):tree[i]["Npart_progen"][j][k]=np.uint32(data[2])

    elif(ibinary==2):

        snaptreelist=open(treefilename,'r')
        #read the first file, get number of snaps from hdf file
        snaptreename = snaptreelist.readline().strip()+".tree.hdf5"
        treedata=h5py.File(snaptreename,"r")
        numsnaps=treedata.attrs['Number_of_snapshots']
        treedata.close()
        snaptreelist.close()

        snaptreelist=open(treefilename,'r')
        for snap in range(numsnaps):
            snaptreename = snaptreelist.readline().strip()+".tree.hdf5"
            if (iverbose): print("Reading",snaptreename)
            treedata = h5py.File(snaptreename,"r")

            tree[snap]["haloID"] = np.asarray(treedata["ID"])
            tree[snap]["Num_progen"] = np.asarray(treedata["NumProgen"])
            if(inpart):tree[snap]["Npart"] = np.asarray(treedata["Npart"])

            #See if the dataset exits
            if("ProgenOffsets" in treedata.keys()):

                #Find the indices to split the array
                split = np.add(np.asarray(treedata["ProgenOffsets"]),tree[snap]["Num_progen"],dtype=np.uint64,casting="unsafe")

                #Read in the progenitors, splitting them as reading them in
                tree[snap]["Progen"] = np.split(treedata["Progenitors"][:],split[:-1])

                if(inpart): tree[snap]["Npart_progen"] = np.split(treedata["ProgenNpart"],split[:-1])
                if(imerit): tree[snap]["Merit"] =  np.split(treedata["Merits"],split[:-1])

        snaptreelist.close()
    if (iverbose): print("done reading tree file ",time.clock()-start)
    return tree


def ReadHaloMergerTreeDescendant(numsnaps,treefilename,ireverseorder=True,ibinary=0,iverbose=0):
    """
    VELOCIraptor/STF descendant based merger tree in ascii format contains
    a header with
        number_of_snapshots
        a description of how the tree was built
        total number of halos across all snapshots

    then followed by data
    for each snapshot
        snapshotvalue numhalos
        haloid_1 numprogen_1
        progenid_1
        progenid_2
        ...
        progenid_numprogen_1
        haloid_2 numprogen_2
        .
        .
        .
    one can also have an output format that has an additional field for each progenitor, the meritvalue

    """
    start = time.clock()
    tree=[]
    if (iverbose): print("reading Tree file",treefilename,os.path.isfile(treefilename))
    if (os.path.isfile(treefilename)==False):
        print("Error, file not found")
        return tree
    #if ascii format
    if (ibinary==0):
        treefile = open(treefilename, 'r')
        numsnap=int(treefile.readline())
        descrip=treefile.readline().strip()
        tothalos=int(treefile.readline())
        tree=[{"haloID": [], "Num_descen": [], "Descen": [], "Rank": []} for i in range(numsnap)]
        offset=0
        totalnumdescen=0
        for i in range(numsnap):
            ii=i
            if (ireverseorder): ii=numsnap-1-i
            [snapval,numhalos]=treefile.readline().strip().split('\t')
            snapval=int(snapval);numhalos=int(numhalos)
            #if really verbose
            if (iverbose==2): print(snapval,numhalos)
            tree[ii]["haloID"]=np.zeros(numhalos, dtype=np.int64)
            tree[ii]["Num_descen"]=np.zeros(numhalos, dtype=np.int32)
            tree[ii]["Descen"]=[[] for j in range(numhalos)]
            tree[ii]["Rank"]=[[] for j in range(numhalos)]
            for j in range(numhalos):
                [hid,ndescen]=treefile.readline().strip().split('\t')
                hid=np.int64(hid);ndescen=int(ndescen)
                tree[ii]["haloID"][j]=hid
                tree[ii]["Num_descen"][j]=ndescen
                totalnumdescen+=ndescen
                if (ndescen>0):
                    tree[ii]["Descen"][j]=np.zeros(ndescen,dtype=np.int64)
                    tree[ii]["Rank"][j]=np.zeros(ndescen,dtype=np.uint32)
                    for k in range(ndescen):
                        data=treefile.readline().strip().split(' ')
                        tree[ii]["Descen"][j][k]=np.int64(data[0])
                        tree[ii]["Rank"][j][k]=np.uint32(data[1])

    elif(ibinary==2):

        tree=[{"haloID": [], "Num_descen": [], "Descen": [], "Rank": []} for i in range(numsnaps)]
        snaptreelist=open(treefilename,'r')
        for snap in range(numsnaps):
            snaptreename = snaptreelist.readline().strip()+".tree.hdf5"
            if (iverbose): print("Reading",snaptreename)
            treedata = h5py.File(snaptreename,"r")

            tree[snap]["haloID"] = np.array(treedata["ID"])
            tree[snap]["Num_descen"] = np.array(treedata["NumDesc"])

            #See if the dataset exits
            if("Descendants" in treedata.keys()):

                #Find the indices to split the array
                split = np.zeros(len(tree[snap]["Num_descen"]),dtype=int)
                for i,numdesc in enumerate(tree[snap]["Num_descen"]):
                    split[i] = split[i-1] + numdesc

                # Read in the data splitting it up as reading it in
                tree[snap]["Descen"] = np.split(treedata["Descendants"][:],split)
                tree[snap]["Rank"] = np.split(treedata["Ranks"][:],split)
 
        snaptreelist.close()
    if (iverbose): print("done reading tree file ",time.clock()-start)
    return tree


def SetForestID(numsnaps,halodata,rootheadid,ForestID,AllRootHead,
    TEMPORALHALOIDVAL = 1000000000000,searchSnapLim = 5, ireversesnaporder=True):
    """
    Sets the forest id of halos using a roothead as a start point.
    Given an initial root head and end snapshot,
    First append the roothead to the AllRootHead list.
    search all previous snapshots for any haloes that share the same roothead.
    Also at each snapshot, find all subhaloes of all haloes sharing the same
    root head
    if the roothead of a subhalo is not present in the AllRootHead list
    then recursively call SetForestID with this subhalo's root head as start point
    if a subhalo's current host is not within the tree defined by rootheadid
    then recursively call SetForestID with this host's root head as start point

    Parameters
    ----------
    numsnaps : numpy.int32
        the number of snapshots
    halodata : dict
        the halodata dictionary structure which must contain the halo merger tree based keys (Head, RootHead), etc.
    rootheadid : numpy.int64
        the rootheadid of the tree that will be explored and have its forestID set
    AllRootHead : list
        a list that stores the current set of rootheadid values that have been searched

    Optional Parameters
    -------------------
    TEMPORALHALOIDVAL : numpy.int64
        Temporal ID value that makes Halo IDs temporally unique, adding a snapshot num* this value.
        Allows one to quickly parse a Halo ID to determine the snapshot it exists at and its index.
    searchSnapLim : numpy.int32
        Maximum number of snapshots to keep searching if no new halos are identified as beloning to
        a rootheadid's tree, moving backwards in time
    ireversesnaporder : bool
        Whether dictionary data has late times starting at 0 (True, default) or at end of dictionary (False)

    Returns
    -------
    AllRootHead : list
        Updated list
    halodata : dict
        Updated halo data

    """


    if (ireversesnaporder): endSnap = numsnaps-int(rootheadid/TEMPORALHALOIDVAL)-1
    else : endSnap = int(rootheadid/TEMPORALHALOIDVAL)
    rootheadindex=int(rootheadid%TEMPORALHALOIDVAL-1)

    AllRootHead.append(rootheadid)

    #set the forest level of this searcheed
    #if this object is a host at final snap then set the forest level to 0
    #otherwise set the ForestLevel to 1
    ForestLevel=1*(halodata[endSnap]["hostHaloID"][rootheadindex]!=-1)

    #Indicator for amount of snapshots searcheed
    iSearchSnap = 0

    #set the direction of how the data will be processed
    if (ireversesnaporder): snaplist=np.arange(endSnap,numsnaps,dtype=np.int32)
    else : snaplist=np.arange(endsnap,-1,-1)
    for snap in snaplist:
        #Find which halos at this snapshot point to the RootDescedant
        sel = np.where(halodata[snap]["RootHead"]==rootheadid)[0]

        #keep track of how many snapshots there have been where there is nothing in the tree
        if(sel.size==0):
            iSearchSnap+=1
        if(iSearchSnap==searchSnapLim): break
        else: iSearchSnap = 0

        # Set all the halos within this tree within this snapshot to this forest ID
        halodata[snap]["ForestID"][sel] = ForestID
        halodata[snap]["ForestLevel"][sel] = ForestLevel

        #Lets find which halos are subhalos of the halos within the tree defined by
        #halos with the same rootheadid
        subHaloIndxs = np.where(np.in1d(halodata[snap]["hostHaloID"],halodata[snap]["ID"][sel]))[0]
        #Lets loop over all the subhalos within this selection, which contains
        #all subhalos of any host halos within the tree defined by rootheadid
        for subHaloIndx in subHaloIndxs:
            #See if this tree has already been set
            if(halodata[snap]["RootHead"][subHaloIndx] not in AllRootHead):
                #Lets walk the subhalo's tree setting the forest ID
                AllRootHead,halodata = SetForestID(numsnaps,halodata,halodata[snap]["RootHead"][subHaloIndx],ForestID,AllRootHead)

        #Extract the hosts of all subhalos in this selection that are not already in the tree defined by rootheadid
        treeSubhaloSel = (halodata[snap]["hostHaloID"][sel]!=-1) & (np.invert(np.in1d(halodata[snap]["hostHaloID"][sel],halodata[snap]["ID"][sel])))
        #Get the index of these hosts that lie outside the tree
        hostIndxs = np.unique(halodata[snap]["hostHaloID"][sel][treeSubhaloSel]%TEMPORALHALOIDVAL-1).astype(int)
        #Loop over all the index for the host halos
        for hostIndx in hostIndxs:
            #See if this tree has already been set
            if(halodata[snap]["RootHead"][hostIndx] not in AllRootHead):
                #Lets walk the hosts tree setting the forrest ID
                AllRootHead,halodata = SetForestID(numsnaps,halodata,halodata[snap]["RootHead"][hostIndx],ForestID,AllRootHead)

    return AllRootHead,halodata

def GenerateForest(numsnaps,numhalos,halodata,cosmo,atime,
    TEMPORALHALOIDVAL=1000000000000, iverbose=1, interactiontime=2, ispatialintflag=False, pos_tree=[]):
    """
    This code traces all root heads back in time identifying all interacting haloes and bundles them together into the same forest id
    The idea is to have in the halodata dictionary an associated unique forest id for all related (sub)haloes. The code also allows
    for some cleaning of the forest, specifically if a (sub)halo is only interacting for some small fraction of time, then it is not
    assigned to the forest. This can limit the size of a forest, which could otherwise become the entire halo catalog.

    Parameters
    ----------
    numsnaps : numpy.int32
        the number of snapshots
    numhalos : array
        array of the number of haloes per snapshot.
    halodata : dict
        the halodata dictionary structure which must contain the halo merger tree based keys (Head, RootHead), etc.
    cosmo : dict
        dictionary which has cosmological information such as box size, hval, Omega_m
    atime : array
        an array of scale factors

    Optional Parameters
    -------------------
    TEMPORALHALOIDVAL : numpy.int64
        Temporal ID value that makes Halo IDs temporally unique, adding a snapshot num* this value.
        Allows one to quickly parse a Halo ID to determine the snapshot it exists at and its index.
    iverbose : int
        verbosity of function (0, minimal, 1, verbose, 2 chatterbox)
    interactiontime : int
        Optional functionality not implemented yet. Allows forest to be split if connections do not span
        more than this number of snapshots
    ispatialintflag : bool
        Flag indicating whether spatial information should be used to join forests.
    pos_tree : scikit.spatial.cKDTree
        Optional functionality not implemented yet. Allows forests to be joined if haloes
        are spatially close.

    Returns
    -------
    ForestSize : numpy.array
        Update the halodata dictionary with ForestID information and also returns the size of
        the forests

    """

    #initialize the dictionaries
    for j in range(numsnaps):
        #store id and snap and mass of last major merger and while we're at it, store number of major mergers
        halodata[j]["ForestID"]=np.ones(numhalos[j],dtype=np.int64)*-1
        halodata[j]["ForestLevel"]=np.ones(numhalos[j],dtype=np.int32)*-1
    #built KD tree to quickly search for near neighbours. only build if not passed.
    if (ispatialintflag):
        start=time.clock()
        boxsize=cosmo['BoxSize']
        hval=cosmo['Hubble_param']
        if (len(pos_tree)==0):
            pos=[[]for j in range(numsnaps)]
            pos_tree=[[]for j in range(numsnaps)]
            start=time.clock()
            if (iverbose): print("KD tree build")
            for j in range(numsnaps):
                if (numhalos[j]>0):
                    boxval=boxsize*atime[j]/hval
                    pos[j]=np.transpose(np.asarray([halodata[j]["Xc"],halodata[j]["Yc"],halodata[j]["Zc"]]))
                    pos_tree[j]=spatial.cKDTree(pos[j],boxsize=boxval)
            if (iverbose): print("done ",time.clock()-start)

    #now start marching backwards in time from root heads
    #identifying all subhaloes that have every been subhaloes for long enough
    #and all progenitors and group them together into the same forest id
    forestidval=1
    start=time.clock()
    for j in range(numsnaps):
        start2=time.clock()
        if (numhalos[j]==0): continue
        #now with tree start at last snapshot and identify all root heads
        #only look at halos that are their own root head and are not subhalos
        rootheads=np.where((halodata[j]['ID']==halodata[j]['RootHead'])*(halodata[j]['hostHaloID']==-1)*(halodata[j]['ForestID']==-1))
        if (iverbose): print("At snapshot",j,len(rootheads[0]))
        for iroothead in rootheads[0]:
            #if a halo has been processed as part of a forest as a
            #result of walking the subhalo branches of a different root head
            #then move on to the next object
            if (halodata[j]['ForestID'][iroothead]!=-1): continue
            AllRootHead = []
            #begin recursively searching and setting the forest using the the roothead
            AllRootHead,halodata = SetForestID(numsnaps,halodata,halodata[j]["RootHead"][iroothead],forestidval,AllRootHead)
            #update forest id
            forestidval+=1
        if (iverbose): print("Done snap",j,time.clock()-start2)

    #get the size of each forest
    ForestSize=np.zeros(forestidval,dtype=int64)
    for j in range(numsnaps):
        if (numhalos[j]==0): continue
        uniqueforest,counts=np.unique(halodata[j]['ForestID'],return_counts=True)
        for icount in range(len(uniqueforest)):
            ForestSize[uniqueforest[icount]-1]+=counts[icount]
        if (iverbose): print("Finished processing forest size for snap",j)
    start2=time.clock()

    #first identify all subhalos and see if any have subhalo connections with different than their host
    for j in range(numsnaps):
        if (numhalos[j]==0): continue
        #now with tree start at last snapshot and identify all root heads
        #only look at halos that are their own root head and are not subhalos
        missingforest=np.where((halodata[j]['ForestID']==-1))
        rootheads=np.where((halodata[j]['ID']==halodata[j]['RootHead'])*(halodata[j]['ForestID']==-1))
        subrootheads=np.where((halodata[j]['ForestID']==-1)*(halodata[j]['hostHaloID']!=-1))
        if (iverbose): print("At snapshot",j," still have ",halodata[j]['ForestID'].size,len(missingforest[0]), " with no forest id ! Of which ",len(rootheads[0])," are root heads", len(subrootheads[0]),"are subhalos")
        #if (iverbose and len(missingforest[0])>0): print("At snapshot",j," still have ",len(missingforest[0]), " with no forest id ! Of which ",len(rootheads[0])," are root heads", len(subrootheads[0]),"are subhalos")
        if (len(subrootheads[0])>0):
            for isub in subrootheads[0]:
                hostid=halodata[j]['hostHaloID'][isub]
                hostindex=int(hostid%TEMPORALHALOIDVAL-1)
                halodata[j]['ForestID'][isub]=halodata[j]['ForestID'][hostindex]
                halodata[j]['ForestLevel'][isub]=halodata[j]['ForestLevel'][hostindex]+1
    #then return this
    print("Done generating forest",time.clock()-start)
    return ForestSize


def TraceMainDescendant(istart,ihalo,numsnaps,numhalos,halodata,tree,TEMPORALHALOIDVAL,ireverseorder=False):
    """
    Follows a halo along descendant tree to root tails
    if reverse order than late times start at 0 and as one moves up in index
    one moves backwards in time
    """

    #start at this snapshot
    halosnap=istart

    #see if halo does not have a Head set
    if (halodata[halosnap]['Head'][ihalo]==0):
        #if halo has not had a Head set the branch needs to be walked along the main branch
        haloid=halodata[halosnap]['ID'][ihalo]
        #only set the Root Tail if it has not been set. Here if halo has not had
        #tail set, then must be the the first progenitor
        #otherwise it should have already been set and just need to store the root tail
        if (halodata[halosnap]['Tail'][ihalo]==0):
            halodata[halosnap]['Tail'][ihalo]=haloid
            halodata[halosnap]['TailSnap'][ihalo]=halosnap
            halodata[halosnap]['RootTail'][ihalo]=haloid
            halodata[halosnap]['RootTailSnap'][ihalo]=halosnap
            roottail,rootsnap,rootindex=haloid,halosnap,ihalo
        else:
            roottail=halodata[halosnap]['RootTail'][ihalo]
            rootsnap=halodata[halosnap]['RootTailSnap'][ihalo]
            rootindex=int(roottail%TEMPORALHALOIDVAL)-1
        #now move along tree first pass to store head and tails and root tails of main branch
        while (True):

            #ids contain index information
            haloindex=int(haloid%TEMPORALHALOIDVAL)-1
            halodata[halosnap]['Num_descen'][haloindex]=tree[halosnap]['Num_descen'][haloindex]
            #if no more descendants, break from search
            if (halodata[halosnap]['Num_descen'][haloindex]==0):
                #store for current halo its tail and root tail info (also store root tail for root head)
                halodata[halosnap]['Head'][haloindex]=haloid
                halodata[halosnap]['HeadSnap'][haloindex]=halosnap
                halodata[halosnap]['RootHead'][haloindex]=haloid
                halodata[halosnap]['RootHeadSnap'][haloindex]=halosnap
                rootheadid,rootheadsnap,rootheadindex=haloid,halosnap,haloindex
                #only set the roots head of the root tail
                #if it has not been set before (ie: along the main branch of root halo)
                if (halodata[rootsnap]['RootHead'][rootindex]==0):
                    halosnap,haloindex,haloid=rootsnap,rootindex,roottail
                    #set the root head of the main branch
                    while(True):
                        halodata[halosnap]['RootHead'][haloindex]=rootheadid
                        halodata[halosnap]['RootHeadSnap'][haloindex]=rootheadsnap
                        descen=halodata[halosnap]['Head'][haloindex]
                        descenindex=int(descen%TEMPORALHALOIDVAL)-1
                        descensnap=int(((descen-descen%TEMPORALHALOIDVAL))/TEMPORALHALOIDVAL)
                        if (ireverseorder):
                            descensnap=numsnaps-1-descensnap
                        if (haloid==descen):
                            break
                        halosnap,haloindex,haloid=descensnap,descenindex,descen
                break
            #now store the rank of the of the descandant.
            descenrank=tree[halosnap]['Rank'][haloindex][0]
            halodata[halosnap]['HeadRank'][haloindex]=descenrank
            #as we are only moving along main branches stop if object is rank is not 0
            if (descenrank>0):
                break
            #otherwise, get the descendant
            #store main progenitor
            maindescen=tree[halosnap]['Descen'][haloindex][0]
            maindescenindex=int(maindescen%TEMPORALHALOIDVAL)-1
            maindescensnap=int(((maindescen-maindescen%TEMPORALHALOIDVAL))/TEMPORALHALOIDVAL)
            #if reverse order, then higher snap values correspond to lower index
            if (ireverseorder):
                maindescensnap=numsnaps-1-maindescensnap
            #calculate stepsize in time based on the halo ids
            stepsize=maindescensnap-halosnap

            #store descendant
            halodata[halosnap]['Head'][haloindex]=maindescen
            halodata[halosnap]['HeadSnap'][haloindex]=maindescensnap

            #and update the root tails of the object
            halodata[maindescensnap]['Tail'][maindescenindex]=haloid
            halodata[maindescensnap]['TailSnap'][maindescenindex]=halosnap
            halodata[maindescensnap]['RootTail'][maindescenindex]=roottail
            halodata[maindescensnap]['RootTailSnap'][maindescenindex]=rootsnap
            halodata[maindescensnap]['Num_progen'][maindescenindex]+=1

            #then move to the next descendant
            haloid=maindescen
            halosnap=maindescensnap

def ProduceUnifiedTreeandHaloCatalog(fname,numsnaps,tree,numhalos,halodata,atime,
    descripdata={'Title':'Tree and Halo catalog of sim', 'VELOCIraptor_version':1.15, 'Tree_version':1.1, 'Particle_num_threshold':20, 'Temporal_linking_length':1, 'Flag_gas':False, 'Flag_star':False, 'Flag_bh':False},
    cosmodata={'Omega_m':1.0, 'Omega_b':0., 'Omega_Lambda':0., 'Hubble_param':1.0,'BoxSize':1.0, 'Sigma8':1.0},
    unitdata={'UnitLength_in_Mpc':1.0, 'UnitVelocity_in_kms':1.0,'UnitMass_in_Msol':1.0, 'Flag_physical_comoving':True,'Flag_hubble_flow':False},
    partdata={'Flag_gas':False, 'Flag_star':False, 'Flag_bh':False},
    ibuildheadtail=0, icombinefile=1):

    """

    produces a unifed HDF5 formatted file containing the full catalog plus information to walk the tree
    \ref BuildTemporalHeadTail must have been called before otherwise it is called.
    Code produces a file for each snapshot
    The keys are the same as that contained in the halo catalog dictionary with the addition of
    Num_of_snaps, and similar header info contain in the VELOCIraptor hdf files, ie Num_of_groups, Total_num_of_groups

    \todo don't know if I should use multiprocessing here to write files in parallel. IO might not be ideal

    """
    if (ibuildheadtail==1):
        BuildTemporalHeadTail(numsnaps,tree,numhalos,halodata)
    totnumhalos=sum(numhalos)
    if (icombinefile==1):
        hdffile=h5py.File(fname+".snap.hdf.data",'w')
        headergrp=hdffile.create_group("Header")
        #store useful information such as number of snapshots, halos,
        #cosmology (Omega_m,Omega_b,Hubble_param,Omega_Lambda, Box size)
        #units (Physical [1/0] for physical/comoving flag, length in Mpc, km/s, solar masses, Gravity
        #and TEMPORALHALOIDVAL used to traverse tree information (converting halo ids to haloindex or snapshot), Reverse_order [1/0] for last snap listed first)
        #set the attributes of the header
        headergrp.attrs["NSnaps"]=numsnaps
        #overall description
        #simulation box size

        #cosmological params
        cosmogrp=headergrp.create_group("Cosmology")
        for key in cosmodata.keys():
            cosmogrp.attrs[key]=cosmodata[key]
        #unit params
        unitgrp=headergrp.create_group("Units")
        for key in unitdata.keys():
            unitgrp.attrs[key]=unitdata[key]
        #particle types
        partgrp=headergrp.create_group("Parttypes")
        partgrp.attrs["Flag_gas"]=descripdata["Flag_gas"]
        partgrp.attrs["Flag_star"]=descripdata["Flag_star"]
        partgrp.attrs["Flag_bh"]=descripdata["Flag_bh"]

        for i in range(numsnaps):
            snapgrp=hdffile.create_group("Snap_%03d"%(numsnaps-1-i))
            snapgrp.attrs["Snapnum"]=i
            snapgrp.attrs["NHalos"]=numhalos[i]
            snapgrp.attrs["scalefactor"]=atime[i]
            for key in halodata[i].keys():
                snapgrp.create_dataset(key,data=halodata[i][key])
        hdffile.close()
    else:
        for i in range(numsnaps):
            hdffile=h5py.File(fname+".snap_%03d.hdf.data"%(numsnaps-1-i),'w')
            hdffile.create_dataset("Snap_value",data=np.array([i],dtype=np.uint32))
            hdffile.create_dataset("NSnaps",data=np.array([numsnaps],dtype=np.uint32))
            hdffile.create_dataset("NHalos",data=np.array([numhalos[i]],dtype=np.uint64))
            hdffile.create_dataset("TotalNHalos",data=np.array([totnumhalos],dtype=np.uint64))
            hdffile.create_dataset("scalefactor",data=np.array([atime[i]],dtype=np.float64))
            for key in halodata[i].keys():
                hdffile.create_dataset(key,data=halodata[i][key])
            hdffile.close()

    # hdffile=h5py.File(fname+".tree.hdf.data",'w')
    # hdffile.create_dataset("NSnaps",data=np.array([numsnaps],dtype=np.uint32))
    # hdffile.create_dataset("TotalNHalos",data=np.array([totnumhalos],dtype=np.uint64))
    # hdffile.create_dataset("NHalos",data=np.array([numhalos],dtype=np.uint64))
    # for i in range(numsnaps):
    #     snapgrp=hdffile.create_group("Snap_%03d"%(numsnaps-1-i))
    #     for key in tree[i].keys():
    #         """
    #         #to be completed for progenitor list
    #         if (key=="Progen"):
    #             for j in range(numhalos[i]):
    #                 halogrp=snapgrp.create_group("Halo"+str(j))
    #                 halogrp.create_dataset(key,data=tree[i][key][j])
    #         else:
    #             snapgrp.create_dataset(key,data=tree[i][key])
    #         """
    #         if (key=="Progen"): continue
    #         snapgrp.create_dataset(key,data=tree[i][key])
    # hdffile.close()

def ReadUnifiedTreeandHaloCatalog(fname, desiredfields=[], icombinedfile=1,iverbose=1):
    """
    Read Unified Tree and halo catalog from HDF file with base filename fname.

    Parameters
    ----------

    Returns
    -------
    """
    if (icombinedfile):
        hdffile=h5py.File(fname,'r')

        #load data sets containing number of snaps
        headergrpname="Header/"
        numsnaps=hdffile[headergrpname].attrs["NSnaps"]

        #allocate memory
        halodata=[dict() for i in range(numsnaps)]
        numhalos=[0 for i in range(numsnaps)]
        atime=[0 for i in range(numsnaps)]
        tree=[[] for i in range(numsnaps)]
        cosmodata=dict()
        unitdata=dict()

        #load cosmology data
        cosmogrpname="Cosmology/"
        fieldnames=[str(n) for n in hdffile[headergrpname+cosmogrpname].attrs.keys()]
        for fieldname in fieldnames:
            cosmodata[fieldname]=hdffile[headergrpname+cosmogrpname].attrs[fieldname]

        #load unit data
        unitgrpname="Units/"
        fieldnames=[str(n) for n in hdffile[headergrpname+unitgrpname].attrs.keys()]
        for fieldname in fieldnames:
            unitdata[fieldname]=hdffile[headergrpname+unitgrpname].attrs[fieldname]

        #for each snap load the appropriate group
        start=time.clock()
        for i in range(numsnaps):
            snapgrpname="Snap_%03d/"%(numsnaps-1-i)
            if (iverbose==1):
                print("Reading ",snapgrpname)
            isnap=hdffile[snapgrpname].attrs["Snapnum"]
            atime[isnap]=hdffile[snapgrpname].attrs["scalefactor"]
            numhalos[isnap]=hdffile[snapgrpname].attrs["NHalos"]
            if (len(desiredfields)>0):
                fieldnames=desiredfields
            else:
                fieldnames=[str(n) for n in hdffile[snapgrpname].keys()]
            for catvalue in fieldnames:
                halodata[isnap][catvalue]=np.array(hdffile[snapgrpname+catvalue])
        hdffile.close()
        print("read halo data ",time.clock()-start)
    else :
        hdffile=h5py.File(fname+".snap_000.hdf.data",'r')
        numsnaps=int(hdffile["NSnaps"][0])
        #get field names
        fieldnames=[str(n) for n in hdffile.keys()]
        #clean of header info
        fieldnames.remove("Snapnum")
        fieldnames.remove("NSnaps")
        fieldnames.remove("NHalos")
        fieldnames.remove("TotalNHalos")
        fieldnames.remove("scalefactor")
        if (len(desiredfields)>0):
            fieldnames=desiredfields
        hdffile.close()
        halodata=[[] for i in range(numsnaps)]
        numhalos=[0 for i in range(numsnaps)]
        atime=[0 for i in range(numsnaps)]
        tree=[[] for i in range(numsnaps)]
        start=time.clock()
        for i in range(numsnaps):
            hdffile=h5py.File(fname+".snap_%03d.hdf.data"%(numsnaps-1-i),'r')
            atime[i]=(hdffile["scalefactor"])[0]
            numhalos[i]=(hdffile["NHalos"])[0]
            halodata[i]=dict()
            for catvalue in fieldnames:
                halodata[i][catvalue]=np.array(hdffile[catvalue])
            hdffile.close()
        print("read halo data ",time.clock()-start)
    #lets ignore the tree file for now
    for i in range(numsnaps):
        tree[i]=dict()
    return atime,tree,numhalos,halodata,cosmodata,unitdata
    if (icombinedfile==1):
        hdffile=h5py.File(fname+".tree.hdf.data",'r')
        treefields=["haloID", "Num_progen"]
        #do be completed for Progenitor list although information is contained in the halo catalog by searching for things with the same head
        #treefields=["haloID", "Num_progen", "Progen"]
        for i in range(numsnaps):
            snapgrpname="Snap_%03d/"%(numsnaps-1-i)
            tree[i]=dict()
            for catvalue in treefields:
                """
                if (catvalue==treefields[-1]):
                    tree[i][catvalue]=[[]for j in range(numhalos[i])]
                    for j in range(numhalos[i]):
                        halogrpname=snapgrpname+"/Halo"+str(j)
                        tree[i][catvalue]=np.array(hdffile[halogrpname+catvalue])
                else:
                    tree[i][catvalue]=np.array(hdffile[snapgrpname+catvalue])
                """
                tree[i][catvalue]=np.array(hdffile[snapgrpname+catvalue])
        hdffile.close()
    return atime,tree,numhalos,halodata,cosmodata,unitdata

# atime,tree,numhaloes,orbvel,cosmodata,unitdata = vpt.ReadUnifiedTreeandHaloCatalog('/mnt/sshfs/pleiades/CosmRun/9p/Hydro/nonrad/Orbweaver/hydro9pAll.0.orbweaver.preprocessed.hdf', desiredfields=['Xc', 'Yc', 'Zc', 'Mass_200crit', 'ID'])

# for i in range(1, 67):
#     atime,tree,numhaloes,orbveltemp,cosmodata,unitdata = vpt.ReadUnifiedTreeandHaloCatalog('/mnt/sshfs/pleiades/CosmRun/9p/Hydro/nonrad/Orbweaver/hydro9pAll.%i.orbweaver.preprocessed.hdf' %i, desiredfields=['Xc', 'Yc', 'Zc', 'Mass_200crit', 'ID'])
#     for key in orbvel[200].keys():
#         orbvel[200][key] = np.append(orbvel[200][key], orbveltemp[200][key])

def TraceMainProgen(istart,ihalo,numsnaps,numhalos,halodata,tree,TEMPORALHALOIDVAL):
    """
    Follows a halo along tree to identify main progenitor
    """
    #start at this snapshot
    k=istart
    #see if halo does not have a tail (descendant set).
    if (halodata[k]['Tail'][ihalo]==0):
        #if halo has not had a tail set the branch needs to be walked along the main branch
        haloid=halodata[k]['ID'][ihalo]
        #only set the head if it has not been set
        #otherwise it should have already been set and just need to store the root head
        if (halodata[k]['Head'][ihalo]==0):
            halodata[k]['Head'][ihalo]=haloid
            halodata[k]['HeadSnap'][ihalo]=k
            halodata[k]['RootHead'][ihalo]=haloid
            halodata[k]['RootHeadSnap'][ihalo]=k
            roothead,rootsnap,rootindex=haloid,k,ihalo
        else:
            roothead=halodata[k]['RootHead'][ihalo]
            rootsnap=halodata[k]['RootHeadSnap'][ihalo]
            rootindex=int(roothead%TEMPORALHALOIDVAL)-1
        #now move along tree first pass to store head and tails and root heads of main branch
        while (True):
            #instead of seraching array make use of the value of the id as it should be in id order
            #wdata=np.where(tree[k]['haloID']==haloid)
            #w2data=np.where(halodata[k]['ID']==haloid)[0][0]
            wdata=w2data=int(haloid%TEMPORALHALOIDVAL)-1
            halodata[k]['Num_progen'][wdata]=tree[k]['Num_progen'][wdata]
            #if no more progenitors, break from search
            #if (tree[k]['Num_progen'][wdata[0][0]]==0 or len(wdata[0])==0):
            if (tree[k]['Num_progen'][wdata]==0):
                #store for current halo its tail and root tail info (also store root tail for root head)
                halodata[k]['Tail'][w2data]=haloid
                halodata[k]['TailSnap'][w2data]=k
                halodata[k]['RootTail'][w2data]=haloid
                halodata[k]['RootTailSnap'][w2data]=k
                #only set the roots tail if it has not been set before (ie: along the main branch of root halo)
                #if it has been set then we are walking along a secondary branch of the root halo's tree
                if (halodata[rootsnap]['RootTail'][rootindex]==0):
                    halodata[rootsnap]['RootTail'][rootindex]=haloid
                    halodata[rootsnap]['RootTailSnap'][rootindex]=k
                break

            #store main progenitor
            #mainprog=tree[k]['Progen'][wdata[0][0]][0]
            mainprog=tree[k]['Progen'][wdata][0]
            #calculate stepsize based on the halo ids
            stepsize=int(((haloid-haloid%TEMPORALHALOIDVAL)-(mainprog-mainprog%TEMPORALHALOIDVAL))/TEMPORALHALOIDVAL)
            #store tail
            halodata[k]['Tail'][w2data]=mainprog
            halodata[k]['TailSnap'][w2data]=k+stepsize
            k+=stepsize

            #instead of searching array make use of the value of the id as it should be in id order
            #for progid in tree[k-stepsize]['Progen'][wdata[0][0]]:
            #    wdata3=np.where(halodata[k]['ID']==progid)[0][0]
            for progid in tree[k-stepsize]['Progen'][wdata]:
                wdata3=int(progid%TEMPORALHALOIDVAL)-1
                halodata[k]['Head'][wdata3]=haloid
                halodata[k]['HeadSnap'][wdata3]=k-stepsize
                halodata[k]['RootHead'][wdata3]=roothead
                halodata[k]['RootHeadSnap'][wdata3]=rootsnap

            #then store next progenitor
            haloid=mainprog

def TraceMainProgenParallelChunk(istart,ihalochunk,numsnaps,numhalos,halodata,tree,TEMPORALHALOIDVAL):
    """
    Wrapper to allow for parallelisation
    """
    for ihalo in ihalochunk:
        TraceMainProgen(istart,ihalo,numsnaps,numhalos,halodata,tree,TEMPORALHALOIDVAL)