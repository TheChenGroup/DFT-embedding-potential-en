# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 12:02:32 2021

@author: 19011
"""

import numpy as np
import numpy
import pyscf
from functools import reduce
from pyscf import gto, scf, dft, qmmm
from pyscf import lo
from pyscf.tools import molden
from pyscf.scf import atom_hf
from pyscf.dft import numint
import matplotlib.pyplot as plt
import time
from scipy import optimize
import os
import shutil


mol_ref_coords = 'C2H6O.xyz'  # Coordinate file of the initial system (xyz file)
Number_of_Atom = 9             # Total number of atoms in the initial system
idx_A = [0, 8]                 # Indices of "embedding region" atoms in the coordinate file (starting from 0)
DFT_type = 'RKS'               # Specific type of DFT
xcfunc = 'hse06'               # Functional
mol_basis = '6-31G'            # Basis set
a_smear_sigma = 0              # Gaussian smearing width for the "embedding region". If smearing was successfully reduced to 0 previously, it can be set to 0 here.
mol_a_spin = 0                 # Spin of the "embedding region"
mol_a_charge = -1              # Charge of the "embedding region"
pc_coords_a = np.array(np.loadtxt('charge_A.txt').reshape(-1, 3))  # XYZ coordinate file of capping charges for the "embedding region", "charge_A.txt". If there are multiple charges, list them line by line, resulting in an N x 3 array.
pc_charges_a = np.array([1 for i in range(len(pc_coords_a))])      # Charge value for each capping charge in the "embedding region", default is +1
miu = 10000                    # Energy-shift parameter for projection

# Default items that can be left unchanged
shutil.copyfile(mol_ref_coords, "part_A.xyz")  # Automatically generate coordinate file "part_A.xyz" for the "embedding region" based on idx_A. Atoms in the "environment" region are replaced with Ghost atoms.
conv_tol = 1e-11               # Energy convergence tolerance in DFT calculation
conv_tol_grad = 1e-7           # Gradient convergence tolerance in DFT calculation

def choose_atoms(atom_index, xyzfile):
    # Choose Ghost Atoms, atom_index is the index of non-ghost atoms
    with open(xyzfile, 'r') as f:
        xyz = f.readlines()
        xyz_title = xyz[:2]
        xyz_coord = xyz[2:]
        atom_tot = [j for j in range(len(xyz_coord))]
        for k in atom_tot:
            if k not in atom_index:
                xyz_coord[k] = 'ghost-' + xyz_coord[k]
    new_xyz = xyz_title + xyz_coord
    
    with open(xyzfile, 'w') as f:
        f.writelines(new_xyz)
    return new_xyz

def DFT_cal(xyzfile, spin, charge, smear_sigma=0.005, dm_init_guess=None, pc_coords=None, pc_charges=None, h1=None, vmat_ext=None, moldenname=None, coeffname=None):
    mol = gto.Mole()
    mol.atom = open(xyzfile).read()
    mol.basis = mol_basis
    mol.charge = charge
    mol.spin = spin
    mol.verbose = 3
    mol.symmetry = True
    mol.build()
    
    if DFT_type == 'RKS':
        mf = dft.RKS(mol)
    elif DFT_type == 'ROKS':
        mf = dft.ROKS(mol)
    elif DFT_type == 'UKS':
        mf = dft.UKS(mol)
    else:
        raise ValueError('Please input DFT type as RKS/ROKS/UKS')
    
    if (pc_coords is not None) and (pc_charges is not None):       
        mf = qmmm.mm_charge(mf, pc_coords, pc_charges)

    if (h1 is not None) and (vmat_ext is not None):    
        vmat_ext = vmat_ext.reshape(len(ha_ref), len(ha_ref))
        h2 = h1 + vmat_ext       
        mf.get_hcore = lambda *args: h2
    
    mf.occ_path = os.getcwd()     
    mf.max_cycle = 200
#    mf.verbose = 5
    mf.xc = xcfunc
    mf.conv_tol = conv_tol
    mf.conv_tol_grad = conv_tol_grad
    mf.smear_sigma = smear_sigma
#    mf = mf.newton()

    if dm_init_guess is not None:
        E = mf.kernel(dm0=dm_init_guess)
    else:
        E = mf.kernel()  
    
    dm = mf.make_rdm1() 
    h = mf.get_hcore()   
    s = mf.get_ovlp(mol)
    occ = mf.get_occ()
    if coeffname is not None:
        numpy.savetxt(coeffname, mf.mo_coeff[:, mf.mo_occ > 0.5])  # Save only occupied mo_coeff   
    if moldenname is not None:
        molden.from_scf(mf, moldenname)
    return E, dm, h, s, occ

mol_a_coords = choose_atoms(idx_A, 'part_A.xyz')

Vemb = numpy.loadtxt('V_DMFET.txt')
dma = numpy.loadtxt('dma.txt')
dmb = numpy.loadtxt('dmb.txt')
ha_ref = numpy.loadtxt('ha_ref.txt')
s_ref = numpy.loadtxt('s_ref.txt')

# Construct the projection operator P, add miu*P to Vemb for calculation, and get the final result
Projector = s_ref.dot(dmb).dot(s_ref)
Vemb_with_P = (Vemb + miu * Projector).reshape(len(ha_ref), len(ha_ref))
Ea_P, dma_P, ha_P, sa_P, occa_P = DFT_cal(mol_a_coords, mol_a_spin, mol_a_charge, coeffname='a-vext_P-coeff.txt', smear_sigma=a_smear_sigma, dm_init_guess=np.loadtxt('dma.txt'), pc_coords=pc_coords_a, pc_charges=pc_charges_a, vmat_ext=Vemb_with_P, h1=ha_ref, moldenname='a-vext_P.molden')
numpy.savetxt('Vemb_with_P.txt', Vemb_with_P)  # This is the final potential with the projection added
numpy.savetxt('dma_P.txt', dma_P)
numpy.savetxt('occa_P.txt', occa_P)

print('dma_deta Max = ', max(abs((dma_P - dma).reshape(1,-1)[0])))  # Check the change in the density matrix of the embedded region with projection added compared to without projection; the change should be very small.

