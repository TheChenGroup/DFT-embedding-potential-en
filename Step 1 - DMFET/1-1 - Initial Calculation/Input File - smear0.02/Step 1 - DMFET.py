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
#import matplotlib.pyplot as plt
import time
from scipy import optimize
import os
import shutil

isFristOEP = True             # Whether it is the first DMFET calculation for this system (whether Vemb is inherited), True means it is the first time, subsequent calculations are all False
mol_ref_coords = 'C2H6O.xyz'  # Coordinate file of the initial system (xyz file)
Number_of_Atom = 9             # Total number of atoms in the initial system
idx_A = [0, 8]          # Index of the "embedding region" atoms in the coordinate file (starting from 0)
DFT_type = 'RKS'                # Specific type of DFT
xcfunc = 'hse06'                # Functional
mol_basis = '6-31G'             # Basis set
total_smear_sigma = 0.02        # Gaussian smearing width for the entire system, set to 0 if no smearing
a_smear_sigma = 0.02            # Gaussian smearing width for the "embedding region"
b_smear_sigma = 0.02            # Gaussian smearing width for the "environment"
mol_ref_spin = 0                                                       # Spin of the total system
mol_ref_charge = 0                                                     # Charge of the total system
mol_a_spin = 0                                                         # Spin of the "embedding region"
mol_a_charge = -1                                                      # Charge of the "embedding region"
pc_coords_a = np.array(np.loadtxt('charge_A.txt').reshape(-1,3))       # XYZ coordinate file of the capping charges for the "embedding region" "charge_A.txt", if there are multiple charges, separate them by line, resulting in N rows and 3 columns
pc_charges_a = np.array([1 for i in range(len(pc_coords_a))])          # Charge amount of each capping charge for the "embedding region", default is +1
mol_b_spin = 0                                                         # Spin of the "environment"
mol_b_charge = +1                                                      # Charge of the "environment" (Note that the sum of the charges of the "embedding region" and "environment" should be the charge of the total system)
pc_coords_b = np.array(np.loadtxt('charge_B.txt').reshape(-1,3))       # Coordinate file of the capping charges for the "environment" "charge_B.txt"
pc_charges_b = np.array([-1 for i in range(len(pc_coords_b))])         # Charge amount of each capping charge for the "environment", default is -1

# Default items that can be left unchanged
shutil.copyfile(mol_ref_coords, "part_A.xyz")                         # Automatically generate the coordinate file "part_A.xyz" for the "embedded region" based on the above settings for idx_A, where atoms in the "environment" region are replaced with Ghost atoms
shutil.copyfile(mol_ref_coords, "part_B.xyz")                         # Automatically generate the coordinate file "part_B.xyz" for the "environment" based on the above settings for idx_A, where atoms in the "embedded" region are replaced with Ghost atoms
number_iter = 0                 # Current iteration count of DMFET's OEP
OvlpCutValue = 0.01             # Manually set threshold for the "overlapping orbital space," default is 0.01
conv_tol = 1e-11                # Energy convergence precision during DFT calculations
conv_tol_grad = 1e-7            # Gradient convergence precision during DFT calculations

def choose_atoms(atom_index, xyzfile):                                 # Function to construct "part_A.xyz" and "part_B.xyz"
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
    
    with open(xyzfile, 'w') as g:
        for j in new_xyz:
            g.write(j)
    return xyzfile

def DFT_cal(mol_coords, mol_spin, mol_charge, smear_sigma, coeffname = None, dm_init_guess = None, moldenname = None, pc_coords = None, pc_charges = None, vmat_ext = None, h1 = None, mol_ecp = None):
    # DFT Calculator   
    # There are many optional parameters here
    # coeffname represents whether the molecular orbital coefficients need to be output, if so, name them
    # dm_init_guess is the initial guess when starting DMFET, default is None for the first DMFET calculation, for subsequent calculations, input the density matrix obtained from the previous run here
    # moldenname represents whether the molecular orbital molden file needs to be output, if so, name it
    # pc_coords is the coordinate file for capping charges
    # pc_charges is the array of charges corresponding one-to-one with the coordinates in pc_coords
    # vmat_ext is the Vemb external field matrix
    # h1 is the single-electron part of the Hamiltonian
    # mol_ecp indicates whether to use pseudopotentials, if so, specify the type (refer to PySCF)
    # The function outputs E: energy; dm: density matrix; h: single-electron part h1; s: overlap matrix of atomic orbitals; occ: molecular orbital occupancy


    global DFT_type, conv_tol, conv_tol_grad, xcfunc, DFT_type, mol_basis
    ni = dft.numint.NumInt()
    mol = gto.Mole()
    mol.atom = mol_coords
    mol.spin = mol_spin
    mol.charge = mol_charge
    mol.basis = mol_basis
#    mol.unit = mol_unit
    if mol_ecp is not None:
        mol.ecp = mol_ecp    
    mol.build()  
    
    if DFT_type == 'RKS':
        mf = dft.RKS(mol)
    elif DFT_type == 'ROKS':
        mf = dft.ROKS(mol)
    elif DFT_type == 'UKS':
        mf = dft.UKS(mol)
    else:
        print('Please input DFTtype as RKS/ROKS/UKS')
    
    if (pc_coords is not None) and (pc_charges is not None):        
        mf = qmmm.mm_charge(mf, pc_coords, pc_charges)

    if (h1 is not None) and (vmat_ext is not None):    
        vmat_ext = vmat_ext.reshape(len(h_ref) , len(h_ref))
        h2 = h1 + vmat_ext        
        mf.get_hcore = lambda *args:h2        

    mf.occ_path = os.getcwd()      
    mf.max_cycle = 200
#    mf.verbose = 5
    mf.xc = xcfunc
    mf.conv_tol = conv_tol
    mf.conv_tol_grad = conv_tol_grad
    mf.smear_sigma = smear_sigma
#    mf = mf.newton()

    if dm_init_guess is not None:
        E = mf.kernel(dm0 = dm_init_guess)
    else:
        E = mf.kernel()  
        
    dm = mf.make_rdm1() 
    h = mf.get_hcore()   
    s = mf.get_ovlp(mol)
    occ = mf.get_occ()
    if coeffname is not None:
        numpy.savetxt(coeffname, mf.mo_coeff[:,mf.mo_occ>0.5]) # Save only the mo_coeff of occupied orbitals
    if moldenname is not None:
        molden.from_scf(mf, moldenname)
    return E, dm, h, s, occ

def Lagrangian(V_DMFET): # OEP process
    # Get Functional and Gradient
    global isFirstOEP, OvlpCutValue, s_ref, number_iter, ha_ref, hb_ref, mol_a_coords, mol_b_coords, mol_a_spin, mol_a_charge, mol_b_spin, mol_b_charge, pc_coords_a, pc_charges_a, pc_coords_b, pc_charges_b    
    V_DMFET = V_DMFET.reshape(len(h_ref), len(h_ref))
    if isFirstOEP and number_iter == 0:
        E_a, dm_a, h_a, s_a, occ_a = DFT_cal(mol_a_coords, mol_a_spin, mol_a_charge, coeffname='a-vext-coeff.txt', smear_sigma=a_smear_sigma, pc_coords=pc_coords_a, pc_charges=pc_charges_a, vmat_ext=V_DMFET, h1=ha_ref, moldenname='a-vext.molden')
        E_b, dm_b, h_b, s_b, occ_b = DFT_cal(mol_b_coords, mol_b_spin, mol_b_charge, coeffname='b-vext-coeff.txt', smear_sigma=b_smear_sigma, pc_coords=pc_coords_b, pc_charges=pc_charges_b, vmat_ext=V_DMFET, h1=hb_ref, moldenname='b-vext.molden') 
    else:    
        E_a, dm_a, h_a, s_a, occ_a = DFT_cal(mol_a_coords, mol_a_spin, mol_a_charge, coeffname='a-vext-coeff.txt', smear_sigma=a_smear_sigma, dm_init_guess=np.loadtxt('dma.txt'), pc_coords=pc_coords_a, pc_charges=pc_charges_a, vmat_ext=V_DMFET, h1=ha_ref, moldenname='a-vext.molden')
        E_b, dm_b, h_b, s_b, occ_b = DFT_cal(mol_b_coords, mol_b_spin, mol_b_charge, coeffname='b-vext-coeff.txt', smear_sigma=b_smear_sigma, dm_init_guess=np.loadtxt('dmb.txt'), pc_coords=pc_coords_b, pc_charges=pc_charges_b, vmat_ext=V_DMFET, h1=hb_ref, moldenname='b-vext.molden')         

    grad_dm = -(dm_a + dm_b - dm_ref)
    W = -(E_a + E_b - np.sum(V_DMFET * dm_ref))
    grad_dm = np.array(grad_dm.reshape(1, -1)[0])
    print('grad_max = ', max(abs(grad_dm)))         # Current gradient value
    print('W = ', W)                                # Current functional value W
    numpy.savetxt('V_DMFET.txt', V_DMFET)           # Save the current embedding potential "V_DMFET.txt"
    numpy.savetxt('dma.txt', dm_a)                  # Save the current density matrix of the "embedded region" as "dma.txt"
    numpy.savetxt('dmb.txt', dm_b)                  # Save the current density matrix of the "environment" as "dmb.txt"
    numpy.savetxt('occa.txt', occ_a)                # Save the current molecular orbital occupancy of the "embedded region" as "occa.txt"
    numpy.savetxt('occb.txt', occ_b)                # Save the current molecular orbital occupancy of the "environment" as "occb.txt"

    a_vext_coeff = np.loadtxt('a-vext-coeff.txt')                             # Load the current molecular orbital coefficients of the "embedded region" from "a-vext-coeff.txt"
    b_vext_coeff = np.loadtxt('b-vext-coeff.txt')                             # Load the current molecular orbital coefficients of the "environment" from "b-vext-coeff.txt"
    MO_ovlp_vext = np.dot(a_vext_coeff.T, s_ref).dot(b_vext_coeff)
    print('Max Abs Ovlp Value = ', np.max(abs(MO_ovlp_vext)))                 # Maximum overlap value between the molecular orbitals of the "embedded region" and the "environment"
    print('shape of a-vext-coeff = ', np.shape(a_vext_coeff))
    print('shape of b-vext-coeff = ', np.shape(b_vext_coeff))
    np.savetxt('MO_ovlp_vext.txt', MO_ovlp_vext)                              # Save the overlap matrix of the molecular orbitals between the "embedded region" and the "environment" as 'MO_ovlp_vext.txt'
    print('shape of MO-vext-ovlp = ', np.shape(MO_ovlp_vext))
    for i in range(len(MO_ovlp_vext)):
        for j in range(len(MO_ovlp_vext[i])):
            if abs(MO_ovlp_vext[i][j]) > OvlpCutValue:
                print('Ovlp of MO ', i+1, ' & ', j+1, ' = ', MO_ovlp_vext[i][j])    # Indices of molecular orbitals with overlap values greater than the threshold; the first index is for the "embedded region" and the second for the "environment"
    
    number_iter += 1
    print('N_iteration = ', number_iter)                                      # Current iteration count of OEP
    return W, grad_dm

atom_idx = [i for i in range(Number_of_Atom)]
idx_B = []
for j in atom_idx:
    if j not in idx_A:
        idx_B.append(j)
mol_a_coords = choose_atoms(idx_A, 'part_A.xyz')
mol_b_coords = choose_atoms(idx_B, 'part_B.xyz')




# Set OEP from here. If this is the first time doing DMFET, dm_init_guess does not need to be set. 
# If continuing OEP, the following settings are needed:
# First, perform a DFT on the total system, then perform a DFT on the "embedding region" (to obtain ha without embedding potential), 
# and finally perform a DFT on the "environment" (to obtain hb without embedding potential).
# If starting OEP for the first time, the initial guess for Vemb is a matrix of all zeros.
# If not starting for the first time, inherit the Vemb obtained from the previous calculation. The filename can be customized, here it is 'V_DMFET.txt'.

if isFristOEP:
    E_ref, dm_ref, h_ref, s_ref, occ_ref = DFT_cal(mol_ref_coords, mol_ref_spin, mol_ref_charge, smear_sigma = total_smear_sigma, moldenname = 'ref.molden', dm_init_guess = None) 
    Ea_ref, dma_ref, ha_ref, sa_ref, occa_ref = DFT_cal(mol_a_coords, mol_a_spin, mol_a_charge, smear_sigma = a_smear_sigma, pc_coords = pc_coords_a, pc_charges = pc_charges_a, moldenname = 'a-ref.molden', dm_init_guess = None)
    Eb_ref, dmb_ref, hb_ref, sb_ref, occb_ref = DFT_cal(mol_b_coords, mol_b_spin, mol_b_charge, smear_sigma = b_smear_sigma, pc_coords = pc_coords_b, pc_charges = pc_charges_b, moldenname = 'b-ref.molden', dm_init_guess = None)    
    V_DMFET0 = np.zeros(len(h_ref) * len(h_ref))
else:
    E_ref, dm_ref, h_ref, s_ref, occ_ref = DFT_cal(mol_ref_coords, mol_ref_spin, mol_ref_charge, smear_sigma = total_smear_sigma, moldenname = 'ref.molden', dm_init_guess = np.loadtxt('dm_ref.txt')) 
    Ea_ref, dma_ref, ha_ref, sa_ref, occa_ref = DFT_cal(mol_a_coords, mol_a_spin, mol_a_charge, smear_sigma = a_smear_sigma, pc_coords = pc_coords_a, pc_charges = pc_charges_a, moldenname = 'a-ref.molden', dm_init_guess = np.loadtxt('dma_ref.txt'))
    Eb_ref, dmb_ref, hb_ref, sb_ref, occb_ref = DFT_cal(mol_b_coords, mol_b_spin, mol_b_charge, smear_sigma = b_smear_sigma, pc_coords = pc_coords_b, pc_charges = pc_charges_b, moldenname = 'b-ref.molden', dm_init_guess = np.loadtxt('dmb_ref.txt'))
    V_DMFET0 = np.loadtxt('V_DMFET.txt').reshape(1,-1)

# All variables with "ref" are "reference", i.e., "without embedding potential".
numpy.savetxt('dm_ref.txt', dm_ref)
numpy.savetxt('h_ref.txt', h_ref)
numpy.savetxt('s_ref.txt', s_ref)
numpy.savetxt('ha_ref.txt', ha_ref)
numpy.savetxt('hb_ref.txt', hb_ref)
numpy.savetxt('dma_ref.txt', dma_ref)
numpy.savetxt('dmb_ref.txt', dmb_ref)

# Scipy's L-BFG-S algorithm iteration
x_final, w_final, d = optimize.fmin_l_bfgs_b(Lagrangian, x0=V_DMFET0, args=(), factr=1e4, pgtol=1e-05, maxls=1000)
# x_final, w_final, d = optimize.minimize(Lagrangian, x0=V_DMFET0, args=(), method='CG')
print(x_final)
print(w_final)
print(d)
