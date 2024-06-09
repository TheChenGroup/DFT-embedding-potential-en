# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 15:11:17 2021

@author: 19011
"""

import numpy as np
import scipy
from scipy.linalg import sqrtm

# Orbital indices that need to be orthogonalized (starting from 1)
id_A = [19, 20]          # Overlapping orbitals in the "embedding region"
id_B = [131, 132]        # Overlapping orbitals in the "environment"

id_A_ovlp = []
id_B_ovlp = []
for _ in range(len(id_A)):
    id_A_ovlp.append(id_A[_]-1)
for _ in range(len(id_B)):
    id_B_ovlp.append(id_B[_]-1)

AoOvlp = np.loadtxt('s_reff.txt')
MoOvlpOcc_a = np.loadtxt('occa.txt').reshape(-1, 1)[:int(len(AoOvlp)), :]
MoOvlpOcc_b = np.loadtxt('occb.txt').reshape(-1, 1)[:int(len(AoOvlp)), :]
MoCoeff_A = np.loadtxt('a-vext-coeff.txt')
MoCoeff_B = np.loadtxt('b-vext-coeff.txt')
MatrixOcc_Aovlp = np.eye(np.shape(id_A)[0])
MatrixOcc_Bovlp = np.eye(np.shape(id_B)[0])

# Construct X matrix for Lowdin orthogonalization
s, w = np.linalg.eigh(AoOvlp)
X_dia = np.eye(np.shape(AoOvlp)[0])
for _ in range(len(s)):
    X_dia[_][_] *= s[_]**(-0.5)

X = np.matrix(np.dot(w, X_dia).dot(w.T))

# Construct Pa and Pb in the original AO basis, with overlapping orbitals

for _ in range(len(id_A)):
    MatrixOcc_Aovlp[_][_] *= MoOvlpOcc_a[id_A_ovlp[_]]
for _ in range(len(id_B)):
    MatrixOcc_Bovlp[_][_] *= MoOvlpOcc_b[id_B_ovlp[_]]    

Pa = np.dot(MoCoeff_A[:, id_A_ovlp], MatrixOcc_Aovlp).dot(MoCoeff_A[:, id_A_ovlp].T)
Pb = np.dot(MoCoeff_B[:, id_B_ovlp], MatrixOcc_Bovlp).dot(MoCoeff_B[:, id_B_ovlp].T)

P = Pa + Pb

# Construct P* in the orthogonalized AO basis
P_orth = np.dot(X.I, P).dot(X.I)

# The following results in v = U.T = U.I
OrthAoOcc, v = np.linalg.eigh(P_orth)
for _ in range(len(OrthAoOcc)): # Correct numerical instability issues
    if abs(OrthAoOcc[_]) < 0.00000005:
        OrthAoOcc[_] = 0
    if abs(OrthAoOcc[_]) < 2.5 and abs(OrthAoOcc[_]) > 1.5:
        OrthAoOcc[_] = 2
    if abs(OrthAoOcc[_]) < 1.5 and abs(OrthAoOcc[_]) > 0.5:
        OrthAoOcc[_] = 1    

print('New Ovlp-Mo Occ = ', OrthAoOcc)

# Construct D
D = np.eye(np.shape(OrthAoOcc)[0])
for _ in range(len(OrthAoOcc)):
    D[_][_] *= OrthAoOcc[_]

# Construct Da and Db, using 0.5 and 1.5 as thresholds. This should be correct, but it's best to confirm to avoid errors.
Da = D.copy()
Db = D.copy()

for i in range(len(Da)):
    for j in range(len(Da[i])):
        if Da[i][j] == 2:
            Da[i][j] = 0
        if Da[i][j] == 1:
            Da[i][j] = 0

for i in range(len(Db)):
    for j in range(len(Db[i])):
        if Db[i][j] == 1:
            Db[i][j] = 0

# Construct Pa* and Pb* in the orthogonal AO basis
Pa_orth = np.dot(v, Da).dot(v.T)
Pb_orth = np.dot(v, Db).dot(v.T)

# Construct new Pa and Pb in the original AO basis, with overlapping orbitals
Pa_ovlp = np.dot(X, Pa_orth).dot(X)
Pb_ovlp = np.dot(X, Pb_orth).dot(X)

# Construct Pa_unovlp and Pb_unovlp in the original AO basis, with non-overlapping orbitals
MoA = [i for i in range(np.shape(MoCoeff_A)[1])]
MoB = [j for j in range(np.shape(MoCoeff_B)[1])]

id_A_unovlp = []
for _ in MoA:
    if _ not in id_A_ovlp:
        id_A_unovlp.append(_)

id_B_unovlp = []
for _ in MoB:
    if _ not in id_B_ovlp:
        id_B_unovlp.append(_)

MatrixOcc_Aunovlp = np.eye(np.shape(id_A_unovlp)[0])
MatrixOcc_Bunovlp = np.eye(np.shape(id_B_unovlp)[0])

for _ in range(len(id_A_unovlp)):
    MatrixOcc_Aunovlp[_][_] *= 2
for _ in range(len(id_B_unovlp)):
    MatrixOcc_Bunovlp[_][_] *= 2    

Pa_unovlp = np.dot(MoCoeff_A[:, id_A_unovlp], MatrixOcc_Aunovlp).dot(MoCoeff_A[:, id_A_unovlp].T)
Pb_unovlp = np.dot(MoCoeff_B[:, id_B_unovlp], MatrixOcc_Bunovlp).dot(MoCoeff_B[:, id_B_unovlp].T)        

# The final new reference Pa_new and Pb_new are the sum of the adjusted overlapping part and the unadjusted non-overlapping part
Pa_new = Pa_ovlp + Pa_unovlp
P_reff = np.loadtxt('dm_reff.txt')
Pb_new = Pb_ovlp + Pb_unovlp
Pa_new = P_reff - Pb_new

# Construct new Mo_coeff
New_coeff_ovlp = np.dot(X, v)
new_b_id = []
for _ in range(len(OrthAoOcc)):
    if OrthAoOcc[_] == 2:
        new_b_id.append(_)
        
New_coeff_B_ovlp = New_coeff_ovlp[:, new_b_id]
np.savetxt('New_coeff_ovlp.txt', New_coeff_ovlp)
New_coeff_B = MoCoeff_B.copy()
i = 0 # Index for the new B overlapping orbitals
for idB in range(int(np.shape(MoCoeff_B)[1])):

    if idB in id_B_ovlp:
        for _ in range(len(New_coeff_B[:, idB])):
            New_coeff_B[_, idB] = New_coeff_B_ovlp[_, i]
        i += 1

# Obtain the final orthogonalized target density matrix: Pa_new (for embedding region), Pb_new (for environment), and new coefficients for the environment's molecular orbitals, used for subsequent overlap calculations
np.savetxt('Pa_new.txt', Pa_new)  
np.savetxt('Pb_new.txt', Pb_new)
np.savetxt('New_coeff_B.txt', New_coeff_B)










