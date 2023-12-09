# -*- coding: utf-8 -*-
"""
Created on Jan  25 11:06:18 2023

@author: gauser
"""

from clifford import MVArray
from QuantumGA import _e1, _e2, _e3, _e12, _e13, _e23, _e123, _zero, _one
from QuantumGA import tract, dyad, triad, places, linelen, ecount
from numpy import sqrt, e, pi, sin, cos, tan, sinh, cosh, tanh, log, array

import QuantumGA as qga
import numpy as np

# set default formatting options

places(12)
linelen(150)

# define some primitive blades

e1 = _e1
e2 = _e2
e3 = _e3

e12 = _e12
e23 = _e23
e13 = _e13

e123 = _e123

zero = _zero
one = _one
i = e123

# define the primitive specular states

u_ket = (1+e3)/2.0
n_ket = (1-e3)/2.0
d_ket = (e1+e13)/2.0
p_ket = (e1-e13)/2.0

# define some primitive spinor states

r_ket = (u_ket+d_ket)/sqrt(2.0)
l_ket = (u_ket-d_ket)/sqrt(2.0)

i_ket = (u_ket+i*d_ket)/sqrt(2.0)
o_ket = (u_ket-i*d_ket)/sqrt(2.0)

s_ket = (1+e1)/2.0
s_bar = (1-e1)/2.0

t_ket = (1+e2)/2.0
t_bar = (1-e2)/2.0

u_bar = (1-e3)/2.0

# define some primitive dyads

oo = dyad(one, one)

ii = dyad(i, i)
oi = dyad(one, i)
io = dyad(i, one)

ox = dyad(one, e1)
xo = dyad(e1, one)
xx = dyad(e1, e1)

oy = dyad(one, e2)
yo = dyad(e2, one)
yy = dyad(e2, e2)

oz = dyad(one, e3)
zo = dyad(e3, one)
zz = dyad(e3, e3)

xy = dyad(e1, e2)
xz = dyad(e1, e3)
yx = dyad(e2, e1)
yz = dyad(e2, e3)
zx = dyad(e3, e1)
zy = dyad(e3, e2)

# define the specular dyadic states

u_u = dyad(u_ket, u_ket)
u_d = dyad(u_ket, d_ket)
d_u = dyad(d_ket, u_ket)
d_d = dyad(d_ket, d_ket)

u_p = dyad(u_ket, p_ket)
u_n = dyad(u_ket, n_ket)
d_p = dyad(d_ket, p_ket)
d_n = dyad(d_ket, n_ket)

p_u = dyad(p_ket, u_ket)
p_d = dyad(p_ket, d_ket)
n_u = dyad(n_ket, u_ket)
n_d = dyad(n_ket, d_ket)

p_p = dyad(p_ket, p_ket)
p_n = dyad(p_ket, n_ket)
n_p = dyad(n_ket, p_ket)
n_n = dyad(n_ket, n_ket)

# define some useful dyadic states

r_u = dyad(r_ket, u_ket)
r_d = dyad(r_ket, d_ket)
l_u = dyad(l_ket, u_ket)
l_d = dyad(l_ket, d_ket)

u_r = dyad(u_ket, r_ket)
d_r = dyad(d_ket, r_ket)
u_l = dyad(u_ket, l_ket)
d_l = dyad(d_ket, l_ket)

r_r = dyad(r_ket, r_ket)
r_l = dyad(r_ket, l_ket)
l_r = dyad(l_ket, r_ket)
l_l = dyad(l_ket, l_ket)

u_s = dyad(u_ket, s_ket)
s_u = dyad(s_ket, u_ket)
n_s = dyad(n_ket, s_ket)
s_n = dyad(s_ket, n_ket)

i_i = dyad(i_ket, i_ket)
i_o = dyad(i_ket, o_ket)
o_i = dyad(o_ket, i_ket)
o_o = dyad(o_ket, o_ket)

# define the dyadic Bell states

phi_p = (u_u + d_d)/sqrt(2.0)
phi_m = (u_u - d_d)/sqrt(2.0)
psi_p = (u_d + d_u)/sqrt(2.0)
psi_m = (u_d - d_u)/sqrt(2.0)

PHI_P = phi_p*~phi_p
PHI_M = phi_m*~phi_m
PSI_P = psi_p*~psi_p
PSI_M = psi_m*~psi_m

# define some dydic projection operators

P1 = (dyad(one, one) + dyad(e3, one))/2
M1 = (dyad(one, one) - dyad(e3, one))/2

P2 = (dyad(one, one) + dyad(one, e3))/2
M2 = (dyad(one, one) - dyad(one, e3))/2

P3 = (dyad(one, one) + dyad(e3, e3))/2
M3 = (dyad(one, one) - dyad(e3, e3))/2

P5 = (dyad(one, one) + dyad(e1, one))/2
M5 = (dyad(one, one) - dyad(e1, one))/2

P6 = (dyad(one, one) + dyad(e2, one))/2
M6 = (dyad(one, one) - dyad(e2, one))/2

PL = M5
PR = P5

# define the 2x2 spectral matrix

S2 = array(MVArray([u_ket, p_ket, d_ket, n_ket]).reshape(2, 2))

"""
# define the 3x3 (reduced 4x4) spectral matrix

S3 = array([[u_u,               (u_p+p_u)/sqrt(2),    p_p],
            [(u_d+d_u)/sqrt(2), (u_n+p_d+d_p+n_u)/2, (p_n+n_p)/sqrt(2)],
            [d_d,               (d_n+n_d)/sqrt(2),    n_n]])
"""

# define the 3x3 (4x4 subspace) spectral matrix

S3 = array([[u_n, p_d, p_n],
            [d_p, n_u, n_p],
            [d_n, n_d, n_n]])

# define the 4x4 spectral matrix

S4 = array([[u_u, u_p, p_u, p_p],
            [u_d, u_n, p_d, p_n],
            [d_u, d_p, n_u, n_p],
            [d_d, d_n, n_d, n_n]])

# define the 4x4 dyadic (Bloch) matrix

D4 = array([[dyad(one, one), dyad(one, e1), dyad(one,e2), dyad(one, e3)],
            [dyad(e1, one), dyad(e1, e1), dyad(e1, e2), dyad(e1, e3)],
            [dyad(e2, one), dyad(e2, e1), dyad(e2, e2), dyad(e2, e3)],
            [dyad(e3, one), dyad(e3, e1), dyad(e3, e2), dyad(e3, e3)]])

# define the 4x4 orthogonal matrix (single-qubit)

O4 = array(MVArray([one,  e1,   e2,  e3,
                    e1,  one,  e12, e13,
                    e2, -e12,  one, e23,
                    e3, -e13, -e23, one]).reshape(4, 4))

# define the 4x4 bell matrix

B4 = array([[phi_p*~phi_p, phi_p*~psi_p, phi_p*~psi_m, phi_p*~phi_m],
            [psi_p*~phi_p, psi_p*~psi_p, psi_p*~psi_m, psi_p*~phi_m],
            [psi_m*~phi_p, psi_m*~psi_p, psi_m*~psi_m, psi_m*~phi_m],
            [phi_m*~phi_p, phi_m*~psi_p, phi_m*~psi_m, phi_m*~phi_m]])

"""
B5 = array([[phi_p*~phi_p, phi_p*~psi_p, phi_p*~phi_m, phi_p*~psi_m],
            [psi_p*~phi_p, psi_p*~psi_p, psi_p*~phi_m, psi_p*~psi_m],
            [phi_m*~phi_p, phi_m*~psi_p, phi_m*~phi_m, phi_m*~psi_m],
            [psi_m*~phi_p, psi_m*~psi_p, psi_m*~phi_m, psi_m*~psi_m]])
"""

# define the standard Gamma matrices

g0 = array([[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0,-1, 0],
            [0, 0, 0,-1]])

g1 = array([[ 0, 0, 0, 1],
            [ 0, 0, 1, 0],
            [ 0,-1, 0, 0],
            [-1, 0, 0, 0]])

g2 = array([[  0,  0,  0,-1j],
            [  0,  0, 1j,  0],
            [  0, 1j,  0,  0],
            [-1j,  0,  0,  0]])

g3 = array([[ 0, 0, 1, 0],
            [ 0, 0, 0,-1],
            [-1, 0, 0, 0],
            [ 0, 1, 0, 0]])

g5 = array([[0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0]])

g6 = array([[0, 0,-1, 0],
            [0, 0, 0,-1],
            [1, 0, 0, 0],
            [0, 1, 0, 0]])

g7 = array([[ 0,  0,-1j,  0],
            [ 0,  0,  0,-1j],
            [1j,  0,  0,  0],
            [ 0, 1j,  0,  0]])

G0 = tract(g0, S4)
G1 = tract(g1, S4)
G2 = tract(g2, S4)*-ii
G3 = tract(g3, S4)

G5 = tract(g5, S4)

G6 = tract(g6, S4)
G7 = tract(g7, S4)

I = G0*G1*G2*G3

PG0 = -io*G5
PG1 = -io*G1
PG2 = -io*G2
PG3 = -io*G3
PG5 = -io*G0

# define Dirac's original matrices

a1 = array([[0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0]])

a2 = array([[ 0,  0,  0,-1j],
            [ 0,  0, 1j,  0],
            [ 0,-1j,  0,  0],
            [1j,  0,  0,  0]])

a3 = array([[0, 0, 1, 0],
            [0, 0, 0,-1],
            [1, 0, 0, 0],
            [0,-1, 0, 0]])

A0 = oo
A1 = tract(a1, S4)
A2 = tract(a2, S4)*-ii
A3 = tract(a3, S4)
A4 = G0

# define sigma matrices for dyads

O1 = -A1
O2 = -A2
O3 = -A3

# define Dirac spin operators

sx = array([[0,1,0,0],
            [1,0,0,0],
            [0,0,0,1],
            [0,0,1,0]])

sy = array([[0,-1j,0,0],
            [1j,0,0,0],
            [0,0,0,-1j],
            [0,0,1j,0]])
 
sz = array([[1,0,0,0],
            [0,-1,0,0],
            [0,0,1,0],
            [0,0,0,-1]]) 

SX = tract(sx, S4)/2
SY = tract(sy, S4).ii()/2
SZ = tract(sz, S4)/2

# define some single-qubit quantum gates

i2 = array([[1,0],
            [0,1]])

I2 = tract(i2,S2)

pauli_x = array([[0,1],
                 [1,0]])

pauli_y = array([[0, -1j],
                 [1j, 0]])

pauli_z = array([[1,0],
                 [0,-1]])

Pauli_x = tract(pauli_x, S2)
Pauli_y = tract(pauli_y, S2)
Pauli_z = tract(pauli_z, S2)

hadamard = array([[1,1],
                  [1,-1]])/sqrt(2)

Hadamard = tract(hadamard, S2)

phase_t = array([[1, 0],
                 [0, e**(1j*pi/4)]])

phase_s = array([[1, 0],
                 [0, e**(1j*pi/2)]])

T = tract(phase_t, S2)
S = tract(phase_s, S2)

X = Pauli_x
Y = Pauli_y
Z = Pauli_z

H = Hadamard

bs = array([[1, 1j],
            [1j, 1]])/sqrt(2)

BS = tract(bs,S2)

# define some two-qubit quantum gates

BS2 = dyad(BS, BS)

i4 = array([[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])

I4 = tract(i4,S4)

swap  = array([[1, 0, 0, 0],
               [0, 0, 1, 0],
               [0, 1, 0, 0],
               [0, 0, 0, 1]])

cnot1 = array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 0, 1],
               [0, 0, 1, 0]])

cnot2 = array([[1, 0, 0, 0],
               [0, 0, 0, 1],
               [0, 0, 1, 0],
               [0, 1, 0, 0]])

not1  = array([[0, 0, 1, 0],
               [0, 0, 0, 1],
               [1, 0, 0, 0],
               [0, 1, 0, 0]])

not2  = array([[0, 1, 0, 0],
               [1, 0, 0, 0],
               [0, 0, 0, 1],
               [0, 0, 1, 0]])

not12 = array([[0, 0, 0, 1],
               [0, 0, 1, 0],
               [0, 1, 0, 0],
               [1, 0, 0, 0]])

SWAP  = tract(swap, S4)
CNOT1 = tract(cnot1, S4)
CNOT2 = tract(cnot2, S4)
NOT1  = tract(not1, S4)
NOT2  = tract(not2, S4)
NOT12 = tract(not12, S4)

CNOT  = CNOT1

hadamard1  = array([[1, 0, 1, 0],
                    [0, 1, 0, 1],
                    [1, 0,-1, 0],
                    [0, 1, 0,-1]])/sqrt(2)

hadamard2  = array([[1, 1, 0, 0],
                    [1,-1, 0, 0],
                    [0, 0, 1, 1],
                    [0, 0, 1,-1]])/sqrt(2)

HADAMARD1 = tract(hadamard1, S4)
HADAMARD2 = tract(hadamard2, S4)

H1 = HADAMARD1
H2 = HADAMARD2

HH = H1*H2

BELL1 = CNOT1 * H1

bell = array([[1, 0, 0, 1],
              [0, 1,-1, 0],
              [0, 1, 1, 0],
              [-1,0, 0, 1]])

BELL2 = tract(bell, S4)

swap_root = array([[1,       0,       0, 0],
                   [0,(1+1j)/2,(1-1j)/2, 0],
                   [0,(1-1j)/2,(1+1j)/2, 0],
                   [0,       0,       0, 1]])

cv        = array([[1, 0,       0,       0],
                   [0, 1,       0,       0],
                   [0, 0,(1+1j)/2,(1-1j)/2],
                   [0, 0,(1-1j)/2,(1+1j)/2]])

dcnot     = array([[1, 0, 0, 0],
                   [0, 0, 0, 1],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0]])

swap_bar  = array([[1, 0, 0, 0],
                   [0, 0,-1, 0],
                   [0,-1, 0, 0],
                   [0, 0, 0, 1]])

cnot1_bar = array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 0,-1],
                   [0, 0,-1, 0]])

cnot2_bar = array([[1, 0, 0, 0],
                   [0, 0, 0,-1],
                   [0, 0, 1, 0],
                   [0,-1, 0, 0]])

not1_bar  = array([[ 0, 0, 1, 0],
                   [ 0, 0, 0,-1],
                   [ 1, 0, 0, 0],
                   [ 0,-1, 0, 0]])

not2_bar  = array([[0, 1, 0, 0],
                   [1, 0, 0, 0],
                   [0, 0, 0,-1],
                   [0, 0,-1, 0]])

SWAP_ROOT = tract(swap_root, S4)
CV        = tract(cv, S4)
DCNOT     = tract(dcnot, S4)

SWAP_BAR  = tract(swap_bar, S4)
CNOT1_BAR = tract(cnot1_bar, S4)
CNOT2_BAR = tract(cnot2_bar, S4)
NOT1_BAR  = tract(not1_bar, S4)
NOT2_BAR  = tract(not2_bar, S4)

# define a set of single-sided Lorentz generators

j1 = array([[ 0, 0, 0, 0],
            [ 0, 0, 0, 0],
            [ 0, 0, 0,-1],
            [ 0, 0, 1, 0]])

j2 = array([[ 0, 0, 0, 0],
            [ 0, 0, 0, 1],
            [ 0, 0, 0, 0],
            [ 0,-1, 0, 0]])

j3 = array([[ 0, 0, 0, 0],
            [ 0, 0,-1, 0],
            [ 0, 1, 0, 0],
            [ 0, 0, 0, 0]])

k1 = array([[ 0, 1, 0, 0],
            [ 1, 0, 0, 0],
            [ 0, 0, 0, 0],
            [ 0, 0, 0, 0]])

k2 = array([[ 0, 0, 1, 0],
            [ 0, 0, 0, 0],
            [ 1, 0, 0, 0],
            [ 0, 0, 0, 0]])

k3 = array([[ 0, 0, 0, 1],
            [ 0, 0, 0, 0],
            [ 0, 0, 0, 0],
            [ 1, 0, 0, 0]])

# Chiral basis generators

J1 = oi*tract(j1, S4)
J2 = io*tract(j2, S4)
J3 = oi*tract(j3, S4)

K1 = oi*tract(k1, S4)
K2 = io*tract(k2, S4)
K3 = oi*tract(k3, S4)

# Euclidean basis generators

J1 = tract(j1, S4)
J2 = tract(j2, S4)
J3 = tract(j3, S4)

K1 = tract(-k1, S4)
K2 = tract(-k2, S4)
K3 = tract(-k3, S4)

J1p = (J1+io*K1)/2
J2p = (J2+io*K2)/2
J3p = (J3+io*K3)/2

J1m = (J1-io*K1)/2
J2m = (J2-io*K2)/2
J3m = (J3-io*K3)/2

# define some Bell space matrices

phi_plus=array([[ 1, 0, 0, 1],
                [ 0, 0, 0, 0],
                [ 0, 0, 0, 0],
                [ 1, 0, 0, 1]])

phi_minus=array([[ 1, 0, 0,-1],
                 [ 0, 0, 0, 0],
                 [ 0, 0, 0, 0],
                 [-1, 0, 0, 1]])

psi_plus=array([[ 0, 0, 0, 0],
                [ 0, 1, 1, 0],
                [ 0, 1, 1, 0],
                [ 0, 0, 0, 0]])

psi_minus=array([[ 0, 0, 0, 0],
                 [ 0, 1,-1, 0],
                 [ 0,-1, 1, 0],
                 [ 0, 0, 0, 0]])

# define the traditional Gell-Mann matrices

gm1 = array([[ 0,  1,  0],
             [ 1,  0,  0],
             [ 0,  0,  0]])

gm2 = array([[ 0,-1j,  0],
             [1j,  0,  0],
             [ 0,  0,  0]])

gm3 = array([[ 1,  0,  0],
             [ 0, -1,  0],
             [ 0,  0,  0]])

gm4 = array([[ 0,  0,  1],
             [ 0,  0,  0],
             [ 1,  0,  0]])

gm5 = array([[ 0,  0,-1j],
             [ 0,  0,  0],
             [1j,  0,  0]])

gm6 = array([[ 0,  0,  0],
             [ 0,  0,  1],
             [ 0,  1,  0]])

gm7 = array([[ 0,  0,  0],
             [ 0,  0,-1j],
             [ 0, 1j,  0]])

gm8 = array([[ 1,  0,  0],
             [ 0,  1,  0],
             [ 0,  0, -2]])/sqrt(3)

gma = array([[ 1, 0, 0],
             [ 0, 0, 0],
             [ 0, 0,-1]])

gmb = array([[ 0, 0, 0],
             [ 0, 1, 0],
             [ 0, 0,-1]])


GM1 = tract(gm1, S3)
GM2 = tract(gm2, S3)
GM3 = tract(gm3, S3)
GM4 = tract(gm4, S3)
GM5 = tract(gm5, S3)
GM6 = tract(gm6, S3)
GM7 = tract(gm7, S3)
GM8 = tract(gm8, S3)

GMA = tract(gma, S3)
GMB = tract(gmb, S3)

T1 = GM1/2.
T2 = GM2/2.
T3 = GM3/2.
T4 = GM4/2.
T5 = GM5/2.
T6 = GM6/2.
T7 = GM7/2.
T8 = GM8/2.
TA = GMA/2.
TB = GMB/2.

TP = (GM1+oi*GM2)/2.
TM = (GM1-oi*GM2)/2.

UP = (GM6+oi*GM7)/2.
UM = (GM6-oi*GM7)/2.

VP = (GM4+io*GM5)/2.
VM = (GM4-io*GM5)/2.

TZ = GM3/2.
Y8 = GM8/sqrt(3)

i3 = array([[1,0,0],
            [0,1,0],
            [0,0,1]])

I3 = tract(i3, S3)

# define the specular triadic states

uuu=triad(u_ket, u_ket, u_ket)
uud=triad(u_ket, u_ket, d_ket)
udu=triad(u_ket, d_ket, u_ket)
udd=triad(u_ket, d_ket, d_ket)
duu=triad(d_ket, u_ket, u_ket)
dud=triad(d_ket, u_ket, d_ket)
ddu=triad(d_ket, d_ket, u_ket)
ddd=triad(d_ket, d_ket, d_ket)

uup=triad(u_ket, u_ket, p_ket)
uun=triad(u_ket, u_ket, n_ket)
udp=triad(u_ket, d_ket, p_ket)
udn=triad(u_ket, d_ket, n_ket)
dup=triad(d_ket, u_ket, p_ket)
dun=triad(d_ket, u_ket, n_ket)
ddp=triad(d_ket, d_ket, p_ket)
ddn=triad(d_ket, d_ket, n_ket)

upu=triad(u_ket, p_ket, u_ket)
upd=triad(u_ket, p_ket, d_ket)
unu=triad(u_ket, n_ket, u_ket)
und=triad(u_ket, n_ket, d_ket)
dpu=triad(d_ket, p_ket, u_ket)
dpd=triad(d_ket, p_ket, d_ket)
dnu=triad(d_ket, n_ket, u_ket)
dnd=triad(d_ket, n_ket, d_ket)

upp=triad(u_ket, p_ket, p_ket)
upn=triad(u_ket, p_ket, n_ket)
unp=triad(u_ket, n_ket, p_ket)
unn=triad(u_ket, n_ket, n_ket)
dpp=triad(d_ket, p_ket, p_ket)
dpn=triad(d_ket, p_ket, n_ket)
dnp=triad(d_ket, n_ket, p_ket)
dnn=triad(d_ket, n_ket, n_ket)

puu=triad(p_ket, u_ket, u_ket)
pud=triad(p_ket, u_ket, d_ket)
pdu=triad(p_ket, d_ket, u_ket)
pdd=triad(p_ket, d_ket, d_ket)
nuu=triad(n_ket, u_ket, u_ket)
nud=triad(n_ket, u_ket, d_ket)
ndu=triad(n_ket, d_ket, u_ket)
ndd=triad(n_ket, d_ket, d_ket)

pup=triad(p_ket, u_ket, p_ket)
pun=triad(p_ket, u_ket, n_ket)
pdp=triad(p_ket, d_ket, p_ket)
pdn=triad(p_ket, d_ket, n_ket)
nup=triad(n_ket, u_ket, p_ket)
nun=triad(n_ket, u_ket, n_ket)
ndp=triad(n_ket, d_ket, p_ket)
ndn=triad(n_ket, d_ket, n_ket)

ppu=triad(p_ket, p_ket, u_ket)
ppd=triad(p_ket, p_ket, d_ket)
pnu=triad(p_ket, n_ket, u_ket)
pnd=triad(p_ket, n_ket, d_ket)
npu=triad(n_ket, p_ket, u_ket)
npd=triad(n_ket, p_ket, d_ket)
nnu=triad(n_ket, n_ket, u_ket)
nnd=triad(n_ket, n_ket, d_ket)

ppp=triad(p_ket, p_ket, p_ket)
ppn=triad(p_ket, p_ket, n_ket)
pnp=triad(p_ket, n_ket, p_ket)
pnn=triad(p_ket, n_ket, n_ket)
npp=triad(n_ket, p_ket, p_ket)
npn=triad(n_ket, p_ket, n_ket)
nnp=triad(n_ket, n_ket, p_ket)
nnn=triad(n_ket, n_ket, n_ket)

# define some simple triadic states

rrr=triad(r_ket, r_ket, r_ket)
rrl=triad(r_ket, r_ket, l_ket)
rlr=triad(r_ket, l_ket, r_ket)
rll=triad(r_ket, l_ket, l_ket)
lrr=triad(l_ket, r_ket, r_ket)
lrl=triad(l_ket, r_ket, l_ket)
llr=triad(l_ket, l_ket, r_ket)
lll=triad(l_ket, l_ket, l_ket)

iii=triad(i_ket, i_ket, i_ket)
iio=triad(i_ket, i_ket, o_ket)
ioi=triad(i_ket, o_ket, i_ket)
ioo=triad(i_ket, o_ket, o_ket)
oii=triad(o_ket, i_ket, i_ket)
oio=triad(o_ket, i_ket, o_ket)
ooi=triad(o_ket, o_ket, i_ket)
ooo=triad(o_ket, o_ket, o_ket)

# define the 8x8 spectral matrix

S8 = array([[uuu, uup, upu, upp, puu, pup, ppu, ppp],
            [uud, uun, upd, upn, pud, pun, ppd, ppn],
            [udu, udp, unu, unp, pdu, pdp, pnu, pnp],
            [udd, udn, und, unn, pdd, pdn, pnd, pnn],
            [duu, dup, dpu, dpp, nuu, nup, npu, npp],
            [dud, dun, dpd, dpn, nud, nun, npd, npn],
            [ddu, ddp, dnu, dnp, ndu, ndp, nnu, nnp],
            [ddd, ddn, dnd, dnn, ndd, ndn, nnd, nnn]])


# define some three-qubit quantum gates

ccnot = array([[1, 0, 0, 0, 0, 0, 0, 0],
               [0, 1, 0, 0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 0],
               [0, 0, 0, 1, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 1, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 1],
               [0, 0, 0, 0, 0, 0, 1, 0]])

cswap = array([[1, 0, 0, 0, 0, 0, 0, 0],
               [0, 1, 0, 0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 0],
               [0, 0, 0, 1, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 0, 1, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 1]])

CCNOT = tract(ccnot, S8)
CSWAP = tract(cswap, S8)

Tiffoli = CCNOT
Fredkin = CSWAP

i8  = array([[ 1, 0, 0, 0, 0, 0, 0, 0],
             [ 0, 1, 0, 0, 0, 0, 0, 0],
             [ 0, 0, 1, 0, 0, 0, 0, 0],
             [ 0, 0, 0, 1, 0, 0, 0, 0],
             [ 0, 0, 0, 0, 1, 0, 0, 0],
             [ 0, 0, 0, 0, 0, 1, 0, 0],
             [ 0, 0, 0, 0, 0, 0, 1, 0],
             [ 0, 0, 0, 0, 0, 0, 0, 1]])

I8 = tract(i8,S8)

# define some three-qubit entangled states

GHZ1 = (uuu + ddd)/sqrt(2)
GHZ2 = (rrr + lll)/sqrt(2)
GHZ3 = (iii + ooo)/sqrt(2)

GHZ = GHZ1

# define some triadic projection operators

P311 = (triad(one, one, one) + triad(e3, one, one))/2.
P131 = (triad(one, one, one) + triad(one, e3, one))/2.
P113 = (triad(one, one, one) + triad(one, one, e3))/2.
P331 = (triad(one, one, one) + triad(e3, e3, one))/2.
P313 = (triad(one, one, one) + triad(e3, one, e3))/2.
P133 = (triad(one, one, one) + triad(one, e3, e3))/2.
P333 = (triad(one, one, one) + triad(e3, e3, e3))/2.

M311 = (triad(one, one, one) - triad(e3, one, one))/2.
M131 = (triad(one, one, one) - triad(one, e3, one))/2.
M113 = (triad(one, one, one) - triad(one, one, e3))/2.
M331 = (triad(one, one, one) - triad(e3, e3, one))/2.
M313 = (triad(one, one, one) - triad(e3, one, e3))/2.
M133 = (triad(one, one, one) - triad(one, e3, e3))/2.
M333 = (triad(one, one, one) - triad(e3, e3, e3))/2.

# define G2 generators

c1  = array([[ 0, 0, 0, 0, 0, 0, 0, 0],
             [ 0, 0, 0, 0, 0, 0, 0, 0],
             [ 0, 0, 0, 0, 0, 0, 0, 0],
             [ 0, 0, 0, 0, 0, 0, 0, 0],
             [ 0, 0, 0, 0, 0, 0, 0,-1],
             [ 0, 0, 0, 0, 0, 0,-1, 0],
             [ 0, 0, 0, 0, 0, 1, 0, 0],
             [ 0, 0, 0, 0, 1, 0, 0, 0]])

c2  = array([[ 0, 0, 0, 0, 0, 0, 0, 0],
             [ 0, 0, 0, 0, 0, 0, 0, 0],
             [ 0, 0, 0, 0, 0, 0, 0, 0],
             [ 0, 0, 0, 0, 0, 0, 0, 0],
             [ 0, 0, 0, 0, 0, 0, 1, 0],
             [ 0, 0, 0, 0, 0, 0, 0,-1],
             [ 0, 0, 0, 0,-1, 0, 0, 0],
             [ 0, 0, 0, 0, 0, 1, 0, 0]])

c3  = array([[ 0, 0, 0, 0, 0, 0, 0, 0],
             [ 0, 0, 0, 0, 0, 0, 0, 0],
             [ 0, 0, 0, 0, 0, 0, 0, 0],
             [ 0, 0, 0, 0, 0, 0, 0, 0],
             [ 0, 0, 0, 0, 0,-1, 0, 0],
             [ 0, 0, 0, 0, 1, 0, 0, 0],
             [ 0, 0, 0, 0, 0, 0, 0,-1],
             [ 0, 0, 0, 0, 0, 0, 1, 0]])

c4  = array([[ 0, 0, 0, 0, 0, 0, 0, 0],
             [ 0, 0, 0, 0, 0, 0, 0, 0],
             [ 0, 0, 0, 0, 0, 0, 0, 1],
             [ 0, 0, 0, 0, 0, 0, 1, 0],
             [ 0, 0, 0, 0, 0, 0, 0, 0],
             [ 0, 0, 0, 0, 0, 0, 0, 0],
             [ 0, 0, 0,-1, 0, 0, 0, 0],
             [ 0, 0,-1, 0, 0, 0, 0, 0]])

c5  = array([[ 0, 0, 0, 0, 0, 0, 0, 0],
             [ 0, 0, 0, 0, 0, 0, 0, 0],
             [ 0, 0, 0, 0, 0, 0,-1, 0],
             [ 0, 0, 0, 0, 0, 0, 0, 1],
             [ 0, 0, 0, 0, 0, 0, 0, 0],
             [ 0, 0, 0, 0, 0, 0, 0, 0],
             [ 0, 0, 1, 0, 0, 0, 0, 0],
             [ 0, 0, 0,-1, 0, 0, 0, 0]])

c6  = array([[ 0, 0, 0, 0, 0, 0, 0, 0],
             [ 0, 0, 0, 0, 0, 0, 0, 0],
             [ 0, 0, 0, 0, 0, 1, 0, 0],
             [ 0, 0, 0, 0,-1, 0, 0, 0],
             [ 0, 0, 0, 1, 0, 0, 0, 0],
             [ 0, 0,-1, 0, 0, 0, 0, 0],
             [ 0, 0, 0, 0, 0, 0, 0, 0],
             [ 0, 0, 0, 0, 0, 0, 0, 0]])

c7  = array([[ 0, 0, 0, 0, 0, 0, 0, 0],
             [ 0, 0, 0, 0, 0, 0, 0, 0],
             [ 0, 0, 0, 0,-1, 0, 0, 0],
             [ 0, 0, 0, 0, 0,-1, 0, 0],
             [ 0, 0, 1, 0, 0, 0, 0, 0],
             [ 0, 0, 0, 1, 0, 0, 0, 0],
             [ 0, 0, 0, 0, 0, 0, 0, 0],
             [ 0, 0, 0, 0, 0, 0, 0, 0]])

c8  = array([[ 0, 0, 0, 0, 0, 0, 0, 0],
             [ 0, 0, 0, 0, 0, 0, 0, 0],
             [ 0, 0, 0,-2, 0, 0, 0, 0],
             [ 0, 0, 2, 0, 0, 0, 0, 0],
             [ 0, 0, 0, 0, 0, 1, 0, 0],
             [ 0, 0, 0, 0,-1, 0, 0, 0],
             [ 0, 0, 0, 0, 0, 0, 0,-1],
             [ 0, 0, 0, 0, 0, 0, 1, 0]])/sqrt(3)

c9  = array([[ 0, 0, 0, 0, 0, 0, 0, 0],
             [ 0, 0,-2, 0, 0, 0, 0, 0],
             [ 0, 2, 0, 0, 0, 0, 0, 0],
             [ 0, 0, 0, 0, 0, 0, 0, 0],
             [ 0, 0, 0, 0, 0, 0, 0, 1],
             [ 0, 0, 0, 0, 0, 0,-1, 0],
             [ 0, 0, 0, 0, 0, 1, 0, 0],
             [ 0, 0, 0, 0,-1, 0, 0, 0]])/sqrt(3)

c10 = array([[ 0, 0, 0, 0, 0, 0, 0, 0],
             [ 0, 0, 0,-2, 0, 0, 0, 0],
             [ 0, 0, 0, 0, 0, 0, 0, 0],
             [ 0, 2, 0, 0, 0, 0, 0, 0],
             [ 0, 0, 0, 0, 0, 0,-1, 0],
             [ 0, 0, 0, 0, 0, 0, 0,-1],
             [ 0, 0, 0, 0, 1, 0, 0, 0],
             [ 0, 0, 0, 0, 0, 1, 0, 0]])/sqrt(3)

c11 = array([[ 0, 0, 0, 0, 0, 0, 0, 0],
             [ 0, 0, 0, 0,-2, 0, 0, 0],
             [ 0, 0, 0, 0, 0, 0, 0,-1],
             [ 0, 0, 0, 0, 0, 0, 1, 0],
             [ 0, 2, 0, 0, 0, 0, 0, 0],
             [ 0, 0, 0, 0, 0, 0, 0, 0],
             [ 0, 0, 0,-1, 0, 0, 0, 0],
             [ 0, 0, 1, 0, 0, 0, 0, 0]])/sqrt(3)

c12 = array([[ 0, 0, 0, 0, 0, 0, 0, 0],
             [ 0, 0, 0, 0, 0,-2, 0, 0],
             [ 0, 0, 0, 0, 0, 0, 1, 0],
             [ 0, 0, 0, 0, 0, 0, 0, 1],
             [ 0, 0, 0, 0, 0, 0, 0, 0],
             [ 0, 2, 0, 0, 0, 0, 0, 0],
             [ 0, 0,-1, 0, 0, 0, 0, 0],
             [ 0, 0, 0,-1, 0, 0, 0, 0]])/sqrt(3)

c13 = array([[ 0, 0, 0, 0, 0, 0, 0, 0],
             [ 0, 0, 0, 0, 0, 0,-2, 0],
             [ 0, 0, 0, 0, 0,-1, 0, 0],
             [ 0, 0, 0, 0,-1, 0, 0, 0],
             [ 0, 0, 0, 1, 0, 0, 0, 0],
             [ 0, 0, 1, 0, 0, 0, 0, 0],
             [ 0, 2, 0, 0, 0, 0, 0, 0],
             [ 0, 0, 0, 0, 0, 0, 0, 0]])/sqrt(3)

c14 = array([[ 0, 0, 0, 0, 0, 0, 0, 0],
             [ 0, 0, 0, 0, 0, 0, 0,-2],
             [ 0, 0, 0, 0, 1, 0, 0, 0],
             [ 0, 0, 0, 0, 0,-1, 0, 0],
             [ 0, 0,-1, 0, 0, 0, 0, 0],
             [ 0, 0, 0, 1, 0, 0, 0, 0],
             [ 0, 0, 0, 0, 0, 0, 0, 0],
             [ 0, 2, 0, 0, 0, 0, 0, 0]])/sqrt(3)

C1  = tract(c1, S8)
C2  = tract(c2, S8)
C3  = tract(c3, S8)
C4  = tract(c4, S8)
C5  = tract(c5, S8)
C6  = tract(c6, S8)
C7  = tract(c7, S8)
C8  = tract(c8, S8)
C9  = tract(c9, S8)
C10 = tract(c10, S8)
C11 = tract(c11, S8)
C12 = tract(c12, S8)
C13 = tract(c13, S8)
C14 = tract(c14, S8)

# define some physical constants

c = 299792458
h = 6.62607015e-34
G = 6.674e-11

h_bar = h/(2*pi)
alpha = 0.0072973525693

Me  = 9.10938370153e-31
Mk  = 8.800596e-28
Mpi = 2.4883e-28
Mu  = 1.883531e-28
Mp  = 1.67262192369e-27
Mn  = 1.67492749804e-27
Mw  = 1.432886e-25
Mt  = 3.1675419787e-27

Mc = Me*c

mu_0 = 4e-7*pi
e_0  = 1/(mu_0*c**2)
q_0  = 1.60217662e-19
m_0  = h_bar/c
m0   = m_0

GeV = 6.2415064799632E+9
MeV = GeV*1e3
KeV = GeV*1e6
eV  = GeV*1e9

# working definitions

q_u = S3[0, 0]
q_d = S3[1, 0]
q_s = S3[2, 0]

# define some boosted spinors and operators

r1 = e**(i*e1*pi/6)
r2 = e**(i*e2*pi/6)
r3 = e**(i*e3*pi/6)

b = 1.0
r = r3*r1
v = (r*e3*~r)(1)

boost = e**(dyad(e3*b/2.0, v))

r_1 = boost*r_u
r_2 = boost*r_d
l_2 = boost*l_u
l_1 = boost*l_d

wp = e**dyad(e3, b*v)*G5 + oo
wm = e**dyad(e3, b*v)*G5 - oo

i_u = dyad(i_ket, u_ket)
i_d = dyad(i_ket, d_ket)
o_u = dyad(o_ket, u_ket)
o_d = dyad(o_ket, d_ket)

i_1 = boost*i_u
i_2 = boost*i_d
o_2 = boost*o_u
o_1 = boost*o_d

fp = e**dyad(e3, b*v)*G6 + oo
fm = e**dyad(e3, b*v)*G6 - oo

r = r3*r1
v = (r3*r1*e3*~r1*~r3)(1)
w = (r3*~r2*e3*r2*~r3)(1)

b = 1.0

boost = e**dyad(e1*b/2.0, v)

u_1 = boost*u_u
u_2 = boost*u_d
v_2 = boost*d_u
v_1 = boost*d_d

dp = e**dyad(e1,b*v)*G0 + oo
dm = e**dyad(e1,b*v)*G0 - oo

Boost = e**(dyad(e1*b, v))

U_1 = Boost*u_u
U_2 = Boost*u_d
V_1 = Boost*d_u
V_2 = Boost*d_d

DP = e**dyad(e1,2.0*v)*G0 + oo
DM = e**dyad(e1,2.0*v)*G0 - oo

m=cos(pi/6)*e1+sin(pi/6)*e2
n=cos(pi*2/6)*e1+sin(pi*2/6)*e2
bm=e**dyad(e1, m/2.0)
bn=e**dyad(e1, n/2.0)
bv=e**dyad(e1, v/2.0)


bw=e**dyad(e1, w/2.0)

pm=e**dyad(e2, m/2.0)
pn=e**dyad(e2, n/2.0)
pv=e**dyad(e2, v/2.0)
pw=e**dyad(e2, w/2.0)

bM=e**dyad(e1, m*2.0)
bN=e**dyad(e1, n*2.0)
bV=e**dyad(e1, v*2.0)
bW=e**dyad(e1, w*2.0)

pM=e**dyad(e2, m*2.0)
pN=e**dyad(e2, n*2.0)
pV=e**dyad(e2, v*2.0)
pW=e**dyad(e2, w*2.0)

U1=dyad(u_ket, one)
U2=dyad(u_ket, e1)
V1=dyad(d_ket, one)
V2=dyad(d_ket, e1)

S1=dyad(s_ket, e1)
T2=dyad(t_ket, e2)
U3=dyad(u_ket, e3)

m_state=bm*P1
n_state=bn*P1
v_state=bv*P1
w_state=bw*P1

M_state=bM*P1
N_state=bN*P1
V_state=bV*P1
W_state=bW*P1

nm_state=bn*bm*P1
mn_state=bm*bn*P1

nM_state=bn*bM*P1
Mn_state=bM*bn*P1

Nm_state=bN*bm*P1
mN_state=bm*bN*P1

NM_state=bN*bM*P1
MN_state=bM*bN*P1

uu_ket = (oo+bn**2*G0)/2.0
nn_ket = (oo-bn**2*G0)/2.0

ww_ket = (oo+pn**2*G0)/2.0

dd_ket = G5*uu_ket
ee_ket = G6*ww_ket

rr_ket = (uu_ket+dd_ket)/sqrt(2)
ll_ket = (uu_ket-dd_ket)/sqrt(2)

ii_ket = (ww_ket+ee_ket)/sqrt(2)
oo_ket = (ww_ket-ee_ket)/sqrt(2)

uuu_ket = (oo+bv**2*G0)/2.0
nnn_ket = (oo-bv**2*G0)/2.0

www_ket = (oo+pv**2*G0)/2.0
mmm_ket = (oo-pv**2*G0)/2.0

ddd_ket = G5*uuu_ket
eee_ket = G6*www_ket

rrr_ket = (uuu_ket+ddd_ket)/sqrt(2)
lll_ket = (uuu_ket-ddd_ket)/sqrt(2)

iii_ket = (www_ket+eee_ket)/sqrt(2)
ooo_ket = (www_ket-eee_ket)/sqrt(2)

II = dyad(e12, one)

Alice = (oo + dyad(e3, one))/2.0
Bob   = (oo + dyad(one, e3))/2.0
John  = (oo + dyad(e3, e3))/2.0

# define some dyadic projectors

Pv = (oo+dyad(one,v))/2.
Qv = (oo-dyad(one,v))/2.

Pw = (oo+dyad(w, one))/2.
Qw = (oo-dyad(w, one))/2.

u = cos(pi/4)*e1+sin(pi/4)*e2

Pu = (oo+dyad(u, one))/2.
Qu = (oo-dyad(u, one))/2.

v_ket = (1+v)/2.0
v_bar = (1-v)/2.0
w_ket = (1+w)/2.0
w_bar = (1-w)/2.0

bw  = e**(w*0.25)
#Psi = e**(v*0.5)
#Phi = bw*Psi*bw

bm  = e**(m*0.25)
bn  = e**(n*0.25)

psi = bn*u_ket
phi = bm*n_ket

r12 = e**(e12*pi/4)
r13 = e**(e13*pi/4)
r23 = e**(e23*pi/4)

ra = e**(e13*pi/16)
rb = e**(e13*pi*9/16)
rc = e**(e13*pi*3/16)

a_ket = ra*u_ket
a_bar = ra*u_bar

b_ket = rb*u_ket
b_bar = rb*u_bar

c_ket = rc*u_ket
c_bar = rc*u_bar

# two-qubit ladder operators

lab = array([[0, 0, 0, 0],
             [0, 0, 1, 1],
             [1, 0, 0, 0],
             [1, 0, 0, 0]])/sqrt(2)

lac = array([[0, 0, 0, 0],
             [1, 0, 0, 0],
             [0, 1, 0, 1],
             [1, 0, 0, 0]])/sqrt(2)

lad = array([[0, 0, 0, 0],
             [1, 0, 0, 0],
             [1, 0, 0, 0],
             [0, 1, 1, 0]])/sqrt(2)

lbc = array([[0, 1, 0, 0],
             [0, 0, 0, 0],
             [1, 0, 0, 1],
             [0, 1, 0, 0]])/sqrt(2)

lbd = array([[0, 1, 0, 0],
             [0, 0, 0, 0],
             [0, 1, 0, 0],
             [1, 0, 1, 0]])/sqrt(2)

lcd = array([[0, 0, 1, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 0],
             [1, 1, 0, 0]])/sqrt(2)

LAB = tract(lab, S4)
LAC = tract(lac, S4)
LAD = tract(lad, S4)
LBC = tract(lbc, S4)
LBD = tract(lbd, S4)
LCD = tract(lcd, S4)

mab = array([[ 0, 0, 0, 0],
             [ 0, 0,-1, 1],
             [ 1, 0, 0, 0],
             [-1, 0, 0, 0]])/sqrt(2)

mac = array([[ 0, 0, 0, 0],
             [ 1, 0, 0, 0],
             [ 0,-1, 0, 1],
             [-1, 0, 0, 0]])/sqrt(2)

mad = array([[ 0, 0, 0, 0],
             [ 1, 0, 0, 0],
             [-1, 0, 0, 0],
             [ 0,-1, 1, 0]])/sqrt(2)

mbc = array([[ 0, 1, 0, 0],
             [ 0, 0, 0, 0],
             [-1, 0, 0, 1],
             [ 0,-1, 0, 0]])/sqrt(2)

mbd = array([[ 0, 1, 0, 0],
             [ 0, 0, 0, 0],
             [ 0,-1, 0, 0],
             [-1, 0, 1, 0]])/sqrt(2)

mcd = array([[ 0, 0, 1, 0],
             [ 0, 0,-1, 0],
             [ 0, 0, 0, 0],
             [-1, 1, 0, 0]])/sqrt(2)

MAB = tract(mab, S4)
MAC = tract(mac, S4)
MAD = tract(mad, S4)
MBC = tract(mbc, S4)
MBD = tract(mbd, S4)
MCD = tract(mcd, S4)

# tetrahedron face-rotation operators (120-degrees)

face1 = array([[0, 0, 1, 0],
               [1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 0, 1]])

face2 = array([[0, 1, 0, 0],
               [0, 0, 0, 1],
               [0, 0, 1, 0],
               [1, 0, 0, 0]])

face3 = array([[0, 0, 0, 1],
               [0, 1, 0, 0],
               [1, 0, 0, 0],
               [0, 0, 1, 0]])

face4 = array([[1, 0, 0, 0],
               [0, 0, 0, 1],
               [0, 1, 0, 0],
               [0, 0, 1, 0]])

FACE1 = tract(face1, S4)
FACE2 = tract(face2, S4)
FACE3 = tract(face3, S4)
FACE4 = tract(face4, S4)

XX = dyad(X, X)
YY = dyad(Y, Y)
ZZ = dyad(Z, Z)

XY = XX+YY
QQ = XX+YY+ZZ

places(12)
ecount(100)

vx = v.value[1]
vy = v.value[2]
vz = v.value[3]

wx = w.value[1]
wy = w.value[2]
wz = w.value[3]

# we might need a bigger float for this

v1 = e1                                 # photon direction
f1 = 2*pi/2.00e-11                      # photon wave number
w1 = e1*cos(pi*8/16)+e2*sin(pi*8/16)    # particle one boost direction
w2 = e1*cos(pi*0/16)+e2*sin(pi*0/16)    # particle two boost direction
b1 = 0.00000000001                      # particle one boost rapidity
b2 = np.arcsinh(f1)                     # particle two boost rapidity
# b2 = np.log(2*f1-1)                     # particle two boost rapidity

m1 = Me                                 # particle one is an electron
m2 = m_0                                # particle two is a massive photon
m3 = Me                                 # particle three is a electron
m4 = m_0                                # particle four is a massive photon
theta = pi*30/180

# A few global functions

def randState():
    """ generate normalized random state in the up/down basis """
    a,b = qga.compPair()
    return a*u_ket + b*d_ket

def randDirac(left=True):
    """ generate normalized random state in the up/down basis """
    a,b,c,d = qga.compQuad()
    if (left == True):
            return (dyad(a,one)*u_u+dyad(b,one)*u_d+dyad(c,one)*d_u+dyad(d,one)*d_d)
    return (dyad(one,a)*u_u+dyad(one,b)*u_d+dyad(one,c)*d_u+dyad(one,d)*d_d)

def randWeyl():
    """ generate normalized random state in the in/out basis """
    a,b,c,d = qga.compQuad()
    return (dyad(a,one)*l_l+dyad(b,one)*l_r+dyad(c,one)*r_l+dyad(d,one)*r_r)

def det(updn):
    """ find the determinant for a updn tuple """
    return updn[0]*updn[3]-updn[1]*updn[2]    

def Array2(mv):
    """ convert multivector to 2x2 complex array """
    Array = np.zeros((2,2),dtype=complex)
    for i in range(2):
        for j in range(2):
            Array[i,j] = (2*mv*S2.T[i,j]).comp()
    return Array

def Array4(mv):
    """ convert multivector to 4x4 complex array """
    Array = np.zeros((4,4),dtype=complex)
    for i in range(4):
        for j in range(4):
            Array[i,j] = ((4*mv*S4.T[i,j]).comp).comp()
    return Array

