import math

import numpy as np
import rdkit.Chem.Descriptors as rdcd
import rdkit.Chem.rdMolDescriptors as rdcmd
from rdkit import Chem
from rdkit.Chem import Fragments

from polymon.estimator.base import BaseEstimator
from polymon.model.register import register_init_params


@register_init_params
class DensityIBMEstimator(BaseEstimator):
    """Density estimator using the IBM method. This is a group contribution
    method that assigns a contribution to each group of atoms in the molecule.
    The original implementation is from https://github.com/IBM/polymer_property_prediction.
    """
    def __init__(self):
        pass

    def estimated_y(self, smiles: str) -> float:
        return estimate_density(smiles)


def BetaFunction(mol, lin):
    #
    #  calculating beta -> beta_ij = delta_i*delta_j
    #

    nbond = mol.GetNumBonds()
    betaf = np.zeros(nbond + 1, dtype=np.float64)
    bondf = BondFunction(mol, lin)
    deltaf = DeltaFunction(mol, lin)
    for i in range(0, nbond + 1):
        ii = bondf[i][0]
        jj = bondf[i][1]
        betaf[i] = deltaf[ii] * deltaf[jj]

    return betaf


def BetavFunction(mol, lin):
    #
    #  calculating betav -> betav_ij = deltav_i*deltav_j
    #

    nbond = mol.GetNumBonds()
    betavf = np.zeros(nbond + 1, dtype=np.float64)
    bondf = BondFunction(mol, lin)
    deltavf = DeltavFunction(mol, lin)
    for i in range(0, nbond + 1):
        ii = bondf[i][0]
        jj = bondf[i][1]
        betavf[i] = deltavf[ii] * deltavf[jj]

    return betavf

def BondFunction(mol, lin):
    #
    #   This function calculates the bonds between the atoms for
    #   molecule mol, obtained from SMILES
    #

    nbond = mol.GetNumBonds()
    bondf = np.zeros([nbond + 1, 2], dtype=np.int64)
    for i in range(0, nbond):
        bondf[i][0] = mol.GetBondWithIdx(i).GetBeginAtomIdx()
        bondf[i][1] = mol.GetBondWithIdx(i).GetEndAtomIdx()
    # defining the periodic bond between monomers
    bondf[nbond][0] = int(lin[2])
    bondf[nbond][1] = int(lin[3])

    return bondf

def DeltaFunction(mol, lin):
    #
    #   This function calculates the number of non-hydrogen atoms to
    #   which a given non-hydrogen atom is bonded
    #

    natom = mol.GetNumAtoms()
    nbond = mol.GetNumBonds()
    deltaf = np.zeros(natom, dtype=np.float64)
    bondf = BondFunction(mol, lin)
    # calculating delta -> number of non-hydrogen atoms bonded to the non-hydrogen atom
    for i in range(0, natom):
        nn = 0
        for j in range(0, nbond + 1):
            if bondf[j][0] == i:
                nn = nn + 1
            if bondf[j][1] == i:
                nn = nn + 1
        deltaf[i] = nn

    return deltaf

def DeltavFunction(mol, lin):
    #
    #   This function calculates deltav, the atom connectivity index
    #   associated with the electronic environment
    #

    natom = mol.GetNumAtoms()
    nbond = mol.GetNumBonds()
    deltavf = np.zeros(natom, dtype=np.float64)
    nh = np.zeros(natom, dtype=np.int64)
    bondf = BondFunction(mol, lin)
    # calculating the number of hydrogens
    for atom in mol.GetAtoms():
        kk = atom.GetIdx()
        nh[kk] = atom.GetImplicitValence()
        z = atom.GetAtomicNum()
        if z == 14:
            nh[kk] = nh[kk] + 1
        if (kk == bondf[nbond][0] or kk == bondf[nbond][1]):
            nh[kk] = nh[kk] - 1

    # calculating deltav
    valence_electron = {
        6: 4,
        7: 5,
        8: 6,
        9: 7,
        14: 4,
        15: 5,
        16: 6,
        17: 7,
        35: 7,
        53: 7,
    }  # dictionary relating atomic number : valence electron
    deltaf = DeltaFunction(mol, lin)
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        z = atom.GetAtomicNum()
        if z == 14:
            z = 6  # replacing Si -> C according to Table 2.3 Josef Bicerano's book
        if z not in valence_electron:
            continue
        zv = valence_electron[z]
        deltavf[i] = (zv - nh[i]) / (z - zv - 1)  # deltav definition
        atom_charge = atom.GetFormalCharge()
        if (z == 7 and atom_charge == 1):  # Nitrogen with +1 charge
            deltavf[i] = deltavf[i] + 1
        if (z == 8 and atom_charge == -1):  # Oxygen with -1 charge
            deltavf[i] = deltavf[i] - 1
        if (z == 16 and deltaf[i] == 3):  # oxidation state +4 of Sulfur
            deltavf[i] = deltavf[i] + 1
        if (z == 16 and deltaf[i] == 4):  # oxidation state +6 of Sulfur
            deltavf[i] = deltavf[i] + 2

    return deltavf

def ConnectivityIndex(mol, lin):
    #
    #   This function calculates the zero and first order connectivity
    #   indices of a polymer monomer. This is the first step to
    #   calculate polymer properties.
    #
    #   INPUT: m is a rdkit.Chem molecule class with informations based
    #          on SMILES code.
    #
    #   OUTPUT: x0, x0v, x1, x1v
    #
    #   Ronaldo Giro  6, February 2019

    v_connec_indexf = np.zeros(4, dtype=np.float64)
    natom = mol.GetNumAtoms()
    nbond = mol.GetNumBonds()
    # calculating x0 and x0v -> zero order connectivity indices
    x0 = 0
    x0v = 0
    deltaf = DeltaFunction(mol, lin)
    deltavf = DeltavFunction(mol, lin)
    for i in range(0, natom):
        if deltaf[i] == 0 or deltavf[i] == 0:
            continue
        x0 = x0 + 1 / math.sqrt(deltaf[i])
        x0v = x0v + 1 / math.sqrt(deltavf[i])

    # calculating x1 and x1v -> first order connectivity indices
    x1 = 0
    x1v = 0
    betaf = BetaFunction(mol, lin)
    betavf = BetavFunction(mol, lin)
    for i in range(0, nbond + 1):
        if betaf[i] == 0 or betavf[i] == 0:
            continue
        x1 = x1 + 1 / math.sqrt(betaf[i])
        x1v = x1v + 1 / math.sqrt(betavf[i])

    v_connec_indexf[0] = x0
    v_connec_indexf[1] = x0v
    v_connec_indexf[2] = x1
    v_connec_indexf[3] = x1v
    return v_connec_indexf


def molextFunction(mol, lin):
    # This function add the head and tail atoms of a molecule to a new
    # molecule in order to take into account the polymer periodicity
    # effect. So, molpf is mol with two added atoms: the head and the
    # tail atoms from mol
    #
    # smiles_string is the OPSIN: Open Parser for Systematic IUPAC
    # nomenclature SMILES with the periodic markers: [*:1] and [*:2]
    # for head and tail ghost atoms
    #
    # defining the molecule with periodic bonds to calculate fragments
    bi = int(lin[2])
    bf = int(lin[3])
    atomi = mol.GetAtomWithIdx(bi)
    atomf = mol.GetAtomWithIdx(bf)
    sti = atomi.GetSymbol()
    stf = atomf.GetSymbol()
    if sti == 'Si':
        sti = '[Si]'
    if stf == 'Si':
        stf = '[Si]'

    # replacing the head and tail ghost atoms [*:1] and [*:2] with the sti and stf atoms
    sttmp1 = lin[1]
    sttmp2 = sttmp1.replace("*", stf)
    st = sttmp2.replace("*", sti)
    molpf = Chem.MolFromSmiles(st)
    return molpf

def RingsFunction(mol):
    #
    # This function return a tuple (ringstuple) with atom indexes belonging to
    # all the rings of a molecule mol. To obtain the number of rings in a molecule
    # just use len(ringstuple)
    #
    # Finding rings at molecule m
    ri = mol.GetRingInfo()
    ringstuple = ri.AtomRings()
    return ringstuple


def FusedRingsFunction(mol):
    #
    # This function return a tuple (fusedtupleindex) with the ring indexes belonging to
    # all the fused rings of a molecule mol. To obtain the number of fused rings in a molecule
    # just use len(fusedtupleindex)
    #
    # Finding fused rings
    ringstuple = RingsFunction(mol)
    nrings = len(ringstuple)
    fusedtupleindex = np.ones([1000], dtype=np.int64)
    fusedtupleindex = -1 * fusedtupleindex
    fused = np.zeros([1000, 2], dtype=np.int64)
    nn = 0
    for i in range(0, nrings):
        for k in range(0, nrings):
            if (k > i):
                for j in range(0, len(ringstuple[i])):
                    for n in range(0, len(ringstuple[k])):
                        if (ringstuple[i][j] == ringstuple[k][n]):
                            fusedtupleindex[nn] = i
                            fused[nn][0] = i
                            fused[nn][1] = i
                            nn = nn + 1
                            fusedtupleindex[nn] = k
                            fused[nn][0] = k
                            fused[nn][1] = i
                            nn = nn + 1

    fusedtupleindex = np.sort(fusedtupleindex)
    fusedtupleindex = np.unique(fusedtupleindex)
    fusedtupleindex = np.flip(fusedtupleindex)
    fusedtupleindex = fusedtupleindex[fusedtupleindex > -1]

    return fusedtupleindex


def NmvFunction(mol, lin):
    # ---------------------------------------------------------------------
    #
    # 	    Nmv is a group contribution correction term for Molar Volume
    #
    # ---------------------------------------------------------------------
    deltaf = DeltaFunction(mol, lin)
    deltavf = DeltavFunction(mol, lin)
    Nsi = 0
    Ns = 0
    Nsulfone = 0
    Ncl = 0
    Nbr = 0
    Nester = 0
    Nether = 0
    Ncarbonate = 0
    Ncdouble = 0
    Ncyc = 0
    Nfused = 0
    # calculating the number of Si (Silicon) atoms
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        if (z == 14):  # atomic number of Si (Silicon)
            Nsi = Nsi + 1

    # calculating the number of -S- (Sulfor) atoms
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        i = atom.GetIdx()
        if (z == 16):
            if (deltaf[i] == 1 and deltavf[i] == 5 / 9):
                Ns = Ns + 1
            if (deltaf[i] == 2 and deltavf[i] == 2 / 3):
                Ns = Ns + 1

    # calculating the number of Sulfone groups
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        i = atom.GetIdx()
        if (z == 16):
            if (deltaf[i] == 4 and deltavf[i] == 8 / 3):
                Nsulfone = Nsulfone + 1

    # calculating the number of Cl (Chlorine) atoms
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        if (z == 17):  # atomic number of Cl (Chlorine)
            Ncl = Ncl + 1

    # calculating the number of Br (Bromine) atoms
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        if (z == 35):  # atomic number of Br (Bromine)
            Nbr = Nbr + 1

    # defining the molecule with periodic bonds to calculate fragments
    molp = mol

    # calculating the number of backbone esters (R-C(=O)OR) groups
    Nester = Fragments.fr_ester(molp)

    # calculating the number of ether (R-O-R) groups
    Nether = Fragments.fr_ether(molp)

    # calculating the number of carbonate (-OCOO-) groups
    Ncarbonate = Fragments.fr_COO2(molp)

    nbond = mol.GetNumBonds()
    for i in range(0, nbond):
        bi = mol.GetBondWithIdx(i).GetBeginAtomIdx()
        atomi = mol.GetAtomWithIdx(bi)
        zi = atomi.GetAtomicNum()
        bf = mol.GetBondWithIdx(i).GetEndAtomIdx()
        atomf = mol.GetAtomWithIdx(bf)
        zf = atomf.GetAtomicNum()
        nbondtype = mol.GetBondWithIdx(i).GetBondTypeAsDouble()
        if (zi == 6 and zf == 6):
            if nbondtype == 2:
                if not atomi.IsInRing():
                    if not atomf.IsInRing():
                        Ncdouble = Ncdouble + 1

    # calculating the number of saturated rings Ncyc
    Ncyc = rdcmd.CalcNumSaturatedRings(mol)

    # calculating the number of fused ring Nfuse
    fusedtupleindex = FusedRingsFunction(mol)
    Nfused = len(fusedtupleindex)

    # calculating Nmv
    if Nfused >= 2:
        Nmvf = 24 * Nsi - 18 * Ns - 5 * Nsulfone - 7 * Ncl - 16 * Nbr
        + 2 * Nester + 3 * Nether + 5 * Ncarbonate + 5 * Ncdouble - 11 * Ncyc - 7 * (Nfused - 1)
    else:
        Nmvf = 24 * Nsi - 18 * Ns - 5 * Nsulfone - 7 * Ncl - 16 * Nbr + 2 * \
            Nester + 3 * Nether + 5 * Ncarbonate + 5 * Ncdouble - 11 * Ncyc

    return Nmvf


def estimate_density(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    attachments = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            attachments.append(atom.GetNeighbors()[0].GetIdx())
    if len(attachments) != 2:
        raise ValueError(f'Invalid SMILES: {smiles}')
    attachments.sort()
    lin = ['name', smiles, attachments[0], attachments[1]]
    v_connec_index = ConnectivityIndex(mol, lin)
    x0 = v_connec_index[0]
    x0v = v_connec_index[1]
    x1 = v_connec_index[2]
    x1v = v_connec_index[3]
    Nmv = NmvFunction(mol, lin)
    molar_volume = 3.642770 * x0 + 9.798697 * x0v - \
        8.542819 * x1 + 21.693912 * x1v + 0.978655 * Nmv + 20
    molar_mass = rdcd.ExactMolWt(mol)
    density_monomer = molar_mass / molar_volume
    return density_monomer