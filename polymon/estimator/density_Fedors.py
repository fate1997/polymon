import re
import rdkit.Chem.Descriptors as rdcd
from rdkit import Chem
from polymon.estimator.base import BaseEstimator
from polymon.model.register import register_init_params

fedors_allowed_atoms = frozenset(['C', 'H', 'O', 'N', 'F', 'Cl', 'Br', 'I', 'S'])
fedors_contributions = {'C': 34.426, 'H': 9.172, 'O': 20.291,
                       'O_alcohol': 18, 'N': 48.855, 'N_amine': 47.422,
                       'F': 22.242, 'Cl': 52.801, 'Br': 71.774, 'I': 96.402,
                       'S': 50.866, '3_ring': -15.824, '4_ring': -17.247,
                       '5_ring': -39.126, '6_ring': -39.508,
                       'double_bond': 5.028, 'triple_bond': 0.7973,
                       'ring_ring_bonds': 35.524}
alcohol_smarts = '[#6][OX2H]'
amine_smarts = '[NX3+0,NX4+;!$([N]~[!#6]);!$([N]*~[#7,#8,#15,#16])]'



@register_init_params
class DensityFedorsEstimator(BaseEstimator):
    def __init__(self):
        pass

    def estimated_y(self, smiles: str) -> float:
        return get_fedors_density(Chem.MolFromSmiles(smiles))


def get_fedors_density(mol):
    r'''Estimate the critical volume of a molecule
    using the Fedors [1]_ method, which is a basic
    group contribution method that also uses certain
    bond count features and the number of different
    types of rings.

    Parameters
    ----------
    mol : str or rdkit.Chem.rdchem.Mol, optional
        Smiles string representing a chemical or a rdkit molecule, [-]

    Returns
    -------
    Vc : float
        Estimated critical volume, [m^3/mol]
    status : str
        A string holding an explanation of why the molecule failed to be
        fragmented, if it fails; 'OK' if it suceeds, [-]
    unmatched_atoms : bool
        Whether or not all atoms in the molecule were matched successfully;
        if this is True, the results should not be trusted, [-]
    unrecognized_bond : bool
        Whether or not all bonds in the molecule were matched successfully;
        if this is True, the results should not be trusted, [-]
    unrecognized_ring_size : bool
        Whether or not all rings in the molecule were matched successfully;
        if this is True, the results should not be trusted, [-]

    Notes
    -----
    Raises an exception if rdkit is not installed, or `smi` or `rdkitmol` is
    not defined.

    Examples
    --------
    Example for sec-butanol in [2]_:

    >>> Vc, status, _, _, _ = Fedors('CCC(C)O') # doctest:+SKIP
    >>> Vc, status # doctest:+SKIP
    (0.000274024, 'OK')

    References
    ----------
    .. [1] Fedors, R. F. "A Method to Estimate Critical Volumes." AIChE
       Journal 25, no. 1 (1979): 202-202. https://doi.org/10.1002/aic.690250129.
    .. [2] Green, Don, and Robert Perry. Perry's Chemical Engineers' Handbook,
       Eighth Edition. McGraw-Hill Professional, 2007.
    '''
    from rdkit import Chem
    if type(mol) is Chem.rdchem.Mol:
        rdkitmol = Chem.Mol(mol)
        no_H_mol = Chem.RemoveHs(rdkitmol)
    else:
        rdkitmol = Chem.MolFromSmiles(mol)
        no_H_mol = Chem.RemoveHs(rdkitmol)

    # Canont modify the molecule we are given
    rdkitmol = Chem.AddHs(rdkitmol)

    ri = no_H_mol.GetRingInfo()
    atom_rings = ri.AtomRings()

    UNRECOGNIZED_RING_SIZE = False
    three_rings = four_rings = five_rings = six_rings = 0
    for ring in atom_rings:
        ring_size = len(ring)
        if ring_size == 3:
            three_rings += 1
        elif ring_size == 4:
            four_rings += 1
        elif ring_size == 5:
            five_rings += 1
        elif ring_size == 6:
            six_rings += 1
        else:
            UNRECOGNIZED_RING_SIZE = True

    rings_attatched_to_rings = count_rings_attatched_to_rings(no_H_mol, atom_rings=atom_rings)

    UNRECOGNIZED_BOND_TYPE = False
    DOUBLE_BOND = Chem.rdchem.BondType.DOUBLE
    TRIPLE_BOND = Chem.rdchem.BondType.TRIPLE
    SINGLE_BOND = Chem.rdchem.BondType.SINGLE
    AROMATIC_BOND = Chem.rdchem.BondType.AROMATIC

    double_bond_count = triple_bond_count = 0
    # GetBonds is very slow; we can make it a little faster by iterating
    # over a copy without hydrogens
    for bond in no_H_mol.GetBonds():
        bond_type = bond.GetBondType()
        if bond_type is DOUBLE_BOND:
            double_bond_count += 1
        elif bond_type is TRIPLE_BOND:
            triple_bond_count += 1
        elif bond_type is SINGLE_BOND or bond_type is AROMATIC_BOND:
            pass
        else:
            UNRECOGNIZED_BOND_TYPE = True

    alcohol_matches = rdkitmol.GetSubstructMatches(Chem.MolFromSmarts(alcohol_smarts))
    amine_matches = rdkitmol.GetSubstructMatches(Chem.MolFromSmarts(amine_smarts))

    # This was the fastest way to get the atom counts
    atoms = simple_formula_parser(Chem.rdMolDescriptors.CalcMolFormula(rdkitmol))
    # For the atoms with functional groups, they always have to be there
    if 'N' not in atoms:
        atoms['N'] = 0
    if 'O' not in atoms:
        atoms['O'] = 0
    found_atoms = set(atoms.keys())
    UNKNOWN_ATOMS = bool(not found_atoms.issubset(fedors_allowed_atoms))

    atoms['O_alcohol'] = len(alcohol_matches)
    atoms['O'] -= len(alcohol_matches)
    atoms['N_amine'] = len(amine_matches)
    atoms['N'] -= len(amine_matches)
    atoms['3_ring'] = three_rings
    atoms['4_ring'] = four_rings
    atoms['5_ring'] = five_rings
    atoms['6_ring'] = six_rings
    atoms['double_bond'] = double_bond_count
    atoms['triple_bond'] = triple_bond_count
    atoms['ring_ring_bonds'] = rings_attatched_to_rings

    Vc = 26.6
    for k, v in fedors_contributions.items():
        try:
            Vc += atoms[k]*v
        except KeyError:
            pass

    # Vc *= 1e-6

    molar_mass = rdcd.ExactMolWt(rdkitmol)
    return molar_mass / Vc * 3.1


def count_rings_attatched_to_rings(mol, allow_neighbors=True, atom_rings=None):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, count the number
    of rings in the molecule that are attatched to another ring.
    if `allow_neighbors` is True, any bond to another atom that is part of a
    ring is allowed; if it is False, the rings have to share a wall.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]
    allow_neighbors : bool
        Whether or not to count neighboring rings or just ones sharing a wall, [-]
    atom_rings : rdkit.Chem.rdchem.RingInfo, optional
        Internal parameter, used for performance only

    Returns
    -------
    rings_attatched_to_rings : bool
        The number of rings bonded to other rings, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> count_rings_attatched_to_rings(MolFromSmiles('C12C3C4C1C5C2C3C45')) # doctest:+SKIP
    6
    '''
    if atom_rings is None:
        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()
    ring_count = len(atom_rings)
    ring_ids = [frozenset(t) for t in atom_rings]
    rings_attatched_to_rings = 0
    other_ring_atoms = set()
    for i in range(ring_count):
        other_ring_atoms.clear()
        attatched_to_ring = False
        for j in range(ring_count):
            if i != j:
                other_ring_atoms.update(atom_rings[j])


        for atom in atom_rings[i]:
            if attatched_to_ring:
                break

            if atom in other_ring_atoms:
                attatched_to_ring = True
                break
            if allow_neighbors:
                atom_obj = mol.GetAtomWithIdx(atom)
                neighbors = atom_obj.GetNeighbors()
                for n in neighbors:
                    if n.GetIdx() in other_ring_atoms:
                        attatched_to_ring = True
                        break
                if attatched_to_ring:
                    break
        if attatched_to_ring:
            rings_attatched_to_rings+= 1
    return rings_attatched_to_rings


_simple_formula_parser_re_str = r'([A-Z][a-z]{0,2})([\d\.\d]+)?'
_simple_formula_parser_re = None
def simple_formula_parser(formula):
    r'''Basic formula parser, primarily for obtaining element counts from
    formulas as formated in PubChem. Handles formulas with integer or decimal
    counts (with period separator), but no brackets, no hydrates, no charges,
    no isotopes, and no group multipliers.

    Strips charges from the end of a formula first. Accepts repeated chemical
    units. Performs no sanity checking that elements are actually elements.
    As it uses regular expressions for matching, errors are mostly just ignored.

    Parameters
    ----------
    formula : str
        Formula string, very simply formats only.

    Returns
    -------
    atoms : dict
        dictionary of counts of individual atoms, indexed by symbol with
        proper capitalization, [-]

    Notes
    -----
    Inspiration taken from the thermopyl project, at
    https://github.com/choderalab/thermopyl.

    Examples
    --------
    >>> simple_formula_parser('CO2')
    {'C': 1, 'O': 2}
    '''
    global _simple_formula_parser_re
    if not _simple_formula_parser_re:
        _simple_formula_parser_re = re.compile(_simple_formula_parser_re_str)
    formula = formula.split('+')[0].split('-')[0]
    counts = {}
    for element, count in _simple_formula_parser_re.findall(formula):
        if count.isdigit():
            count = int(count)
        elif count:
            count = float(count)
        else:
            count = 1
        if element in counts:
            counts[element] += count
        else:
            counts[element] = count
    return counts