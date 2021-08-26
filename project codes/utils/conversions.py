"""

Module for constants that convert one unit to another.

"""

# Energy conversions:

HARTREE_TO_EV = 27.21138602
EV_TO_HARTREE = 1.0 / HARTREE_TO_EV

# Distance conversions:

ANGSTROM_TO_BOHR = 1.8897259886
BOHR_TO_ANGSTROM = 1.0 / ANGSTROM_TO_BOHR

ANGSTROM_SQ_TO_BOHR_SQ = ANGSTROM_TO_BOHR * ANGSTROM_TO_BOHR
BOHR_SQ_TO_ANGSTROM_SQ = 1.0 / ANGSTROM_SQ_TO_BOHR_SQ

ANGSTROM_TO_CM = 1.0E8
