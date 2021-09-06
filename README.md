# Optimising-SCOP
## Basic structure 

The file `Updated version` contains the orginal data about charge densities ` updated_charge_densities`, utility functions `utils` and graphical and numerical resulting cross sections`cross_section_figs`.
We model the Spherical Complex Optical Potential (V_st, V_ex, V_pol and V_abs) and formulate the solution to the total (elastic + inelastic) cross sections in `util.py`.
The results can be obtained by directly running `cross_section.py` just change the Atom categories for the five different atoms or molecules.
We convert the unit from Hartree to electronVolt using the conversion function `conversions.py` 
