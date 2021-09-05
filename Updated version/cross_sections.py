#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 21:37:16 2021

@author: ruizhebu
"""
import numpy as np
from utils import util
# import modified_utility as util
import tqdm
from matplotlib import pyplot as plt

atom = util.Neon()

plt.figure(figsize=(20,8))
plt.title('Cross Sections for Neon')
# energies = np.linspace(100, 700, num=7)
energies = np.array([100, 200, 300, 400, 500, 600, 700])
cross_sections = []
elastic_cross_section = []
inelastic_cross_section = []
with open('new_new_figs/neon.txt', 'w') as f:
    for energy in tqdm.tqdm(energies):
        # if abs(energy - 400) < 1e-4:
        #     continue
        cs = atom.loop_over_partial_waves(energy)
        print(energy, cs)
        f.write(str(energy) + ' ' + str(cs) + '\n')
        cross_sections.append(cs[0])
        elastic_cross_section.append(cs[1])
        inelastic_cross_section.append(cs[2])
# cross_sections = [atom.loop_over_partial_waves(energy) for energy in energies]
# plt.plot(energies, cross_sections)
plt.plot(energies, cross_sections, label='Total Cross Section')
plt.plot(energies, elastic_cross_section, label='Elastic Cross Section')
plt.plot(energies, inelastic_cross_section, label='Inelastic Cross Section')
plt.xlabel('$eV$')
plt.ylabel('$angstroms^2$')
plt.legend()
plt.savefig('new_new_figs/neon.png')
plt.show()
