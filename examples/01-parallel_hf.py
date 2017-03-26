#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
OMP_NUM_THREADS=4 MASTER='local[4]' python 01-parallel_hf.py
'''

import numpy
import pyscf.pbc.gto as pbcgto
import pyscf.pbc.scf as pscf
from mpi4pyscf.pbc import df as mpidf

cell = pbcgto.Cell()
cell.atom = [['C', ([ 0.,  0.,  0.])],
             ['C', ([ 0.8917,  0.8917,  0.8917])],
             ['C', ([ 1.7834,  1.7834,  0.    ])],
             ['C', ([ 2.6751,  2.6751,  0.8917])],
             ['C', ([ 1.7834,  0.    ,  1.7834])],
             ['C', ([ 2.6751,  0.8917,  2.6751])],
             ['C', ([ 0.    ,  1.7834,  1.7834])],
             ['C', ([ 0.8917,  2.6751,  2.6751])]
            ]
cell.h = numpy.eye(3) * 3.5668
cell.basis = 'sto3g'
cell.gs = [5] * 3
cell.verbose = 4
cell.build()

mydf = mpidf.AFTDF(cell)
mf = pscf.RHF(cell)
mf.exxdiv = 'ewald'
mf.with_df = mydf
mf.kernel()
