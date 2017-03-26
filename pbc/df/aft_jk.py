#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
from pyscf import lib
from pyscf.pbc.df import aft_jk
from pyscf.pbc.df import df_jk
from spark4pyscf.tools import spark

def call_then_reduce(f):
    def ff(mydf, *args, **kwargs):
        def fsub(rank):
            mydf.rank = rank
            return f(mydf, *args, **kwargs)
        return spark.nodes.map(fsub).reduce(lambda x,y: x+y)
    return ff
def call_then_collect(f):
    def ff(mydf, *args, **kwargs):
        def fsub(rank):
            mydf.rank = rank
            return f(mydf, *args, **kwargs)
        return spark.nodes.map(fsub).collect()
    return ff

get_j_kpts = call_then_reduce(aft_jk.get_j_kpts)

def get_k_kpts(mydf, dm_kpts, hermi=1, kpts=numpy.zeros((1,3)), kpts_band=None,
               exxdiv=None):
    fk = call_then_reduce(aft_jk.get_k_kpts)
    vk_kpts = fk(mydf, dm_kpts, hermi, kpts, kpts_band, None)
    if exxdiv is not None:
        dms = df_jk._format_dms(lib.asarray(dm_kpts), kpts)
        df_jk._ewald_exxdiv_for_G0(mydf.cell, kpts, dms,
                                   vk_kpts.reshape(dms.shape), kpts_band)
    return vk_kpts


##################################################
#
# Single k-point
#
##################################################

def get_jk(mydf, dm, hermi=1, kpt=numpy.zeros(3),
           kpt_band=None, with_j=True, with_k=True, exxdiv=None):
    '''JK for given k-point'''
    fjk = call_then_collect(aft_jk.get_jk)
    vjk = fjk(mydf, dm, hermi, kpt, kpt_band, with_j, with_k, None)
    if with_j:
        vj = sum([v[0] for v in vjk])
    if with_k:
        vk = sum([v[1] for v in vjk])
        if exxdiv is not None:
            df_jk._ewald_exxdiv_for_G0(mydf.cell, kpt, dm, vk, kpt_band)
    return vj, vk


if __name__ == '__main__':
    from pyscf.pbc import gto as pgto
    from pyscf.pbc import scf as pscf
    from spark4pyscf.pbc.df import aft

    L = 5.
    n = 5
    cell = pgto.Cell()
    cell.a = numpy.diag([L,L,L])
    cell.gs = numpy.array([n,n,n])

    cell.atom = '''He    3.    2.       3.
                   He    1.    1.       1.'''
    #cell.basis = {'He': [[0, (1.0, 1.0)]]}
    #cell.basis = '631g'
    #cell.basis = {'He': [[0, (2.4, 1)], [1, (1.1, 1)]]}
    cell.basis = 'ccpvdz'
    cell.verbose = 0
    cell.build(0,0)
    cell.verbose = 5

    df = aft.AFTDF(cell)
    df.gs = (15,)*3
    dm = pscf.RHF(cell).get_init_guess()
    vj, vk = df.get_jk(cell, dm)
    print(numpy.einsum('ij,ji->', df.get_nuc(cell), dm), 'ref=-10.384051732669329')
    df.analytic_ft = True
    #print(numpy.einsum('ij,ji->', vj, dm), 'ref=5.3766911667862516')
    #print(numpy.einsum('ij,ji->', vk, dm), 'ref=8.2255177602309022')
    print(numpy.einsum('ij,ji->', df.get_nuc(cell), dm), 'ref=-10.447018516011319')


