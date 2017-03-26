#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''Density expansion on plane waves'''

import time
import ctypes
import copy
import numpy

from pyscf import lib
from pyscf.pbc.gto import pseudo
from pyscf.pbc.df import aft

from spark4pyscf.tools import spark
from spark4pyscf.pbc.df import aft_jk


get_nuc = aft_jk.call_then_reduce(aft.get_nuc)

def get_pp(mydf, kpts=None):
    if kpts is None:
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))

    fpp = aft_jk.call_then_collect(aft.get_pp_loc_part1)
    vpp = sum([lib.asarray(v) for v in fpp(mydf, kpts_lst)])

    vloc2 = pseudo.pp_int.get_pp_loc_part2(mydf.cell, kpts_lst)
    vppnl = pseudo.pp_int.get_pp_nl(mydf.cell, kpts_lst)
    for k in range(len(kpts_lst)):
        vpp[k] += numpy.asarray(vppnl[k] + vloc2[k], dtype=vpp.dtype)

    if kpts is None or numpy.shape(kpts) == (3,):
        vpp = vpp[0]
    return vpp


class AFTDF(aft.AFTDF):

    def prange(self, start, stop, step=None):
        # affect pw_loop and ft_loop function
        size = stop - start
        segsize = (size+spark.size-1) // spark.size
        if step is None:
            step = segsize
        else:
            step = min(step, segsize)
        start = min(size, start + self.rank * segsize)
        stop = min(size, start + segsize)
        return lib.prange(start, stop, step)

    def _int_nuc_vloc(self, nuccell, kpts, intor='cint3c2e_sph'):
        v = aft._int_nuc_vloc(self, nuccell, kpts, intor)
        return v * (1./spark.size)

    get_nuc = get_nuc
    get_pp = get_pp

    def get_jk(self, dm, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, exxdiv='ewald'):
        '''Gamma-point calculation by default'''
        if kpts is None:
            if numpy.all(self.kpts == 0):
                kpts = numpy.zeros(3)
            else:
                kpts = self.kpts
        else:
            kpts = numpy.asarray(kpts)

        if kpts.shape == (3,):
            return aft_jk.get_jk(self, dm, hermi, kpts, kpts_band, with_j,
                                 with_k, exxdiv)

        vj = vk = None
        if with_k:
            vk = aft_jk.get_k_kpts(self, dm, hermi, kpts, kpts_band, exxdiv)
        if with_j:
            vj = aft_jk.get_j_kpts(self, dm, hermi, kpts, kpts_band)
        return vj, vk


if __name__ == '__main__':
    from pyscf.pbc import gto as pgto
    from spark4pyscf.pbc import df
    cell = pgto.Cell()
    cell.atom = 'He 1. .5 .5; C .1 1.3 2.1'
    cell.basis = {'He': [(0, (2.5, 1)), (0, (1., 1))],
                  'C' :'gth-szv',}
    cell.pseudo = {'C':'gth-pade'}
    cell.a = numpy.eye(3) * 2.5
    cell.gs = [5] * 3
    cell.build()
    numpy.random.seed(19)
    kpts = numpy.random.random((5,3))

    def finger(a):
        a = numpy.asarray(a)
        return numpy.dot(numpy.cos(numpy.arange(a.size)), a.ravel())
    mydf = df.AFTDF(cell)
    v = mydf.get_nuc()
    print(finger(v) - -5.8429124013026268)

    v = mydf.get_pp(kpts)
    print(finger(v) - (-7.170338584113936+0.82293729396018711j))

    cell = pgto.M(atom='He 0 0 0; He 0 0 1', a=numpy.eye(3)*4, gs=[5]*3)
    mydf = df.AFTDF(cell)
    nao = cell.nao_nr()
    dm = numpy.ones((nao,nao))
    vj, vk = mydf.get_jk(dm)
    print(finger(vj)- 0.05269466981334303)
    print(finger(vk)- 0.15372821172554785)
    print(numpy.einsum('ij,ji->', vj, dm) - 2.8790714593955942)
    print(numpy.einsum('ij,ji->', vk, dm) - 5.4670155332128649)

    dm_kpts = [dm]*5
    vj, vk = mydf.get_jk(dm_kpts, kpts=kpts)
    print(finger(vj) - (0.44421154967551829+0.00015278759444228032j))
    print(finger(vk) - (0.8085816873329319-0.037783739844583691j)   )

