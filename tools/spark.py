#!/usr/bin/env python

#import sys
#import time
#import threading
#import traceback
#import numpy
#from mpi4py import MPI
#from . import mpi_pool
#from .mpi_pool import MPIPool

from pyscf.gto import mole
from pyscf.pbc.gto import cell

from pyspark import SparkContext, SparkConf
#conf = SparkConf().setMaster("local[4]").setAppName('pyscf')
conf = SparkConf().setAppName('pyscf')
#conf.set("spark.ui.port", "4042")
#conf.set("spark.blockManager.port", "29900")
#conf.set("spark.broadcast.port", "29901")
#conf.set("spark.driver.port", "29902")
#conf.set("spark.executor.port", "29903")
#conf.set("spark.fileserver.port", "29904")
#conf.set("spark.replClassServer.port", "29905")
#sc = SparkContext()
sc = SparkContext.getOrCreate(conf=conf)
#sc.setLogLevel('ERROR')
size = sc.defaultParallelism
nodes = sc.parallelize(range(size))
# Some objects hold temporary files on distributed nodes for intermediate data.
# Keep track of the objects in a global registry.  This is to avoid the objects
# (as well as the temporary files) on workers being destroyed after executing.
_registry_hoods = sc.broadcast(dict())

import atexit
atexit.register(sc.stop)


def del_registry(key):
    def clear(rank):
        _registry = _registry_hoods.value
        _registry.pop(key)
    nodes.map(clear).collect()

def _init_and_register(cls):
    old_init = cls.__init__
    def init(obj, *args, **kwargs):
        key = id(obj)
        def _init_on_workers(rank):
            old_init(obj, *args, **kwargs)
            _registry = _registry_hoods.value
            _registry[key] = obj
        nodes.map(_init_on_workers).collect()
    return init
def _with_enter(obj):
    return obj
def _with_exit(obj):
    del_registry(id(obj))

def register_class(cls):
    cls.__init__ = _init_and_register(cls)
    cls.__enter__ = _with_enter
    cls.__exit__ = _with_exit
    cls.close = _with_exit
    return cls

