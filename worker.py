#!/usr/bin/env python
from mpi4py import MPI
import numpy
import sys

comm = MPI.Comm.Get_parent()
size = comm.Get_size()
rank = comm.Get_rank()
name = MPI.Get_processor_name()

#sys.stdout.write("I am process %d of %d on %s.\n" % (rank, size, name))

# total number to simulate
N = numpy.array(0, dtype='i')
comm.Bcast([N, MPI.INT], root=0)
Var = numpy.array(0, dtype='d')
comm.Bcast([Var, MPI.DOUBLE], root=0)

number = N / size

numpy.random.seed(3081989)
draws = numpy.random.normal(0, scale=numpy.sqrt(Var), size=number)
draws_sum = numpy.sum(draws)
comm.Reduce([draws_sum, MPI.DOUBLE], None,
            op=MPI.SUM, root=0)

comm.Disconnect()
