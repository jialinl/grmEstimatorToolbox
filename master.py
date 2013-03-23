#!/usr/bin/env python
from mpi4py import MPI
import numpy
import sys

# For U0 draws
comm = MPI.COMM_SELF.Spawn(sys.executable,
                           args=['worker.py'],
                           maxprocs=4)

N = numpy.array(10000, 'i')
comm.Bcast([N, MPI.INT], root=MPI.ROOT)
Var = numpy.array(0.5)
comm.Bcast([Var, MPI.DOUBLE], root=MPI.ROOT)

draws_sum = numpy.array(0.0, 'd')
comm.Reduce(None, [draws_sum, MPI.DOUBLE],
            op=MPI.SUM, root=MPI.ROOT)
print(draws_sum/N)

# For U1 draws
comm = MPI.COMM_SELF.Spawn(sys.executable,
                           args=['worker.py'],
                           maxprocs=4)

N = numpy.array(10000, 'i')
comm.Bcast([N, MPI.INT], root=MPI.ROOT)
Var = numpy.array(0.9)
comm.Bcast([Var, MPI.DOUBLE], root=MPI.ROOT)

draws_sum = numpy.array(0.0, 'd')
comm.Reduce(None, [draws_sum, MPI.DOUBLE],
            op=MPI.SUM, root=MPI.ROOT)
print(draws_sum/N)

comm.Disconnect()
