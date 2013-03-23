from mpi4py import MPI
import numpy as np
import sys

def matvec(comm, A, x):
    m = A.shape[0] # local rows
    p = comm.Get_size()
    xg = np.zeros(m*p, dtype='d')
    comm.Allgather([x,  MPI.DOUBLE],
                   [xg, MPI.DOUBLE])
    y = np.dot(A, xg)
    return y
A=np.array([[1,2],[3,4]])
x=1
comm = MPI.COMM_SELF.Spawn(sys.executable,maxprocs=5)
matvec(comm,A,x)