# This script will give us the ATE

# standard library
import os
import sys
# project library
import grmToolbox
import grmReader
from mpi4py import MPI
import numpy

''' Simulation.
'''
grmToolbox.simulate()

''' Estimation.
'''
rslt=grmToolbox.estimate()
Y1_beta    = (rslt['Y1_beta'])
Y0_beta    = (rslt['Y0_beta'])   
U1_var     = rslt['U1_var'] 
U0_var     = rslt['U0_var']

# Get the X's from the data
# Checks.
assert (os.path.exists('grmInit.ini'))     
# Process initialization file.
initDict = grmReader.read()
data = numpy.genfromtxt(initDict['fileName'], dtype = 'float')
trash   = numpy.array(initDict['Y1_beta'])
numCovarsOut  = trash.shape[0]  
X = data[:,2:(numCovarsOut + 2)]
D = data[:,1]
N_type1=sum(D)
index_0=numpy.where(D==0)[0]
index_1=numpy.where(D==1)[0]

# Now Create Y1 and Y0 without the errors; we will generate the errors later
# using Parallel Computing
D1_wo_errors_Y1=numpy.dot(X[index_1,:],Y1_beta)
D1_wo_errors_Y1_avg=numpy.mean(D1_wo_errors_Y1)

D1_wo_errors_Y0=numpy.dot(X[index_1,:],Y0_beta)
D1_wo_errors_Y0_avg=numpy.mean(D1_wo_errors_Y0)

D0_wo_errors_Y1=numpy.dot(X[index_0,:],Y1_beta)
D0_wo_errors_Y1_avg=numpy.mean(D0_wo_errors_Y1)

D0_wo_errors_Y0=numpy.dot(X[index_0,:],Y0_beta)
D0_wo_errors_Y0_avg=numpy.mean(D0_wo_errors_Y0)

# Now we simulate the error terms using Parallel Computing
# For U1_var draws for D=1
comm = MPI.COMM_SELF.Spawn(sys.executable,
                           args=['worker.py'],
                           maxprocs=4)
# Broadcast the total number of individuals we need to simulate
# Broadcast the variance of the random draw
N = numpy.array(N_type1, 'i')
comm.Bcast([N, MPI.INT], root=MPI.ROOT)
Var = numpy.array(U1_var)
comm.Bcast([Var, MPI.DOUBLE], root=MPI.ROOT)
draws_sum = numpy.array(0.0, 'd')
comm.Reduce(None, [draws_sum, MPI.DOUBLE],
            op=MPI.SUM, root=MPI.ROOT)
# From Parallel Computing Get the mean of these error terms
U1_avg_D1=draws_sum/N 

# For U1_var draws for D=0
comm = MPI.COMM_SELF.Spawn(sys.executable,
                           args=['worker.py'],
                           maxprocs=4)
# Broadcast the total number of individuals we need to simulate
# Broadcast the variance of the random draw
N = numpy.array(N_type1, 'i')
comm.Bcast([N, MPI.INT], root=MPI.ROOT)
Var = numpy.array(U1_var)
comm.Bcast([Var, MPI.DOUBLE], root=MPI.ROOT)
draws_sum = numpy.array(0.0, 'd')
comm.Reduce(None, [draws_sum, MPI.DOUBLE],
            op=MPI.SUM, root=MPI.ROOT)
# From Parallel Computing Get the mean of these error terms
U1_avg_D0=draws_sum/N
 
# For U0_var draws for D=1
comm = MPI.COMM_SELF.Spawn(sys.executable,
                           args=['worker.py'],
                           maxprocs=4)
# Broadcast the total number of individuals we need to simulate
# Broadcast the variance of the random draw
N = numpy.array(10000-N_type1, 'i')
comm.Bcast([N, MPI.INT], root=MPI.ROOT)
Var = numpy.array(U0_var)
comm.Bcast([Var, MPI.DOUBLE], root=MPI.ROOT)
draws_sum = numpy.array(0.0, 'd')
comm.Reduce(None, [draws_sum, MPI.DOUBLE],
            op=MPI.SUM, root=MPI.ROOT)
# From Parallel Computing Get the mean of these error terms
U0_avg_D1=draws_sum/N 

# For U0_var draws for D=0
comm = MPI.COMM_SELF.Spawn(sys.executable,
                           args=['worker.py'],
                           maxprocs=4)
# Broadcast the total number of individuals we need to simulate
# Broadcast the variance of the random draw
N = numpy.array(10000-N_type1, 'i')
comm.Bcast([N, MPI.INT], root=MPI.ROOT)
Var = numpy.array(U0_var)
comm.Bcast([Var, MPI.DOUBLE], root=MPI.ROOT)
draws_sum = numpy.array(0.0, 'd')
comm.Reduce(None, [draws_sum, MPI.DOUBLE],
            op=MPI.SUM, root=MPI.ROOT)
# From Parallel Computing Get the mean of these error terms
U0_avg_D0=draws_sum/N 

comm.Disconnect()

# Now, we calculate the ATE, TT, TUT
TT=(D1_wo_errors_Y1_avg+U1_avg_D1)-(D1_wo_errors_Y0_avg+U0_avg_D1)
TUT=(D0_wo_errors_Y1_avg+U1_avg_D0)-(D0_wo_errors_Y0_avg+U0_avg_D0)
ATE=(TT*N_type1+TUT*(10000-N_type1))/10000
print TT 
print TUT 
print ATE