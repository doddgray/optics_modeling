import time
import numpy as np

A = np.random.rand(2000,2000)
B = np.random.rand(2000,2000)

print('Matrix multiplication')
time1 = time.time()
clock1 = time.clock_gettime(0)
C = np.dot(A,B)
clock2 = time.clock_gettime(0)
time2 = time.time()
print('  Elapsed time: %.02f sec.' % (time2-time1) )
print('  CPU time: %.02f sec.' % (clock2-clock1) )

print('Eigenvalue computation')
time1 = time.time()
clock1 = time.clock_gettime(0)
np.linalg.eig(A)
clock2 = time.clock_gettime(0)
time2 = time.time()
print('  Elapsed time: %.02f sec.' % (time2-time1) )
print('  CPU time: %.02f sec.' % (clock2-clock1) )
