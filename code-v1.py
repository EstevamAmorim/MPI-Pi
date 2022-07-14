import sys
from mpi4py import MPI

N = 840
ITERATIONS = 1000

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
proc = MPI.Get_processor_name()


def pi_value(_startInclusive, _endExclusive, _n):
    result = 0

    for i in range(_startInclusive, _endExclusive):
        result += 1 / (1 + pow((i - 1/2)/_n, 2))

    return result*(4/_n)


if (len(sys.argv) > 1):
    n = int(sys.argv[1])
else:
    n = N

comm.Barrier()
tStart = MPI.Wtime()

for i in range(ITERATIONS):
    result = pi_value(1, n+1, n)

comm.Barrier()
tEnd = MPI.Wtime()

avgDeltaTimeMs = (tEnd - tStart) * 1000.0
avgTimePerCompMs = (tEnd - tStart) * (1000.0 / ITERATIONS)

# print('Process {0:02} ({1}) computed Pi (N={2}, Iter={3}) in {4:.2f}ms: {5}'.format(rank, proc, n, ITERATIONS, avgDeltaTimeMs,  result))
print('Process {0:02} ({1}) computed Pi (N={2}, Iter={3}) in {4:.2f}ms: {5}'.format(rank, proc, n, ITERATIONS, avgTimePerCompMs,  result))