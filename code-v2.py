import sys
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
proc = MPI.Get_processor_name()

N = 840
ITERATIONS = 1000


def pi_value(_startInclusive, _endExclusive, _n):
    result = 0

    for i in range(_startInclusive, _endExclusive):
        result += 1 / (1 + pow((i - 1/2)/_n, 2))

    return result*(4/_n)


def get_interval(_rank, _size, _n):
    div = int(_n / _size)
    start = div * _rank + 1
    end = div * (_rank + 1)
    return (start, end)


if (len(sys.argv) > 1):
    n = int(sys.argv[1])
else:
    n = N

(start, end) = get_interval(rank, size, n)

comm.Barrier()
tStart = MPI.Wtime()

for i in range(ITERATIONS):
    result = pi_value(start, end + 1, n)

comm.Barrier()
tEnd = MPI.Wtime()

# avgDeltaTimeMs = (tEnd - tStart) * 1000.0
avgTimePerCompMs = (tEnd - tStart) * (1000.0 / ITERATIONS)

# print('Process {0:02} ({1}) computed Pi (N={2}, Iter={3}), from {4:03} to {5:03}, in {6:.2f}ms: {7}'.format(rank, proc, n, ITERATIONS,  start, end, avgDeltaTimeMs, result))
print('Process {0:02} ({1}) computed Pi (N={2}, Iter={3}), from {4:03} to {5:03}, in {6:.2f}ms: {7}'.format(rank, proc, n, ITERATIONS,  start, end, avgTimePerCompMs, result))