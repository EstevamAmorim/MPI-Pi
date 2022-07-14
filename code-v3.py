import numpy
import sys
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
proc = MPI.Get_processor_name()

recv_buffer = numpy.zeros(1)
result = numpy.zeros(1)

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
    result[0] = pi_value(start, end + 1, n)

    if rank != 0:
        comm.Send(result, 0)
        # print('Result sent: {}'.format(result))
    else:
        total = result[0]
        # print('Initial total: {}'.format(total))

        for i in range(size - 1):
            comm.Recv(recv_buffer, MPI.ANY_SOURCE)
            # print('Received: {}'.format(recv_buffer[0]))
            total += recv_buffer[0]
        # print('Current total: {}'.format(total))

comm.Barrier()
tEnd = MPI.Wtime()

# avgDeltaTimeMs = (tEnd - tStart) * 1000.0
avgTimePerCompMs = (tEnd - tStart) * (1000.0 / ITERATIONS)

# print('Process {0:02} ({1}) computed partial Pi (N={2}), from {3:03} to {4:03}: {5}'.format(rank, proc, n,  start, end, result[0]))

if rank == 0:
    # print('Processes computed Pi value (N={0}, Iterations={1}) in {2:.2f}ms: {3}'.format(n, ITERATIONS, avgDeltaTimeMs, total))
    print('Processes computed Pi value (N={0}, Iterations={1}) in {2:.2f}ms: {3}'.format(n, ITERATIONS, avgTimePerCompMs, total))