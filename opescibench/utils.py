import sys
from datetime import datetime

try:
    from mpi4py import MPI
    mpi_rank = MPI.COMM_WORLD.rank
except ImportError:
    # Assume serial
    mpi_rank = 0


__all__ = ['bench_print', 'mpi_rank']


def bench_print(msg, pre=0, post=0, timestamp=False):
    if sys.stdout.isatty() and sys.stderr.isatty():
        # Blue
        color = '\033[1;37;34m%s\033[0m'
    else:
        color = '%s'

    for i in range(pre):
        if mpi_rank == 0:
            print ("")
    if msg:
        if mpi_rank == 0:
            now = datetime.now()
            ts = ' [%s]' % now.strftime("%H:%M:%S") if timestamp else ''
            print (color % ("OpesciBench%s: %s" % (ts, msg)))
    for i in range(post):
        if mpi_rank == 0:
            print ("")
