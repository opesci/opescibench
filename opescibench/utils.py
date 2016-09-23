import sys

__all__ = ['bench_print']


def bench_print(msg, pre=0, post=0):
    if sys.stdout.isatty() and sys.stderr.isatty():
        # Blue
        color = '\033[1;37;34m%s\033[0m'
    else:
        color = '%s'

    for i in range(pre):
        print ""
    if msg:
        print color % ("OpesciBench: %s" % msg)
    for i in range(post):
        print ""
