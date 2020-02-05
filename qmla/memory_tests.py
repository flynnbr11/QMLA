from __future__ import division
import sys
import os
import inspect
from psutil import virtual_memory

__all__ = [
    'print_loc',
    'print_file_line'
]


def print_loc(print_location=False, sig_figures=2):
    if print_location:
        callerframerecord = inspect.stack()[1]    # 0 represents this line
        frame = callerframerecord[0]
        info = inspect.getframeinfo(frame)
        mem = virtual_memory()
        mem_percent = (mem.used / mem.total) * 100
        print(
            "Line {} of {}; function {}. Memory \% used: {}".format(
                info.lineno,
                info.filename,
                info.function,
                round(
                    mem_percent,
                    sig_figures)
            )
        )


def print_file_line(print_location=False, sig_figures=2):
    if print_location:
        callerframerecord = inspect.stack()[1]    # 0 represents this line
        frame = callerframerecord[0]
        info = inspect.getframeinfo(frame)
        # mem = virtual_memory()
        # mem_percent = (mem.used/mem.total)*100
        print(
            "Line ",
            info.lineno,
            " of ",
            info.filename,
            " function ",
            info.function)
