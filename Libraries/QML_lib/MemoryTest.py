from __future__ import division
import sys, os        
import inspect
from psutil import virtual_memory



def print_loc(print_location=False, sig_figures=2):
    if print_location:
      callerframerecord = inspect.stack()[1]    # 0 represents this line
      frame = callerframerecord[0]
      info = inspect.getframeinfo(frame)
      mem = virtual_memory()
      mem_percent = (mem.used/mem.total)*100
      print("Line ", info.lineno, " of ", info.filename, " function ", info.function, "Mem % used : ", round(mem_percent, sig_figures))

