# initializes src as a module, and imports benchmark_2 from it.
import sys
import os

sys.path.insert(0,os.path.dirname(os.path.realpath(__file__)))

from . import benchmark_2
from . import send_results
