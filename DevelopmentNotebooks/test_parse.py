import argparse
import os
import sys
import pickle 




parser = argparse.ArgumentParser(description='Pass variables for (I)QLE.')


parser.add_argument(
  '-inp', '--input_parameter', 
  help="True operator to be simulated and learned against.",
  type=str,
  default='this'
)

parser.add_argument(
  '-l', '--input_list', 
  help="True operator to be simulated and learned against.",
  # type=str,
  action='append',
  default=[]
)



arguments = parser.parse_args()


args_dict = vars(arguments)
for a in list(args_dict.keys()):
	print(
	    a, 
	    ':', 
	    args_dict[a]
	)
