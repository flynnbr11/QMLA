import sys
import argparse

parser = argparse.ArgumentParser(description='Pass variables for (I)QLE.')

parser.add_argument(
  '-exp', '--num_experiments', 
  type=float,
  default=200
)

arguments = parser.parse_args()

num_experiments = arguments.num_experiments

print("Num experiments = ", num_experiments)


