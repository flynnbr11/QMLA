from jug import TaskGenerator
import time
import os 

@TaskGenerator
def task_one(my_id):
  print("Task type 1 has id ", my_id, " on pid ", os.getpid())
  time.sleep(10)
  print("has slept")
    
@TaskGenerator
def task_two(my_id):
  print("Task type 2 has id ", my_id, " on pid ", os.getpid())
  time.sleep(1)    

output1 = [] 
output2 = [] 

for n in range(1, 10):
  output1.append(task_one(n))
  output2.append(task_two(n))


print("output1 is ", output1)
print("\noutput1 is ", output1)
