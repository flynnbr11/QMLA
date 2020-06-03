I am running a program on a PBS server, where each instance of the program uses a unique [redis-server][1], and [RQ workers][2] to run tasks remotely with Python. This works okay, but on different instances of the program, I see large variance in timing. For example, the same program on one node can take over ten times longer on another. I would like to understand why this is the case; are some remote nodes delayed in reading the memory, or is it due to the structure of my program? 


The tasks given to the RQ workers is reasonably expensive and involves a number of `import`s from the module's directory - I have tried to boil it down to essential steps below. 

Here is the file `my_module.available_classes.py`:

```
import my_module.available_methods

class my_class():
    def __init__(self):
        # set method1 as function object written elsewhere in the module, 
        # so it can be easily replaced with alternatives
        self.method_1 = my_module.available_methods.option_1

    def call_method_1(self, **kwargs):
        # Wrapper for method1 object; expensive calculation
        self.method_1(**kwargs)
```

And `my_module.remote_subroutine.py`:

```
import my_module.available_classes

def expensive_step( args_for_expensive_step ):
    # instance of class defined within the module
    c = my_module.available_classes.my_class()

    # call expensive method repeatedly
    while True:
        c.call_method_1()

        if c.is_finished():
            break

```
Then, `my_module.my_program.py` runs 

```
from rq import Connection, Queue, Worker
import my_module.remote_subroutine

queue = Queue(queue_id,  connection=redis_conn)
queue.enqueue(
    my_module.remote_subroutine.expensive_step, 
    args_for_expensive_step)
```

I read in the [RQ docs][3] that it might be beneficial to use a custom worker script with modules pre-loaded before forking. I have tried the following but still have the timing issue.

```
import redis
from rq import Queue

# Preload libraries:
import my_module

# Parse command line arguments 
# -> set redis_host_name, redis_port_number, queue_id

redis_conn = redis.Redis(
        host = redis_host_name,
        port = redis_port_number
)

with Connection( redis_conn ):
        w = Worker( [str(queue_id)] , connection=redis_conn)
        w.work()

```

My question is: does the structure of my program not allow for pre-loading `my_module`, such that RQ workers have immediate access to `method1` etc?  Otherwise, why are some RQ workers so much slower than others at performing the same task? 
Note that within instances of the program, `expensive_step` is computed for different inputs and timings are consistent *within instances*, i.e. it takes 100s for each of 20 jobs on `my_program instance 1`, and 1000s for each of 20 jobs on `my_program instance 2`. 


  [1]: https://pypi.org/project/redis/
  [2]: https://python-rq.org/
  [3]: https://python-rq-docs-cn.readthedocs.io/en/latest/workers.html