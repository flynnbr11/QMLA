import sys
import os
import numpy as np
import pandas as pd
import time
import pickle

from mpi4py import MPI

from param_sweep import get_all_configurations, run_genetic_algorithm

# globals  
WORKTAG = 0
DIETAG = 1
MASTER_RANK = 0

def master():   
    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    status = MPI.Status()
    num_processes = comm.Get_size()
    
    other_processes = list(range(num_processes))
    other_processes.remove(rank)
    print("Worker process ranks:", other_processes)

    # set up results infrastructure
    ga_results_df = pd.DataFrame() # for storage

    # get configurations to cycle over
    all_configurations = get_all_configurations(
        log_file = os.path.join(
            os.getcwd(), 'output.log'
        ),
    )
    iterable_configurations = iter(all_configurations)
        
    # send first batches of work to the workers
    for r in other_processes:
        try:
            configuration = next(iterable_configurations)
        except:
            print("No remaining configurations")
            break
        comm.send(
            obj = configuration, 
            dest = r, 
            tag = WORKTAG
        )

    # iteratively send configurations to workers, until none remain
    while True: 
        try:
            configuration = next(iterable_configurations)
        except:
            print("No remaining configurations")
            break

        # process result
        incoming_data = comm.recv(
            source = MPI.ANY_SOURCE,
            tag = MPI.ANY_TAG,
            status = status
        )
        result = pd.Series(incoming_data)
        ga_results_df = ga_results_df.append(
            result, 
            ignore_index=True
        )

        # send more work to the process which has just finished
        sender = status.Get_source()
        print("Sending next config to {} : {}".format(
            sender, 
            configuration
        ))
        comm.send(
            obj = configuration, 
            dest = sender, 
            tag = WORKTAG
        )
    
    # wait for outstanding work packets
    print("Waiting on outstanding work packets")
    for r in other_processes:
        incoming_data = comm.recv(
            source = MPI.ANY_SOURCE,
            tag = MPI.ANY_TAG,
            status = status
        )
        result = pd.Series(incoming_data)
        ga_results_df = ga_results_df.append(
            result, 
            ignore_index=True
        )

    # kill the workers 
    print("Killing workers")
    for r in other_processes:
        comm.send(
            obj=None, 
            dest = r,
            tag = DIETAG, 
        )

    # store the result
    path_to_store_result = os.path.join(
        os.getcwd(), 
        'result_param_sweep.csv'
    )
    ga_results_df.to_csv( path_to_store_result )
    print("Results stored at:", path_to_store_result)


def worker():
    comm = MPI.COMM_WORLD
    status = MPI.Status()
    print("I am worker with rank {}".format(rank))

    while True: 

        incoming_data = comm.recv(
            source = 0, 
            tag = MPI.ANY_TAG, 
            status = status, 
        )
        if status.Get_tag() == DIETAG: 
            # received die tag
            print("Received DIE tag; terminating")
            break   
        else:
            result = run_genetic_algorithm(configuration = incoming_data)
            print(
                "Completed genetic alg run; sending result to master. config=", incoming_data, 
                "\nresult=", result
            )
            comm.send(
                obj = result,
                dest = 0
            )
    print("Worker {} received DIETAG so has stopped accepting work.")


# Run the script; this is run on every process. 
if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == MASTER_RANK:
        master()
        print("Master finished")
    else:
        worker()



