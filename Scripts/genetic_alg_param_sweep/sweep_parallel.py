import sys
import os
import numpy as np
import pandas as pd
import time
import pickle
import datetime

from mpi4py import MPI

from param_sweep import get_all_configurations, run_genetic_algorithm

# globals  
WORKTAG = 0
DIETAG = 1
MASTER_RANK = 0

TAGS = {
    'available' : 0, 
    'work' : 1,
    'shutdown' : 2, 
    'failed_job' : 3, 
    'awake' : 4,
    'result' : 5
}

def master():   
    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    status = MPI.Status()
    num_processes = comm.Get_size()
    
    other_processes = list(range(num_processes))
    other_processes.remove(rank)
    num_workers = len(other_processes)
    num_workers_shutdown = 0
    num_workers_registered = 0

    # set up results infrastructure
    ga_results_df = pd.DataFrame() # for storage

    # get configurations to cycle over
    all_configurations = get_all_configurations(
        log_file = os.path.join(
            os.getcwd(), 'output.log'
        ),
    )
    iterable_configurations = iter(all_configurations)
    print("[MASTER] Configurations to cycle through:", all_configurations, flush=True)

    # iteratively send configurations to workers, until none remain
    while num_workers_shutdown < num_workers: 

        # receive a message from any worker
        incoming_data = comm.recv(
            source = MPI.ANY_SOURCE,
            tag = MPI.ANY_TAG,
            status = status
        )
        sender = status.Get_source()
        tag = status.Get_tag()

        # process the message
        if tag == TAGS['shutdown']:
            num_workers_shutdown += 1
        elif tag == TAGS['failed_job'] : 
            print("[MASTER] Job failure recorded from worker ", sender, flush=True)
        elif tag == TAGS['result']:
            result = pd.Series(incoming_data)
            print("[MASTER] received result:", result)
            ga_results_df = ga_results_df.append(
                result, 
                ignore_index=True
            )

        # if the worker is still available, send it the next message
        if tag == TAGS['result'] or tag == TAGS['failed_job'] or tag==TAGS['awake']:
            try:
                configuration = next(iterable_configurations)
                print("[MASTER] Got a new config", flush=True)
                comm.send(
                    obj = configuration, 
                    dest = sender, 
                    tag = TAGS['work']
                )
            except:
                print("[MASTER] No remaining configurations - sending shut down message", flush=True)
                comm.send(
                    obj=None, 
                    dest = sender,
                    tag = TAGS['shutdown'], 
                )

    # store the result
    result_directory = os.path.join(os.getcwd(), 'results')
    if not result_directory: os.makedirs(result_directory)

    now = datetime.datetime.now()
    time = "{}_{}_{}_{}".format(
        now.strftime("%b"),
        now.strftime("%d"),
        now.strftime("%H"),
        now.strftime("%M"),
    )
    path_to_store_result = os.path.join(
        result_directory, 
        'results_{}.csv'.format(time)
    )
    ga_results_df.to_csv( path_to_store_result )
    print("[MASTER] Results stored at:", path_to_store_result, flush=True)


def worker():
    comm = MPI.COMM_WORLD
    status = MPI.Status()
    print("I am worker with rank {}".format(rank), flush=True)
    
    # tell the master I am awake
    comm.send(
        obj = None,
        dest = 0,
        tag = TAGS['awake']
    )

    # accept messages and act based on what they say
    while True: 
        print("Worker {} waiting on message from master".format(rank), flush=True)
        incoming_data = comm.recv(
            source = 0, 
            tag = MPI.ANY_TAG, 
            status = status, 
        )
        tag = status.Get_tag()

        # act based on message
        if tag == TAGS['shutdown']: 
            # Tell master I am shutting down
            print("Received shut down tag; terminating worker ", rank, flush=True)
            comm.send(
                obj = None,
                dest = 0,
                tag = TAGS['shutdown'], 
            )
            break   
        elif tag == TAGS['work']:
            # perform the calculation and send the result to the master
            result = run_genetic_algorithm(configuration = incoming_data)
            if result is None: 
                print("Job failed")
                tag = TAGS['failed_job']
            else: 
                tag = TAGS['result']
            comm.send(
                obj = result,
                dest = 0,
                tag = tag
            )
        else:
            print("Received message with unexpected tag:", tag, "on worker", rank)

    print("Worker {} no longer accepting work.".format(rank), flush=True)


# Run the script; this is run on every process. 
if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    print("__main__ starting, on rank {}".format(rank))
    if rank == MASTER_RANK:
        master()
    else:
        worker()
    print("__main__ finished, on rank {}".format(rank))    

