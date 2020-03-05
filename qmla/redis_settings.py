import os
import sys
import pickle

import redis

"""
This file provides functionality to create multiple databases on a redis server.
Each QMD has a unique port number (though can share hosts).
By calling databases_from_qmd_id, a set of database addresses unique to that qmd id (host/port) are returned in a dictionary.
These can be used to store information required by distinct actors
to perform QMD, without requiring all information be held on all actors
at all times.
This method is quite slow - useful information is stored in dictionaries and
pickled to redis. Pickling and unpickling is quite slow, so should be minimised.

"""

__all__ = [
    'databases_from_qmd_id'
]


databases_required = [
    'qmla_core_info_database',
    'learned_models_info_db',
    'learned_models_ids',
    'bayes_factors_db',
    'bayes_factors_winners_db',
    'active_branches_learning_models',
    'active_branches_bayes',
    'active_interbranch_bayes',
    'any_job_failed'
]


def databases_from_qmd_id(
    host_name,
    port_number,
    qmd_id,
    tree_identifiers=None,
    print_status=False
):
r"""
Function that returns a dictionary containing a set of database addresses unique to the qmd_id, host_name and port_number
passed as arguments
!!! StrictRedis !!! redis-py 3.0 drops support for StrictRedis taht will be renamed Redis

:params string host_name: 
    string with the specific name of the host

:params int port_number: 
    port number

:params qmd_id: 
    identifier of the qmd isntance


"""

    database_dict = {}
    seed = get_seed(
        host_name=host_name,
        port_number=port_number,
        qmd_id=qmd_id)

    if print_status:
        qid_seeds = redis.StrictRedis(host=host_name, port=port_number, db=0)

    for i in range(len(databases_required)):
        new_db = databases_required[i]
        database_dict[new_db] = redis.StrictRedis(
            host=host_name,
            port=port_number,
            db=seed + i
        )

    return database_dict


def get_seed(host_name, port_number, qmd_id, print_status=False):
    qid_seeds = redis.StrictRedis(host=host_name, port=port_number, db=0)
    seed_db_keys = [a.decode() for a in qid_seeds.keys()]
    first_qmd_id = False

    if 'max' not in seed_db_keys:
        # ie the database has not been set yet
        qid_seeds.set('max', 1)
        first_qmd_id = True

    if str(qmd_id) in seed_db_keys:
        seed = int(qid_seeds.get(qmd_id))

    elif qmd_id not in seed_db_keys:
        max_seed = int(qid_seeds.get('max'))
        if first_qmd_id:
            new_qid_seed = 1
        else:
            new_qid_seed = max_seed + len(databases_required)
        qid_seeds.set(qmd_id, int(new_qid_seed))
        qid_seeds.set('max', new_qid_seed)

        seed = new_qid_seed

    if print_status == True:
        print("Seed requested for host/port/id", host_name, '/', port_number,
              '/', qmd_id, ";has seed", seed
              )
        print("Seed keys", seed_db_keys)

    return int(seed)
