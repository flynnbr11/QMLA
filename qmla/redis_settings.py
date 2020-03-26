import os
import sys
import pickle

import redis

"""
Create multiple databases on a redis server.

Redis connections require a port number and host server. 
Each QMLA instance has a unique port number (though can share hosts).
    These are determined by the launch script, 
    either local_launch.sh or parallel_launch.sh.
By calling get_redis_databases_by_qmla_id, a set of database 
    addresses unique to that qmd id (host/port) are returned in a dictionary.
These can be used to store information required by distinct actors
    to perform QMLA, without requiring all information be held on all actors
    at all times.
This method is quite slow - useful information is stored in dictionaries and
pickled to redis. Pickling and unpickling is quite slow, so should be minimised.
"""

__all__ = [
    'get_redis_databases_by_qmla_id'
]


databases_required = [
    'qmla_core_info_database',
    'learned_models_info_db',
    'learned_models_ids',
    'bayes_factors_db',
    'bayes_factors_winners_db',
    'active_branches_learning_models',
    'active_branches_bayes',
    'active_interbranch_bayes', # TODO unused?
    'any_job_failed'
]


def get_redis_databases_by_qmla_id(
    host_name,
    port_number,
    qmla_id,
    tree_identifiers=None,
    print_status=False
):
    r"""
    Gets the set of redis databases unique to this QMLA instance.

    !!! StrictRedis !!! redis-py 3.0 drops support for StrictRedis
        - will be renamed Redis
        - TODO

    :param str host_name: name of host server on which redis database exists.
    :param int port_number: this QMLA instance's unique port number,
        on which redis database exists. 
    :param int qmla_id: QMLA id, unique to a single instance within a run. 
        Used to cosntruct a unique redis database corresponding to this instance.
    :return dict database_dict: set of database addresses unique to the qmla_id, host_name and port_number
    """

    database_dict = {}
    seed = get_seed(
        host_name=host_name,
        port_number=port_number,
        qmla_id=qmla_id)

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


def get_seed(host_name, port_number, qmla_id, print_status=False):
    r"""
    Unique seed for this QMLA id
    """

    qid_seeds = redis.StrictRedis(host=host_name, port=port_number, db=0)
    seed_db_keys = [a.decode() for a in qid_seeds.keys()]
    first_qmla_id = False

    if 'max' not in seed_db_keys:
        # ie the database has not been set yet
        qid_seeds.set('max', 1)
        first_qmla_id = True

    if str(qmla_id) in seed_db_keys:
        seed = int(qid_seeds.get(qmla_id))

    elif qmla_id not in seed_db_keys:
        max_seed = int(qid_seeds.get('max'))
        if first_qmla_id:
            new_qid_seed = 1
        else:
            new_qid_seed = max_seed + len(databases_required)
        qid_seeds.set(qmla_id, int(new_qid_seed))
        qid_seeds.set('max', new_qid_seed)

        seed = new_qid_seed

    if print_status == True:
        print("Seed requested for host/port/id", host_name, '/', port_number,
              '/', qmla_id, ";has seed", seed
              )
        print("Seed keys", seed_db_keys)

    return int(seed)
