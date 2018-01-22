from multiprocessing import Pool, current_process

def square(x):
    """Function to return the square of the argument"""
    print("Worker %s calculating square of %d" % \
             (current_process().pid, x))
    return x*x

if __name__ == "__main__":
    nprocs = 8

    # print the number of cores
    print("Number of workers equals %d" % nprocs)

    # create a pool of workers
    pool = Pool(processes=nprocs)

    # create an array of 10 integers, from 1 to 10
    a = range(1,11)

    result = pool.map( square, a )

    total = reduce( lambda x,y: x+y, result )

    print("The sum of the square of the first 10 integers is %d" % total)

