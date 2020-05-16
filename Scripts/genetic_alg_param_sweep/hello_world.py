from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()


print("Hello world from rank", rank)