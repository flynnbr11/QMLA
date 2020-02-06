

# def get_pr0_array_qle(
#     t_list,
#     modelparams,
#     oplist,
#     probe,
#     # measurement_type='full_access',
#     growth_class=None,
#     use_experimental_data=False,
#     use_exp_custom=True,
#     exp_comparison_tol=None,
#     enable_sparse=True,
#     ham_list=None,
#     log_file='QMDLog.log',
#     log_identifier=None,
#     **kwargs
# ):
#     from rq import timeouts
#     def log_print(
#         self, 
#         to_print_list, 
#     ):
#         qmla.logging.print_to_log(
#             to_print_list = to_print_list, 
#             log_file = log_file, 
#             log_identifier = 'get_pr0'
#         )
    
#     qmla.memory_tests.print_loc(global_print_loc)
#     num_particles = len(modelparams)
#     num_times = len(t_list)
#     output = np.empty([num_particles, num_times])

#     for evoId in range(num_particles):  
#         try:
#             ham = np.tensordot(
#                 modelparams[evoId], oplist, axes=1
#             )
#         except BaseException:
#             self.log_print(
#                 [
#                     "Failed to build Hamiltonian.",
#                     "\nmodelparams:", modelparams[evoId],
#                     "\noplist:", oplist
#                 ],
#             )
#             raise

#         for tId in range(len(t_list)):
#             t = t_list[tId]
#             if t > 1e6:  # Try limiting times to use to 1 million
#                 import random
#                 # random large number but still computable without error
#                 t = random.randint(1e6, 3e6)
#             try:
#                 likel = growth_class.expectation_value(
#                     ham=ham,
#                     t=t,
#                     state=probe,
#                     log_file=log_file,
#                     log_identifier=log_identifier
#                 )
#                 output[evoId][tId] = likel

#             except NameError:
#                 self.log_print(
#                     [
#                         "Error raised; unphysical expecation value.",
#                         "\nHam:\n", ham,
#                         "\nt=", t,
#                         "\nState=", probe
#                     ],
#                 )
#                 sys.exit()
#             except timeouts.JobTimeoutException:
#                 self.log_print(
#                     [
#                         "RQ Time exception. \nprobe=",
#                         probe,
#                         "\nt=", t, "\nHam=",
#                         ham
#                     ],
#                 )
#                 sys.exit()

#             if output[evoId][tId] < 0:
#                 print("NEGATIVE PROB")
#                 self.log_print(
#                     [
#                         "[QLE] Negative probability : \
#                         \t \t probability = ",
#                         output[evoId][tId]
#                     ],
#                 )
#             elif output[evoId][tId] > 1.001:
#                 self.log_print(
#                     [
#                         "[QLE] Probability > 1: \
#                         \t \t probability = ",
#                         output[evoId][tId]
#                     ],
#                 )
#     return output

