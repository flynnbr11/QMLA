import datetime


def time_seconds():
    # return time in h:m:s format for logging.
    now = datetime.date.today()
    hour = datetime.datetime.now().hour
    minute = datetime.datetime.now().minute
    second = datetime.datetime.now().second
    time = str(str(hour) + ':' + str(minute) + ':' + str(second))
    return time


def print_to_log(
    to_print_list,
    log_file,
    log_identifier=''
):
    """
    log_print writes in the log file the string passed as first argument.
    log_print(to_print_list, log_file, log_identifier)

    longhish description: adds the content of the first argument to the log file passed as second argument
    and using the identifier for the log entry specified in the third parameter
    if the first argument is not passed as string will be converted to string.

    :param to_print_list: string you want to print
    :type to_print_list: str()
    :param log_file: path of the log file you want to update
    :type log_file: str()
    :param log_identifier: identifier for the log
    :type log_identifier: str()

    """
    if not isinstance(to_print_list, list):
        to_print_list = list(to_print_list)
    identifier = str(str(time_seconds()) +
                     " [" + log_identifier + "]"
                     )

    print_strings = [str(s) for s in to_print_list]
    to_print = " ".join(print_strings)
    with open(log_file, 'a') as write_log_file:
        print(identifier, str(to_print), file=write_log_file, flush=True)
