import datetime

__all__ = [
    'print_to_log'
]

def time_seconds():
    r"""return current time in h:m:s format for logging."""
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
    Writes to the log file, registering the time and identifier. 

    Adds the content of `to_print_list` to the `log_file`,
    using the `log_identifier` to indicate where a given 
    log entry originated. 

    :param to_print_list: string you want to print
    :type to_print_list: str() or list()
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
