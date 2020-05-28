from rq import Worker


class WorkerQMLA(Worker):
    def __init__(self, queues=None, *args, **kwargs):
        import qmla
        super().__init__(queues=None, *args, **kwargs)


