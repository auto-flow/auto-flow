import multiprocessing as mp


def parse_n_jobs(n_jobs):
    if n_jobs > 0:
        return n_jobs
    elif n_jobs < 0:
        return mp.cpu_count() + 1 - n_jobs
    else:
        return 1
