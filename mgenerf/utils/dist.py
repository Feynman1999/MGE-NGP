from megengine import distributed
import functools

def master_only(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper

def get_dist_info():
    rank = distributed.get_rank()
    world_size =  distributed.get_world_size()
    return rank, world_size
