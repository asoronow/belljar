def run_parallel(rank, size, train_fn, args):
    """Distributed function"""
    train_fn(*args, rank=rank, world_size=size)
