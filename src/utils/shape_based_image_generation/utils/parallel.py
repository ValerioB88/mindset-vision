from tqdm import tqdm
import multiprocessing as mp


def parallel(make_one, n: int, progress_bar: bool = True):
    """
    Run the make_one function in parallel
    params:
        make_one: the function to run in parallel
        n: the number of times to run the function
    """
    if progress_bar:
        with mp.Pool(processes=mp.cpu_count()) as pool:
            for _ in tqdm(pool.imap_unordered(make_one, [()] * n), total=n):
                pass
    else:
        with mp.Pool(processes=mp.cpu_count()) as pool:
            pool.map(make_one, [()] * n)


def parallel_args(make_one, combinations: list, progress_bar: bool = True):
    """
    Run the make_one function in parallel with arguments
    params:
        make_one: the function to run in parallel
        combinations: the list of combinations to be passed to the make_one function
    """
    if progress_bar:
        with mp.Pool(processes=mp.cpu_count()) as pool:
            for _ in tqdm(
                pool.imap_unordered(make_one, combinations), total=len(combinations)
            ):
                pass
    else:
        with mp.Pool(processes=mp.cpu_count()) as pool:
            pool.map(make_one, combinations)
