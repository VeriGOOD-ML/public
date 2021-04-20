from pytools import memoize
from itertools import product
import numpy as np

@memoize
def level_factors(n, level):
    if level <= 1:
        return factors_rand_sort(n)
    else:
        return factors_reversed(n)

@memoize
def factors(n, level):
    return sorted(list(set(x for tup in ([i, n//i]
                for i in range(1, int(n**0.5)+1) if n % i == 0) for x in tup)))


@memoize
def factors_rand_sort(n, level):
    return sorted(list(set(x for tup in ([i, n//i]
                for i in range(1, int(n**0.5)+1) if n % i == 0) for x in tup)), key=lambda x: np.random.random())


@memoize
def factors_reversed(n, level):
    return sorted(list(set(x for tup in ([i, n//i]
                for i in range(1, int(n**0.5)+1) if n % i == 0) for x in tup)), reverse=True)

@memoize
def get_sorted_perms(perms):
    return sorted(list(perms), key=lambda x: np.prod(x), reverse=True)