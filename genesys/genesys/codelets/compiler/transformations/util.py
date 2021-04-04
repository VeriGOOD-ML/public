from pytools import memoize

@memoize
def factors(n):
    return sorted(list(set(x for tup in ([i, n//i]
                for i in range(1, int(n**0.5)+1) if n % i == 0) for x in tup)))