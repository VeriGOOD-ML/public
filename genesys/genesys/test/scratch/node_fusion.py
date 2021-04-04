import numpy as np


def generate_data(b, c):
    A = np.arange(b*c).reshape(b,c)
    av = np.arange(c)
    B = np.arange(b*b).reshape(b,b)
    bv = np.arange(b)

    return A, av, B, bv

def compute(A, av, B, bv):
    X = A.dot(av)
    Y = B.dot(bv)
    return X + Y

def fuse_compute(A, av, B, bv):
    A = np.concatenate((A, B), axis=1)
    b = np.concatenate((av, bv))
    return A.dot(b)

if __name__ == "__main__":
    b = 20
    c = 30
    A, av, B, bv = generate_data(b, c)

    res1 = compute(A, av, B, bv)
    res2 = fuse_compute(A, av, B, bv)
    print(f"A shape: {A.shape_symbols}")
    print(f"av shape: {av.shape}")
    print(f"B shape: {B.shape_symbols}")
    print(f"bv shape: {bv.shape}")
    print(f"Res1 shape: {res1.shape_symbols}")
    print(f"Res2 shape: {res2.shape_symbols}")



    print(res1)
    print(res2)
    print("\n")
    print(np.allclose(res1, res2))
