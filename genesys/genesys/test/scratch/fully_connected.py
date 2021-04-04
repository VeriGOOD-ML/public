import numpy as np
from systolic_array import SystolicArray


def generate_data(input_size, weight_size):
    data = np.random.randint(-3,3, input_size)
    weights = np.random.randint(-3,3, weight_size)
    bias = np.random.randint(-3,3, weight_size[1])
    return data, weights, bias

def stanford_fc(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.
  The input x has shape_symbols (N, d_1, ..., d_k) where x[i] is the ith input.
  We multiply this against a weight matrix of shape_symbols (D, M) where
  D = \prod_i d_i
  Inputs:
  x - Input data, of shape_symbols (N, d_1, ..., d_k)
  w - Weights, of shape_symbols (D, M)
  b - Biases, of shape_symbols (M,)

  Returns a tuple of:
  - out: output, of shape_symbols (N, M)
  - cache: (x, w, b)
  """
  out = x.reshape(x.shape_symbols[0], -1).dot(w) + b
  return out

def naive_fc(I, W, B):
    N = I.shape_symbols[0]
    CHW = I.shape_symbols[1]
    M = W.shape_symbols[1]
    O = np.empty((N, M))
    for m in range(M):
        for n in range(N):
            O[n, m] = B[m]
            for o in range(CHW):
                O[n, m] += I[n,o]*W[o,m]
    return O

def sys_array_fc(I, W, B):
    sys_array = SystolicArray(32)
    TOC = 4
    TIC = 4
    TT_OC = TOC // M + 1
    TT_IC = TIC // N + 1
    O = np.zeros((N, M))
    for t_oc in range(TOC):
        for t_ic in range(TIC):
            sys_array.fill_outputs(O[t_oc + ])
            for tt_oc in range(TT_OC):
                for tt_ic in range(TT_IC):


def main():
    params = []
    lenet = ((128,80), (80,10))
    test = ((32, 32), (32, 32))

    params.append(lenet)
    params.append(test)

    for p in params:
        I, W, B = generate_data(p[0], p[1])
        stanford_O = stanford_fc(I, W, B)
        naive_O = naive_fc(I, W, B)
        np.testing.assert_allclose(naive_O, stanford_O)

if __name__ == "__main__":
    main()
