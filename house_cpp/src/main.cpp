#include "pybind11/pybind11.h"   // Pybind11 import to define Python bindings
#include "pybind11/complex.h"    // not sure why?


#include "xtensor/xtensor.hpp"   // xtensor (fixed dimension)
#define FORCE_IMPORT_ARRAY    // numpy C api loading
#include "xtensor-python/pytensor.hpp"    // Numpy bindings

#include <complex>
#include "xtensor/xcomplex.hpp"

#include "xtensor/xmath.hpp"              // xtensor import for the C++ universal functions

#include <vector>
#include "xtensor/xview.hpp"
#include <numeric>                        // Standard library import for std::accumulate
#include <cmath>

#include "xtensor-blas/xlinalg.hpp"

namespace py = pybind11;

using namespace std::complex_literals;


xt::pytensor<std::complex<double>, 2> test(xt::pytensor<std::complex<double>, 2>& W)
{
  auto s = W.shape();
  auto m = s[0];
  auto n = s[1];

  xt::pytensor<std::complex<double>, 2> Q = xt::eye(m);

  for(int i = 0; i < n; ++i) {
    for(int k = n-1; k >= 0; --k) {
      xt::view(Q, xt::range(k, m), k) = 3;
    }
  }
  return Q;
}




xt::pytensor<std::complex<double>, 2> house(xt::pytensor<std::complex<double>, 2>& A)
{
  auto s = A.shape();
  auto m = s[0];
  auto n = s[1];

  xt::pytensor<std::complex<double>, 2> R = xt::copy(A);
  xt::pytensor<std::complex<double>, 2> W = xt::zeros<std::complex<double>>({m, n});

  // for k in range(n):
  //     v_k = np.copy(R[k:, k])
  //     sgn = np.sign(v_k[0])
  //     if  sgn == 0: sgn = 1
  //     v_k[0] += np.exp(1j*np.angle(v_k[0])) * sgn * np.linalg.norm(np.abs(v_k))
  //     v_k /= np.linalg.norm(v_k)
  //     W[k:, k] = v_k
  //     R[k:, k:] -= 2 * np.outer(v_k, np.dot(np.conjugate(v_k).T, R[k:, k:]))          # 28 ms
  //     #R[k:, k:] -= 2 * np.dot(np.outer(v_k, np.conjugate(v_k).T), R[k:, k:])  # slower 31.5 ms
  // if m > n:
  //     R = np.copy(R[:n,:])
  return R; //, R
}





xt::pytensor<std::complex<double>, 2> formQ(xt::pytensor<std::complex<double>, 2>& W)
{
  auto s = W.shape();
  auto m = s[0];
  auto n = s[1];

  xt::pytensor<std::complex<double>, 2> Q = xt::eye(m);

  for(int i = 0; i < n; ++i) {
    for(int k = n-1; k >= 0; --k) {
      xt::xtensor<std::complex<double>, 1> v_k = xt::view(W, xt::range(k, m), k);
      xt::view(Q, xt::range(k, m), i) -= (std::complex<double>)2 * v_k * xt::linalg::vdot(v_k, xt::view(Q, xt::range(k, m), i));
    }
  }
  return Q;
}



// Python Module and Docstrings

PYBIND11_PLUGIN(house_cpp)
{
    xt::import_numpy();

    py::module m("house_cpp", R"docu(
        Householder orthogonal triangularization in C++ with xtensor

        .. currentmodule:: house_cpp

        .. autosummary::
           :toctree: _generate

           test
           formQ
    )docu");

    m.def("test", test, "test function");

    m.def("house", house, R"pbdoc(
      Computes an implicit representation of a full QR factorization A = QR
      of an m x n matrix A with m >= n using Householder reﬂections.

      Returns
      -------
      - lower-triangular complex matrix W m x n whose columns are the vectors v_k
          defining the successive Householder reﬂections
      - upper-triangular complex matrix R  n x n
    )pbdoc");

    m.def("formQ", formQ, "Generates a corresponding m × m orthogonal matrix Q.");

    return m.ptr();
}
