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

  xt::pytensor<std::complex<double>, 2> Q(W);

  for(int i = 0; i < n; ++i) {
    for(int k = n-1; k >= 0; --k) {
      xt::view(Q, xt::range(k, m), k) = 3;
    }
  }
  return Q;
}


xt::xtensor<std::complex<double>, 1> angle(xt::xtensor<std::complex<double>, 1> complex_number){
  return xt::atan2(xt::imag(complex_number), xt::real(complex_number));
}


xt::pytensor<std::complex<double>, 2> house(xt::pytensor<std::complex<double>, 2>& A)
{
  auto s = A.shape();
  auto m = s[0];
  auto n = s[1];

  xt::pytensor<std::complex<double>, 2> R(A);
  xt::pytensor<std::complex<double>, 2> W = xt::zeros<std::complex<double>>({m, n});

  for(int k=0; k<n; ++k) {
    xt::xtensor<std::complex<double>, 1> v_k(xt::view(R, xt::range(k, m), k));
    // How to get the number/sign??
    std::complex<float> sgn = xt::sign(xt::view(v_k, 0));
    if(sgn == 0){
      sgn = 1;
    }
    xt::view(v_k, 0) += xt::exp(1i*angle(xt::view(v_k, 0))) * sgn * xt::linalg::norm(v_k);
    v_k /= xt::linalg::norm(v_k);
    xt::view(W, xt::range(k, m), k) = v_k;
    // no matching function for call to 'outer'
    xt::view(R, xt::range(k, n), xt::range(k, n)) -= (std::complex<double>)2 * xt::linalg::outer(v_k, xt::linalg::vdot(v_k, xt::view(R, xt::range(k, n), xt::range(k, n))));
  }
  if(m>n){
    R = xt::view(R, xt::range(0, n), xt::all());
  }

  // How to return W, R ??
  return W; //, R
}





xt::pytensor<std::complex<double>, 2> formQ(xt::pytensor<std::complex<double>, 2>& W)
{
  auto s = W.shape();
  auto m = s[0];
  auto n = s[1];

  if(m<n){
    //throw exception("m (rows) must be greater or equal than n (columns)");
  }

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
