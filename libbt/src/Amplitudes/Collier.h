// -*- c++ -*-

#ifndef BLACKTHORN_COLLIER_COLLIER_INTERFACE_H
#define BLACKTHORN_COLLIER_COLLIER_INTERFACE_H

#include <cmath>
#include <complex>
#define dcplx std::complex<double>

// COLLIER subroutines
extern "C" {
/**
 * Initialize the Collier library.
 * @param[in] nmax maximal number of loop propagators.
 * @param[in] rmax maximal rank of loop integrals.
 */
void collier_initialize(int *, int *);

/**
 * Reinialize Collier for tensor integral computations.
 * @param[in] nmax maximal number of loop propagators.
 * @param[in] rmax maximal rank of loop integrals.
 */
void collier_initialize_event();

/**
 * Complex the scalar A0(m12).
 * @param[out] A0 on output, stores the result.
 * @param[in] m12 squared mass.
 */
void collier_scalarA0(dcplx *A0, dcplx *m12);
/**
 * Complex the scalar B0(s,m12,m22).
 * @param[out] B0 on output, stores the result.
 * @param[in] s external invariant.
 * @param[in] m12,m22 squared masses.
 */
void collier_scalarB0(dcplx *B0, dcplx *s, dcplx *m12, dcplx *m22);
/**
 * Complex the scalar C0(s1,s12,s2,m02,m22,m22).
 * @param[out] C0 on output, stores the result.
 * @param[in] s1,s12,s2 external invariants.
 * @param[in] m12,m22,m32 squared masses.
 */
void collier_scalarC0(dcplx *C0, dcplx *s1, dcplx *s12, dcplx *s2, dcplx *m12,
                      dcplx *m22, dcplx *m23);
/**
 * @breif Compute the number of tensor coefficients of an `n`-pt integral of
 * rank `r`.
 *
 * @param[out] nc number of tensor coefficients.
 * @param[in] n number of propagators.
 * @param[in] r rank.
 */
int collier_get_nc(int *n, int *r);
/**
 * Complex the scalar A0(m12).
 * @param[out] A 1D-array of the coefficients of shape (rmax/2+1)
 * @param[out] Auv 1D-array of the coefficients of UV-singular
 * \f$1/\epsilon_{UV}\f$ poles (same shape as `A`)
 * @param[in] m02 squared mass
 * @param[in] rmax maximal rank
 */
void collier_coeffs_a(dcplx *A, dcplx *Auv, dcplx *m12, int *rmax);
/**
 * Complex the scalar B0(s,m12,m22).
 * @param[out] B 2D-array of the coefficients of shape (rmax/2+1, rmax+1)
 * @param[out] Buv 2D-array of the coefficients of UV-singular
 * \f$1/\epsilon_{UV}\f$ poles (same shape as B)
 * @param[in] s1,s12,s2 momentum invariants
 * @param[in] m02,m12,m22 squared masses
 * @param[in] rmax maximal rank
 */
void collier_coeffs_b(dcplx *B, dcplx *Buv, dcplx *s, dcplx *m02, dcplx *m12,
                      int *rmax);
/**
 * Complex the scalar C0(s1,s12,s2,m02,m22,m22).
 * @param[out] C 3D-array of the coefficients of shape (rmax/2+1, rmax+1,
 * rmax+1)
 * @param[out] Cuv 3D-array of the coefficients of UV-singular
 * \f$1/\epsilon_{UV}\f$ poles (same shape as `C`)
 * @param[in] s1,s12,s2 momentum invariants
 * @param[in] m02,m12,m22 squared masses
 * @param[in] rmax maximal rank
 */
void collier_coeffs_c(dcplx *C, dcplx *Cuv, dcplx *s1, dcplx *s12, dcplx *s2,
                      dcplx *m02, dcplx *m12, dcplx *m22, int *rmax);
//,dcplx *Cerr, dcplx *Cerr2);

/**
 * Complex the scalar C0(s1,s12,s2,m02,m22,m22).
 * @param[out] TN array of the coefficients
 * @param[out] TNuv array of the coefficients of UV-singular
 * \f$1/\epsilon_{UV}\f$ poles
 * @param[in] sarr array of momentum invariants of length \f$\binom{N}{2}\f$
 * @param[in] m2arr array of squared mass of length `N`-1
 * @param[in] N number of loop propagators
 * @param[in] rmax maximal rank
 */
void collier_coeffs_tn(dcplx *TN, dcplx *TNuv, dcplx *sarr, dcplx *m2arr,
                       int *n, int *rmax);
}

#endif // BLACKTHORN_COLLIER_COLLIER_INTERFACE_H
