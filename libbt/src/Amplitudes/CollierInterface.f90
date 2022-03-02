module collier_interface
   use iso_c_binding
   use collier
   use combinatorics
   use collier_global
   use collier_init
   use collier_aux
   use reductionTN

   implicit none

contains

   ! =========================================================================
   ! ---- Initialization -----------------------------------------------------
   ! =========================================================================

   !> \breif Initialize the Collier library.
   !!
   !! \param[in] nmax maximal number of loop propagators.
   !! \param[in] rmax maximal rank of loop integrals.
   subroutine collier_initialize(nmax, rmax) bind(c, name='collier_initialize')
      integer, intent(in) :: nmax, rmax
      call Init_cll(nmax, rmax, "")
   end subroutine collier_initialize

   !> \breif Reinitialize Collier's error and accurarcy flags.
   subroutine collier_initialize_event() bind(c, name='collier_initialize_event')
      call InitEvent_cll
   end subroutine collier_initialize_event

   ! =========================================================================
   ! ---- Scalar A0, B0, C0 --------------------------------------------------
   ! =========================================================================

   !> \breif Compute the coefficients \f$T^1\f$ of the Lorentz-covariant
   !! decompostion of the scalar 1-point integral.
   !!
   !! \param[out] A0 result of the scalar integral.
   !! \param[in] m02 squared mass
   subroutine collier_scalarA0(A0, m02) bind(c, name='collier_scalarA0')
      double complex, intent(in) :: m02
      double complex, intent(out) :: A0
      call A0_cll(A0, m02)
   end subroutine collier_scalarA0

   !> \breif Compute the coefficients \f$T^2\f$ of the Lorentz-covariant
   !! decompostion of the scalar 2-point integral.
   !!
   !! \param[out] B0 result of the scalar integral.
   !! \param[in] s momentum invariant.
   !! \param[in] m02,m12 squared masses.
   subroutine collier_scalarB0(B0, s, m02, m12) bind(c, name='collier_scalarB0')
      double complex, intent(in) :: s, m02, m12
      double complex, intent(out) :: B0
      call B0_cll(B0, s, m02, m12)
   end subroutine collier_scalarB0

   !> \breif Compute the coefficients \f$T^3\f$ of the Lorentz-covariant
   !! decompostion of the scalar 3-point integral.
   !!
   !! \param[out] C0 result of the scalar integral.
   !! \param[in] s1,s12,s2 momentum invariants.
   !! \param[in] m02,m12,m22 squared masses.
   subroutine collier_scalarC0(C0, s1, s12, s2, m02, m12, m22) bind(c, name='collier_scalarC0')
      double complex, intent(in) :: s1, s12, s2
      double complex, intent(in) :: m02, m12, m22
      double complex, intent(out) :: C0
      call C0_cll(C0, s1, s12, s2, m02, m12, m22)
   end subroutine collier_scalarC0

   ! =========================================================================
   ! ---- A, B, C ------------------------------------------------------------
   ! =========================================================================

   !> \breif Compute the coefficients \f$T^1_{i_1,\dots,i_P}\f$ of the Lorentz-covariant
   !! decompostion of the 1-point integrals \f$T^{1,P}\f$.
   !!
   !! \param[out] A 1D-array of the coefficients of shape (rmax/2+1)
   !! \param[out] Auv 1D-array of the coefficients of UV-singular \f$1/\epsilon_{UV}\f$ poles (same shape as `A`)
   !! \param[in] m02 squared mass
   !! \param[in] rmax maximal rank
   subroutine collier_coeffs_a(A, Auv, m02, rmax) bind(c, name='collier_coeffs_a')
      integer, intent(in) :: rmax
      double complex, intent(in) :: m02
      double complex, intent(out) :: Auv(0:rmax/2), A(0:rmax/2)
      ! double precision, optional, intent(out) :: Aerr(0:rmax)
      call A_cll(A, Auv, m02, rmax)
   end subroutine collier_coeffs_a

   !> \breif Compute the coefficients \f$T^2_{i_1,\dots,i_P}\f$ of the Lorentz-covariant
   !! decompostion of the 2-point integrals \f$T^{2,P}\f$.
   !!
   !! This is equivalent to calling:
   !! \code{.f90}
   !! N = 2
   !! sarr(1:1) = (/s/)
   !! m2arr(0:1) = (/m02,m12/)
   !! collier_coeffs_tn(TN,TNuv,sarr,m2arr,N,rank)
   !! \endcode
   !! where `TN` and `TNuv` are 1D arrays of the appropriate length.
   !!
   !! \param[out] B 2D-array of the coefficients of shape (rmax/2+1, rmax+1)
   !! \param[out] Buv 2D-array of the coefficients of UV-singular \f$1/\epsilon_{UV}\f$ poles (same shape as B)
   !! \param[in] s1,s12,s2 momentum invariants
   !! \param[in] m02,m12,m22 squared masses
   !! \param[in] rmax maximal rank
   subroutine collier_coeffs_b(B, Buv, s, m02, m12, rmax, Berr) bind(c, name='collier_coeffs_b')
      integer, intent(in) :: rmax
      double complex, intent(in) :: s, m02, m12
      double complex, intent(out) :: Buv(1:NCoefs(rmax, 2)), B(1:NCoefs(rmax, 2))
      double precision, optional, intent(out) :: Berr(0:rmax)
      call B_cll(B, Buv, s, m02, m12, rmax)
   end subroutine collier_coeffs_b

   !> \breif Compute the coefficients \f$T^3_{i_1,\dots,i_P}\f$ of the Lorentz-covariant
   !! decompostion of the 3-point integrals \f$T^{3,P}\f$.
   !!
   !! This is equivalent to calling:
   !! \code{.f90}
   !! N = 3
   !! sarr(1:3) = (/s1,s12,s2/)
   !! m2arr(0:2) = (/m02,m12,m22/)
   !! collier_coeffs_tn(TN,TNuv,sarr,m2arr,N,rank)
   !! \endcode
   !! where `TN` and `TNuv` are 1D arrays of the appropriate length.
   !!
   !! \param[out] C 3D-array of the coefficients of shape (rmax/2+1, rmax+1, rmax+1)
   !! \param[out] Cuv 3D-array of the coefficients of UV-singular \f$1/\epsilon_{UV}\f$ poles (same shape as `C`)
   !! \param[in] s1,s12,s2 momentum invariants
   !! \param[in] m02,m12,m22 squared masses
   !! \param[in] rmax maximal rank
   subroutine collier_coeffs_c(C, Cuv, s1, s12, s2, m02, m12, m22, rmax) bind(c, name='collier_coeffs_c')
      integer, intent(in) :: rmax
      double complex, intent(in) :: s1, s12, s2, m02, m12, m22
      !double precision, optional, intent(out) :: Cerr(0:rmax), Cerr2(0:rmax)
      double complex, intent(out) :: Cuv(NCoefs(rmax, 3)), C(NCoefs(rmax, 3))
      ! call C_cll(C, Cuv, s1, s12, s2, m02, m12, m22, rmax, Cerr)
      call C_cll(C, Cuv, s1, s12, s2, m02, m12, m22, rmax)
   end subroutine collier_coeffs_c

   ! =========================================================================
   ! ---- General ------------------------------------------------------------
   ! =========================================================================

   !> \breif Compute the number of tensor coefficients of an `n`-pt integral of
   !! rank `r`.
   !!
   !! \param[in] n number of propagators.
   !! \param[in] r rank.
   function collier_get_nc(n, r) result(nc) bind(c, name='collier_get_nc')
      integer :: nc
      integer, intent(in) :: n, r
      nc = GetNc_cll(n, r)
   end function collier_get_nc

   !> \breif Compute the coefficients \f$T^N_{i_1,\dots,i_P}\f$ of the Lorentz-covariant
   !! decompostion of the tensor intergrals \f$T^{N,P}\f$.
   !!
   !! \param[out] TN array of the coefficients
   !! \param[out] TNuv array of the coefficients of UV-singular \f$1/\epsilon_{UV}\f$ poles
   !! \param[in] sarr array of momentum invariants of length \f$\binom{N}{2}\f$
   !! \param[in] m2arr array of squared mass of length `N`-1
   !! \param[in] N number of loop propagators
   !! \param[in] rmax maximal rank
   subroutine collier_coeffs_tn(TN, TNuv, sarr, m2arr, N, rmax) bind(c, name='collier_coeffs_tn')
      integer, intent(in) :: N, rmax
      double complex, intent(in) :: sarr(BinomTable(2, N)), m2arr(0:N - 1)
      double complex, intent(out) :: TN(NCoefs(rmax, N))
      double complex, intent(out) :: TNuv(NCoefs(rmax, N))
      call TN_cll(TN, TNuv, sarr, m2arr, N, rmax)
   end subroutine collier_coeffs_tn

end module collier_interface
