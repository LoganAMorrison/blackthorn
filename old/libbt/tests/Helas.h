#ifndef HELAS_INTERFACE_H
#define HELAS_INTERFACE_H

#ifdef __cplusplus
#include <cmath>
#include <complex>
#define cmplx std::complex<double>
#else
#include <complex.h>
#include <math.h>
#define cmplx double _Complex
#endif

#ifdef __cplusplus
extern "C" {
#endif

void sxxxxx_(double[], int *, cmplx[]);
void ixxxxx_(double[], double *, int *, int *, cmplx[]);
void oxxxxx_(double[], double *, int *, int *, cmplx[]);
void vxxxxx_(double[], double *, int *, int *, cmplx[]);

void jioxxx_(cmplx[], cmplx[], cmplx[], double *, double *, cmplx[]);
void hioxxx_(cmplx[], cmplx[], cmplx[], double *, double *, cmplx[]);
void fvixxx_(cmplx[], cmplx[], cmplx[], double *, double *, cmplx[]);
void fvoxxx_(cmplx[], cmplx[], cmplx[], double *, double *, cmplx[]);

void iovxxx_(cmplx[], cmplx[], cmplx[], cmplx[], cmplx *);
void iosxxx_(cmplx[], cmplx[], cmplx[], cmplx[], cmplx *);

#ifdef __cplusplus
}
#endif

#undef cmplx
#endif // HELAS_INTERFACE_H
