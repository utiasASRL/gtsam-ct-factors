#ifndef CEPHES_H
#define CEPHES_H

#ifdef __cplusplus
extern "C" {
#endif

double Gamma(double x);
double lgam(double x);
double lgam_sgn(double x, int *sign);
double gammasgn(double x);

double igamc(double a, double x);
double igam(double a, double x);
double igam_fac(double a, double x);
double igamci(double a, double q);
double igami(double a, double p);

double log1pmx(double x);
double cosm1(double x);
double lgam1p(double x);

double zeta(double x, double q);
double zetac(double x);

double lanczos_sum_expg_scaled(double x);

#ifdef __cplusplus
}
#endif

#endif /* CEPHES_H */
