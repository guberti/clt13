#ifndef __CLT13_H__
#define __CLT13_H__

#include <gmp.h>

extern int g_verbose;

typedef struct crt_tree {
    unsigned long n, n2;
    mpz_t mod;
    mpz_t crt_left;
    mpz_t crt_right;
    struct crt_tree *left;
    struct crt_tree *right;
} crt_tree;

typedef struct {
    gmp_randstate_t rng;
    unsigned long n;
    unsigned long nzs;
    unsigned long rho;
    unsigned long nu;
    mpz_t x0;
    mpz_t pzt;
    mpz_t *gs;
    mpz_t *zinvs;
    crt_tree *crt;
} clt_state;

int clt_state_init(
    clt_state *s,
    unsigned long kappa,
    unsigned long lambda,
    unsigned long nzs,
    const int *pows
);

void clt_state_clear(clt_state *s);

typedef struct {
    mpz_t x0;
    mpz_t pzt;
    unsigned long nu;
} clt_public_parameters;

void clt_pp_init(clt_public_parameters *pp, clt_state *mmap);
void clt_pp_init_from_file(clt_public_parameters *pp, const char *dir);
void clt_pp_clear(clt_public_parameters *pp);

void write_public_params(const clt_public_parameters *pp, const char *dir);

void clt_encode(mpz_t rop, clt_state *s, size_t nins, const mpz_t *ins, const int *pows);
int clt_is_zero(clt_public_parameters *pp, const mpz_t c);

#endif
