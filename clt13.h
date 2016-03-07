#ifndef __CLT13_H__
#define __CLT13_H__

#include <gmp.h>

extern int g_verbose;

typedef unsigned long ulong;

typedef struct crt_tree {
    ulong n, n2;
    mpz_t mod;
    mpz_t crt_left;
    mpz_t crt_right;
    struct crt_tree *left;
    struct crt_tree *right;
} crt_tree;

// state

typedef struct {
    gmp_randstate_t rng;
    ulong n;
    ulong nzs;
    ulong rho;
    ulong nu;
    mpz_t x0;
    mpz_t pzt;
    mpz_t *gs;
    mpz_t *ps;
    mpz_t *zinvs;
    crt_tree *crt;
} clt_state;

void clt_state_init(clt_state *s, ulong kappa, ulong lambda, ulong nzs, const int *pows);
void clt_state_clear(clt_state *s);
void clt_state_read(clt_state *s, const char *dir);
void clt_state_save(const clt_state *s, const char *dir);

// public parameters

typedef struct {
    mpz_t x0;
    mpz_t pzt;
    ulong nu;
} clt_pp;

void clt_pp_init(clt_pp *pp, clt_state *mmap);
void clt_pp_clear(clt_pp *pp);
void clt_pp_read(clt_pp *pp, const char *dir);
void clt_pp_save(const clt_pp *pp, const char *dir);

// encodings

void clt_encode(mpz_t rop, clt_state *s, size_t nins, const mpz_t *ins, const int *pows);
int clt_is_zero(clt_pp *pp, const mpz_t c);

#endif
