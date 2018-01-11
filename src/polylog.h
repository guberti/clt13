#pragma once

#include "clt13.h"

#include <aesrand/aesrand.h>
#include <stdbool.h>
#include <stdlib.h>

#pragma GCC visibility push(hidden)

typedef struct {
    size_t level;
    size_t wordsize;            /* must be a power of two */
    size_t k;                   /* = η_ℓ / log2(wordsize) */
    mpz_t *ys;                  /* [Θ] */
    clt_elem_t ***sigmas;       /* [Θ][ℓ] */
} switch_state_t;

typedef struct {
    size_t n;
    size_t b;
    size_t nlevels;
    size_t theta;
    mpz_t **zs;                 /* [nlevels][n] */
    mpz_t **zinvs;              /* [nlevels][n] */
    mpz_t **ps;                 /* [nlevels][n] */
    mpz_t **crt_coeffs;         /* [nlevels][n] */
    mpz_t *x0s;                 /* [nlevels] */
    switch_state_t **switches;  /* [nmuls] */
    size_t nmuls;
} polylog_state_t;

typedef struct {
    size_t theta;
    mpz_t *x0s;
    switch_state_t **switches;
    size_t nmuls;
    bool local;
} polylog_pp_t;

polylog_state_t *
polylog_state_new(clt_state_t *s, size_t eta, size_t theta, size_t b, size_t wordsize,
                  size_t nlevels, const switch_params_t *sparams, size_t nmuls);
void polylog_state_free(polylog_state_t *state);

polylog_pp_t * polylog_pp_new(const polylog_state_t *s);
void polylog_pp_free(polylog_pp_t *pp);

int polylog_encode(clt_elem_t *rop, const clt_state_t *s, size_t n, mpz_t *xs, const int *ix, size_t idx);

int polylog_switch(clt_elem_t *rop, const clt_pp_t *s, const clt_elem_t *x,
                   const switch_state_t *sstate, bool verbose);


#pragma GCC visibility pop
