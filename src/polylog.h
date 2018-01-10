#pragma once

#include "clt13.h"

#include <aesrand/aesrand.h>
#include <stdbool.h>
#include <stdlib.h>

#pragma GCC visibility push(hidden)

typedef struct {
    size_t level;
    size_t wordsize;
    size_t k;                   /* = η_ℓ / log2(wordsize) */
    mpz_t *ys;                  /* [Θ] */
    clt_elem_t ***sigmas;       /* [Θ][ℓ] */
} switch_params_t;

typedef struct {
    size_t n;
    size_t b;
    size_t nlevels;
    size_t theta;
    size_t *etas;               /* [nlevels] */
    mpz_t **ps;                 /* [nlevels][n] */
    mpz_t **crt_coeffs;         /* [nlevels][n] */
    mpz_t *x0s;                 /* [nlevels] */
    mpz_t **phats;              /* [nlevels][n] */
    switch_params_t *switchs;   /* [nmuls] */
} polylog_params_t;

polylog_params_t *
polylog_params_new(size_t n, size_t eta, size_t b, size_t nlevels, aes_randstate_t *rngs, bool verbose);

void
polylog_params_free(polylog_params_t *params);

int
polylog_encode(clt_elem_t *rop, const clt_state_t *s, mpz_t *xs, const int *ix, size_t level);

#pragma GCC visibility pop
