#pragma once

#include "clt13.h"

#include <aesrand/aesrand.h>
#include <stdbool.h>
#include <stdlib.h>

#pragma GCC visibility push(hidden)

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
} level_params_t;

typedef struct {
    mpz_t pi;
    mpz_t pip;
    size_t k;
    mpz_t *ys;                  /* [Θ] */
    clt_elem_t **sigmas;        /* [Θ][k] */
} switch_params_t;

level_params_t *
level_params_new(size_t n, size_t eta, size_t b, size_t nlevels, aes_randstate_t *rngs, bool verbose);

void
level_params_free(level_params_t *params);

int
polylog_encode(clt_elem_t *rop, const clt_state_t *s, mpz_t *xs, const int *ix, size_t level);

#pragma GCC visibility pop
