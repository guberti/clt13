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
} switch_state_t;

typedef struct {
    size_t n;
    size_t b;
    size_t nlevels;
    size_t theta;
    mpz_t **ps;                 /* [nlevels][n] */
    mpz_t **crt_coeffs;         /* [nlevels][n] */
    mpz_t *x0s;                 /* [nlevels] */
    switch_state_t **switches;  /* [nmuls] */
    size_t nmuls;
} polylog_state_t;

typedef struct {
    mpz_t *x0s;
    switch_state_t **switches;
    size_t nmuls;
} polylog_pp_t;

polylog_state_t *
polylog_state_new(clt_state_t *s, size_t eta, size_t b, size_t wordsize,
                  size_t nlevels, size_t *levels, size_t nops);
void polylog_state_free(polylog_state_t *state);
int polylog_encode(clt_elem_t *rop, const clt_state_t *s, size_t n, mpz_t *xs, const int *ix, size_t idx);

int
polylog_switch(clt_elem_t *rop, const clt_state_t *s, const clt_elem_t *x, const switch_state_t *sstate, bool verbose);


#pragma GCC visibility pop
