#pragma once

#include "clt13.h"
#include "crt_tree.h"
#include "polylog.h"

struct clt_elem_t {
    mpz_t elem;
    int *ix;
    size_t level;
};

struct clt_state_t {
    size_t n;                   /* number of slots */
    size_t nzs;                 /* number of z's in the index set */
    size_t rho;                 /* bitsize of randomness */
    size_t nu;                  /* number of most-significant-bits to extract */
    mpz_t *gs;                  /* plaintext moduli */
    aes_randstate_t *rngs;      /* random number generators (one per slot) */

    mpz_t x0;
    level_params_t *lparams;
    mpz_t pzt;                  /* zero testing parameter */
    mpz_t *zinvs;               /* z inverses */
    union {
        crt_tree *crt;
        mpz_t *crt_coeffs;
    };
    size_t flags;
};

struct clt_pp_t {
    mpz_t x0;
    mpz_t pzt;                  /* zero testing parameter */
    size_t nu;                  /* number of most-significant-bits to extract */
    bool is_polylog;            /* are we using polylog CLT? */
};
