#include "polylog.h"

#include "_clt13.h"
#include "utils.h"

#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>

void
polylog_state_free(polylog_state_t *state)
{
    if (state == NULL)
        return;
    if (state->etas)
        free(state->etas);
    for (size_t i = 0; i < state->nlevels; ++i) {
        for (size_t j = 0; j < state->n; ++j) {
            mpz_clear(state->ps[i][j]);
            mpz_clear(state->phats[i][j]);
        }
        free(state->ps[i]);
        free(state->phats[i]);
    }
    free(state->ps);
    free(state->phats);
    free(state);
}

polylog_state_t *
polylog_state_new(size_t n, size_t eta, size_t b, size_t nlevels, aes_randstate_t *rngs, bool verbose)
{
    polylog_state_t *state;
    int count;

    if ((state = calloc(1, sizeof state[0])) == NULL)
        return NULL;
    state->n = n;
    state->b = b;
    state->nlevels = nlevels;
    if (verbose) {
        fprintf(stderr, "polylog\n");
        fprintf(stderr, "  n: %lu\n", state->n);
        fprintf(stderr, "  b: %lu\n", state->b);
        fprintf(stderr, "  nlevels: %lu\n", state->nlevels);
        fprintf(stderr, "  ηs:");
    }
    state->etas = calloc(nlevels, sizeof state->etas[0]);
    for (size_t i = 0; i < nlevels; ++i) {
        if (i * 2 * b > eta) {
            fprintf(stderr, "error: η - ℓ·2b < 0\n");
            goto error;
        }
        state->etas[i] = eta - i * 2 * b;
        if (verbose)
            fprintf(stderr, " %lu", state->etas[i]);
    }
    if (verbose) {
        fprintf(stderr, "\n  Generating p_i's:\n");
    }
    state->ps = calloc(nlevels, sizeof state->ps[0]);
    for (size_t i = 0; i < nlevels; ++i) {
        state->ps[i] = calloc(n, sizeof state->ps[i][0]);
        if (verbose) {
            count = 0;
            print_progress(count, n);
        }
        for (size_t j = 0; j < n; ++j) {
            mpz_init(state->ps[i][j]);
            mpz_prime(state->ps[i][j], rngs[j], state->etas[i]);
            if (verbose)
                print_progress(++count, n);
        }
        if (verbose)
            fprintf(stderr, "\n");
    }
    if (verbose) {
        fprintf(stderr, "  Generating x0s:\n");
    }
    state->x0s = calloc(nlevels, sizeof state->x0s[0]);
    state->crt_coeffs = calloc(nlevels, sizeof state->crt_coeffs[0]);
    for (size_t i = 0; i < nlevels; ++i) {
        mpz_init(state->x0s[i]);
        product(state->x0s[i], state->ps[i], n, verbose);
        state->crt_coeffs[i] = mpz_vector_new(n);
        crt_coeffs(state->crt_coeffs[i], state->ps[i], n, state->x0s[i], verbose);
    }
    if (verbose) {
        fprintf(stderr, "  Generating p hat's:\n");
        count = 0;
        print_progress(count, nlevels);
    }
    state->phats = calloc(nlevels, sizeof state->phats[0]);
    for (size_t i = 0; i < nlevels; ++i) {
        state->phats[i] = calloc(n, sizeof state->phats[i][0]);
        for (size_t j = 0; j < n; ++j) {
            mpz_init(state->phats[i][j]);
            mpz_div(state->phats[i][j], state->x0s[i], state->ps[i][j]);
        }
        if (verbose)
            print_progress(++count, nlevels);
    }
    if (verbose)
        fprintf(stderr, "\n");
    return state;
error:
    polylog_state_free(state);
    return NULL;
}

switch_state_t *
switch_state_new(clt_state_t *s, size_t wordsize, size_t level)
{
    polylog_state_t *pstate = s->pstate;
    switch_state_t *state;
    mpz_t K;

    if (level > pstate->nlevels - 1) {
        fprintf(stderr, "error: level too large (%lu > %lu)\n", level, pstate->nlevels - 1);
        return NULL;
    }

    if ((state = calloc(1, sizeof state[0])) == NULL)
        return NULL;
    state->level = level;
    state->wordsize = wordsize;
    /* k = η_ℓ / log2(wordsize) */
    state->k = (int) ceil(pstate->etas[level] / log2(wordsize));

    /* K = Π · (wordsize)ᵏ */
    mpz_init(K);
    mpz_ui_pow_ui(K, wordsize, state->k);
    mpz_mul(K, K, pstate->x0s[level]);

    state->ys = calloc(pstate->theta, sizeof state->ys[0]);
    /* Sample y_{n+1}, ..., y_Θ ∈ [0, K) */
    for (size_t i = s->n; i < pstate->theta; ++i) {
        mpz_init(state->ys[i]);
        mpz_urandomm_aes(state->ys[i], s->rngs[0], K);
    }
    mpz_t *fs;
    fs = calloc(s->n, sizeof fs[0]);
    for (size_t i = 0; i < s->n; ++i) {
        mpz_init(fs[i]);
        mpz_invert(fs[i], s->gs[i], pstate->ps[level][i]);
        mpz_mul(fs[i], fs[i], pstate->ps[level + 1][i]);
        mpz_fdiv_q(fs[i], fs[i], pstate->ps[level][i]);
        mpz_mul(fs[i], fs[i], s->gs[i]);
        mpz_mod_near(fs[i], fs[i], pstate->ps[level + 1][i]);
        mpz_invert(fs[i], fs[i], s->gs[i]);
    }
    mpz_t **ss;
    ss = calloc(s->n, sizeof ss[0]);
    for (size_t i = 0; i < s->n; ++i) {
        ss[i] = calloc(pstate->theta, sizeof ss[i][0]);
        for (size_t j = 0; j < s->n; ++j)
            mpz_init_set_ui(ss[i][j], 0);
        for (size_t j = s->n; j < pstate->theta; ++j) {
            mpz_init(ss[i][j]);
            mpz_urandomb_aes(ss[i][j], s->rngs[0], wordsize);
        }
        mpz_set_ui(ss[i][i], 1);

        mpz_t tmp;
        mpz_init(tmp);
        mpz_init(state->ys[i]);
        mpz_invert(tmp, fs[i], s->gs[i]);
        mpz_invert(state->ys[i], s->gs[i], pstate->ps[level][i]);
        mpz_mul(state->ys[i], state->ys[i], tmp);
        mpz_mod_near(state->ys[i], state->ys[i], pstate->ps[level][i]);
        mpz_mul(state->ys[i], state->ys[i], K);
        mpz_div(state->ys[i], state->ys[i], pstate->ps[level][i]);

        mpz_set_ui(tmp, 0);
        for (size_t j = s->n; j < pstate->theta; ++j) {
            mpz_t m;
            mpz_init(m);
            mpz_mul(m, state->ys[j], ss[i][j]);
            mpz_add(tmp, tmp, m);
            mpz_clear(m);
        }
        mpz_sub(state->ys[i], state->ys[i], tmp);
        mpz_mod_near(state->ys[i], state->ys[i], K);
    }

    state->sigmas = calloc(pstate->theta, sizeof state->sigmas[0]);
    for (size_t t = 0; t < pstate->theta; ++t) {
        state->sigmas[t] = calloc(state->k, sizeof state->sigmas[t][0]);
        /* for (size_t j = 0; j < pstate->k, ++j) { */
        /*     mpz_t tmp; */
        /*     mpz_init(tmp); */
        /*     mpz_invert(tmp, pows[j][k], pstate->ps[level+1][j]) */
        /* } */
    }

    return state;
}

int
polylog_encode(clt_elem_t *rop, const clt_state_t *s, size_t n, mpz_t *xs, const int *ix, size_t level)
{
    polylog_state_t *pstate = s->pstate;
    mpz_t b_mpz;
    mpz_init_set_ui(b_mpz, pstate->b);
    mpz_set_ui(rop->elem, 0);
    for (size_t i = 0; i < n; ++i) {
        mpz_t tmp;
        mpz_init(tmp);
        mpz_urandomb_aes(tmp, s->rngs[i], pstate->b);
        mpz_mod_near(tmp, tmp, b_mpz);
        mpz_mul(tmp, tmp, s->gs[i]);
        mpz_add(tmp, tmp, xs[i]); /* XXX */
        mpz_mul(tmp, tmp, pstate->crt_coeffs[level][i]);
        mpz_add(rop->elem, rop->elem, tmp);
        mpz_clear(tmp);
    }
    rop->level = level;
    if (ix) {
        mpz_t tmp;
        mpz_init(tmp);
        /* multiply by appropriate zinvs */
        rop->ix = calloc(s->nzs, sizeof rop->ix[0]);
        for (unsigned long i = 0; i < s->nzs; ++i) {
            if (ix[i] <= 0)
                continue;
            rop->ix[i] = ix[i];
            mpz_powm_ui(tmp, s->zinvs[i], ix[i], pstate->x0s[level]);
            mpz_mul_mod(rop->elem, rop->elem, tmp, pstate->x0s[level]);
        }
        mpz_clear(tmp);
    }
    return CLT_OK;
}

int
polylog_elem_mul(clt_elem_t *rop, const clt_pp_t *pp, const clt_elem_t *a, const clt_elem_t *b, size_t level)
{
    if (a->level != b->level) {
        fprintf(stderr, "error: levels unequal (%lu ≠ %lu), unable to multiply",
                a->level, b->level);
        return CLT_ERR;
    }
    mpz_mul(rop->elem, a->elem, b->elem);
    mpz_mod(rop->elem, rop->elem, pp->pstate->x0s[level]);
    rop->level = a->level + 1;
    return CLT_OK;
}

static void
quotient(mpz_t rop, mpz_t a, mpz_t b)
{
    mpz_mod_near(rop, a, b);
    mpz_sub(rop, a, rop);
    mpz_div(rop, rop, b);
}

static mpz_t *
worddecomp(mpz_t x, size_t wordsize_, size_t k)
{
    mpz_t tmp, wordsize;
    mpz_t *decomp;

    mpz_init_set_ui(wordsize, wordsize_);
    mpz_init_set(tmp, x);
    decomp = calloc(k, sizeof decomp[0]);
    for (size_t i = 0; i < k; ++i) {
        mpz_init(decomp[i]);
        mpz_mod_near(decomp[i], tmp, wordsize);
        quotient(tmp, tmp, wordsize);
    }
    return decomp;
}

int
polylog_switch(clt_elem_t *rop, const clt_state_t *s, clt_elem_t *x, switch_state_t *sstate)
{
    const polylog_state_t *pstate = s->pstate;
    mpz_t *pi, *pip, ct, twok;

    if (rop == NULL)
        return CLT_ERR;
    if (sstate->level != x->level) {
        fprintf(stderr, "error: using switch parameter with a mismatched level\n");
        return CLT_ERR;
    }

    pi = &pstate->x0s[sstate->level];
    pip = &pstate->x0s[sstate->level + 1];

    mpz_inits(ct, twok, NULL);
    mpz_ui_pow_ui(twok, 2, sstate->k);
    mpz_set_ui(rop->elem, 0);
    for (size_t t = 0; t < pstate->theta; ++t) {
        mpz_t *cts;
        /* Compute cₜ = (x · yₜ) / Π mod (wordsize)^ℓ */
        mpz_mul(ct, x->elem, sstate->ys[t]);
        mpz_div(ct, ct, *pi);
        mpz_mod(ct, ct, twok);
        cts = worddecomp(ct, sstate->wordsize, sstate->k);
        for (size_t i = 0; i < sstate->k; ++i) {
            mpz_t tmp;
            mpz_init(tmp);
            mpz_mul(tmp, cts[i], sstate->sigmas[t][i]->elem);
            mpz_add(rop->elem, rop->elem, tmp);
            mpz_mod(rop->elem, rop->elem, *pip);
            mpz_clear(tmp);
        }
    }
    rop->level = x->level + 1;
    return CLT_OK;
}
