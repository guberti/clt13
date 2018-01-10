#include "polylog.h"

#include "_clt13.h"
#include "utils.h"

#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>

void
level_params_free(level_params_t *params)
{
    if (params == NULL)
        return;
    if (params->etas)
        free(params->etas);
    for (size_t i = 0; i < params->nlevels; ++i) {
        for (size_t j = 0; j < params->n; ++j) {
            mpz_clear(params->ps[i][j]);
            mpz_clear(params->phats[i][j]);
        }
        free(params->ps[i]);
        free(params->phats[i]);
    }
    free(params->ps);
    free(params->phats);
    free(params);
}

level_params_t *
level_params_new(size_t n, size_t eta, size_t b, size_t nlevels, aes_randstate_t *rngs, bool verbose)
{
    level_params_t *params;

    if ((params = calloc(1, sizeof params[0])) == NULL)
        return NULL;
    params->n = n;
    params->b = b;
    params->nlevels = nlevels;
    params->etas = calloc(nlevels, sizeof params->etas[0]);
    for (size_t i = 0; i < nlevels; ++i) {
        if (i * 2 * b > eta) {
            fprintf(stderr, "error: η - ℓ·2b < 0\n");
            goto error;
        }
        params->etas[i] = eta - i * 2 * b;
    }
    params->ps = calloc(nlevels, sizeof params->ps[0]);
    for (size_t i = 0; i < nlevels; ++i) {
        int count = 0;
        params->ps[i] = calloc(n, sizeof params->ps[i][0]);
        print_progress(count, n);
        for (size_t j = 0; j < n; ++j) {
            mpz_init(params->ps[i][j]);
            mpz_prime(params->ps[i][j], rngs[j], params->etas[i]);
            if (verbose)
                print_progress(++count, n);
        }
    }
    params->x0s = calloc(nlevels, sizeof params->x0s[0]);
    for (size_t i = 0; i < nlevels; ++i) {
        mpz_init(params->x0s[i]);
        product(params->x0s[i], params->ps[i], n, verbose);
    }
    params->phats = calloc(nlevels, sizeof params->phats[0]);
    for (size_t i = 0; i < nlevels; ++i) {
        params->phats[i] = calloc(n, sizeof params->phats[i][0]);
        for (size_t j = 0; j < n; ++j) {
            mpz_init(params->phats[i][j]);
            mpz_div(params->phats[i][j], params->x0s[i], params->ps[i][j]);
        }
    }
    return params;
error:
    level_params_free(params);
    return NULL;
}

switch_params_t *
switch_params_new(clt_state_t *s, size_t wordsize, size_t level)
{
    level_params_t *lparams = s->lparams;
    switch_params_t *params;
    mpz_t K;

    if (level > lparams->nlevels - 1) {
        fprintf(stderr, "error: level too large (%lu > %lu)\n", level, lparams->nlevels - 1);
        return NULL;
    }

    if ((params = calloc(1, sizeof params[0])) == NULL)
        return NULL;
    /* k = η_{ℓ+1} */
    params->k = lparams->etas[level + 1];
    mpz_init_set(params->pi, lparams->x0s[level]);
    mpz_init_set(params->pip, lparams->x0s[level + 1]);

    /* K = Π · (wordsize)ᵏ */
    mpz_init(K);
    mpz_ui_pow_ui(K, wordsize, params->k);
    mpz_mul(K, K, params->pi);

    params->ys = calloc(lparams->theta, sizeof params->ys[0]);
    /* Sample y_{n+1}, ..., y_Θ ∈ [0, K) */
    for (size_t i = s->n; i < lparams->theta; ++i) {
        mpz_init(params->ys[i]);
        mpz_urandomm_aes(params->ys[i], s->rngs[0], K);
    }
    mpz_t *fs;
    fs = calloc(s->n, sizeof fs[0]);
    for (size_t i = 0; i < s->n; ++i) {
        mpz_init(fs[i]);
        mpz_invert(fs[i], s->gs[i], lparams->ps[level][i]);
        mpz_mul(fs[i], fs[i], lparams->ps[level + 1][i]);
        mpz_fdiv_q(fs[i], fs[i], lparams->ps[level][i]);
        mpz_mul(fs[i], fs[i], s->gs[i]);
        mpz_mod_near(fs[i], fs[i], lparams->ps[level + 1][i]);
        mpz_invert(fs[i], fs[i], s->gs[i]);
    }
    mpz_t **ss;
    ss = calloc(s->n, sizeof ss[0]);
    for (size_t i = 0; i < s->n; ++i) {
        ss[i] = calloc(s->lparams->theta, sizeof ss[i][0]);
        for (size_t j = 0; j < s->n; ++j)
            mpz_init_set_ui(ss[i][j], 0);
        for (size_t j = s->n; j < s->lparams->theta; ++j) {
            mpz_init(ss[i][j]);
            mpz_urandomb_aes(ss[i][j], s->rngs[0], wordsize);
        }
        mpz_set_ui(ss[i][i], 1);

        mpz_t tmp;
        mpz_init(tmp);
        mpz_init(params->ys[i]);
        mpz_invert(tmp, fs[i], s->gs[i]);
        mpz_invert(params->ys[i], s->gs[i], lparams->ps[level][i]);
        mpz_mul(params->ys[i], params->ys[i], tmp);
        mpz_mod_near(params->ys[i], params->ys[i], lparams->ps[level][i]);
        mpz_mul(params->ys[i], params->ys[i], K);
        mpz_div(params->ys[i], params->ys[i], lparams->ps[level][i]);

        mpz_set_ui(tmp, 0);
        for (size_t j = s->n; j < lparams->theta; ++j) {
            mpz_t m;
            mpz_init(m);
            mpz_mul(m, params->ys[j], ss[i][j]);
            mpz_add(tmp, tmp, m);
            mpz_clear(m);
        }
        mpz_sub(params->ys[i], params->ys[i], tmp);
        mpz_mod_near(params->ys[i], params->ys[i], K);
    }

    params->sigmas = calloc(lparams->theta, sizeof params->sigmas[0]);
    for (size_t t = 0; t < lparams->theta; ++t) {
        params->sigmas[t] = calloc(params->k, sizeof params->sigmas[t][0]);
        /* for (size_t j = 0; j < lparams->k, ++j) { */
        /*     mpz_t tmp; */
        /*     mpz_init(tmp); */
        /*     mpz_invert(tmp, pows[j][k], lparams->ps[level+1][j]) */
        /* } */
    }

    return params;
}

int
polylog_encode(clt_elem_t *rop, const clt_state_t *s, mpz_t *xs, const int *ix, size_t level)
{
    mpz_set_ui(rop->elem, 0);
    for (size_t i = 0; i < s->n; ++i) {
        mpz_t tmp;
        mpz_init(tmp);
        mpz_random_(tmp, s->rngs[i], s->lparams->b); /* XXX center around 0 */
        mpz_mul(tmp, tmp, s->gs[i]);
        mpz_add(tmp, tmp, xs[i]); /* XXX */
        mpz_mul(tmp, tmp, s->lparams->crt_coeffs[level][i]);
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
            mpz_powm_ui(tmp, s->zinvs[i], ix[i], s->lparams->x0s[level]);
            mpz_mul_mod(rop->elem, rop->elem, tmp, s->lparams->x0s[level]);
        }
        mpz_clear(tmp);
    }
    return CLT_OK;
}
