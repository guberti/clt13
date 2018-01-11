#include "polylog.h"

#include "_clt13.h"
#include "utils.h"

#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>

polylog_pp_t *
polylog_pp_new(const polylog_state_t *s)
{
    polylog_pp_t *pp;

    if ((pp = calloc(1, sizeof pp[0])) == NULL)
        return NULL;
    pp->theta = s->theta;
    pp->x0s = s->x0s;
    pp->nmuls = s->nmuls;
    pp->switches = s->switches;
    pp->local = false;
    return pp;
}

void
polylog_pp_free(polylog_pp_t *pp)
{
    if (pp && pp->local) {
        /* XXX free stuff */
    }
}

static inline void
mpz_quotient(mpz_t rop, const mpz_t a, const mpz_t b)
{
    mpz_t tmp;
    mpz_init(tmp);
    mpz_mod_near(tmp, a, b);
    mpz_sub(rop, a, tmp);
    mpz_tdiv_q(rop, rop, b);
    mpz_clear(tmp);
}

static inline void
mpz_quotient_2exp(mpz_t rop, const mpz_t a, const size_t b)
{
    mpz_t tmp;
    mpz_init(tmp);
    mpz_mod_near_ui(tmp, a, 1 << b);
    mpz_sub(rop, a, tmp);
    mpz_tdiv_q_2exp(rop, rop, b);
    mpz_clear(tmp);
}

static switch_state_t *
switch_state_new(clt_state_t *s, const int *ix, size_t eta, size_t wordsize,
                 size_t level, bool verbose)
{
    (void) ix;
    polylog_state_t *pstate = s->pstate;
    switch_state_t *state;
    mpz_t wk, K, z1, z2;
    mpz_t **ss;
    double start, _start;

    if (level >= pstate->nlevels - 1) {
        fprintf(stderr, "error: level too large (%lu ≥ %lu)\n", level, pstate->nlevels - 1);
        return NULL;
    }

    if ((state = calloc(1, sizeof state[0])) == NULL)
        return NULL;
    state->level = level;
    state->wordsize = wordsize;
    /* k = η_ℓ / log2(wordsize) */
    state->k = (int) ceil(eta / log2(wordsize));

    if (verbose)
        fprintf(stderr, "  Generating switch state [%lu->%lu]\n", level, level+1);
    start = current_time();

    mpz_inits(wk, K, z1, z2, NULL);
    /* wk = (wordsize)ᵏ */
    mpz_ui_pow_ui(wk, wordsize, state->k);
    /* K = Π^(ℓ) · wk */
    mpz_mul(K, pstate->x0s[level], wk);
    /* compute zs */
    mpz_set_ui(z1, 1);
    mpz_set_ui(z2, 1);
    if (ix) {
        mpz_t tmp;
        mpz_init(tmp);
        for (size_t i = 0; i < s->nzs; ++i) {
            if (ix[i] <= 0) continue;
            mpz_powm_ui(tmp, pstate->zs[level][i], ix[i], pstate->x0s[level]);
            mpz_mul_mod_near(z1, z1, tmp, pstate->x0s[level]);
            mpz_powm_ui(tmp, pstate->zinvs[level + 1][i], ix[i], pstate->x0s[level + 1]);
            mpz_mul_mod_near(z2, z2, tmp, pstate->x0s[level + 1]);
        }
        mpz_clear(tmp);
    }
    _start = current_time();
    state->ys = mpz_vector_new(pstate->theta);
    /* Sample y_{n+1}, ..., y_Θ ∈ [0, K) */
    for (size_t i = s->n; i < pstate->theta; ++i) {
        mpz_urandomm_aes(state->ys[i], s->rngs[0], K);
    }
    if (verbose)
        fprintf(stderr, "    Generating random y values: [%.2fs]\n", current_time() - _start);

    if (verbose)
        fprintf(stderr, "    Generating s and y values: ");
    _start = current_time();
    ss = calloc(s->n, sizeof ss[0]);
#pragma omp parallel for
    for (size_t i = 0; i < s->n; ++i) {
        mpz_t tmp, ginv, f;

        mpz_inits(tmp, ginv, f, NULL);

        mpz_invert(ginv, s->gs[i], pstate->ps[level][i]);
        /* Compute fᵢ */
        mpz_mul(f, ginv, pstate->ps[level + 1][i]);
        mpz_quotient(f, f, pstate->ps[level][i]);
        mpz_mul(f, f, s->gs[i]);
        mpz_mod_near(f, f, pstate->ps[level + 1][i]);
        mpz_mod_near(f, f, s->gs[i]);
        mpz_invert(f, f, s->gs[i]);
        mpz_mul(f, f, ginv);

        ss[i] = mpz_vector_new(pstate->theta);
        for (size_t j = 0; j < s->n; ++j)
            mpz_set_ui(ss[i][j], 0);
        for (size_t j = s->n; j < pstate->theta; ++j) {
            mpz_urandomb_aes(ss[i][j], s->rngs[0], (int) log2(wordsize));
            mpz_mod_near_ui(ss[i][j], ss[i][j], wordsize);
        }
        mpz_set_ui(ss[i][i], 1);

        mpz_mul(state->ys[i], f, z1);
        mpz_mod_near(state->ys[i], state->ys[i], pstate->ps[level][i]);
        mpz_mul(state->ys[i], state->ys[i], K);
        mpz_quotient(state->ys[i], state->ys[i], pstate->ps[level][i]);

        for (size_t t = s->n; t < pstate->theta; ++t) {
            mpz_mul(tmp, state->ys[t], ss[i][t]);
#pragma omp critical
            {
                mpz_sub(state->ys[i], state->ys[i], tmp);
            }
        }
#pragma omp critical
        {
            mpz_mod_near(state->ys[i], state->ys[i], K);
        }
        mpz_clears(tmp, ginv, f, NULL);
    }
    if (verbose)
        fprintf(stderr, "[%.2fs]\n", current_time() - _start);

    if (verbose)
        fprintf(stderr, "    Generating σ values: ");
    _start = current_time();
    state->sigmas = calloc(pstate->theta, sizeof state->sigmas[0]);
    for (size_t t = 0; t < pstate->theta; ++t) {
        mpz_t **rs;
        state->sigmas[t] = calloc(state->k, sizeof state->sigmas[t][0]);
        rs = calloc(state->k * s->n, sizeof rs[0]);
        for (size_t j = 0; j < state->k; ++j) {
            rs[j] = mpz_vector_new(s->n);
            for (size_t i = 0; i < s->n; ++i) {
                mpz_urandomb_aes(rs[j][i], s->rngs[i], 2 * s->rho);
                mpz_mod_near_ui(rs[j][i], rs[j][i], s->rho);
            }
        }
#pragma omp parallel for
        for (size_t j = 0; j < state->k; ++j) {
            mpz_t wkj;
            mpz_init(wkj);
            mpz_ui_pow_ui(wkj, wordsize, state->k - j);
            state->sigmas[t][j] = clt_elem_new();
            for (size_t i = 0; i < s->n; ++i) {
                mpz_t x;
                mpz_init(x);
                mpz_mul(x, ss[i][t], pstate->ps[level + 1][i]);
                mpz_quotient(x, x, wkj);
                mpz_add(x, x, rs[j][i]);
                mpz_mul_mod_near(x, x, s->gs[i], pstate->ps[level + 1][i]);
                mpz_mul(x, x, pstate->crt_coeffs[level + 1][i]);
#pragma omp critical
                {
                    mpz_add(state->sigmas[t][j]->elem, state->sigmas[t][j]->elem, x);
                }
                mpz_clear(x);
            }
            mpz_mul_mod_near(state->sigmas[t][j]->elem, state->sigmas[t][j]->elem, z2, pstate->x0s[level + 1]);
            state->sigmas[t][j]->level = level + 1;
            mpz_clear(wkj);
        }
        for (size_t j = 0; j < state->k; ++j) {
            for (size_t i = 0; i < s->n; ++i)
                mpz_clear(rs[j][i]);
            free(rs[j]);
        }
        free(rs);
    }
    if (verbose)
        fprintf(stderr, "[%.2fs]\n", current_time() - _start);
    for (size_t i = 0; i < s->n; ++i) {
        for (size_t t = 0; t < pstate->theta; ++t) {
            mpz_clear(ss[i][t]);
        }
        free(ss[i]);
    }
    free(ss);
    mpz_clears(wk, K, NULL);
    if (verbose)
        fprintf(stderr, "    Total: [%.2fs]\n", current_time() - start);
    return state;
}

void
polylog_state_free(polylog_state_t *state)
{
    if (state == NULL)
        return;
    for (size_t i = 0; i < state->nlevels; ++i) {
        for (size_t j = 0; j < state->n; ++j) {
            mpz_clear(state->ps[i][j]);
        }
        free(state->ps[i]);
    }
    free(state->ps);
    free(state);
}

polylog_state_t *
polylog_state_new(clt_state_t *s, size_t eta, size_t theta, size_t b, size_t wordsize,
                  size_t nlevels, const switch_params_t *sparams, size_t nmuls)
{
    const bool verbose = s->flags & CLT_FLAG_VERBOSE;
    polylog_state_t *state;
    size_t *etas;
    int count;

    if (log2(wordsize) != floor(log2(wordsize))) {
        fprintf(stderr, "error: wordsize must be a power of two\n");
        return NULL;
    }

    if ((state = calloc(1, sizeof state[0])) == NULL)
        return NULL;
    s->pstate = state;
    state->n = s->n;
    state->nlevels = nlevels + 1;
    state->theta = theta;
    state->b = b;
    if (state->theta <= state->n) {
        fprintf(stderr, "error: θ ≤ n\n");
        goto error;
    }
    /* Compute ηs */
    etas = calloc(state->nlevels, sizeof etas[0]);
    for (size_t i = 0; i < state->nlevels; ++i) {
        if (i * 2 * b > eta) {
            fprintf(stderr, "error: η - ℓ·2B < 0\n");
            goto error;
        }
        etas[i] = eta - i * 2 * b;
    }
    if (verbose) {
        fprintf(stderr, "Polylog CLT initialization:\n");
        fprintf(stderr, "  θ: ..... %lu\n", state->theta);
        fprintf(stderr, "  b: ..... %lu\n", state->b);
        fprintf(stderr, "  nlevels: 0 →  %lu\n", state->nlevels - 1);
        fprintf(stderr, "  ηs: .... ");
        for (size_t i = 0; i < state->nlevels; ++i)
            fprintf(stderr, "%lu ", etas[i]);
        fprintf(stderr, "\n");
    }

    if (verbose)
        fprintf(stderr, "  Generating p_i's:\n");
    state->ps = calloc(state->nlevels, sizeof state->ps[0]);
    for (size_t i = 0; i < state->nlevels; ++i) {
        double start = current_time();
        state->ps[i] = mpz_vector_new(state->n);
        if (verbose) {
            count = 0;
            fprintf(stderr, "%lu", etas[i]);
            print_progress(count, state->n);
        }
        for (size_t j = 0; j < state->n; ++j) {
            mpz_prime(state->ps[i][j], s->rngs[j], etas[i]);
            if (verbose)
                print_progress(++count, state->n);
        }
        if (verbose)
            fprintf(stderr, "\t[%.2fs]\n", current_time() - start);
    }
    if (verbose)
        fprintf(stderr, "  Generating x0s:\n");
    state->x0s = mpz_vector_new(state->nlevels);
    state->crt_coeffs = calloc(state->nlevels, sizeof state->crt_coeffs[0]);
    for (size_t i = 0; i < state->nlevels; ++i) {
        if (verbose)
            fprintf(stderr, "  Computing product:\n");
        product(state->x0s[i], state->ps[i], state->n, verbose);
        state->crt_coeffs[i] = mpz_vector_new(state->n);
        crt_coeffs(state->crt_coeffs[i], state->ps[i], state->n, state->x0s[i], verbose);
    }
    if (verbose)
        fprintf(stderr, "  Generating z's:\n");
    state->zs = calloc(state->nlevels, sizeof state->zs[0]);
    state->zinvs = calloc(state->nlevels, sizeof state->zinvs[0]);
    for (size_t i = 0; i < state->nlevels; ++i) {
        state->zs[i] = mpz_vector_new(s->nzs);
        state->zinvs[i] = mpz_vector_new(s->nzs);
        generate_zs(state->zs[i], state->zinvs[i], s->rngs, s->nzs, state->x0s[i], verbose);
    }
    state->switches = calloc(nmuls, sizeof state->switches[0]);
    for (size_t i = 0; i < nmuls; ++i) {
        size_t level = sparams[i].level;
        int *ix = sparams[i].ix;
        if ((state->switches[i] = switch_state_new(s, ix, etas[level], wordsize, level, verbose)) == NULL)
            goto error;
    }
    free(etas);
    if (verbose)
        fprintf(stderr, "Polylog CLT initialization complete!\n");
    return state;
error:
    polylog_state_free(state);
    return NULL;
}

int
polylog_encode(clt_elem_t *rop, const clt_state_t *s, size_t n, mpz_t *xs, const int *ix, size_t level)
{
    polylog_state_t *pstate = s->pstate;
    mpz_set_ui(rop->elem, 0);
#pragma omp parallel for
    for (size_t i = 0; i < s->n; ++i) {
        mpz_t tmp;
        mpz_init(tmp);
        mpz_urandomb_aes(tmp, s->rngs[i], 2 * pstate->b);
        mpz_mod_near_ui(tmp, tmp, pstate->b);
        mpz_mul(tmp, tmp, s->gs[i]);
        mpz_add(tmp, tmp, xs[slot(i, n, s->n)]);
        mpz_mul(tmp, tmp, pstate->crt_coeffs[level][i]);
#pragma omp critical
        {
            mpz_add(rop->elem, rop->elem, tmp);
        }
        mpz_clear(tmp);
    }
    rop->level = level;
    if (ix) {
        mpz_t tmp;
        mpz_init(tmp);
        /* multiply by appropriate zinvs */
        rop->ix = calloc(s->nzs, sizeof rop->ix[0]);
        for (size_t i = 0; i < s->nzs; ++i) {
            if (ix[i] <= 0) continue;
            rop->ix[i] = ix[i];
            mpz_powm_ui(tmp, pstate->zinvs[level][i], rop->ix[i], pstate->x0s[level]);
            mpz_mul_mod_near(rop->elem, rop->elem, tmp, pstate->x0s[level]);
        }
        rop->nzs = s->nzs;
        mpz_clear(tmp);
    }
    return CLT_OK;
}

int
polylog_elem_add(clt_elem_t *rop, const clt_pp_t *pp, const clt_elem_t *a, const clt_elem_t *b)
{
    if (a->level != b->level) {
        fprintf(stderr, "error: levels unequal (%lu ≠ %lu), unable to add\n",
                a->level, b->level);
        return CLT_ERR;
    }
    if (a->nzs != b->nzs) {
        fprintf(stderr, "error: encoding index sets of different length (%lu ≠ %lu)\n",
                a->nzs, b->nzs);
        return CLT_ERR;
    }
    rop->nzs = a->nzs;
    rop->ix = calloc(rop->nzs, sizeof rop->ix[0]);
    {
        bool valid = true;
        for (size_t i = 0; i < a->nzs; ++i) {
            if (a->ix[i] != b->ix[i]) {
                valid = false;
                break;
            }
            rop->ix[i] = a->ix[i];
        }
        if (!valid) {
            fprintf(stderr, "error: index sets not equal\n");
            free(rop->ix);
            return CLT_ERR;
        }
    }
    mpz_add(rop->elem, a->elem, b->elem);
    mpz_mod(rop->elem, rop->elem, pp->pstate->x0s[a->level]);
    rop->level = a->level;
    return CLT_OK;
}

int
polylog_elem_mul(clt_elem_t *rop, const clt_pp_t *pp, const clt_elem_t *a,
                 const clt_elem_t *b, size_t idx, int verbose)
{
    switch_state_t *sstate;
    if (a->level != b->level) {
        fprintf(stderr, "error: levels unequal (%lu ≠ %lu), unable to multiply\n",
                a->level, b->level);
        return CLT_ERR;
    }
    if (a->nzs != b->nzs) {
        fprintf(stderr, "error: encoding index sets of different length (%lu ≠ %lu)\n",
                a->nzs, b->nzs);
        return CLT_ERR;
    }
    rop->nzs = a->nzs;
    rop->ix = calloc(rop->nzs, sizeof rop->ix[0]);
    for (size_t i = 0; i < rop->nzs; ++i) {
        rop->ix[i] = a->ix[i] + b->ix[i];
    }
    sstate = pp->pstate->switches[idx];
    if (sstate->level != a->level) {
        fprintf(stderr, "error: switch and element levels unequal (%lu ≠ %lu)\n",
                sstate->level, a->level);
        return CLT_ERR;
    }
    /* a · b mod Π^(ℓ) */
    mpz_mul(rop->elem, a->elem, b->elem);
    mpz_mod(rop->elem, rop->elem, pp->pstate->x0s[sstate->level]);
    rop->level = a->level;
    if (polylog_switch(rop, pp, rop, sstate, verbose) == CLT_ERR)
        return CLT_ERR;
    return CLT_OK;
}

int
polylog_switch(clt_elem_t *rop, const clt_pp_t *pp, const clt_elem_t *x_,
               const switch_state_t *sstate, bool verbose)
{
    const polylog_pp_t *pstate = pp->pstate;
    const double start = current_time();
    mpz_t *pi, *pip, ct, wk;
    clt_elem_t *x;
    int ret = CLT_ERR;

    if (rop == NULL)
        return CLT_ERR;
    mpz_inits(ct, wk, NULL);
    x = clt_elem_new();
    clt_elem_set(x, x_);
    if (sstate->level != x->level) {
        fprintf(stderr, "error: switch and element levels unequal (%lu ≠ %lu)\n",
                sstate->level, x->level);
        goto cleanup;
    }

    pi = &pstate->x0s[sstate->level];
    pip = &pstate->x0s[sstate->level + 1];

    mpz_ui_pow_ui(wk, sstate->wordsize, sstate->k);
    mpz_set_ui(rop->elem, 0);
    for (size_t t = 0; t < pstate->theta; ++t) {
        /* Compute cₜ = (x · yₜ) / Π^(ℓ) mod (wordsize)ᵏ */
        mpz_mul(ct, x->elem, sstate->ys[t]);
        mpz_quotient(ct, ct, *pi);
        mpz_mod(ct, ct, wk);
        for (size_t i = 0; i < sstate->k; ++i) {
            mpz_t tmp, decomp;
            mpz_inits(tmp, decomp, NULL);
            /* Compute word decomposition c_{t,i} of cₜ */
            mpz_mod_near_ui(decomp, ct, sstate->wordsize);
            mpz_quotient_2exp(ct, ct, (int) log2(sstate->wordsize));
            /* σ_{t,i} · c_{t,i} */
            mpz_mul(tmp, decomp, sstate->sigmas[t][i]->elem);
            mpz_add(rop->elem, rop->elem, tmp);
            mpz_mod(rop->elem, rop->elem, *pip);
            mpz_clears(tmp, decomp, NULL);
        }
    }
    rop->level = x->level + 1;
    if (verbose)
        fprintf(stderr, "Switch time: %.2fs\n", current_time() - start);
    ret = CLT_OK;
cleanup:
    mpz_clears(ct, wk, NULL);
    clt_elem_free(x);
    return ret;
}

int
polylog_elem_decrypt(clt_elem_t *x, const clt_state_t *s, size_t level)
{
    size_t nbits;
    mpz_t rop, tmp, z;
    mpz_inits(rop, tmp, z, NULL);
    mpz_set_ui(z, 1);
    printf("DECRYPTION @ LEVEL %lu :: ", level);
    for (size_t i = 0; i < x->nzs; ++i) {
        printf("%d ", x->ix[i]);
        if (x->ix[i] <= 0) continue;
        mpz_powm_ui(tmp, s->pstate->zs[level][i], x->ix[i], s->pstate->x0s[level]);
        mpz_mul_mod_near(z, z, tmp, s->pstate->x0s[level]);
    }
    mpz_mul(z, z, x->elem);
    printf(":: ");
    for (size_t i = 0; i < s->n; ++i) {
        mpz_mod_near(rop, z, s->pstate->ps[level][i]);
        nbits = mpz_sizeinbase(x->elem, 2);
        mpz_mod_near(rop, rop, s->gs[i]);
        gmp_printf("%Zd ", rop);
    }
    mpz_clears(rop, tmp, z, NULL);
    printf(": %lu\n", nbits);
    return CLT_OK;
}
