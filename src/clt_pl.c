#include "clt_pl.h"
#include "clt_elem.h"
#include "estimates.h"
#include "utils.h"

#include <stdbool.h>
#include <omp.h>
#include <math.h>
#include <unistd.h>

struct switch_state_t {
    size_t source;              /* starting level */
    size_t target;              /* ending level */
    size_t wordsize;            /* must be a power of two */
    size_t k;                   /* = η_ℓ / log2(wordsize) */
    mpz_t *ys;                  /* [Θ] */
    mpz_t **sigmas;             /* [Θ][k] */
    bool active;
};

struct clt_pl_state_t {
    size_t n;                   /* number of slots */
    size_t nzs;                 /* number of z's in the index set */
    size_t rho;                 /* bitsize of randomness */
    size_t nu;                  /* number of most-significant-bits to extract */
    size_t nlevels;
    size_t theta;
    size_t nswitches;
    size_t b;
    mpz_t pzt;
    mpz_t *gs;                  /* [n] */
    mpz_t **zs;                 /* [nlevels][nzs] */
    mpz_t **zinvs;              /* [nlevels][nzs] */
    mpz_t **ps;                 /* [nlevels][n] */
    mpz_t **crt_coeffs;         /* [nlevels][n] */
    mpz_t *x0s;                 /* [nlevels] */
    switch_state_t ***switches; /* [nswitches][2] */
    aes_randstate_t *rngs;      /* [max{n,nzs}] */
    size_t flags;
};

struct clt_pl_pp_t {
    size_t nu;
    size_t theta;
    size_t nlevels;
    mpz_t *x0s;
    mpz_t pzt;
    size_t nswitches;
    switch_state_t ***switches; /* [nswitches][2] */
    bool verbose;
    bool local;
};

switch_state_t ***
clt_pl_pp_switches(const clt_pl_pp_t *pp)
{
    return pp->switches;
}

size_t
clt_pl_elem_level(const clt_elem_t *x)
{
    return x->level;
}

int
clt_pl_elem_switch(clt_elem_t *rop, const clt_pl_pp_t *pp, const clt_elem_t *x,
                   const switch_state_t *sstate)
{
    const bool verbose = pp->verbose;
    const double start = current_time();
    mpz_t *pi, *pip, ct, wk, tmp;

    if (rop == NULL || pp == NULL || x == NULL || sstate == NULL)
        return CLT_ERR;
    mpz_inits(ct, wk, NULL);
    mpz_init_set(tmp, x->elem);
    /* Copy so we don't overwrite `x` if `rop == x` */

    if (sstate->source == 0 && sstate->target == 0)
        return CLT_OK;

    if (x->level != sstate->source) {
        fprintf(stderr, "error: %s: levels not equal (%lu ≠ %lu)\n",
                __func__, x->level, sstate->source);
        abort();
        return CLT_ERR;
    }

    pi = &pp->x0s[sstate->source];
    pip = &pp->x0s[sstate->target];

    mpz_ui_pow_ui(wk, sstate->wordsize, sstate->k);
    mpz_set_ui(rop->elem, 0);
    for (size_t t = 0; t < pp->theta; ++t) {
        /* Compute cₜ = (x · yₜ) / Π^(ℓ) mod (wordsize)ᵏ */
        mpz_mul(ct, tmp, sstate->ys[t]);
        mpz_quotient(ct, ct, *pi);
        mpz_mod(ct, ct, wk);
        for (size_t i = 0; i < sstate->k; ++i) {
            mpz_t tmp, decomp;
            mpz_inits(tmp, decomp, NULL);
            /* Compute word decomposition c_{t,i} of cₜ */
            mpz_mod_near_ui(decomp, ct, sstate->wordsize);
            mpz_quotient_2exp(ct, ct, (int) log2(sstate->wordsize));
            /* σ_{t,i} · c_{t,i} */
            mpz_mul(tmp, decomp, sstate->sigmas[t][i]);
            mpz_add(rop->elem, rop->elem, tmp);
            mpz_mod(rop->elem, rop->elem, *pip);
            mpz_clears(tmp, decomp, NULL);
        }
    }
    rop->level = sstate->target;
    if (verbose)
        fprintf(stderr, "Switch time: %.2fs\n", current_time() - start);
    mpz_clears(ct, wk, tmp, NULL);
    return CLT_OK;
}

int
clt_pl_elem_add(clt_elem_t *rop, const clt_pl_pp_t *pp, const clt_elem_t *a,
                const clt_elem_t *b)
{
    if (a->level != b->level) {
        fprintf(stderr, "error: %s: levels unequal (%lu ≠ %lu)\n",
                __func__, a->level, b->level);
        abort();
    }
    rop->level = a->level;
    mpz_add(rop->elem, a->elem, b->elem);
    mpz_mod_near(rop->elem, rop->elem, pp->x0s[rop->level]);
    return CLT_OK;
}

int
clt_pl_elem_sub(clt_elem_t *rop, const clt_pl_pp_t *pp, const clt_elem_t *a,
                const clt_elem_t *b)
{
    if (a->level != b->level) {
        fprintf(stderr, "error: %s: levels unequal (%lu ≠ %lu)\n",
                __func__, a->level, b->level);
        abort();
    }
    rop->level = a->level;
    mpz_sub(rop->elem, a->elem, b->elem);
    mpz_mod_near(rop->elem, rop->elem, pp->x0s[rop->level]);
    return CLT_OK;
}

int
clt_pl_elem_mul(clt_elem_t *rop, const clt_pl_pp_t *pp, const clt_elem_t *a, const clt_elem_t *b)
{
    if (a->level != b->level) {
        fprintf(stderr, "error: %s: levels unequal (%lu ≠ %lu)\n",
                __func__, a->level, b->level);
        abort();
    }
    mpz_mul(rop->elem, a->elem, b->elem);
    mpz_mod(rop->elem, rop->elem, pp->x0s[a->level]);
    rop->level = a->level;
    return CLT_OK;
}

/* XXX REMOVE ONCE WE'VE GOT EVERYTHING WORKING */
int
clt_pl_elem_decrypt(clt_elem_t *x, const clt_pl_state_t *s, size_t nzs, const int *ix, size_t level)
{
    size_t nbits = 0;
    mpz_t rop, tmp, z;
    mpz_inits(rop, tmp, z, NULL);
    mpz_set_ui(z, 1);
    printf("DECRYPTION @ LEVEL %lu :: ", level);
    if (ix) {
        for (size_t i = 0; i < nzs; ++i) {
            printf("%d ", ix[i]);
            if (ix[i] <= 0) continue;
            mpz_powm_ui(tmp, s->zs[level][i], ix[i], s->x0s[level]);
            mpz_mul_mod_near(z, z, tmp, s->x0s[level]);
        }
    }
    mpz_mul(z, z, x->elem);
    printf(":: ");
    for (size_t i = 0; i < s->n; ++i) {
        mpz_mod_near(rop, z, s->ps[level][i]);
        nbits = mpz_sizeinbase(x->elem, 2);
        mpz_mod_near(rop, rop, s->gs[i]);
        gmp_printf("%Zd ", rop);
    }
    mpz_clears(rop, tmp, z, NULL);
    printf(": %lu\n", nbits);
    return CLT_OK;
}

int
clt_pl_is_zero(const clt_elem_t *c, const clt_pl_pp_t *pp)
{
    int ret;

    mpz_t tmp, *x0;
    mpz_init(tmp);
    x0 = &pp->x0s[pp->nlevels - 1];

    mpz_mul(tmp, c->elem, pp->pzt);
    mpz_mod_near(tmp, tmp, *x0);

    ret = mpz_sizeinbase(tmp, 2) < mpz_sizeinbase(*x0, 2) - pp->nu;
    mpz_clear(tmp);
    return ret ? 1 : 0;
}

static void
switch_state_free(switch_state_t *s, size_t theta)
{
    if (s == NULL)
        return;
    if (s->active) {
        mpz_vector_free(s->ys, theta);
        for (size_t t = 0; t < theta; ++t)
            mpz_vector_free(s->sigmas[t], s->k);
        free(s->sigmas);
    }
    free(s);
}

static switch_state_t *
switch_state_fread(FILE *fp, size_t theta)
{
    switch_state_t *s;

    if ((s = calloc(1, sizeof s[0])) == NULL)
        return NULL;
    if (bool_fread(fp, &s->active) == CLT_ERR) goto error;
    if (!s->active)
        return s;
    if (size_t_fread(fp, &s->source) == CLT_ERR) goto error;
    if (size_t_fread(fp, &s->target) == CLT_ERR) goto error;
    if (size_t_fread(fp, &s->wordsize) == CLT_ERR) goto error;
    if (size_t_fread(fp, &s->k) == CLT_ERR) goto error;
    s->ys = mpz_vector_new(theta);
    if (mpz_vector_fread(s->ys, theta, fp) == CLT_ERR) goto error;
    s->sigmas = calloc(theta, sizeof s->sigmas[0]);
    for (size_t i = 0; i < theta; ++i) {
        s->sigmas[i] = mpz_vector_new(s->k);
        if (mpz_vector_fread(s->sigmas[i], s->k, fp) == CLT_ERR) goto error;
    }
    return s;
error:
    switch_state_free(s, theta);
    return NULL;
}

static int
switch_state_fwrite(FILE *fp, switch_state_t *s, size_t theta)
{
    if (bool_fwrite(fp, s->active) == CLT_ERR) return CLT_ERR;
    if (!s->active)
        return CLT_OK;
    if (size_t_fwrite(fp, s->source) == CLT_ERR) return CLT_ERR;
    if (size_t_fwrite(fp, s->target) == CLT_ERR) return CLT_ERR;
    if (size_t_fwrite(fp, s->wordsize) == CLT_ERR) return CLT_ERR;
    if (size_t_fwrite(fp, s->k) == CLT_ERR) return CLT_ERR;
    if (mpz_vector_fwrite(s->ys, theta, fp) == CLT_ERR) return CLT_ERR;
    for (size_t i = 0; i < theta; ++i)
        if (mpz_vector_fwrite(s->sigmas[i], s->k, fp) == CLT_ERR) return CLT_ERR;
    return CLT_OK;
}

static switch_state_t *
switch_state_new(clt_pl_state_t *s, const int ix[s->nzs], size_t eta,
                 size_t wordsize, size_t source, size_t target, bool verbose)
{
    switch_state_t *state;
    mpz_t wk, K, z1, z2;
    size_t **ss;
    double start, _start;

    /* XXX fix this so we are not generating empty switch statements */
    if (source == 0 && target == 0)
        return calloc(1, sizeof state[0]);

    if (source >= target) {
        fprintf(stderr, "error: %s: source ≥ target (%lu ≥ %lu)\n",
                __func__, source, target);
        return NULL;
    }
    if (target >= s->nlevels) {
        fprintf(stderr, "error: %s: target level too large (%lu ≥ %lu)\n",
                __func__, target, s->nlevels);
        return NULL;
    }

    if ((state = calloc(1, sizeof state[0])) == NULL)
        return NULL;
    state->source = source;
    state->target = target;
    state->wordsize = wordsize;
    /* k = η_ℓ / log2(wordsize) */
    state->k = (int) ceil(eta / log2(wordsize));

    if (verbose)
        fprintf(stderr, "\n    Generating switch state [%lu →  %lu]:\n", source, target);
    start = current_time();

    mpz_inits(wk, K, z1, z2, NULL);
    /* wk = (wordsize)ᵏ */
    mpz_ui_pow_ui(wk, wordsize, state->k);
    /* K = Π^(ℓ) · wk */
    mpz_mul(K, s->x0s[source], wk);
    /* compute zs */
    mpz_set_ui(z1, 1);
    mpz_set_ui(z2, 1);
    if (ix) {
        mpz_t tmp;
        mpz_init(tmp);
        for (size_t i = 0; i < s->nzs; ++i) {
            if (ix[i] <= 0) continue;
            mpz_powm_ui(tmp, s->zs[source][i], ix[i], s->x0s[source]);
            mpz_mul_mod_near(z1, z1, tmp, s->x0s[source]);
            mpz_powm_ui(tmp, s->zinvs[target][i], ix[i], s->x0s[target]);
            mpz_mul_mod_near(z2, z2, tmp, s->x0s[target]);
        }
        mpz_clear(tmp);
    }
    if (verbose)
        fprintf(stderr, "      Generating random y values: ");
    _start = current_time();
    state->ys = mpz_vector_new(s->theta);
    for (size_t i = s->n; i < s->theta; ++i) {
        mpz_urandomm_aes(state->ys[i], s->rngs[0], K);
    }
    if (verbose)
        fprintf(stderr, "[%.2fs]\n", current_time() - _start);

    if (verbose)
        fprintf(stderr, "      Generating s and y values: ");
    _start = current_time();
    ss = calloc(s->n, sizeof ss[0]);
#pragma omp parallel for
    for (size_t i = 0; i < s->n; ++i) {
        mpz_t tmp, ginv, f;

        mpz_inits(tmp, ginv, f, NULL);

        mpz_invert(ginv, s->gs[i], s->ps[source][i]);
        /* Compute fᵢ */
        mpz_mul(f, ginv, s->ps[target][i]);
        mpz_quotient(f, f, s->ps[source][i]);
        mpz_mul(f, f, s->gs[i]);
        mpz_mod_near(f, f, s->ps[target][i]);
        mpz_mod_near(f, f, s->gs[i]);
        mpz_invert(f, f, s->gs[i]);
        mpz_mul(f, f, ginv);

        ss[i] = calloc(s->theta, sizeof ss[i][0]);
        for (size_t j = s->n; j < s->theta; ++j) {
            mpz_urandomb_aes(tmp, s->rngs[0], (int) log2(wordsize));
            ss[i][j] = mpz_get_ui(tmp);
        }
        ss[i][i] = 1;

        mpz_mul(state->ys[i], f, z1);
        mpz_mod_near(state->ys[i], state->ys[i], s->ps[source][i]);
        mpz_mul(state->ys[i], state->ys[i], K);
        mpz_quotient(state->ys[i], state->ys[i], s->ps[source][i]);

        for (size_t t = s->n; t < s->theta; ++t) {
            mpz_mul_ui(tmp, state->ys[t], ss[i][t]);
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
        fprintf(stderr, "      Generating σ values: ");
    _start = current_time();
    state->sigmas = calloc(s->theta, sizeof state->sigmas[0]);
    for (size_t t = 0; t < s->theta; ++t) {
        mpz_t **rs;
        state->sigmas[t] = mpz_vector_new(state->k);
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
            for (size_t i = 0; i < s->n; ++i) {
                mpz_t x;
                mpz_init(x);
                mpz_mul_ui(x, s->ps[target][i], ss[i][t]);
                mpz_quotient(x, x, wkj);
                mpz_add(x, x, rs[j][i]);
                mpz_mul_mod_near(x, x, s->gs[i], s->ps[target][i]);
                mpz_mul(x, x, s->crt_coeffs[target][i]);
#pragma omp critical
                {
                    mpz_add(state->sigmas[t][j], state->sigmas[t][j], x);
                }
                mpz_clear(x);
            }
            mpz_mul_mod_near(state->sigmas[t][j], state->sigmas[t][j], z2, s->x0s[target]);
            mpz_clear(wkj);
        }
        for (size_t j = 0; j < state->k; ++j) {
            for (size_t i = 0; i < s->n; ++i)
                mpz_clear(rs[j][i]);
            free(rs[j]);
        }
        free(rs);
    }
    state->active = true;
    if (verbose)
        fprintf(stderr, "[%.2fs]\n", current_time() - _start);
    for (size_t i = 0; i < s->n; ++i) {
        free(ss[i]);
    }
    free(ss);
    mpz_clears(wk, K, z1, z2, NULL);
    if (verbose)
        fprintf(stderr, "    Total: [%.2fs]\n", current_time() - start);
    return state;
}

clt_pl_pp_t *
clt_pl_pp_new(const clt_pl_state_t *s)
{
    clt_pl_pp_t *pp;

    if ((pp = calloc(1, sizeof pp[0])) == NULL)
        return NULL;
    pp->nu = s->nu;
    pp->theta = s->theta;
    pp->nlevels = s->nlevels;
    pp->x0s = s->x0s;
    mpz_init_set(pp->pzt, s->pzt);
    pp->nswitches = s->nswitches;
    pp->switches = s->switches;
    pp->verbose = s->flags & CLT_PL_FLAG_VERBOSE;
    pp->local = false;
    return pp;
}

void
clt_pl_pp_free(clt_pl_pp_t *pp)
{
    if (pp == NULL)
        return;
    if (pp->local) {
        mpz_clear(pp->pzt);
        mpz_vector_free(pp->x0s, pp->nlevels);
        for (size_t i = 0; i < pp->nswitches; ++i) {
            for (size_t j = 0; j < 2; ++j) {
                switch_state_free(pp->switches[i][j], pp->theta);
            }
            free(pp->switches[i]);
        }
        free(pp->switches);
    }
    free(pp);
}

clt_pl_pp_t *
clt_pl_pp_fread(FILE *fp)
{
    clt_pl_pp_t *pp;

    if ((pp = calloc(1, sizeof pp[0])) == NULL)
        return NULL;
    if (size_t_fread(fp, &pp->nu) == CLT_ERR) goto cleanup;
    if (size_t_fread(fp, &pp->theta) == CLT_ERR) goto cleanup;
    if (size_t_fread(fp, &pp->nlevels) == CLT_ERR) goto cleanup;
    if (size_t_fread(fp, &pp->nswitches) == CLT_ERR) goto cleanup;
    mpz_init(pp->pzt);
    if (mpz_fread(pp->pzt, fp) == CLT_ERR) goto cleanup;
    pp->x0s = mpz_vector_new(pp->nlevels);
    if (mpz_vector_fread(pp->x0s, pp->nlevels, fp) == CLT_ERR) goto cleanup;
    pp->switches = calloc(pp->nswitches, sizeof pp->switches[0]);
    for (size_t i = 0; i < pp->nswitches; ++i) {
        pp->switches[i] = calloc(2, sizeof pp->switches[0]);
        for (size_t j = 0; j < 2; ++j)
            if ((pp->switches[i][j] = switch_state_fread(fp, pp->theta)) == NULL) goto cleanup;
    }
    if (bool_fread(fp, &pp->verbose) == CLT_ERR) goto cleanup;
    pp->local = true;
    return pp;
cleanup:
    clt_pl_pp_free(pp);
    return NULL;
}

int
clt_pl_pp_fwrite(clt_pl_pp_t *pp, FILE *fp)
{
    if (size_t_fwrite(fp, pp->nu) == CLT_ERR) goto cleanup;
    if (size_t_fwrite(fp, pp->theta) == CLT_ERR) goto cleanup;
    if (size_t_fwrite(fp, pp->nlevels) == CLT_ERR) goto cleanup;
    if (size_t_fwrite(fp, pp->nswitches) == CLT_ERR) goto cleanup;
    if (mpz_fwrite(pp->pzt, fp) == CLT_ERR) goto cleanup;
    if (mpz_vector_fwrite(pp->x0s, pp->nlevels, fp) == CLT_ERR) goto cleanup;
    for (size_t i = 0; i < pp->nswitches; ++i)
        for (size_t j = 0; j < 2; ++j)
            if (switch_state_fwrite(fp, pp->switches[i][j], pp->theta) == CLT_ERR) goto cleanup;
    if (bool_fwrite(fp, pp->verbose) == CLT_ERR) goto cleanup;
    return CLT_OK;
cleanup:
    return CLT_ERR;
}

void
clt_pl_state_free(clt_pl_state_t *s)
{
    if (s == NULL)
        return;
    if (s->pzt)
        mpz_clear(s->pzt);
    if (s->gs)
        mpz_vector_free(s->gs, s->n);
    if (s->zs) {
        for (size_t i = 0; i < s->nlevels; ++i)
            mpz_vector_free(s->zs[i], s->nzs);
        free(s->zs);
    }
    if (s->zinvs) {
        for (size_t i = 0; i < s->nlevels; ++i)
            mpz_vector_free(s->zinvs[i], s->nzs);
        free(s->zinvs);
    }
    if (s->ps) {
        for (size_t i = 0; i < s->nlevels; ++i)
            mpz_vector_free(s->ps[i], s->n);
        free(s->ps);
    }
    if (s->crt_coeffs) {
        for (size_t i = 0; i < s->nlevels; ++i)
            mpz_vector_free(s->crt_coeffs[i], s->n);
        free(s->crt_coeffs);
    }
    if (s->x0s)
        mpz_vector_free(s->x0s, s->nlevels);
    if (s->switches) {
        for (size_t i = 0; i < s->nswitches; ++i) {
            for (size_t j = 0; j < 2; ++j) {
                switch_state_free(s->switches[i][j], s->theta);
            }
            free(s->switches[i]);
        }
    }
    if (s->rngs) {
        for (size_t i = 0; i < max(s->n, s->nzs); ++i)
            aes_randclear(s->rngs[i]);
        free(s->rngs);
    }
    free(s);
}

static inline size_t
max4(size_t a, size_t b, size_t c, size_t d)
{
    size_t max = 0;
    max = max > a ? max : a;
    max = max > b ? max : b;
    max = max > c ? max : c;
    max = max > d ? max : d;
    return max;
}

static inline size_t
log2_(size_t n)
{
    return (size_t) floor(log2(n));
}

clt_pl_state_t *
clt_pl_state_new(const clt_pl_params_t *params, const clt_pl_opt_params_t *opts,
                 size_t nthreads, size_t flags, aes_randstate_t rng)
{
    const bool verbose = flags & CLT_PL_FLAG_VERBOSE;
    size_t alpha, beta, eta_L, eta, h, wordsize;
    size_t *etas = NULL;
    clt_pl_state_t *s;
    const size_t slots = opts ? opts->slots : 0;
    int count = 0;

    if ((s = calloc(1, sizeof s[0])) == NULL)
        return NULL;

    if (nthreads == 0)
        nthreads = sysconf(_SC_NPROCESSORS_ONLN);
    (void) omp_set_num_threads(nthreads);

    /* calculate CLT parameters */
    s->nzs     = params->nzs;
    s->nlevels = params->nlevels + 1;
    s->rho     = params->lambda;
    alpha      = params->lambda;
    eta        = s->nlevels * params->lambda;
    s->n       = max(estimate_n(params->lambda, eta, flags), slots);
    h          = params->lambda;
    s->theta   = max(s->n + 1, s->nlevels * pow(params->lambda, 3 / 2.0));
    beta       = max4(s->rho + 3, s->rho + alpha + 1, log2_(s->theta) + log2_(eta) + 3, 2 * alpha + 3);
    s->nu      = params->lambda;
    eta_L = max(beta + 1, beta - alpha + h + log2_(s->n) + s->n + s->nu);
    s->nswitches = params->nswitches;
    s->flags = flags;

    /* Loop until a fixed point reached */
    {
        const size_t nloops = 10;
        size_t old_beta = 0, old_eta_L = 0, old_eta = 0, old_theta = 0,
               old_n = 0, old_nu = 0;
        size_t i = 0;
        for (; i < nloops && (old_beta != beta
                              || old_eta_L != eta_L
                              || old_eta != eta
                              || old_theta != s->theta
                              || old_n != s->n
                              || old_nu != s->nu); ++i) {
            old_beta = beta, old_eta_L = eta_L, old_eta = eta;
            old_theta = s->theta, old_n = s->n, old_nu = s->nu;
            beta = max4(s->rho + 3, s->rho + alpha + 1, log2_(s->theta) + log2_(eta) + 3, 2 * alpha + 3);
            assert(beta >= alpha);
            eta_L = max(beta + 1, beta - alpha + h + log2_(s->n) + s->n + s->nu);
            eta = eta_L + (beta + 4) * s->nlevels;
            s->n = max(estimate_n(params->lambda, eta, flags), slots);
            s->theta = max(s->n + 1, s->nlevels * pow(params->lambda, 3 / 2.0));
            /* s->theta = sqrt(s->n * eta * params->lambda); */
        }
        if (i == nloops && (old_beta != beta
                            || old_eta_L != eta_L
                            || old_eta != eta
                            || old_theta != s->theta
                            || old_n != s->n
                            || old_nu != s->nu)) {
            fprintf(stderr, "error: unable to find valid η_L, η, n, Θ, and ν choices\n");
            goto error;
        }
    }

    s->b = beta + 4;

    wordsize = opts && opts->wordsize ? opts->wordsize : 2;
    if (log2(wordsize) != floor(log2(wordsize))) {
        fprintf(stderr, "error: wordsize must be a power of two\n");
        goto error;
    }

    /* Compute ηs */
    etas = calloc(s->nlevels, sizeof etas[0]);
    for (size_t i = 0; i < s->nlevels; ++i) {
        if (i * s->b > eta) {
            fprintf(stderr, "error: %s: η - ℓ·r = %lu - %lu·%lu < 0\n",
                    __func__, eta, i, s->b);
            goto error;
        }
        etas[i] = eta - i * s->b;
    }
    if (verbose) {
        fprintf(stderr, "Polylog CLT initialization:\n");
        fprintf(stderr, "  λ: ...... %ld\n", params->lambda);
        fprintf(stderr, "  L: ...... 0 →  %lu\n", s->nlevels - 1);
        fprintf(stderr, "  ηs: ..... ");
        for (size_t i = 0; i < s->nlevels; ++i)
            fprintf(stderr, "%lu ", etas[i]);
        fprintf(stderr, "\n");
        fprintf(stderr, "  α: ...... %ld\n", alpha);
        fprintf(stderr, "  β: ...... %ld\n", beta);
        fprintf(stderr, "  η_L: .... %ld\n", eta_L);
        fprintf(stderr, "  η: ...... %ld\n", eta);
        fprintf(stderr, "  ν: ...... %ld\n", s->nu);
        fprintf(stderr, "  ρ: ...... %ld\n", s->rho);
        fprintf(stderr, "  n: ...... %ld\n", s->n);
        fprintf(stderr, "  θ: ...... %lu\n", s->theta);
        fprintf(stderr, "  b: ...... %lu\n", s->b);
        fprintf(stderr, "  wordsize: %lu\n", wordsize);
    }

    assert(eta_L >= beta + 1);
    /* assert(eta_L >= beta - alpha + h + log2(s->n) + s->n + s->nu); */
    assert(s->n >= slots);
    assert(s->theta > s->n);

    /* Generate randomness for each core */
    s->rngs = calloc(max(s->n, s->nzs), sizeof s->rngs[0]);
    for (size_t i = 0; i < max(s->n, s->nzs); ++i) {
        unsigned char *buf;
        size_t nbytes;

        buf = random_aes(rng, 128, &nbytes);
        aes_randinit_seedn(s->rngs[i], (char *) buf, nbytes, NULL, 0);
        free(buf);
    }

    /* Generate "plaintext" moduli */
    s->gs = mpz_vector_new(s->n);
    if (verbose)
        fprintf(stderr, "  Generating gs:\n");
    if (opts && opts->moduli && opts->nmoduli) {
        for (size_t i = 0; i < opts->nmoduli; ++i)
            mpz_set(s->gs[i], opts->moduli[i]);
        generate_primes(s->gs + opts->nmoduli, s->rngs, s->n - opts->nmoduli, alpha, verbose);
    } else {
        generate_primes(s->gs, s->rngs, s->n, alpha, verbose);
    }

    if (verbose)
        fprintf(stderr, "  Generating %lu ps:\n", s->nlevels);
    s->ps = calloc(s->nlevels, sizeof s->ps[0]);
    for (size_t i = 0; i < s->nlevels; ++i) {
        double start = current_time();
        s->ps[i] = mpz_vector_new(s->n);
        if (verbose) {
            count = 0;
            fprintf(stderr, "%lu", etas[i]);
            print_progress(count, s->n);
        }
        for (size_t j = 0; j < s->n; ++j) {
            mpz_prime(s->ps[i][j], s->rngs[j], etas[i]);
            if (verbose)
                print_progress(++count, s->n);
        }
        if (verbose)
            fprintf(stderr, "\t[%.2fs]\n", current_time() - start);
    }
    if (verbose)
        fprintf(stderr, "  Generating %lu Πs:\n", s->nlevels);
    s->x0s = mpz_vector_new(s->nlevels);
    s->crt_coeffs = calloc(s->nlevels, sizeof s->crt_coeffs[0]);
    for (size_t i = 0; i < s->nlevels; ++i) {
        if (verbose)
            fprintf(stderr, "%lu", i);
        product(s->x0s[i], s->ps[i], s->n, verbose);
        s->crt_coeffs[i] = mpz_vector_new(s->n);
        if (verbose)
            fprintf(stderr, "%lu", i);
        crt_coeffs(s->crt_coeffs[i], s->ps[i], s->n, s->x0s[i], verbose);
    }
    if (verbose)
        fprintf(stderr, "  Generating %lu zs:\n", s->nlevels);
    s->zs = calloc(s->nlevels, sizeof s->zs[0]);
    s->zinvs = calloc(s->nlevels, sizeof s->zinvs[0]);
    for (size_t i = 0; i < s->nlevels; ++i) {
        s->zs[i] = mpz_vector_new(s->nzs);
        s->zinvs[i] = mpz_vector_new(s->nzs);
        if (verbose)
            fprintf(stderr, "%lu", i);
        generate_zs(s->zs[i], s->zinvs[i], s->rngs, s->nzs, s->x0s[i], verbose);
    }
    if (s->nswitches) {
        const double start = current_time();
        int count = 0;
        if (verbose) {
            fprintf(stderr, "  Generating %lu switches:\n", s->nswitches);
            print_progress(count, s->nswitches);
        }
        s->switches = calloc(s->nswitches, sizeof s->switches[0]);
/* #pragma omp parallel for */
        for (size_t i = 0; i < s->nswitches; ++i) {
            s->switches[i] = calloc(2, sizeof s->switches[i][0]);
            for (size_t j = 0; j < 2; ++j) {
                size_t source = params->sparams[i][j].source;
                size_t target = params->sparams[i][j].target;
                int *ix = params->sparams[i][j].ix;
                s->switches[i][j] = switch_state_new(s, ix, etas[source], wordsize, source, target, verbose);
            }
            if (verbose)
/* #pragma omp critical */
            {
                print_progress(++count, s->nswitches);
            }
        }
        if (verbose)
            fprintf(stderr, "\t[%.2fs]\n", current_time() - start);
    }
    mpz_init(s->pzt);
    generate_pzt(s->pzt, beta, s->n, s->ps[s->nlevels - 1], s->gs, s->nzs,
                 s->zs[s->nlevels - 1], params->pows, s->x0s[s->nlevels - 1],
                 s->rngs, verbose);
    free(etas);
    if (verbose)
        fprintf(stderr, "Polylog CLT initialization complete!\n");
    return s;
error:
    clt_pl_state_free(s);
    return NULL;
}

clt_pl_state_t *
clt_pl_state_fread(FILE *fp)
{
    clt_pl_state_t *s;

    if ((s = calloc(1, sizeof s[0])) == NULL)
        return NULL;
    if (size_t_fread(fp, &s->n) == CLT_ERR) goto error;
    if (size_t_fread(fp, &s->nzs) == CLT_ERR) goto error;
    if (size_t_fread(fp, &s->rho) == CLT_ERR) goto error;
    if (size_t_fread(fp, &s->nu) == CLT_ERR) goto error;
    if (size_t_fread(fp, &s->b) == CLT_ERR) goto error;
    if (size_t_fread(fp, &s->nlevels) == CLT_ERR) goto error;
    if (size_t_fread(fp, &s->theta) == CLT_ERR) goto error;
    if (size_t_fread(fp, &s->nswitches) == CLT_ERR) goto error;
    mpz_init(s->pzt);
    if (mpz_fread(s->pzt, fp) == CLT_ERR) goto error;
    s->gs = mpz_vector_new(s->n);
    if (mpz_vector_fread(s->gs, s->n, fp) == CLT_ERR) goto error;
    s->zs = calloc(s->nlevels, sizeof s->zs[0]);
    for (size_t i = 0; i < s->nlevels; ++i) {
        s->zs[i] = mpz_vector_new(s->nzs);
        if (mpz_vector_fread(s->zs[i], s->nzs, fp) == CLT_ERR) goto error;
    }
    s->zinvs = calloc(s->nlevels, sizeof s->zinvs[0]);
    for (size_t i = 0; i < s->nlevels; ++i) {
        s->zinvs[i] = mpz_vector_new(s->nzs);
        if (mpz_vector_fread(s->zinvs[i], s->nzs, fp) == CLT_ERR) goto error;
    }
    s->ps = calloc(s->nlevels, sizeof s->ps[0]);
    for (size_t i = 0; i < s->nlevels; ++i) {
        s->ps[i] = mpz_vector_new(s->n);
        if (mpz_vector_fread(s->ps[i], s->n, fp) == CLT_ERR) goto error;
    }
    s->crt_coeffs = calloc(s->nlevels, sizeof s->crt_coeffs[0]);
    for (size_t i = 0; i < s->nlevels; ++i) {
        s->crt_coeffs[i] = mpz_vector_new(s->n);
        if (mpz_vector_fread(s->crt_coeffs[i], s->n, fp) == CLT_ERR) goto error;
    }
    s->x0s = mpz_vector_new(s->nlevels);
    if (mpz_vector_fread(s->x0s, s->nlevels, fp) == CLT_ERR) goto error;
    s->switches = calloc(s->nswitches, sizeof s->switches[0]);
    for (size_t i = 0; i < s->nswitches; ++i) {
        s->switches[i] = calloc(2, sizeof s->switches[i][0]);
        for (size_t j = 0; j < 2; ++j)
            if ((s->switches[i][j] = switch_state_fread(fp, s->theta)) == NULL) goto error;
    }
    s->rngs = calloc(max(s->n, s->nzs), sizeof s->rngs[0]);
    for (size_t i = 0; i < max(s->n, s->nzs); ++i)
        aes_randstate_fread(s->rngs[i], fp);
    return s;
error:
    clt_pl_state_free(s);
    return NULL;
}

int
clt_pl_state_fwrite(clt_pl_state_t *s, FILE *fp)
{
    if (size_t_fwrite(fp, s->n) == CLT_ERR) return CLT_ERR;
    if (size_t_fwrite(fp, s->nzs) == CLT_ERR) return CLT_ERR;
    if (size_t_fwrite(fp, s->rho) == CLT_ERR) return CLT_ERR;
    if (size_t_fwrite(fp, s->nu) == CLT_ERR) return CLT_ERR;
    if (size_t_fwrite(fp, s->b) == CLT_ERR) return CLT_ERR;
    if (size_t_fwrite(fp, s->nlevels) == CLT_ERR) return CLT_ERR;
    if (size_t_fwrite(fp, s->theta) == CLT_ERR) return CLT_ERR;
    if (size_t_fwrite(fp, s->nswitches) == CLT_ERR) return CLT_ERR;
    if (mpz_fwrite(s->pzt, fp) == CLT_ERR) return CLT_ERR;
    if (mpz_vector_fwrite(s->gs, s->n, fp) == CLT_ERR) return CLT_ERR;
    for (size_t i = 0; i < s->nlevels; ++i)
        if (mpz_vector_fwrite(s->zs[i], s->nzs, fp) == CLT_ERR) return CLT_ERR;
    for (size_t i = 0; i < s->nlevels; ++i)
        if (mpz_vector_fwrite(s->zinvs[i], s->nzs, fp) == CLT_ERR) return CLT_ERR;
    for (size_t i = 0; i < s->nlevels; ++i)
        if (mpz_vector_fwrite(s->ps[i], s->n, fp) == CLT_ERR) return CLT_ERR;
    for (size_t i = 0; i < s->nlevels; ++i)
        if (mpz_vector_fwrite(s->crt_coeffs[i], s->n, fp) == CLT_ERR) return CLT_ERR;
    if (mpz_vector_fwrite(s->x0s, s->nlevels, fp) == CLT_ERR) return CLT_ERR;
    for (size_t i = 0; i < s->nswitches; ++i)
        for (size_t j = 0; j < 2; ++j)
            if (switch_state_fwrite(fp, s->switches[i][j], s->theta) == CLT_ERR) return CLT_ERR;
    for (size_t i = 0; i < max(s->n, s->nzs); ++i)
        aes_randstate_fwrite(s->rngs[i], fp);
    return CLT_OK;
}

mpz_t *
clt_pl_state_moduli(const clt_pl_state_t *s)
{
    return s->gs;
}

size_t
clt_pl_state_nslots(const clt_pl_state_t *s)
{
    return s->n;
}

size_t
clt_pl_state_nzs(const clt_pl_state_t *s)
{
    return s->nzs;
}

int
clt_pl_encode(clt_elem_t *rop, const clt_pl_state_t *s, size_t n, mpz_t xs[n],
              clt_pl_encode_params_t *params)
{
    mpz_set_ui(rop->elem, 0);
    rop->level = params ? params->level : 0;
/* #pragma omp parallel for */
    for (size_t i = 0; i < s->n; ++i) {
        mpz_t tmp;
        mpz_init(tmp);
        mpz_urandomb_aes(tmp, s->rngs[i], 2 * s->b);
        mpz_mod_near_ui(tmp, tmp, s->b);
        mpz_mul(tmp, tmp, s->gs[i]);
        /* mpz_add(tmp, tmp, xs[slot(i, n, s->n)]); */
        if (i < n)
            mpz_add(tmp, tmp, xs[i]);
        mpz_mul(tmp, tmp, s->crt_coeffs[rop->level][i]);
/* #pragma omp critical */
        {
            mpz_add(rop->elem, rop->elem, tmp);
        }
        mpz_clear(tmp);
    }
    if (params && params->ix) {
        const int *ix = params->ix;
        mpz_t tmp;
        mpz_init(tmp);
        /* multiply by appropriate zinvs */
        for (size_t i = 0; i < s->nzs; ++i) {
            if (ix[i] <= 0) continue;
            mpz_powm_ui(tmp, s->zinvs[rop->level][i], ix[i], s->x0s[rop->level]);
            mpz_mul_mod_near(rop->elem, rop->elem, tmp, s->x0s[rop->level]);
        }
        mpz_clear(tmp);
    }
    return CLT_OK;
}
