#include "clt13.h"
#include "crt_tree.h"
#include "estimates.h"
#include "utils.h"

#include <assert.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <omp.h>

/* etap is the size of the factors in the p_i's when using the composite ps
 * optimization.  We default this to 420, as is done in
 * https://github.com/tlepoint/new-multilinear-maps/blob/master/generate_pp.cpp */
#define ETAP_DEFAULT 420

struct clt_elem_t {
    mpz_t elem;
};

struct clt_state_t {
    size_t n;
    size_t nzs;
    size_t rho;
    size_t nu;
    union {
        mpz_t x0;
        struct {
            mpz_t *x0s;
            size_t nmults;
        } polylog_t;
    };
    mpz_t pzt;
    mpz_t *gs;
    mpz_t *zinvs;
    aes_randstate_t *rngs;
    union {
        crt_tree *crt;
        mpz_t *crt_coeffs;
    };
    size_t flags;
};

struct clt_pp_t {
    mpz_t x0;
    mpz_t pzt;
    size_t nu;
};

static double current_time(void);
static int size_t_fread(FILE *const fp, size_t *x);
static int size_t_fwrite(FILE *const fp, size_t x);

static inline void
mpz_mod_near(mpz_t rop, const mpz_t a, const mpz_t p)
{
    mpz_t p_;
    mpz_init(p_);
    mpz_mod(rop, a, p);
    mpz_cdiv_q_ui(p_, p, 2);
    if (mpz_cmp(rop, p_) > 0)
        mpz_sub(rop, rop, p);
    mpz_clear(p_);
}

static inline void
mpz_mul_mod(mpz_t rop, mpz_t a, const mpz_t b, const mpz_t p)
{
    mpz_mul(rop, a, b);
    mpz_mod_near(rop, rop, p);
}

static inline void
mpz_random_(mpz_t rop, aes_randstate_t rng, size_t len)
{
    mpz_urandomb_aes(rop, rng, len);
    mpz_setbit(rop, len-1);
}

static inline void
mpz_prime(mpz_t rop, aes_randstate_t rng, size_t len)
{
    mpz_t p_unif;
    mpz_init(p_unif);
    do {
        mpz_random_(p_unif, rng, len);
        mpz_nextprime(rop, p_unif);
    } while (mpz_tstbit(rop, len) == 1);
    assert(mpz_tstbit(rop, len-1) == 1);
    mpz_clear(p_unif);
}

////////////////////////////////////////////////////////////////////////////////
// encodings

void
clt_encode(clt_elem_t *rop, const clt_state_t *s, size_t nins, mpz_t *ins, const int *pows)
{
    if (!(s->flags & CLT_FLAG_OPT_PARALLEL_ENCODE)) {
        omp_set_num_threads(1);
    }

    if (s->flags & CLT_FLAG_OPT_CRT_TREE) {
        /* slots[i] = m[i] + r*g[i] */
        mpz_t *slots = mpz_vector_new(s->n);
#pragma omp parallel for
        for (size_t i = 0; i < s->n; i++) {
            mpz_random_(slots[i], s->rngs[i], s->rho);
            mpz_mul(slots[i], slots[i], s->gs[i]);
            if (i < nins)
                mpz_add(slots[i], slots[i], ins[i]);
        }
        crt_tree_do_crt(rop->elem, s->crt, slots);
        mpz_vector_free(slots, s->n);
    } else {
        /* Don't use CRT tree */
        mpz_set_ui(rop->elem, 0);
#pragma omp parallel for
        for (unsigned long i = 0; i < s->n; ++i) {
            mpz_t tmp;
            mpz_init(tmp);
            mpz_random_(tmp, s->rngs[i], s->rho);
            mpz_mul(tmp, tmp, s->gs[i]);
            if (i < nins)
                mpz_add(tmp, tmp, ins[i]);
            mpz_mul(tmp, tmp, s->crt_coeffs[i]);
#pragma omp critical
            {
                mpz_add(rop->elem, rop->elem, tmp);
            }
            mpz_clear(tmp);
        }
    }
    {
        mpz_t tmp;
        mpz_init(tmp);
        /* multiply by appropriate zinvs */
        for (unsigned long i = 0; i < s->nzs; ++i) {
            if (pows[i] <= 0)
                continue;
            mpz_powm_ui(tmp, s->zinvs[i], pows[i], s->x0);
            mpz_mul_mod(rop->elem, rop->elem, tmp, s->x0);
        }
        mpz_clear(tmp);
    }
}

int
clt_is_zero(const clt_elem_t *c, const clt_pp_t *pp)
{
    int ret;

    mpz_t tmp, x0_;
    mpz_inits(tmp, x0_, NULL);

    mpz_mul(tmp, c->elem, pp->pzt);
    mpz_mod_near(tmp, tmp, pp->x0);

    ret = mpz_sizeinbase(tmp, 2) < mpz_sizeinbase(pp->x0, 2) - pp->nu;
    mpz_clears(tmp, x0_, NULL);
    return ret ? 1 : 0;
}

clt_elem_t *
clt_elem_new(void)
{
    clt_elem_t *e = calloc(1, sizeof e[0]);
    mpz_init(e->elem);
    return e;
}

void
clt_elem_free(clt_elem_t *e)
{
    mpz_clear(e->elem);
    free(e);
}

void
clt_elem_set(clt_elem_t *a, const clt_elem_t *b)
{
    mpz_set(a->elem, b->elem);
}

void
clt_elem_add(clt_elem_t *rop, const clt_pp_t *pp, const clt_elem_t *a, const clt_elem_t *b)
{
    mpz_add(rop->elem, a->elem, b->elem);
    mpz_mod(rop->elem, rop->elem, pp->x0);
}

void
clt_elem_sub(clt_elem_t *rop, const clt_pp_t *pp, const clt_elem_t *a, const clt_elem_t *b)
{
    mpz_sub(rop->elem, a->elem, b->elem);
    mpz_mod(rop->elem, rop->elem, pp->x0);
}

void
clt_elem_mul(clt_elem_t *rop, const clt_pp_t *pp, const clt_elem_t *a, const clt_elem_t *b)
{
    mpz_mul(rop->elem, a->elem, b->elem);
    mpz_mod(rop->elem, rop->elem, pp->x0);
}

void
clt_elem_mul_ui(clt_elem_t *rop, const clt_pp_t *pp, const clt_elem_t *a, unsigned int b)
{
    mpz_mul_ui(rop->elem, a->elem, b);
    mpz_mod(rop->elem, rop->elem, pp->x0);
}

void
clt_elem_print(clt_elem_t *a)
{
    gmp_printf("%Zd", a->elem);
}

int
clt_elem_fread(clt_elem_t *x, FILE *fp)
{
    x = clt_elem_new();
    return mpz_fread(x->elem, fp);
}

int
clt_elem_fwrite(clt_elem_t *x, FILE *fp)
{
    return mpz_fwrite(x->elem, fp);
}

clt_state_t *
clt_state_fread(FILE *fp)
{
    clt_state_t *s;
    int ret = 1;

    s = calloc(1, sizeof s[0]);
    if (s == NULL)
        return NULL;

    if (size_t_fread(fp, &s->flags) == CLT_ERR) {
        fprintf(stderr, "[%s] couldn't read flags!\n", __func__);
        goto cleanup;
    }
    if (size_t_fread(fp, &s->n) == CLT_ERR) {
        fprintf(stderr, "[%s] couldn't read n!\n", __func__);
        goto cleanup;
    }
    if (size_t_fread(fp, &s->nzs) == CLT_ERR) {
        fprintf(stderr, "[%s] couldn't read nzs!\n", __func__);
        goto cleanup;
    }
    if (size_t_fread(fp, &s->rho) == CLT_ERR) {
        fprintf(stderr, "[%s] couldn't read rho!\n", __func__);
        goto cleanup;
    }
    if (size_t_fread(fp, &s->nu) == CLT_ERR) {
        fprintf(stderr, "[%s] couldn't read nu!\n", __func__);
        goto cleanup;
    }

    mpz_init(s->x0);
    if (mpz_fread(s->x0, fp) == CLT_ERR) {
        fprintf(stderr, "[%s] couldn't read x0!\n", __func__);
        goto cleanup;
    }
    mpz_init(s->pzt);
    if (mpz_fread(s->pzt, fp) == CLT_ERR) {
        fprintf(stderr, "[%s] couldn't read pzt!\n", __func__);
        goto cleanup;
    }

    s->gs = mpz_vector_new(s->n);
    if (mpz_vector_fread(s->gs, s->n, fp) == CLT_ERR) {
        fprintf(stderr, "[%s] couldn't read gs!\n", __func__);
        goto cleanup;
    }

    s->zinvs = mpz_vector_new(s->nzs);
    if (mpz_vector_fread(s->zinvs, s->nzs, fp) == CLT_ERR) {
        fprintf(stderr, "[%s] couldn't read zinvs!\n", __func__);
        goto cleanup;
    }

    if (s->flags & CLT_FLAG_OPT_CRT_TREE) {
        if ((s->crt = crt_tree_fread(fp, s->n)) == NULL) {
            fprintf(stderr, "[%s] couldn't read crt_tree!\n", __func__);
            goto cleanup;
        }
    } else {
        s->crt_coeffs = mpz_vector_new(s->n);
        if (mpz_vector_fread(s->crt_coeffs, s->n, fp) != 0) {
            fprintf(stderr, "[%s] couldn't read crt_coeffs!\n", __func__);
            goto cleanup;
        }
    }

    s->rngs = calloc(MAX(s->n, s->nzs), sizeof s->rngs[0]);
    for (size_t i = 0; i < MAX(s->n, s->nzs); ++i) {
        aes_randstate_fread(s->rngs[i], fp);
    }
    ret = 0;
cleanup:
    if (ret) {
        free(s);
        return NULL;
    } else {
        return s;
    }
}

int
clt_state_fwrite(clt_state_t *const s, FILE *const fp)
{
    int ret = CLT_ERR;

    if (size_t_fwrite(fp, s->flags) == CLT_ERR) {
        fprintf(stderr, "[%s] failed to save flags!\n", __func__);
        goto cleanup;
    }
    if (size_t_fwrite(fp, s->n) == CLT_ERR) {
        fprintf(stderr, "[%s] failed to save n!\n", __func__);
        goto cleanup;
    }
    if (size_t_fwrite(fp, s->nzs) == CLT_ERR) {
        fprintf(stderr, "[%s] failed to save n!\n", __func__);
        goto cleanup;
    }
    if (size_t_fwrite(fp, s->rho) == CLT_ERR) {
        fprintf(stderr, "[%s] failed to save rho!\n", __func__);
        goto cleanup;
    }
    if (size_t_fwrite(fp, s->nu) == CLT_ERR) {
        fprintf(stderr, "[%s] failed to save nu!\n", __func__);
        goto cleanup;
    }
    if (mpz_fwrite(s->x0, fp) == CLT_ERR) {
        fprintf(stderr, "[%s] failed to save x0!\n", __func__);
        goto cleanup;
    }
    if (mpz_fwrite(s->pzt, fp) == CLT_ERR) {
        fprintf(stderr, "[%s] failed to save pzt!\n", __func__);
        goto cleanup;
    }
    if (mpz_vector_fwrite(s->gs, s->n, fp) == CLT_ERR) {
        fprintf(stderr, "[%s] failed to save gs!\n", __func__);
        goto cleanup;
    }
    if (mpz_vector_fwrite(s->zinvs, s->nzs, fp) == CLT_ERR) {
        fprintf(stderr, "[%s] failed to save zinvs!\n", __func__);
        goto cleanup;
    }
    if (s->flags & CLT_FLAG_OPT_CRT_TREE) {
        if (crt_tree_fwrite(fp, s->crt, s->n) != 0) {
            fprintf(stderr, "[%s] failed to save crt_tree!\n", __func__);
            goto cleanup;
        }
    } else {
        if (mpz_vector_fwrite(s->crt_coeffs, s->n, fp) != 0) {
            fprintf(stderr, "[%s] failed to save crt_coefs!\n", __func__);
            goto cleanup;
        }
    }

    for (size_t i = 0; i < MAX(s->n, s->nzs); ++i) {
        aes_randstate_fwrite(s->rngs[i], fp);
    }

    ret = CLT_OK;
cleanup:
    return ret;
}

////////////////////////////////////////////////////////////////////////////////
// public parameters

clt_pp_t *
clt_pp_new(const clt_state_t *mmap)
{
    clt_pp_t *pp;

    pp = calloc(1, sizeof pp[0]);
    if (pp == NULL)
        return NULL;
    mpz_inits(pp->x0, pp->pzt, NULL);
    mpz_set(pp->x0, mmap->x0);
    mpz_set(pp->pzt, mmap->pzt);
    pp->nu = mmap->nu;
    return pp;
}

void
clt_pp_free(clt_pp_t *pp)
{
    mpz_clears(pp->x0, pp->pzt, NULL);
    free(pp);
}

clt_pp_t *
clt_pp_fread(FILE *fp)
{
    clt_pp_t *pp;
    int ret = CLT_ERR;

    if ((pp = calloc(1, sizeof pp[0])) == NULL)
        return NULL;
    mpz_inits(pp->x0, pp->pzt, NULL);

    if (size_t_fread(fp, &pp->nu) == CLT_ERR)
        goto cleanup;
    if (mpz_fread(pp->x0, fp) == CLT_ERR)
        goto cleanup;
    if (mpz_fread(pp->pzt, fp) == CLT_ERR)
        goto cleanup;
    ret = CLT_OK;
cleanup:
    if (ret == CLT_OK) {
        return pp;
    } else {
        clt_pp_free(pp);
        return NULL;
    }
}

int
clt_pp_fwrite(clt_pp_t *pp, FILE *fp)
{
    int ret = CLT_ERR;

    if (size_t_fwrite(fp, pp->nu) == CLT_ERR)
        goto cleanup;
    if (mpz_fwrite(pp->x0, fp) == CLT_ERR)
        goto cleanup;
    if (mpz_fwrite(pp->pzt, fp) == CLT_ERR)
        goto cleanup;
    ret = CLT_OK;
cleanup:
    return ret;
}

////////////////////////////////////////////////////////////////////////////////
// helper functions

static int
size_t_fread(FILE *fp, size_t *x)
{
    if (fread(x, sizeof x[0], 1, fp) != 1)
        return CLT_ERR;
    return CLT_OK;
}

static int
size_t_fwrite(FILE *fp, size_t x)
{
    if (fwrite(&x, sizeof x, 1, fp) != 1)
        return CLT_ERR;
    return CLT_OK;
}


static double
current_time(void)
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + (double) (t.tv_usec / 1000000.0);
}

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

static void
print_progress(size_t cur, size_t total)
{
    static int last_val = 0;
    double percentage = (double) cur / total;
    int val  = percentage * 100;
    int lpad = percentage * PBWIDTH;
    int rpad = PBWIDTH - lpad;
    if (val != last_val) {
        fprintf(stderr, "\r\t%3d%% [%.*s%*s] %lu/%lu", val, lpad, PBSTR, rpad, "", cur, total);
        fflush(stderr);
        last_val = val;
    }
}

static inline size_t nb_of_bits(size_t x)
{
    size_t nb = 0;
    while (x > 0) {
        x >>= 1;
        nb++;
    }
    return nb;
}

static int
gen_primes(mpz_t *v, aes_randstate_t *rngs, size_t n, size_t len, bool verbose)
{
    double start = current_time();
    int count = 0;
    fprintf(stderr, "%lu", len);
    print_progress(count, n);
#pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        mpz_prime(v[i], rngs[i], len);
        if (verbose) {
#pragma omp critical
            print_progress(++count, n);
        }
    }
    if (verbose)
        fprintf(stderr, "\t[%.2fs]\n", current_time() - start);
    return CLT_OK;
}

static int
gen_primes_composite_ps(mpz_t *v, aes_randstate_t *rngs, size_t n, size_t eta, bool verbose)
{
    int count = 0;
    double start = current_time();
    size_t etap = ETAP_DEFAULT;
    if (eta > 350)
        /* TODO: change how we set etap, should be resistant to factoring x_0 */
        for (/* */; eta % etap < 350; etap++)
            ;
    if (verbose) {
        fprintf(stderr, " [eta_p: %lu] ", etap);
    }
    size_t nchunks = eta / etap;
    size_t leftover = eta - nchunks * etap;
    if (verbose) {
        fprintf(stderr, "[nchunks=%lu leftover=%lu]\n", nchunks, leftover);
        print_progress(count, n);
    }
#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        mpz_t p_unif;
        mpz_set_ui(v[i], 1);
        mpz_init(p_unif);
        /* generate a p_i */
        for (size_t j = 0; j < nchunks; j++) {
            mpz_prime(p_unif, rngs[i], etap);
            mpz_mul(v[i], v[i], p_unif);
        }
        mpz_prime(p_unif, rngs[i], leftover);
        mpz_mul(v[i], v[i], p_unif);
        mpz_clear(p_unif);

        if (verbose) {
#pragma omp critical
            print_progress(++count, n);
        }
    }
    if (verbose) {
        fprintf(stderr, "\t[%.2fs]\n", current_time() - start);
    }
    return CLT_OK;
}

static void
crt_coeffs(mpz_t *coeffs, mpz_t *ps, size_t n, mpz_t x0, bool verbose)
{
    const double start = current_time();
    int count = 0;
    if (verbose)
        fprintf(stderr, "  Generating CRT coefficients:\n");
#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        mpz_t q;
        mpz_init(q);
        mpz_div(q, x0, ps[i]);
        mpz_invert(coeffs[i], q, ps[i]);
        mpz_mul_mod(coeffs[i], coeffs[i], q, x0);
        mpz_clear(q);
        if (verbose) {
#pragma omp critical
            print_progress(++count, n);
        }
    }
    if (verbose)
        fprintf(stderr, "\t[%.2fs]\n", current_time() - start);
}

static void
clt_state_new_polylog(clt_state_t *s, size_t nmults, size_t eta, bool verbose)
{
    mpz_t **ps;
    eta += 50 * nmults;

    ps = calloc(nmults + 1, sizeof ps[0]);
    for (size_t i = 0; i < nmults + 1; ++i) {
        ps[i] = mpz_vector_new(s->n);
        gen_primes(ps[i], s->rngs, s->n, eta, verbose);
        eta -= 50;
    }
}

clt_state_t *
clt_state_new(const clt_params_t *params, const clt_params_opt_t *opts,
              size_t ncores, size_t flags, aes_randstate_t rng)
{
    clt_state_t *s;
    size_t alpha, beta, eta, rho_f;
    mpz_t *ps, *zs;
    double start_time = 0.0;
    int count;
    const bool verbose = flags & CLT_FLAG_VERBOSE;
    const size_t min_slots = opts ? opts->min_slots : 0;

    if (flags & CLT_FLAG_POLYLOG &&
        (flags & CLT_FLAG_OPT_CRT_TREE || flags & CLT_FLAG_OPT_PARALLEL_ENCODE
         || flags & CLT_FLAG_OPT_COMPOSITE_PS)) {
        fprintf(stderr, "error: polylog not (yet) compatible with CLT optimizations\n");
        return NULL;
    }

    s = calloc(1, sizeof s[0]);
    if (s == NULL)
        return NULL;

    if (ncores == 0)
        ncores = sysconf(_SC_NPROCESSORS_ONLN);
    (void) omp_set_num_threads(ncores);

    /* calculate CLT parameters */
    s->nzs = params->nzs;
    alpha  = params->lambda;           /* bitsize of g_i primes */
    beta   = params->lambda;           /* bitsize of h_i entries */
    s->rho = params->lambda;           /* bitsize of randomness */
    rho_f  = params->kappa * (s->rho + alpha); /* max bitsize of r_i's */
    eta    = rho_f + alpha + beta + 9; /* bitsize of primes p_i */
    s->n   = MAX(estimate_n(params->lambda, eta, flags), min_slots); /* number of primes */
    eta    = rho_f + alpha + beta + nb_of_bits(s->n) + 9; /* bitsize of primes p_i */
    s->nu  = eta - beta - rho_f - nb_of_bits(s->n) - 3; /* number of msbs to extract */
    s->flags = flags;

    /* Loop until a fixed point reached for choosing eta, n, and nu */
    {
        size_t old_eta = 0, old_n = 0, old_nu = 0;
        int i = 0;
        for (; i < 10 && (old_eta != eta || old_n != s->n || old_nu != s->nu);
             ++i) {
            old_eta = eta, old_n = s->n, old_nu = s->nu;
            eta = rho_f + alpha + beta + nb_of_bits(s->n) + 9;
            s->n = MAX(estimate_n(params->lambda, eta, flags), min_slots);
            s->nu = eta - beta - rho_f - nb_of_bits(s->n) - 3;
        }

        if (i == 10 && (old_eta != eta || old_n != s->n || old_nu != s->nu)) {
            fprintf(stderr, "error: unable to find valid η, n, and ν choices\n");
            free(s);
            return NULL;
        }
    }

    /* Make sure the proper bounds are hit [CLT13, Lemma 8] */
    assert(s->nu >= alpha + 6);
    assert(beta + alpha + rho_f + nb_of_bits(s->n) <= eta - 9);
    assert(s->n >= min_slots);

    if (verbose) {
        fprintf(stderr, "  λ: %ld\n", params->lambda);
        fprintf(stderr, "  κ: %ld\n", params->kappa);
        fprintf(stderr, "  α: %ld\n", alpha);
        fprintf(stderr, "  β: %ld\n", beta);
        fprintf(stderr, "  η: %ld\n", eta);
        fprintf(stderr, "  ν: %ld\n", s->nu);
        fprintf(stderr, "  ρ: %ld\n", s->rho);
        fprintf(stderr, "  ρ_f: %ld\n", rho_f);
        fprintf(stderr, "  n: %ld\n", s->n);
        fprintf(stderr, "  nzs: %ld\n", s->nzs);
        fprintf(stderr, "  ncores: %ld\n", ncores);
        fprintf(stderr, "  Flags: \n");
        if (s->flags & CLT_FLAG_OPT_CRT_TREE)
            fprintf(stderr, "    CRT TREE\n");
        if (s->flags & CLT_FLAG_OPT_PARALLEL_ENCODE)
            fprintf(stderr, "    PARALLEL ENCODE\n");
        if (s->flags & CLT_FLAG_OPT_COMPOSITE_PS)
            fprintf(stderr, "    COMPOSITE PS\n");
        if (s->flags & CLT_FLAG_SEC_IMPROVED_BKZ)
            fprintf(stderr, "    IMPROVED BKZ\n");
        if (s->flags & CLT_FLAG_SEC_CONSERVATIVE)
            fprintf(stderr, "    CONSERVATIVE\n");
        if (s->flags & CLT_FLAG_POLYLOG)
            fprintf(stderr, "    POLYLOG\n");
    }

    /* Generate randomness for each core */
    s->rngs = calloc(MAX(s->n, s->nzs), sizeof s->rngs[0]);
    for (size_t i = 0; i < MAX(s->n, s->nzs); ++i) {
        unsigned char *buf;
        size_t nbytes;

        buf = random_aes(rng, 128, &nbytes);
        aes_randinit_seedn(s->rngs[i], (char *) buf, nbytes, NULL, 0);
        free(buf);
    }

    s->zinvs = mpz_vector_new(s->nzs);
    s->gs = mpz_vector_new(s->n);
    if (!(s->flags & CLT_FLAG_OPT_CRT_TREE)) {
        s->crt_coeffs = mpz_vector_new(s->n);
    }
    mpz_init_set_ui(s->x0,  1);
    mpz_init_set_ui(s->pzt, 0);

    if (s->flags & CLT_FLAG_POLYLOG) {
        clt_state_new_polylog(s, 1 /* FIXME: */, eta, verbose);
        return s;
    }

    ps = mpz_vector_new(s->n);
    zs = mpz_vector_new(s->nzs);

    if (verbose) {
        fprintf(stderr, "  Generating p_i's and g_i's:");
        start_time = current_time();
    }

generate_ps:
    if (s->flags & CLT_FLAG_OPT_COMPOSITE_PS) {
        gen_primes_composite_ps(ps, s->rngs, s->n, eta, verbose);
    } else {
        if (verbose) fprintf(stderr, "\n");
        gen_primes(ps, s->rngs, s->n, eta, verbose);
    }
    if (opts && opts->moduli && opts->nmoduli) {
        for (size_t i = 0; i < opts->nmoduli; ++i)
            mpz_set(s->gs[i], opts->moduli[i]);
        gen_primes(s->gs + opts->nmoduli, s->rngs, s->n - opts->nmoduli, alpha, verbose);
    } else {
        gen_primes(s->gs, s->rngs, s->n, alpha, verbose);
    }

    if (s->flags & CLT_FLAG_OPT_CRT_TREE) {
        if (verbose) {
            fprintf(stderr, "  Generating CRT-Tree: ");
            start_time = current_time();
        }
        s->crt = crt_tree_new(ps, s->n);
        if (s->crt == NULL) {
            /* if crt_tree_init fails, regenerate with new p_i's */
            if (verbose)
                fprintf(stderr, "(restarting) ");
            goto generate_ps;
        }
        /* crt_tree_init succeeded, set x0 */
        mpz_set(s->x0, s->crt->mod);
        if (verbose) {
            fprintf(stderr, "[%.2fs]\n", current_time() - start_time);
        }
    } else {
        /* Don't use CRT tree optimization */
        if (verbose) {
            fprintf(stderr, "  Computing x0: \n");
            start_time = current_time();
        }

        /* calculate x0 the hard way */
        /* TODO: could parallelize this if desired */
        for (size_t i = 0; i < s->n; i++) {
            mpz_mul(s->x0, s->x0, ps[i]);
            if (verbose)
                print_progress(i, s->n-1);
        }

        if (verbose)
            fprintf(stderr, "\t[%.2fs]\n", current_time() - start_time);

        crt_coeffs(s->crt_coeffs, ps, s->n, s->x0, verbose);
    }

    /* Compute z_i's */
    if (verbose) {
        fprintf(stderr, "  Generating z_i's:\n");
        start_time = current_time();
        count = 0;
    }
#pragma omp parallel for
    for (size_t i = 0; i < s->nzs; ++i) {
        do {
            mpz_urandomm_aes(zs[i], s->rngs[i], s->x0);
        } while (mpz_invert(s->zinvs[i], zs[i], s->x0) == 0);
        if (verbose) {
#pragma omp critical
            print_progress(++count, s->nzs);
        }
    }
    if (verbose) {
        fprintf(stderr, "\t[%.2fs]\n", current_time() - start_time);
    }

    /* Compute pzt */
    if (verbose) {
        fprintf(stderr, "  Generating pzt:\n");
        start_time = current_time();
        count = 0;
    }

    {
        mpz_t zk;
        mpz_init_set_ui(zk, 1);
        /* compute z_1^t_1 ... z_k^t_k mod x0 */
        for (size_t i = 0; i < s->nzs; ++i) {
            mpz_t tmp;
            mpz_init(tmp);
            mpz_powm_ui(tmp, zs[i], params->pows[i], s->x0);
            mpz_mul_mod(zk, zk, tmp, s->x0);
            mpz_clear(tmp);
            if (verbose) {
                print_progress(++count, s->n + s->nzs);
            }
        }
#pragma omp parallel for
        for (size_t i = 0; i < s->n; ++i) {
            mpz_t tmp, qpi, rnd;
            mpz_inits(tmp, qpi, rnd, NULL);
            /* compute ((g_i^{-1} mod p_i) * z * r_i * (x0 / p_i) */
            mpz_invert(tmp, s->gs[i], ps[i]);
            mpz_mul_mod(tmp, tmp, zk, ps[i]);
            do {
                mpz_random_(rnd, s->rngs[i], beta);
            } while (mpz_cmp(rnd, s->gs[i]) == 0);
            mpz_mul(tmp, tmp, rnd);
            mpz_div(qpi, s->x0, ps[i]);
            mpz_mul_mod(tmp, tmp, qpi, s->x0);
#pragma omp critical
            {
                mpz_add(s->pzt, s->pzt, tmp);
            }
            mpz_clears(tmp, qpi, rnd, NULL);
            if (verbose) {
#pragma omp critical
                print_progress(++count, s->n + s->nzs);
            }
        }
        mpz_mod_near(s->pzt, s->pzt, s->x0);
        mpz_clear(zk);
    }
    if (verbose) {
        fprintf(stderr, "\t[%.2fs]\n", current_time() - start_time);
    }

    mpz_vector_free(ps, s->n);
    mpz_vector_free(zs, s->nzs);

    return s;
}

void
clt_state_free(clt_state_t *s)
{
    mpz_clears(s->x0, s->pzt, NULL);
    mpz_vector_free(s->gs, s->n);
    mpz_vector_free(s->zinvs, s->nzs);
    if (s->flags & CLT_FLAG_OPT_CRT_TREE) {
        crt_tree_free(s->crt);
    } else {
        mpz_vector_free(s->crt_coeffs, s->n);
    }
    if (s->rngs) {
        for (size_t i = 0; i < MAX(s->n, s->nzs); ++i) {
            aes_randclear(s->rngs[i]);
        }
        free(s->rngs);
    }
    free(s);
}

mpz_t *
clt_state_moduli(const clt_state_t *s)
{
    return s->gs;
}

size_t
clt_state_nslots(const clt_state_t *s)
{
    return s->n;
}

size_t
clt_state_nzs(const clt_state_t *s)
{
    return s->nzs;
}
