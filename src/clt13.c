#include "clt13.h"
#include <assert.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#ifndef RANDFILE
#  define RANDFILE "/dev/urandom"
#endif

#if OPTIMIZATION_CRT_TREE
static int  crt_tree_init   (crt_tree *crt, const mpz_t *ps, size_t nps);
static void crt_tree_clear  (crt_tree *crt);
static void crt_tree_do_crt (mpz_t rop, const crt_tree *crt, const mpz_t *cs);
static void crt_tree_save   (const char *fname, crt_tree *crt, size_t n);
static void crt_tree_load   (const char *fname, crt_tree *crt, size_t n);
#endif

static int seed_rng (gmp_randstate_t *rng);

static double current_time();

static int load_ulong (const char *fname, ulong *x);
static int save_ulong (const char *fname, const ulong x);
static int load_mpz_scalar (const char *fname, mpz_t x);
static int save_mpz_scalar (const char *fname, const mpz_t x);
static int load_mpz_vector (const char *fname, mpz_t *m, const int len);
static int save_mpz_vector (const char *fname, const mpz_t *m, const int len);

////////////////////////////////////////////////////////////////////////////////
// state

void clt_state_init (clt_state *s, ulong kappa, ulong lambda, ulong nzs, const int *pows)
{
    ulong alpha, beta, eta, rho_f;
    mpz_t *ps, *zs;
    double start_time;

    // calculate CLT parameters
    s->nzs = nzs;
    alpha  = lambda;
    beta   = lambda;
    s->rho = lambda;
    rho_f  = kappa * (s->rho + alpha + 2);
    eta    = rho_f + alpha + 2 * beta + lambda + 8;
    s->nu  = eta - beta - rho_f - lambda - 3;
    s->n   = eta * log2((float) lambda);

    if (g_verbose) {
        fprintf(stderr, "  Security Parameter: %ld\n", lambda);
        fprintf(stderr, "  Kappa: %ld\n", kappa);
        fprintf(stderr, "  Alpha: %ld\n", alpha);
        fprintf(stderr, "  Beta: %ld\n", beta);
        fprintf(stderr, "  Eta: %ld\n", eta);
        fprintf(stderr, "  Nu: %ld\n", s->nu);
        fprintf(stderr, "  Rho: %ld\n", s->rho);
        fprintf(stderr, "  Rho_f: %ld\n", rho_f);
        fprintf(stderr, "  N: %ld\n", s->n);
        fprintf(stderr, "  Number of Zs: %ld\n", s->nzs);
    }

    ps       = malloc(sizeof(mpz_t) * s->n);
    s->gs    = malloc(sizeof(mpz_t) * s->n);
    zs       = malloc(sizeof(mpz_t) * s->nzs);
    s->zinvs = malloc(sizeof(mpz_t) * s->nzs);

#if OPTIMIZATION_CRT_TREE
    s->crt = malloc(sizeof(crt_tree));
#else
    s->crt_coeffs = malloc(s->n * sizeof(mpz_t));
    for (ulong i = 0; i < s->n; i++) {
        mpz_init(s->crt_coeffs[i]);
    }
#endif

    seed_rng(&s->rng);

    // initialize gmp variables
    mpz_init_set_ui(s->x0,  1);
    mpz_init_set_ui(s->pzt, 0);
    for (ulong i = 0; i < s->n; ++i) {
        mpz_init_set_ui(ps[i], 1);
        mpz_init(s->gs[i]);
    }
    for (ulong i = 0; i < s->nzs; ++i) {
        mpz_inits(zs[i], s->zinvs[i], NULL);
    }

    // Generate p_i's and g_i's, as well as x0 = \prod p_i
    if (g_verbose) fprintf(stderr, "  Generating p_i's and g_i's");
    start_time = current_time();
GEN_PIS:
#pragma omp parallel for
#if OPTIMIZATION_COMPOSITE_PS
    for (ulong i = 0; i < s->n; i++) {
        // TODO: add chunking optimization
    }
#else
    for (ulong i = 0; i < s->n; i++) {
        mpz_t p_unif;
        mpz_init(p_unif);
        mpz_urandomb(p_unif, s->rng, eta);
        mpz_nextprime(ps[i], p_unif);
        mpz_urandomb(p_unif, s->rng, alpha);
        mpz_nextprime(s->gs[i], p_unif);
        mpz_clear(p_unif);
    }
#endif
#if OPTIMIZATION_CRT_TREE
    // use crt_tree to find x0
    int ok = crt_tree_init(s->crt, ps, s->n);
    if (!ok) {
        // if crt_tree_init fails, regenerate with new p_i's
        crt_tree_clear(s->crt);
        if (g_verbose) fprintf(stderr, " (restarting)");
        goto GEN_PIS;
    }
    // crt_tree_init succeeded, set x0
    mpz_set(s->x0, s->crt->mod);
    if (g_verbose) fprintf(stderr, ": %f\n", current_time() - start_time);
#else
    // find x0 the hard way
    for (ulong i = 0; i < s->n; i++) {
        mpz_mul(s->x0, s->x0, ps[i]);
    }
    if (g_verbose) fprintf(stderr, ": %f\n", current_time() - start_time);

    // Compute CRT coefficients
    if (g_verbose) fprintf(stderr, "  Generating CRT coefficients: ");
    start_time = current_time();
#pragma omp parallel for
    for (unsigned long i = 0; i < s->n; i++) {
        mpz_t q;
        mpz_init(q);
        mpz_tdiv_q(q, s->x0, ps[i]);
        mpz_invert(s->crt_coeffs[i], q, ps[i]);
        mpz_mul(s->crt_coeffs[i], s->crt_coeffs[i], q);
        mpz_mod(s->crt_coeffs[i], s->crt_coeffs[i], s->x0);
        mpz_clear(q);
    }
    if (g_verbose) fprintf(stderr, ": %f\n", current_time() - start_time);
#endif

    // Compute z_i's
    if (g_verbose) fprintf(stderr, "  Generating z_i's");
    start_time = current_time();
#pragma omp parallel for
    for (ulong i = 0; i < s->nzs; ++i) {
        do {
            mpz_urandomm(zs[i], s->rng, s->x0);
        } while (mpz_invert(s->zinvs[i], zs[i], s->x0) == 0);
    }
    if (g_verbose) fprintf(stderr, ": %f\n", current_time() - start_time);

    // Compute pzt
    if (g_verbose) fprintf(stderr, "  Generating pzt");
    start_time = current_time();
    {
        mpz_t zk;
        mpz_init_set_ui(zk, 1);
        // compute z_1^t_1 ... z_k^t_k mod q
        for (ulong i = 0; i < s->nzs; ++i) {
            mpz_t tmp;
            mpz_init(tmp);
            mpz_powm_ui(tmp, zs[i], pows[i], s->x0);
            mpz_mul(zk, zk, tmp);
            mpz_mod(zk, zk, s->x0);
            mpz_clear(tmp);
        }
#pragma omp parallel for
        for (ulong i = 0; i < s->n; ++i) {
            mpz_t tmp, qpi, rnd;
            mpz_inits(tmp, qpi, rnd, NULL);
            // compute (((g_i)^{-1} mod p_i) * z^k mod p_i) * r_i * (q / p_i)
            mpz_invert(tmp, s->gs[i], ps[i]);
            mpz_mul(tmp, tmp, zk);
            mpz_mod(tmp, tmp, ps[i]);
            mpz_urandomb(rnd, s->rng, beta);
            mpz_mul(tmp, tmp, rnd);
            mpz_div(qpi, s->x0, ps[i]);
            mpz_mul(tmp, tmp, qpi);
            mpz_mod(tmp, tmp, s->x0);
#pragma omp critical
            {
                mpz_add(s->pzt, s->pzt, tmp);
            }
            mpz_clears(tmp, qpi, rnd, NULL);
        }
        mpz_mod(s->pzt, s->pzt, s->x0);
        mpz_clear(zk);
    }
    if (g_verbose) fprintf(stderr, ": %f\n", current_time() - start_time);

    for (ulong i = 0; i < s->n; i++)
        mpz_clear(ps[i]);
    free(ps);

    for (ulong i = 0; i < s->nzs; ++i)
        mpz_clear(zs[i]);
    free(zs);
}

void clt_state_clear(clt_state *s)
{
    gmp_randclear(s->rng);
    mpz_clears(s->x0, s->pzt, NULL);
    for (ulong i = 0; i < s->n; i++) {
        mpz_clear(s->gs[i]);
    }
    free(s->gs);
    for (ulong i = 0; i < s->nzs; i++) {
        mpz_clear(s->zinvs[i]);
    }
    free(s->zinvs);
#if OPTIMIZATION_CRT_TREE
    crt_tree_clear(s->crt);
    free(s->crt);
#else
    for (ulong i = 0; i < s->n; i++) {
        mpz_clear(s->crt_coeffs[i]);
    }
    free(s->crt_coeffs);
#endif
}

void clt_state_read(clt_state *s, const char *dir)
{
    char *fname;
    int len = strlen(dir) + 10;
    fname = malloc(sizeof(char) + len);

    seed_rng(&s->rng);

    snprintf(fname, len, "%s/n", dir);
    load_ulong(fname, &s->n);

    snprintf(fname, len, "%s/nzs", dir);
    load_ulong(fname, &s->nzs);

    snprintf(fname, len, "%s/rho", dir);
    load_ulong(fname, &s->rho);

    snprintf(fname, len, "%s/nu", dir);
    load_ulong(fname, &s->nu);

    s->gs    = malloc(sizeof(mpz_t) * s->n);
    s->zinvs = malloc(sizeof(mpz_t) * s->nzs);

    mpz_inits(s->x0, s->pzt, NULL);
    for (ulong i = 0; i < s->n; i++)
        mpz_init(s->gs[i]);
    for (ulong i = 0; i < s->nzs; i++)
        mpz_init(s->zinvs[i]);

    snprintf(fname, len, "%s/x0", dir);
    load_mpz_scalar(fname, s->x0);

    snprintf(fname, len, "%s/pzt", dir);
    load_mpz_scalar(fname, s->pzt);

    snprintf(fname, len, "%s/gs", dir);
    load_mpz_vector(fname, s->gs, s->n);

    snprintf(fname, len, "%s/zinvs", dir);
    load_mpz_vector(fname, s->zinvs, s->nzs);

#if OPTIMIZATION_CRT_TREE
    s->crt = malloc(sizeof(crt_tree));
    snprintf(fname, len, "%s/crt_tree", dir);
    crt_tree_load(fname, s->crt, s->n);
#else
    s->crt_coeffs = malloc(sizeof(mpz_t) * s->n);
    for (ulong i = 0; i < s->n; i++)
        mpz_init(s->crt_coeffs[i]);
    snprintf(fname, len, "%s/crt_coeffs", dir);
    load_mpz_vector(fname, s->crt_coeffs, s->n);
#endif
    free(fname);
}


void clt_state_save(const clt_state *s, const char *dir)
{
    char *fname;
    int len = strlen(dir) + 10;
    fname = malloc(sizeof(char) + len);

    snprintf(fname, len, "%s/n", dir);
    save_ulong(fname, s->n);

    snprintf(fname, len, "%s/nzs", dir);
    save_ulong(fname, s->nzs);

    snprintf(fname, len, "%s/rho", dir);
    save_ulong(fname, s->rho);

    snprintf(fname, len, "%s/nu", dir);
    save_ulong(fname, s->nu);

    snprintf(fname, len, "%s/x0", dir);
    save_mpz_scalar(fname, s->x0);

    snprintf(fname, len, "%s/pzt", dir);
    save_mpz_scalar(fname, s->pzt);

    snprintf(fname, len, "%s/gs", dir);
    save_mpz_vector(fname, s->gs, s->n);

    snprintf(fname, len, "%s/zinvs", dir);
    save_mpz_vector(fname, s->zinvs, s->nzs);

#if OPTIMIZATION_CRT_TREE
    snprintf(fname, len, "%s/crt_tree", dir);
    crt_tree_save(fname, s->crt, s->n);
#else
    snprintf(fname, len, "%s/crt_coeffs", dir);
    save_mpz_vector(fname, s->crt_coeffs, s->n);
#endif
    free(fname);
}

////////////////////////////////////////////////////////////////////////////////
// public parameters

void clt_pp_init(clt_pp *pp, clt_state *mmap)
{
    mpz_inits(pp->x0, pp->pzt, NULL);
    mpz_set(pp->x0, mmap->x0);
    mpz_set(pp->pzt, mmap->pzt);
    pp->nu = mmap->nu;
}

void clt_pp_clear( clt_pp *pp )
{
    mpz_clears(pp->x0, pp->pzt, NULL);
}

void clt_pp_read(clt_pp *pp, const char *dir)
{
    char *fname;
    int len = strlen(dir) + 10;
    fname = malloc(sizeof(char) + len);

    mpz_inits(pp->x0, pp->pzt, NULL);

    // load nu
    snprintf(fname, len, "%s/nu", dir);
    load_ulong(fname, &pp->nu);

    // load x0
    snprintf(fname, len, "%s/x0", dir);
    load_mpz_scalar(fname, pp->x0);

    // load pzt
    snprintf(fname, len, "%s/pzt", dir);
    load_mpz_scalar(fname, pp->pzt);

    free(fname);
}

void clt_pp_save(const clt_pp *pp, const char *dir)
{
    char *fname;
    int len = strlen(dir) + 10;
    fname = malloc(sizeof(char) * len);

    // save nu
    snprintf(fname, len, "%s/nu", dir);
    save_ulong(fname, pp->nu);

    // save x0
    snprintf(fname, len, "%s/x0", dir);
    save_mpz_scalar(fname, pp->x0);

    // save pzt
    snprintf(fname, len, "%s/pzt", dir);
    save_mpz_scalar(fname, pp->pzt);

    free(fname);
}

////////////////////////////////////////////////////////////////////////////////
// encodings

#if OPTIMIZATION_CRT_TREE
void clt_encode(mpz_t rop, clt_state *s, size_t nins, const mpz_t *ins, const int *pows)
{
    // slots[i] = m[i] + r*g[i]
    mpz_t *slots = malloc(s->n * sizeof(mpz_t));
    for (int i = 0; i < s->n; i++) {
        mpz_init(slots[i]);
        mpz_urandomb(slots[i], s->rng, s->rho);
        mpz_mul(slots[i], slots[i], s->gs[i]);
        if (i < nins)
            mpz_add(slots[i], slots[i], ins[i]);
    }

    crt_tree_do_crt(rop, s->crt, slots);

    for (int i = 0; i < s->n; i++)
        mpz_clear(slots[i]);
    free(slots);

    // multiply by appropriate zinvs
    mpz_t tmp, zinv;
    mpz_inits(tmp, zinv, NULL);
    mpz_set_ui(zinv, 1);
    for (int i = 0; i < s->nzs; i++) {
        if (pows[i] <= 0)
            continue;
        mpz_powm_ui(tmp, s->zinvs[i], pows[i], s->x0);
        mpz_mul(zinv, zinv, tmp);
        mpz_mod(zinv, zinv, s->x0);
    }

    mpz_mul(rop, rop, zinv);
    mpz_mod(rop, rop, s->x0);

    mpz_clears(tmp, zinv, NULL);
}
#else
void clt_encode(mpz_t rop, clt_state *s, size_t nins, const mpz_t *ins, const int *pows)
{
    mpz_t tmp;
    mpz_init(tmp);
    mpz_set_ui(rop, 0);
    for (unsigned long i = 0; i < s->n; ++i) {
        mpz_urandomb(tmp, s->rng, s->rho);
        mpz_mul(tmp, tmp, s->gs[i]);
        if (i < nins)
            mpz_add(tmp, tmp, ins[i]);
        mpz_mul(tmp, tmp, s->crt_coeffs[i]);
        mpz_add(rop, rop, tmp);
    }
    for (unsigned long i = 0; i < s->nzs; ++i) {
        if (pows[i] <= 0)
            continue;
        mpz_powm_ui(tmp, s->zinvs[i], pows[i], s->x0);
        mpz_mul(rop, rop, tmp);
        mpz_mod(rop, rop, s->x0);
    }
    mpz_clear(tmp);
}
#endif

int clt_is_zero(clt_pp *pp, const mpz_t c)
{
    int ret;

    mpz_t tmp, x0_;
    mpz_inits(tmp, x0_, NULL);

    mpz_mul(tmp, c, pp->pzt);
    mpz_mod(tmp, tmp, pp->x0);

    mpz_cdiv_q_ui(x0_, pp->x0, 2);
    if (mpz_cmp(tmp, x0_) > 0)
        mpz_sub(tmp, tmp, pp->x0);

    ret = mpz_sizeinbase(tmp, 2) < mpz_sizeinbase(pp->x0, 2) - pp->nu;
    mpz_clears(tmp, x0_, NULL);
    return ret;
}

////////////////////////////////////////////////////////////////////////////////
// crt_tree

#if OPTIMIZATION_CRT_TREE

int crt_tree_init (crt_tree *crt, const mpz_t *ps, size_t nps)
{
    int ok = 1;
    crt->n  = nps;
    crt->n2 = nps/2;
    assert(crt->n > 0);

    mpz_init(crt->mod);

    if (crt->n == 1) {
        crt->left  = NULL;
        crt->right = NULL;
        mpz_set(crt->mod, ps[0]);
    } else {
        crt->left  = malloc(sizeof(crt_tree));
        crt->right = malloc(sizeof(crt_tree));

        ok &= crt_tree_init(crt->left,  ps,           crt->n2);
        ok &= crt_tree_init(crt->right, ps + crt->n2, crt->n - crt->n2);

        mpz_t g;
        mpz_inits(g, crt->crt_left, crt->crt_right, NULL);

        mpz_set_ui(g, 0);
        mpz_gcdext(g, crt->crt_right, crt->crt_left, crt->left->mod, crt->right->mod);
        if (! (mpz_cmp_ui(g, 1) == 0)) // if g != 1, raise error
            ok &= 0;

        mpz_clear(g);

        mpz_mul(crt->crt_left,  crt->crt_left,  crt->right->mod);
        mpz_mul(crt->crt_right, crt->crt_right, crt->left->mod);
        mpz_mul(crt->mod, crt->left->mod, crt->right->mod);
    }
    return ok;
}

void crt_tree_clear (crt_tree *crt)
{
    if (crt->n != 1) {
        crt_tree_clear(crt->left);
        crt_tree_clear(crt->right);
        mpz_clears(crt->crt_left, crt->crt_right, NULL);
        free(crt->left);
        free(crt->right);
    }
    mpz_clear(crt->mod);
}

void crt_tree_do_crt (mpz_t rop, const crt_tree *crt, const mpz_t *cs)
{
    if (crt->n == 1) {
        mpz_set(rop, cs[0]);
        return;
    }

    mpz_t val_left, val_right, tmp;
    mpz_inits(val_left, val_right, tmp, NULL);

    crt_tree_do_crt(val_left,  crt->left,  cs);
    crt_tree_do_crt(val_right, crt->right, cs + crt->n2);

    mpz_mul(rop, val_left,  crt->crt_left);
    mpz_mul(tmp, val_right, crt->crt_right);
    mpz_add(rop, rop, tmp);
    mpz_mod(rop, rop, crt->mod);

    mpz_clears(val_left, val_right, tmp, NULL);
}

void _crt_tree_get_leafs (mpz_t *leafs, int *i, crt_tree *crt)
{
    if (crt->n == 1) {
        mpz_set(leafs[(*i)++], crt->mod);
        return;
    }
    _crt_tree_get_leafs(leafs, i, crt->left);
    _crt_tree_get_leafs(leafs, i, crt->right);
}

void crt_tree_save (const char *fname, crt_tree *crt, size_t n)
{
    mpz_t *ps = malloc(n * sizeof(mpz_t));
    for (int i = 0; i < n; i++)
        mpz_init(ps[i]);
    int ctr = 0;

    _crt_tree_get_leafs(ps, &ctr, crt);
    save_mpz_vector(fname, ps, n);

    for (int i = 0; i < n; i++)
        mpz_clear(ps[i]);
    free(ps);
}

void crt_tree_load (const char *fname, crt_tree *crt, size_t n)
{
    mpz_t *ps = malloc(n * sizeof(mpz_t));
    for (int i = 0; i < n; i++)
        mpz_init(ps[i]);

    load_mpz_vector(fname, ps, n);
    crt_tree_init(crt, ps, n);

    for (int i = 0; i < n; i++)
        mpz_clear(ps[i]);
    free(ps);
}

#endif

////////////////////////////////////////////////////////////////////////////////
// helper functions

int seed_rng (gmp_randstate_t *rng)
{
    int file;
    if ((file = open(RANDFILE, O_RDONLY)) == -1) {
        fprintf(stderr, "Error opening %s\n", RANDFILE);
        return 1;
    } else {
        ulong seed;
        if (read(file, &seed, sizeof seed) == -1) {
            fprintf(stderr, "Error reading from %s\n", RANDFILE);
            close(file);
            return 1;
        } else {
            if (g_verbose)
                fprintf(stderr, "  Seed: %lu\n", seed);

            gmp_randinit_default(*rng);
            gmp_randseed_ui(*rng, seed);
        }
    }
    if (file != -1)
        close(file);
    return 0;
}

static int load_ulong(const char *fname, ulong *x)
{
    FILE *f;
    if ((f = fopen(fname, "r")) == NULL) {
        perror(fname);
        return 1;
    }
    fscanf(f, "%lu", x);
    fclose(f);
    return 0;
}

static int save_ulong(const char *fname, const ulong x)
{
    FILE *f;
    if ((f = fopen(fname, "w")) == NULL) {
        perror(fname);
        return 1;
    }
    fprintf(f, "%lu", x);
    fclose(f);
    return 0;
}

int load_mpz_scalar(const char *fname, mpz_t x)
{
    FILE *f;
    if ((f = fopen(fname, "r")) == NULL) {
        perror(fname);
        return 1;
    }
    mpz_inp_raw(x, f);
    fclose(f);
    return 0;
}

int save_mpz_scalar(const char *fname, const mpz_t x)
{
    FILE *f;
    if ((f = fopen(fname, "w")) == NULL) {
        perror(fname);
        return 1;
    }
    if (mpz_out_raw(f, x) == 0) {
        fclose(f);
        return 1;
    }
    fclose(f);
    return 0;
}

int load_mpz_vector(const char *fname, mpz_t *m, const int len)
{
    FILE *f;
    if ((f = fopen(fname, "r")) == NULL) {
        perror(fname);
        return 1;
    }
    for (int i = 0; i < len; ++i) {
        mpz_inp_raw(m[i], f);
    }
    fclose(f);
    return 0;
}

int save_mpz_vector(const char *fname, const mpz_t *m, const int len)
{
    FILE *f;
    if ((f = fopen(fname, "w")) == NULL) {
        perror(fname);
        return 1;
    }
    for (int i = 0; i < len; ++i) {
        if (mpz_out_raw(f, m[i]) == 0) {
            (void) fclose(f);
            return 1;
        }
    }
    fclose(f);
    return 0;
}

double current_time(void)
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + (double) (t.tv_usec / 1000000.0);
}

