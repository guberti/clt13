#include "clt13.h"

#include <assert.h>
#include <fcntl.h>
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

typedef struct crt_tree {
    ulong n, n2;
    clt_elem_t mod;
    clt_elem_t crt_left;
    clt_elem_t crt_right;
    struct crt_tree *left;
    struct crt_tree *right;
} crt_tree;

#define GET_NEWLINE(fp) (fscanf(fp, "\n"))
#define PUT_NEWLINE(fp) (!(fprintf(fp, "\n") > 0))

static int  crt_tree_init   (crt_tree *crt, clt_elem_t *ps, size_t nps);
static void crt_tree_clear  (crt_tree *crt);
static void crt_tree_do_crt (clt_elem_t rop, const crt_tree *crt, clt_elem_t *cs);
static void crt_tree_read   (const char *fname, crt_tree *crt, size_t n);
static void crt_tree_save   (const char *fname, crt_tree *crt, size_t n);
static int  crt_tree_fread  (FILE *const fp, crt_tree *crt, size_t n);
static int  crt_tree_fsave  (FILE *const fp, crt_tree *crt, size_t n);

static double current_time(void);

static int ulong_read  (const char *fname, ulong *x);
static int ulong_save  (const char *fname, ulong x);
static int ulong_fread (FILE *const fp, ulong *x);
static int ulong_fsave (FILE *const fp, ulong x);

static inline ulong nb_of_bits(ulong x)
{
    ulong nb = 0;
    while (x > 0) {
        x >>= 1;
        nb++;
    }
    return nb;
}

////////////////////////////////////////////////////////////////////////////////
// state

void
clt_state_init (clt_state *s, ulong kappa, ulong lambda, ulong nzs,
                const int *pows, ulong flags, aes_randstate_t rng)
{
    ulong alpha, beta, eta, rho_f;
    clt_elem_t *ps, *zs;
    double start_time = 0.0;

    // calculate CLT parameters
    s->nzs = nzs;
    alpha  = lambda;            /* bitsize of g_i primes */
    beta   = lambda;            /* bitsize of matrix H entries */
    s->rho = lambda;            /* bitsize of randomness */
    rho_f = kappa * (s->rho + alpha);  /* max bitsize of r_i's */
    eta    = rho_f + alpha + beta;     /* bitsize of primes p_i */
    s->n   = eta * nb_of_bits(lambda); /* number of primes */
    s->nu = alpha - nb_of_bits(s->n) - 3; /* number of msbs to extract */
    s->flags = flags;

    if (s->flags & CLT_FLAG_VERBOSE) {
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

    ps       = malloc(sizeof(clt_elem_t) * s->n);
    s->gs    = malloc(sizeof(clt_elem_t) * s->n);
    zs       = malloc(sizeof(clt_elem_t) * s->nzs);
    s->zinvs = malloc(sizeof(clt_elem_t) * s->nzs);

    if (s->flags & CLT_FLAG_OPT_CRT_TREE) {
        s->crt = malloc(sizeof(crt_tree));
    } else {
        s->crt_coeffs = malloc(s->n * sizeof(clt_elem_t));
        for (ulong i = 0; i < s->n; i++) {
            mpz_init(s->crt_coeffs[i]);
        }
    }

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
    if (s->flags & CLT_FLAG_VERBOSE) {
        fprintf(stderr, "  Generating p_i's and g_i's: ");
        start_time = current_time();
    }

GEN_PIS:
    if (s->flags & CLT_FLAG_OPT_COMPOSITE_PS) {
        ulong etap = ETAP_DEFAULT;
        /* ignore if eta <= 350 for testing with small lambdas */
        if (eta > 350)
            for (/* */; eta % etap < 350; etap++)
                ;
        if (s->flags & CLT_FLAG_VERBOSE) {
            fprintf(stderr, "[eta_p: %lu] ", etap);
        }
        ulong nchunks = eta / etap;
        ulong leftover = eta - nchunks * etap;
        fprintf(stderr, "[nchunks=%lu leftover=%lu] ", nchunks, leftover);
#pragma omp parallel for
        for (ulong i = 0; i < s->n; i++) {
            clt_elem_t p_unif;
            mpz_set_ui(ps[i], 1);
            mpz_init(p_unif);
            // generate a p_i
            for (ulong j = 0; j < nchunks; j++) {
                mpz_urandomb_aes(p_unif, rng, etap);
                mpz_nextprime(p_unif, p_unif);
                mpz_mul(ps[i], ps[i], p_unif);
            }
            mpz_urandomb_aes(p_unif, rng, leftover);
            mpz_nextprime(p_unif, p_unif);
            mpz_mul(ps[i], ps[i], p_unif);
            // generate a g_i
            mpz_urandomb_aes(p_unif, rng, alpha);
            mpz_nextprime(s->gs[i], p_unif);
            mpz_clear(p_unif);
        }
    } else {
#pragma omp parallel for
        for (ulong i = 0; i < s->n; i++) {
            clt_elem_t p_unif;
            mpz_init(p_unif);
            mpz_urandomb_aes(p_unif, rng, eta);
            mpz_nextprime(ps[i], p_unif);
            mpz_urandomb_aes(p_unif, rng, alpha);
            mpz_nextprime(s->gs[i], p_unif);
            mpz_clear(p_unif);
        }
    }

    if (s->flags & CLT_FLAG_OPT_CRT_TREE) {
        // use crt_tree to find x0
        int ok = crt_tree_init(s->crt, ps, s->n);
        if (!ok) {
            // if crt_tree_init fails, regenerate with new p_i's
            crt_tree_clear(s->crt);
            if (s->flags & CLT_FLAG_VERBOSE) {
                fprintf(stderr, "(restarting) ");
            }
            goto GEN_PIS;
        }
        // crt_tree_init succeeded, set x0
        mpz_set(s->x0, s->crt->mod);
        if (s->flags & CLT_FLAG_VERBOSE) {
            fprintf(stderr, "%f\n", current_time() - start_time);
        }
    } else {
        // find x0 the hard way
        for (ulong i = 0; i < s->n; i++) {
            mpz_mul(s->x0, s->x0, ps[i]);
        }
        if (s->flags & CLT_FLAG_VERBOSE) {
            fprintf(stderr, "%f\n", current_time() - start_time);
        }

        // Compute CRT coefficients
        if (s->flags & CLT_FLAG_VERBOSE) {
            fprintf(stderr, "  Generating CRT coefficients: ");
            start_time = current_time();
        }
#pragma omp parallel for
        for (unsigned long i = 0; i < s->n; i++) {
            clt_elem_t q;
            mpz_init(q);
            mpz_tdiv_q(q, s->x0, ps[i]);
            mpz_invert(s->crt_coeffs[i], q, ps[i]);
            mpz_mul(s->crt_coeffs[i], s->crt_coeffs[i], q);
            mpz_mod(s->crt_coeffs[i], s->crt_coeffs[i], s->x0);
            mpz_clear(q);
        }
        if (s->flags & CLT_FLAG_VERBOSE) {
            fprintf(stderr, "%f\n", current_time() - start_time);
        }
    }

    // Compute z_i's
    if (s->flags & CLT_FLAG_VERBOSE) {
        fprintf(stderr, "  Generating z_i's: ");
        start_time = current_time();
    }
#pragma omp parallel for
    for (ulong i = 0; i < s->nzs; ++i) {
        do {
            mpz_urandomm_aes(zs[i], rng, s->x0);
        } while (mpz_invert(s->zinvs[i], zs[i], s->x0) == 0);
    }
    if (s->flags & CLT_FLAG_VERBOSE) {
        fprintf(stderr, "%f\n", current_time() - start_time);
    }

    // Compute pzt
    if (s->flags & CLT_FLAG_VERBOSE) {
        fprintf(stderr, "  Generating pzt: ");
        start_time = current_time();
    }
    {
        clt_elem_t zk;
        mpz_init_set_ui(zk, 1);
        // compute z_1^t_1 ... z_k^t_k mod q
        for (ulong i = 0; i < s->nzs; ++i) {
            clt_elem_t tmp;
            mpz_init(tmp);
            mpz_powm_ui(tmp, zs[i], pows[i], s->x0);
            mpz_mul(zk, zk, tmp);
            mpz_mod(zk, zk, s->x0);
            mpz_clear(tmp);
        }
#pragma omp parallel for
        for (ulong i = 0; i < s->n; ++i) {
            clt_elem_t tmp, qpi, rnd;
            mpz_inits(tmp, qpi, rnd, NULL);
            // compute (((g_i)^{-1} mod p_i) * z^k mod p_i) * r_i * (q / p_i)
            mpz_invert(tmp, s->gs[i], ps[i]);
            mpz_mul(tmp, tmp, zk);
            mpz_mod(tmp, tmp, ps[i]);
            mpz_urandomb_aes(rnd, rng, beta);
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
    if (s->flags & CLT_FLAG_VERBOSE) {
        fprintf(stderr, "%f\n", current_time() - start_time);
    }

    for (ulong i = 0; i < s->n; i++)
        mpz_clear(ps[i]);
    free(ps);

    for (ulong i = 0; i < s->nzs; ++i)
        mpz_clear(zs[i]);
    free(zs);
}

void clt_state_clear(clt_state *s)
{
    mpz_clears(s->x0, s->pzt, NULL);
    for (ulong i = 0; i < s->n; i++) {
        mpz_clear(s->gs[i]);
    }
    free(s->gs);
    for (ulong i = 0; i < s->nzs; i++) {
        mpz_clear(s->zinvs[i]);
    }
    free(s->zinvs);
    if (s->flags & CLT_FLAG_OPT_CRT_TREE) {
        crt_tree_clear(s->crt);
        free(s->crt);
    } else {
        for (ulong i = 0; i < s->n; i++) {
            mpz_clear(s->crt_coeffs[i]);
        }
        free(s->crt_coeffs);
    }
}

void clt_state_read(clt_state *s, const char *dir)
{
    char *fname;
    int len = strlen(dir) + 13;
    fname = malloc(sizeof(char) + len);

    snprintf(fname, len, "%s/flags", dir);
    ulong_read(fname, &s->flags);

    snprintf(fname, len, "%s/n", dir);
    ulong_read(fname, &s->n);

    snprintf(fname, len, "%s/nzs", dir);
    ulong_read(fname, &s->nzs);

    snprintf(fname, len, "%s/rho", dir);
    ulong_read(fname, &s->rho);

    snprintf(fname, len, "%s/nu", dir);
    ulong_read(fname, &s->nu);

    s->gs    = malloc(sizeof(clt_elem_t) * s->n);
    s->zinvs = malloc(sizeof(clt_elem_t) * s->nzs);

    mpz_inits(s->x0, s->pzt, NULL);
    for (ulong i = 0; i < s->n; i++)
        mpz_init(s->gs[i]);
    for (ulong i = 0; i < s->nzs; i++)
        mpz_init(s->zinvs[i]);

    snprintf(fname, len, "%s/x0", dir);
    clt_elem_read(fname, s->x0);

    snprintf(fname, len, "%s/pzt", dir);
    clt_elem_read(fname, s->pzt);

    snprintf(fname, len, "%s/gs", dir);
    clt_vector_read(fname, s->gs, s->n);

    snprintf(fname, len, "%s/zinvs", dir);
    clt_vector_read(fname, s->zinvs, s->nzs);

    if (s->flags & CLT_FLAG_OPT_CRT_TREE) {
        s->crt = malloc(sizeof(crt_tree));
        snprintf(fname, len, "%s/crt_tree", dir);
        crt_tree_read(fname, s->crt, s->n);
    } else {
        s->crt_coeffs = malloc(sizeof(clt_elem_t) * s->n);
        for (ulong i = 0; i < s->n; i++)
            mpz_init(s->crt_coeffs[i]);
        snprintf(fname, len, "%s/crt_coeffs", dir);
        clt_vector_read(fname, s->crt_coeffs, s->n);
    }
    free(fname);
}


void clt_state_save(const clt_state *s, const char *dir)
{
    char *fname;
    int len = strlen(dir) + 13;
    fname = malloc(sizeof(char) + len);

    snprintf(fname, len, "%s/flags", dir);
    ulong_save(fname, s->flags);

    snprintf(fname, len, "%s/n", dir);
    ulong_save(fname, s->n);

    snprintf(fname, len, "%s/nzs", dir);
    ulong_save(fname, s->nzs);

    snprintf(fname, len, "%s/rho", dir);
    ulong_save(fname, s->rho);

    snprintf(fname, len, "%s/nu", dir);
    ulong_save(fname, s->nu);

    snprintf(fname, len, "%s/x0", dir);
    clt_elem_save(fname, s->x0);

    snprintf(fname, len, "%s/pzt", dir);
    clt_elem_save(fname, s->pzt);

    snprintf(fname, len, "%s/gs", dir);
    clt_vector_save(fname, s->gs, s->n);

    snprintf(fname, len, "%s/zinvs", dir);
    clt_vector_save(fname, s->zinvs, s->nzs);

    if (s->flags & CLT_FLAG_OPT_CRT_TREE) {
        snprintf(fname, len, "%s/crt_tree", dir);
        crt_tree_save(fname, s->crt, s->n);
    } else {
        snprintf(fname, len, "%s/crt_coeffs", dir);
        clt_vector_save(fname, s->crt_coeffs, s->n);
    }
    free(fname);
}

int
clt_state_fread(FILE *const fp, clt_state *s)
{
    int ret = 1;

    if (ulong_fread(fp, &s->flags) || GET_NEWLINE(fp) != 0) {
        fprintf(stderr, "[clt_state_fread] couldn't read flags!\n");
        goto cleanup;
    }

    if (ulong_fread(fp, &s->n) || GET_NEWLINE(fp) != 0) {
        fprintf(stderr, "[clt_state_fread] couldn't read n!\n");
        goto cleanup;
    }

    if (ulong_fread(fp, &s->nzs) || GET_NEWLINE(fp) != 0) {
        fprintf(stderr, "[clt_state_fread] couldn't read nzs!\n");
        goto cleanup;
    }

    if (ulong_fread(fp, &s->rho) || GET_NEWLINE(fp) != 0) {
        fprintf(stderr, "[clt_state_fread] couldn't read rho!\n");
        goto cleanup;
    }

    if (ulong_fread(fp, &s->nu) || GET_NEWLINE(fp) != 0) {
        fprintf(stderr, "[clt_state_fread] couldn't read nu!\n");
        goto cleanup;
    }

    mpz_init(s->x0);

    if (clt_elem_fread(fp, s->x0) || GET_NEWLINE(fp) != 0) {
        fprintf(stderr, "[clt_state_fread] couldn't read x0!\n");
        goto cleanup;
    }

    mpz_init(s->pzt);

    if (clt_elem_fread(fp, s->pzt) || GET_NEWLINE(fp) != 0) {
        fprintf(stderr, "[clt_state_fread] couldn't read pzt!\n");
        goto cleanup;
    }

    s->gs = malloc(sizeof(clt_elem_t) * s->n);
    for (ulong i = 0; i < s->n; i++)
        mpz_init(s->gs[i]);
    if (clt_vector_fread(fp, s->gs, s->n) != 0) {
        fprintf(stderr, "[clt_state_fread] couldn't read gs!\n");
        goto cleanup;
    }

    if (GET_NEWLINE(fp) != 0) {
        fprintf(stderr, "[clt_state_fread] couldn't read newline!\n");
        goto cleanup;
    }

    s->zinvs = malloc(sizeof(clt_elem_t) * s->nzs);
    for (ulong i = 0; i < s->nzs; i++)
        mpz_init(s->zinvs[i]);
    if (clt_vector_fread(fp, s->zinvs, s->nzs) != 0) {
        fprintf(stderr, "[clt_state_fread] couldn't read zinvs!\n");
        goto cleanup;
    }

    if (GET_NEWLINE(fp) != 0) {
        fprintf(stderr, "[clt_state_fread] couldn't read newline!\n");
        goto cleanup;
    }

    if (s->flags & CLT_FLAG_OPT_CRT_TREE) {
        s->crt = malloc(sizeof(crt_tree));
        if (crt_tree_fread(fp, s->crt, s->n) != 0) {
            fprintf(stderr, "[clt_state_fread] couldn't read crt_tree!\n");
            goto cleanup;
        }
    } else {
        s->crt_coeffs = malloc(sizeof(clt_elem_t) * s->n);
        for (ulong i = 0; i < s->n; i++)
            mpz_init(s->crt_coeffs[i]);
        if (clt_vector_fread(fp, s->crt_coeffs, s->n) != 0) {
            fprintf(stderr, "[clt_state_fread] couldn't read crt_coeffs!\n");
            goto cleanup;
        }
    }
    ret = 0;
cleanup:
    return ret;
}

int
clt_state_fsave(FILE *const fp, const clt_state *s)
{
    int ret = 1;

    if (ulong_fsave(fp, s->flags) || PUT_NEWLINE(fp) != 0) {
        fprintf(stderr, "[clt_state_fsave] failed to save flags!\n");
        goto cleanup;
    }

    if (ulong_fsave(fp, s->n) || PUT_NEWLINE(fp) != 0) {
        fprintf(stderr, "[clt_state_fsave] failed to save n!\n");
        goto cleanup;
    }

    if (ulong_fsave(fp, s->nzs) || PUT_NEWLINE(fp) != 0) {
        fprintf(stderr, "[clt_state_fsave] failed to save n!\n");
        goto cleanup;
    }

    if (ulong_fsave(fp, s->rho) || PUT_NEWLINE(fp) != 0) {
        fprintf(stderr, "[clt_state_fsave] failed to save rho!\n");
        goto cleanup;
    }

    if (ulong_fsave(fp, s->nu) || PUT_NEWLINE(fp) != 0) {
        fprintf(stderr, "[clt_state_fsave] failed to save nu!\n");
        goto cleanup;
    }

    if (clt_elem_fsave(fp, s->x0) || PUT_NEWLINE(fp) != 0) {
        fprintf(stderr, "[clt_state_fsave] failed to save x0!\n");
        goto cleanup;
    }

    if (clt_elem_fsave(fp, s->pzt) || PUT_NEWLINE(fp) != 0) {
        fprintf(stderr, "[clt_state_fsave] failed to save pzt!\n");
        goto cleanup;
    }

    if (clt_vector_fsave(fp, s->gs, s->n) || PUT_NEWLINE(fp) != 0) {
        fprintf(stderr, "[clt_state_fsave] failed to save gs!\n");
        goto cleanup;
    }

    if (clt_vector_fsave(fp, s->zinvs, s->nzs) || PUT_NEWLINE(fp) != 0) {
        fprintf(stderr, "[clt_state_fsave] failed to save zinvs!\n");
        goto cleanup;
    }

    if (s->flags & CLT_FLAG_OPT_CRT_TREE) {
        if (crt_tree_fsave(fp, s->crt, s->n) != 0) {
            fprintf(stderr, "[clt_state_fsave] failed to save crt_tree!\n");
            goto cleanup;
        }
    } else {
        if (clt_vector_fsave(fp, s->crt_coeffs, s->n) != 0) {
            fprintf(stderr, "[clt_state_fsave] failed to save crt_coefs!\n");
            goto cleanup;
        }
    }
    ret = 0;
cleanup:
    return ret;
}

////////////////////////////////////////////////////////////////////////////////
// public parameters

void
clt_pp_init(clt_pp *pp, const clt_state *mmap)
{
    mpz_inits(pp->x0, pp->pzt, NULL);
    mpz_set(pp->x0, mmap->x0);
    mpz_set(pp->pzt, mmap->pzt);
    pp->nu = mmap->nu;
}

void
clt_pp_clear( clt_pp *pp )
{
    mpz_clears(pp->x0, pp->pzt, NULL);
}

int
clt_pp_read(clt_pp *pp, const char *dir)
{
    char *fname;
    int ret = 1;
    int len = strlen(dir) + 10;
    fname = malloc(sizeof(char) + len);

    mpz_inits(pp->x0, pp->pzt, NULL);

    // load nu
    snprintf(fname, len, "%s/nu", dir);
    if (ulong_read(fname, &pp->nu) != 0)
        goto cleanup;

    // load x0
    snprintf(fname, len, "%s/x0", dir);
    if (clt_elem_read(fname, pp->x0) != 0)
        goto cleanup;

    // load pzt
    snprintf(fname, len, "%s/pzt", dir);
    if (clt_elem_read(fname, pp->pzt) != 0)
        goto cleanup;

    ret = 0;
cleanup:
    free(fname);
    return ret;
}

int
clt_pp_save(const clt_pp *pp, const char *dir)
{
    char *fname;
    int ret = 1;
    int len = strlen(dir) + 10;
    fname = malloc(sizeof(char) * len);

    // save nu
    snprintf(fname, len, "%s/nu", dir);
    if (ulong_save(fname, pp->nu) != 0)
        goto cleanup;

    // save x0
    snprintf(fname, len, "%s/x0", dir);
    if (clt_elem_save(fname, pp->x0) != 0)
        goto cleanup;

    // save pzt
    snprintf(fname, len, "%s/pzt", dir);
    if (clt_elem_save(fname, pp->pzt) != 0)
        goto cleanup;

    ret = 0;
cleanup:
    free(fname);
    return ret;
}

int
clt_pp_fread(FILE *const fp, clt_pp *pp)
{
    int ret = 1;

    mpz_inits(pp->x0, pp->pzt, NULL);

    if (ulong_fread(fp, &pp->nu) != 0)
        goto cleanup;

    if (GET_NEWLINE(fp) != 0)
        goto cleanup;

    if (clt_elem_fread(fp, pp->x0) != 0)
        goto cleanup;

    if (GET_NEWLINE(fp) != 0)
        goto cleanup;

    if (clt_elem_fread(fp, pp->pzt) != 0)
        goto cleanup;

    ret = 0;
cleanup:
    return ret;
}

int
clt_pp_fsave(FILE *const fp, const clt_pp *pp)
{
    int ret = 1;

    if (ulong_fsave(fp, pp->nu) || PUT_NEWLINE(fp) != 0)
        goto cleanup;

    if (clt_elem_fsave(fp, pp->x0) || PUT_NEWLINE(fp) != 0)
        goto cleanup;

    if (clt_elem_fsave(fp, pp->pzt) || PUT_NEWLINE(fp) != 0)
        goto cleanup;

    ret = 0;
cleanup:
    return ret;
}

////////////////////////////////////////////////////////////////////////////////
// encodings

void
clt_encode(clt_elem_t rop, const clt_state *s, size_t nins, clt_elem_t *ins,
           const int *pows, aes_randstate_t rng)
{
    clt_elem_t tmp;
    mpz_init(tmp);

    if (s->flags & CLT_FLAG_OPT_CRT_TREE) {
        // slots[i] = m[i] + r*g[i]
        clt_elem_t *slots = malloc(s->n * sizeof(clt_elem_t));
        if (s->flags & CLT_FLAG_OPT_PARALLEL_ENCODE) {
#pragma omp parallel for
            for (ulong i = 0; i < s->n; i++) {
                mpz_init(slots[i]);
                mpz_urandomb_aes(slots[i], rng, s->rho);
                mpz_mul(slots[i], slots[i], s->gs[i]);
                if (i < nins)
                    mpz_add(slots[i], slots[i], ins[i]);
            }
        } else {
            for (ulong i = 0; i < s->n; i++) {
                mpz_init(slots[i]);
                mpz_urandomb_aes(slots[i], rng, s->rho);
                mpz_mul(slots[i], slots[i], s->gs[i]);
                if (i < nins)
                    mpz_add(slots[i], slots[i], ins[i]);
            }
        }

        crt_tree_do_crt(rop, s->crt, slots);

        for (ulong i = 0; i < s->n; i++)
            mpz_clear(slots[i]);
        free(slots);
    } else {
        mpz_set_ui(rop, 0);
        for (unsigned long i = 0; i < s->n; ++i) {
            mpz_urandomb_aes(tmp, rng, s->rho);
            mpz_mul(tmp, tmp, s->gs[i]);
            if (i < nins)
                mpz_add(tmp, tmp, ins[i]);
            mpz_mul(tmp, tmp, s->crt_coeffs[i]);
            mpz_add(rop, rop, tmp);
        }
    }
    // multiply by appropriate zinvs
    for (unsigned long i = 0; i < s->nzs; ++i) {
        if (pows[i] <= 0)
            continue;
        mpz_powm_ui(tmp, s->zinvs[i], pows[i], s->x0);
        mpz_mul(rop, rop, tmp);
        mpz_mod(rop, rop, s->x0);
    }
    mpz_clear(tmp);
}

int
clt_is_zero(const clt_pp *pp, const clt_elem_t c)
{
    int ret;

    clt_elem_t tmp, x0_;
    mpz_inits(tmp, x0_, NULL);

    mpz_mul(tmp, c, pp->pzt);
    mpz_mod(tmp, tmp, pp->x0);

    mpz_cdiv_q_ui(x0_, pp->x0, 2);
    if (mpz_cmp(tmp, x0_) > 0)
        mpz_sub(tmp, tmp, pp->x0);

    ret = mpz_sizeinbase(tmp, 2) < mpz_sizeinbase(pp->x0, 2) - pp->nu;
    mpz_clears(tmp, x0_, NULL);
    return ret ? 1 : 0;
}

////////////////////////////////////////////////////////////////////////////////
// crt_tree

static int
crt_tree_init(crt_tree *crt, clt_elem_t *ps, size_t nps)
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

        clt_elem_t g;
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

static void
crt_tree_clear(crt_tree *crt)
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

static void
crt_tree_do_crt(clt_elem_t rop, const crt_tree *crt, clt_elem_t *cs)
{
    if (crt->n == 1) {
        mpz_set(rop, cs[0]);
        return;
    }

    clt_elem_t val_left, val_right, tmp;
    mpz_inits(val_left, val_right, tmp, NULL);

    crt_tree_do_crt(val_left,  crt->left,  cs);
    crt_tree_do_crt(val_right, crt->right, cs + crt->n2);

    mpz_mul(rop, val_left,  crt->crt_left);
    mpz_mul(tmp, val_right, crt->crt_right);
    mpz_add(rop, rop, tmp);
    mpz_mod(rop, rop, crt->mod);

    mpz_clears(val_left, val_right, tmp, NULL);
}

static void
_crt_tree_get_leafs(clt_elem_t *leafs, int *i, crt_tree *crt)
{
    if (crt->n == 1) {
        mpz_set(leafs[(*i)++], crt->mod);
        return;
    }
    _crt_tree_get_leafs(leafs, i, crt->left);
    _crt_tree_get_leafs(leafs, i, crt->right);
}

static void
crt_tree_save(const char *fname, crt_tree *crt, size_t n)
{
    clt_elem_t *ps = malloc(n * sizeof(clt_elem_t));
    for (ulong i = 0; i < n; i++)
        mpz_init(ps[i]);
    int ctr = 0;

    _crt_tree_get_leafs(ps, &ctr, crt);
    clt_vector_save(fname, ps, n);

    for (ulong i = 0; i < n; i++)
        mpz_clear(ps[i]);
    free(ps);
}

static void
crt_tree_read(const char *fname, crt_tree *crt, size_t n)
{
    clt_elem_t *ps = malloc(n * sizeof(clt_elem_t));
    for (ulong i = 0; i < n; i++)
        mpz_init(ps[i]);

    clt_vector_read(fname, ps, n);
    crt_tree_init(crt, ps, n);

    for (ulong i = 0; i < n; i++)
        mpz_clear(ps[i]);
    free(ps);
}

static int
crt_tree_fread (FILE *const fp, crt_tree *crt, size_t n)
{
    int ret = 1;

    clt_elem_t *ps = malloc(n * sizeof(clt_elem_t));
    for (ulong i = 0; i < n; i++)
        mpz_init(ps[i]);

    if (clt_vector_fread(fp, ps, n) != 0) {
        fprintf(stderr, "[crt_tree_fread] couldn't read ps!\n");
        goto cleanup;
    }

    if (crt_tree_init(crt, ps, n) == 0) {
        fprintf(stderr, "[crt_tree_fread] couldn't initialize crt_tree!\n");
        goto cleanup;
    }

    for (ulong i = 0; i < n; i++)
        mpz_clear(ps[i]);
    free(ps);

    ret = 0;
cleanup:
    return ret;
}

static int
crt_tree_fsave(FILE *const fp, crt_tree *crt, size_t n)
{
    int ret = 1;

    clt_elem_t *ps = malloc(n * sizeof(clt_elem_t));
    for (ulong i = 0; i < n; i++)
        mpz_init(ps[i]);
    int ctr = 0;

    _crt_tree_get_leafs(ps, &ctr, crt);
    if (clt_vector_fsave(fp, ps, n) != 0)
        goto cleanup;

    for (ulong i = 0; i < n; i++)
        mpz_clear(ps[i]);
    free(ps);

    ret = 0;
cleanup:
    return ret;
}

////////////////////////////////////////////////////////////////////////////////
// helper functions

static int
ulong_read(const char *fname, ulong *x)
{
    FILE *f;
    if ((f = fopen(fname, "r")) == NULL) {
        perror(fname);
        return 1;
    }
    ulong_fread(f, x);
    fclose(f);
    return 0;
}

static int
ulong_save(const char *fname, ulong x)
{
    FILE *f;
    if ((f = fopen(fname, "w")) == NULL) {
        perror(fname);
        return 1;
    }
    ulong_fsave(f, x);
    fclose(f);
    return 0;
}

static int
ulong_fread(FILE *const fp, ulong *x)
{
    return !(fscanf(fp, "%lu", x) > 0);
}

static int
ulong_fsave(FILE *const fp, ulong x)
{
    return !(fprintf(fp, "%lu", x) > 0);
}

int
clt_elem_read(const char *fname, clt_elem_t x)
{
    FILE *f;
    if ((f = fopen(fname, "r")) == NULL) {
        perror(fname);
        return 1;
    }
    clt_elem_fread(f, x);
    fclose(f);
    return 0;
}

int
clt_elem_save(const char *fname, const clt_elem_t x)
{
    FILE *f;
    if ((f = fopen(fname, "w")) == NULL) {
        perror(fname);
        return 1;
    }
    if (clt_elem_fsave(f, x) == 0) {
        fclose(f);
        return 1;
    }
    fclose(f);
    return 0;
}

int
clt_elem_fread(FILE *const fp, clt_elem_t x)
{
    return !(mpz_inp_raw(x, fp) > 0);
}

int
clt_elem_fsave(FILE *const fp, const clt_elem_t x)
{
    return !(mpz_out_raw(fp, x) > 0);
}

int
clt_vector_read(const char *fname, clt_elem_t *m, ulong len)
{
    FILE *f;
    if ((f = fopen(fname, "r")) == NULL) {
        perror(fname);
        return 1;
    }
    for (ulong i = 0; i < len; ++i) {
        if (mpz_inp_raw(m[i], f) == 0) {
            fclose(f);
            return 1;
        }
    }
    fclose(f);
    return 0;
}

int
clt_vector_save(const char *fname, clt_elem_t *m, ulong len)
{
    FILE *f;
    if ((f = fopen(fname, "w")) == NULL) {
        perror(fname);
        return 1;
    }
    for (ulong i = 0; i < len; ++i) {
        if (mpz_out_raw(f, m[i]) == 0) {
            (void) fclose(f);
            return 1;
        }
    }
    fclose(f);
    return 0;
}

int
clt_vector_fread(FILE *const fp, clt_elem_t *m, ulong len)
{
    for (ulong i = 0; i < len; ++i) {
        if (mpz_inp_raw(m[i], fp) == 0)
            return 1;
    }
    return 0;
}

int
clt_vector_fsave(FILE *const fp, clt_elem_t *m, ulong len)
{
    for (ulong i = 0; i < len; ++i) {
        if (mpz_out_raw(fp, m[i]) == 0)
            return 1;
    }
    return 0;
}

static double
current_time(void)
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + (double) (t.tv_usec / 1000000.0);
}
