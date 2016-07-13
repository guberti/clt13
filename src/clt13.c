#include "clt13.h"
#include "crt_tree.h"

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
/* when trying to find a fixed point, loop MAX_LOOP_LENGTH times before aborting */
#define MAX_LOOP_LENGTH 10

#define GET_NEWLINE(fp) (fscanf(fp, "\n"))
#define PUT_NEWLINE(fp) (!(fprintf(fp, "\n") > 0))

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

int
clt_state_init (clt_state *s, ulong kappa, ulong lambda, ulong nzs,
                const int *pows, ulong flags, aes_randstate_t rng)
{
    ulong alpha, beta, eta, rho_f;
    clt_elem_t *ps, *zs;
    double start_time = 0.0;

    /* calculate CLT parameters */
    s->nzs = nzs;
    alpha  = lambda;                   /* bitsize of g prime */
    beta   = lambda;                   /* bitsize of h_i entries */
    s->rho = lambda;                   /* bitsize of randomness */
    rho_f = kappa * (s->rho + alpha);  /* max bitsize of r_i's */
    eta    = rho_f + alpha + beta + 9; /* bitsize of primes p_i */
    s->n   = eta * nb_of_bits(lambda); /* number of primes */
    s->nu = eta - beta - rho_f - nb_of_bits(s->n) - 3; /* number of msbs to extract */
    {
        /* Loop until fixed point reached */
        ulong old_eta = 0, old_n = 0, old_nu = 0;
        int i = 0;
        for (i = 0;
             i < MAX_LOOP_LENGTH && (old_eta != eta || old_n != s->n || old_nu != s->nu);
             ++i) {
            old_eta = eta, old_n = s->n, old_nu = s->nu;
            eta  = rho_f + alpha + beta + nb_of_bits(s->n) + 9;
            s->n = eta * nb_of_bits(lambda);
            s->nu = eta - beta - rho_f - nb_of_bits(s->n) - 3;
        }
        if (i == MAX_LOOP_LENGTH
            && (old_eta != eta || old_n != s->n || old_nu != s->nu)) {
            fprintf(stderr, "Error: unable to find valid eta, n, and nu choices\n");
            return CLT_ERR;
        }
    }
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

    /* Make sure the proper bounds are hit */
    assert(s->nu >= alpha + 6);
    assert(beta + alpha + rho_f + nb_of_bits(s->n) <= eta - 9);

    ps       = malloc(sizeof(clt_elem_t) * s->n);
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
    mpz_init(s->g);
    for (ulong i = 0; i < s->n; ++i) {
        mpz_init_set_ui(ps[i], 1);
    }
    for (ulong i = 0; i < s->nzs; ++i) {
        mpz_inits(zs[i], s->zinvs[i], NULL);
    }

    // Generate p_i's and g, as well as x0 = \prod p_i
    if (s->flags & CLT_FLAG_VERBOSE) {
        fprintf(stderr, "  Generating p_i's and g: ");
        start_time = current_time();
    }

    /* Generate a single g element, see [CHLRS15, footnote 5] */
    {
        clt_elem_t p_unif;
        mpz_init(p_unif);
        mpz_urandomb_aes(p_unif, rng, alpha);
        mpz_nextprime(s->g, p_unif);
        mpz_clear(p_unif);
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
            mpz_clear(p_unif);
        }
    } else {
#pragma omp parallel for
        for (ulong i = 0; i < s->n; i++) {
            clt_elem_t p_unif;
            mpz_init(p_unif);
            mpz_urandomb_aes(p_unif, rng, eta);
            mpz_nextprime(ps[i], p_unif);
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
            // compute (((g)^{-1} mod p_i) * z^k mod p_i) * r_i * (q / p_i)
            mpz_invert(tmp, s->g, ps[i]);
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

    return CLT_OK;
}

void clt_state_clear(clt_state *s)
{
    mpz_clears(s->x0, s->pzt, s->g, NULL);
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

    s->zinvs = malloc(sizeof(clt_elem_t) * s->nzs);

    mpz_inits(s->x0, s->pzt, s->g, NULL);
    for (ulong i = 0; i < s->nzs; i++)
        mpz_init(s->zinvs[i]);

    snprintf(fname, len, "%s/x0", dir);
    clt_elem_read(fname, s->x0);

    snprintf(fname, len, "%s/pzt", dir);
    clt_elem_read(fname, s->pzt);

    snprintf(fname, len, "%s/g", dir);
    clt_elem_read(fname, s->g);

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

    snprintf(fname, len, "%s/g", dir);
    clt_elem_save(fname, s->g);

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

    mpz_init(s->g);

    if (clt_elem_fread(fp, s->g) || GET_NEWLINE(fp) != 0) {
        fprintf(stderr, "[clt_state_fread] couldn't read g!\n");
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

    if (clt_elem_fsave(fp, s->g) || PUT_NEWLINE(fp) != 0) {
        fprintf(stderr, "[clt_state_fsave] failed to save g!\n");
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
        // slots[i] = m[i] + r*g
        clt_elem_t *slots = malloc(s->n * sizeof(clt_elem_t));
        if (s->flags & CLT_FLAG_OPT_PARALLEL_ENCODE) {
#pragma omp parallel for
            for (ulong i = 0; i < s->n; i++) {
                mpz_init(slots[i]);
                mpz_urandomb_aes(slots[i], rng, s->rho);
                mpz_mul(slots[i], slots[i], s->g);
                if (i < nins)
                    mpz_add(slots[i], slots[i], ins[i]);
            }
        } else {
            for (ulong i = 0; i < s->n; i++) {
                mpz_init(slots[i]);
                mpz_urandomb_aes(slots[i], rng, s->rho);
                mpz_mul(slots[i], slots[i], s->g);
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
            mpz_mul(tmp, tmp, s->g);
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
