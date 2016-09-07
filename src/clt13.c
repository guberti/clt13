#include "clt13.h"
#include "crt_tree.h"
#include "estimates.h"

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

#define GET_NEWLINE(fp) (fscanf(fp, "\n"))
#define PUT_NEWLINE(fp) (!(fprintf(fp, "\n") > 0))

typedef unsigned long ulong;

struct clt_state {
    size_t n;
    size_t nzs;
    size_t rho;
    size_t nu;
    clt_elem_t x0;
    clt_elem_t pzt;
    clt_elem_t *gs;
    clt_elem_t *zinvs;
    aes_randstate_t *rngs;
    union {
        crt_tree *crt;
        clt_elem_t *crt_coeffs;
    };
    ulong flags;
};

struct clt_pp {
    clt_elem_t x0;
    clt_elem_t pzt;
    size_t nu;
};

static double current_time(void);

static int ulong_read  (const char *fname, ulong *x);
static int ulong_save  (const char *fname, ulong x);
static int ulong_fread (FILE *const fp, ulong *x);
static int ulong_fsave (FILE *const fp, ulong x);

static void print_progress (size_t cur, size_t total);

static inline ulong nb_of_bits(ulong x)
{
    ulong nb = 0;
    while (x > 0) {
        x >>= 1;
        nb++;
    }
    return nb;
}

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
mpz_random_(mpz_t rop, aes_randstate_t rng, ulong len)
{
    mpz_urandomb_aes(rop, rng, len);
    mpz_setbit(rop, len-1);
}

static inline void
mpz_prime(mpz_t rop, aes_randstate_t rng, ulong len)
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
// state

clt_state *
clt_state_new(size_t kappa, size_t lambda, size_t min_slots, size_t nzs,
              const int *const pows, size_t ncores, size_t flags,
              aes_randstate_t rng)
{
    clt_state *s;
    size_t alpha, beta, eta, rho_f;
    clt_elem_t *ps, *zs;
    double start_time = 0.0;
    int count = 0;

    s = malloc(sizeof(clt_state));
    if (s == NULL)
        return NULL;

    if (ncores == 0)
        ncores = sysconf(_SC_NPROCESSORS_ONLN);
    (void) omp_set_num_threads(ncores);

    /* calculate CLT parameters */
    s->nzs = nzs;
    alpha  = lambda;                   /* bitsize of g_i primes */
    beta   = lambda;                   /* bitsize of h_i entries */
    s->rho = lambda;                   /* bitsize of randomness */
    rho_f  = kappa * (s->rho + alpha); /* max bitsize of r_i's */
    eta    = rho_f + alpha + beta + 9; /* bitsize of primes p_i */
    s->n   = estimate_n(lambda, eta, flags);  /* number of primes */
    eta    = rho_f + alpha + beta + nb_of_bits(s->n) + 9; /* bitsize of primes p_i */
    s->nu  = eta - beta - rho_f - nb_of_bits(s->n) - 3; /* number of msbs to extract */
    s->flags = flags;

    /* Loop until a fixed point reached for choosing eta, n, and nu */
    {
        ulong old_eta = 0, old_n = 0, old_nu = 0;
        int i = 0;
        for (; i < 10 && (old_eta != eta || old_n != s->n || old_nu != s->nu);
             ++i) {
            old_eta = eta, old_n = s->n, old_nu = s->nu;
            eta = rho_f + alpha + beta + nb_of_bits(s->n) + 9;
            s->n = estimate_n(lambda, eta, flags);
            s->nu = eta - beta - rho_f - nb_of_bits(s->n) - 3;
        }

        if (i == 10 && (old_eta != eta || old_n != s->n || old_nu != s->nu)) {
            fprintf(stderr, "Error: unable to find valid η, n, and ν choices\n");
            free(s);
            return NULL;
        }
    }

    /* Make sure the proper bounds are hit [CLT13, Lemma 8] */
    assert(s->nu >= alpha + 6);
    assert(beta + alpha + rho_f + nb_of_bits(s->n) <= eta - 9);

    if (min_slots != 0 && s->n < min_slots) {
        fprintf(stderr, "Error: number of slots is less than required (%lu < %lu)\n",
                s->n, min_slots);
        free(s);
        return NULL;
    }

    if (s->flags & CLT_FLAG_VERBOSE) {
        fprintf(stderr, "  λ: %ld\n", lambda);
        fprintf(stderr, "  κ: %ld\n", kappa);
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
    }

    /* Generate randomness for each core */
    s->rngs = malloc(sizeof(aes_randstate_t) * MAX(s->n, s->nzs));
    for (ulong i = 0; i < s->n; ++i) {
        unsigned char *buf;
        size_t n = 128;

        buf = random_aes(rng, &n);
        aes_randinit_seedn(s->rngs[i], (char *) buf + 5, n - 5, NULL, 0);
        free(buf);
    }

    ps       = malloc(sizeof(clt_elem_t) * s->n);
    zs       = malloc(sizeof(clt_elem_t) * s->nzs);
    s->zinvs = malloc(sizeof(clt_elem_t) * s->nzs);
    s->gs    = malloc(sizeof(clt_elem_t) * s->n);

    if (s->flags & CLT_FLAG_OPT_CRT_TREE) {
        s->crt = malloc(sizeof(crt_tree));
    } else {
        s->crt_coeffs = malloc(s->n * sizeof(clt_elem_t));
        for (ulong i = 0; i < s->n; i++) {
            mpz_init(s->crt_coeffs[i]);
        }
    }

    /* initialize gmp variables */
    mpz_init_set_ui(s->x0,  1);
    mpz_init_set_ui(s->pzt, 0);
    for (ulong i = 0; i < s->n; ++i) {
        mpz_init_set_ui(ps[i], 1);
        mpz_init(s->gs[i]);
    }
    for (ulong i = 0; i < s->nzs; ++i) {
        mpz_inits(zs[i], s->zinvs[i], NULL);
    }

    /* Generate p_i's and g_i's, as well as x0 = \prod p_i */
    if (s->flags & CLT_FLAG_VERBOSE) {
        fprintf(stderr, "  Generating p_i's and g_i's:");
        start_time = current_time();
    }

GEN_PIS:
    if (s->flags & CLT_FLAG_OPT_COMPOSITE_PS) {
        ulong etap = ETAP_DEFAULT;
        if (eta > 350)
            /* TODO: change how we set etap, should be resistant to factoring x_0 */
            for (/* */; eta % etap < 350; etap++)
                ;
        if (s->flags & CLT_FLAG_VERBOSE) {
            fprintf(stderr, "[eta_p: %lu] ", etap);
        }
        ulong nchunks = eta / etap;
        ulong leftover = eta - nchunks * etap;
        if (s->flags & CLT_FLAG_VERBOSE) {
            fprintf(stderr, "[nchunks=%lu leftover=%lu]\n", nchunks, leftover);
            print_progress(count, s->n);
        }
#pragma omp parallel for
        for (ulong i = 0; i < s->n; i++) {
            clt_elem_t p_unif;
            mpz_set_ui(ps[i], 1);
            mpz_init(p_unif);
            /* generate a p_i */
            for (ulong j = 0; j < nchunks; j++) {
                mpz_prime(p_unif, s->rngs[i], etap);
                mpz_mul(ps[i], ps[i], p_unif);
            }
            mpz_prime(p_unif, s->rngs[i], leftover);
            mpz_mul(ps[i], ps[i], p_unif);
            /* generate a g_i */
            mpz_prime(s->gs[i], s->rngs[i], alpha);
            mpz_clear(p_unif);

            if (s->flags & CLT_FLAG_VERBOSE) {
#pragma omp critical
                print_progress(++count, s->n);
            }
        }
    } else {
        /* Don't use composite p's optimization */
        if (s->flags & CLT_FLAG_VERBOSE) {
            fprintf(stderr, "\n");
            print_progress(count, s->n);
        }
#pragma omp parallel for
        for (ulong i = 0; i < s->n; i++) {
            /* generate a p_i */
            mpz_prime(ps[i], s->rngs[i], eta);
            /* generate a g_i */
            mpz_prime(s->gs[i], s->rngs[i], alpha);
            if (s->flags & CLT_FLAG_VERBOSE) {
#pragma omp critical
                print_progress(++count, s->n);
            }
        }
    }
    if (s->flags & CLT_FLAG_VERBOSE) {
        fprintf(stderr, "\t[%.2fs]\n", current_time() - start_time);
    }

    if (s->flags & CLT_FLAG_OPT_CRT_TREE) {
        if (s->flags & CLT_FLAG_VERBOSE) {
            fprintf(stderr, "  Generating CRT-Tree: ");
            start_time = current_time();
        }
        /* use crt_tree to find x0 */
        int ok = crt_tree_init(s->crt, ps, s->n);
        if (!ok) {
            /* if crt_tree_init fails, regenerate with new p_i's */
            crt_tree_clear(s->crt);
            if (s->flags & CLT_FLAG_VERBOSE) {
                fprintf(stderr, "(restarting) ");
            }
            goto GEN_PIS;
        }
        /* crt_tree_init succeeded, set x0 */
        mpz_set(s->x0, s->crt->mod);
        if (s->flags & CLT_FLAG_VERBOSE) {
            fprintf(stderr, "[%.2fs]\n", current_time() - start_time);
        }
    } else {
        /* Don't use CRT tree optimization */
        if (s->flags & CLT_FLAG_VERBOSE) {
            fprintf(stderr, "  Computing x0: \n");
            start_time = current_time();
        }

        /* calculate x0 the hard way */
        for (ulong i = 0; i < s->n; i++) {
            mpz_mul(s->x0, s->x0, ps[i]);
            if (s->flags & CLT_FLAG_VERBOSE)
                print_progress(i, s->n-1);
        }

        if (s->flags & CLT_FLAG_VERBOSE) {
            fprintf(stderr, "\t[%.2fs]\n", current_time() - start_time);
        }

        /* Compute CRT coefficients */
        if (s->flags & CLT_FLAG_VERBOSE) {
            fprintf(stderr, "  Generating CRT coefficients:\n");
            start_time = current_time();
            count = 0;
        }
#pragma omp parallel for
        for (unsigned long i = 0; i < s->n; i++) {
            clt_elem_t q;
            mpz_init(q);
            mpz_div(q, s->x0, ps[i]);
            mpz_invert(s->crt_coeffs[i], q, ps[i]);
            mpz_mul_mod(s->crt_coeffs[i], s->crt_coeffs[i], q, s->x0);
            mpz_clear(q);

            if (s->flags & CLT_FLAG_VERBOSE) {
#pragma omp critical
                print_progress(++count, s->n);
            }
        }
        if (s->flags & CLT_FLAG_VERBOSE) {
            fprintf(stderr, "\t[%.2fs]\n", current_time() - start_time);
        }
    }

    /* Compute z_i's */
    if (s->flags & CLT_FLAG_VERBOSE) {
        fprintf(stderr, "  Generating z_i's:\n");
        start_time = current_time();
        count = 0;
    }
#pragma omp parallel for
    for (ulong i = 0; i < s->nzs; ++i) {
        do {
            mpz_urandomm_aes(zs[i], s->rngs[i], s->x0);
        } while (mpz_invert(s->zinvs[i], zs[i], s->x0) == 0);
        if (s->flags & CLT_FLAG_VERBOSE) {
#pragma omp critical
            print_progress(++count, s->nzs);
        }
    }
    if (s->flags & CLT_FLAG_VERBOSE) {
        fprintf(stderr, "\t[%.2fs]\n", current_time() - start_time);
    }

    /* Compute pzt */
    if (s->flags & CLT_FLAG_VERBOSE) {
        fprintf(stderr, "  Generating pzt:\n");
        start_time = current_time();
        count = 0;
    }

    {
        clt_elem_t zk;
        mpz_init_set_ui(zk, 1);
        /* compute z_1^t_1 ... z_k^t_k mod x0 */
        for (ulong i = 0; i < s->nzs; ++i) {
            clt_elem_t tmp;
            mpz_init(tmp);
            mpz_powm_ui(tmp, zs[i], pows[i], s->x0);
            mpz_mul_mod(zk, zk, tmp, s->x0);
            mpz_clear(tmp);
            if (s->flags & CLT_FLAG_VERBOSE) {
                print_progress(++count, s->n + s->nzs);
            }
        }
#pragma omp parallel for
        for (ulong i = 0; i < s->n; ++i) {
            clt_elem_t tmp, qpi, rnd;
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
            if (s->flags & CLT_FLAG_VERBOSE) {
#pragma omp critical
                print_progress(++count, s->n + s->nzs);
            }
        }
        mpz_mod_near(s->pzt, s->pzt, s->x0);
        mpz_clear(zk);
    }
    if (s->flags & CLT_FLAG_VERBOSE) {
        fprintf(stderr, "\t[%.2fs]\n", current_time() - start_time);
    }

    for (ulong i = 0; i < s->n; i++)
        mpz_clear(ps[i]);
    free(ps);

    for (ulong i = 0; i < s->nzs; ++i)
        mpz_clear(zs[i]);
    free(zs);

    return s;
}

void
clt_state_delete(clt_state *s)
{
    mpz_clears(s->x0, s->pzt, NULL);
    for (ulong i = 0; i < s->n; ++i) {
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
    if (s->rngs) {
        for (ulong i = 0; i < MAX(s->n, s->nzs); ++i) {
            aes_randclear(s->rngs[i]);
        }
        free(s->rngs);
    }
    free(s);
}

clt_elem_t *
clt_state_moduli(const clt_state *const s)
{
    return s->gs;
}

size_t
clt_state_nslots(const clt_state *const s)
{
    return s->n;
}

size_t
clt_state_nzs(const clt_state *const s)
{
    return s->nzs;
}

////////////////////////////////////////////////////////////////////////////////
// encodings

void
clt_encode(clt_elem_t rop, const clt_state *s, size_t nins, mpz_t *ins,
           const int *pows)
{
    if (!(s->flags & CLT_FLAG_OPT_PARALLEL_ENCODE)) {
        omp_set_num_threads(1);
    }

    if (s->flags & CLT_FLAG_OPT_CRT_TREE) {
        /* slots[i] = m[i] + r*g[i] */
        clt_elem_t *slots = malloc(s->n * sizeof(clt_elem_t));
#pragma omp parallel for
        for (ulong i = 0; i < s->n; i++) {
            mpz_init(slots[i]);
            mpz_random_(slots[i], s->rngs[i], s->rho);
            mpz_mul(slots[i], slots[i], s->gs[i]);
            if (i < nins)
                mpz_add(slots[i], slots[i], ins[i]);
        }
        crt_tree_do_crt(rop, s->crt, slots);
        for (ulong i = 0; i < s->n; i++)
            mpz_clear(slots[i]);
        free(slots);
    } else {
        /* Don't use CRT tree */
        mpz_set_ui(rop, 0);
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
                mpz_add(rop, rop, tmp);
            }
            mpz_clear(tmp);
        }
    }
    {
        clt_elem_t tmp;
        mpz_init(tmp);
        /* multiply by appropriate zinvs */
        for (unsigned long i = 0; i < s->nzs; ++i) {
            if (pows[i] <= 0)
                continue;
            mpz_powm_ui(tmp, s->zinvs[i], pows[i], s->x0);
            mpz_mul_mod(rop, rop, tmp, s->x0);
        }
        mpz_clear(tmp);
    }
}

int
clt_is_zero(const clt_elem_t c, const clt_pp *pp)
{
    int ret;

    clt_elem_t tmp, x0_;
    mpz_inits(tmp, x0_, NULL);

    mpz_mul(tmp, c, pp->pzt);
    mpz_mod_near(tmp, tmp, pp->x0);

    ret = mpz_sizeinbase(tmp, 2) < mpz_sizeinbase(pp->x0, 2) - pp->nu;
    mpz_clears(tmp, x0_, NULL);
    return ret ? 1 : 0;
}

void
clt_elem_init(clt_elem_t rop)
{
    mpz_init(rop);
}

void
clt_elem_clear(clt_elem_t rop)
{
    mpz_clear(rop);
}

void
clt_elem_set(clt_elem_t a, const clt_elem_t b)
{
    mpz_set(a, b);
}

void
clt_elem_add(clt_elem_t rop, const clt_pp *pp, const clt_elem_t a, const clt_elem_t b)
{
    mpz_add(rop, a, b);
    mpz_mod(rop, rop, pp->x0);
}

void
clt_elem_sub(clt_elem_t rop, const clt_pp *pp, const clt_elem_t a, const clt_elem_t b)
{
    mpz_sub(rop, a, b);
    mpz_mod(rop, rop, pp->x0);
}

void
clt_elem_mul(clt_elem_t rop, const clt_pp *pp, const clt_elem_t a, const clt_elem_t b)
{
    mpz_mul(rop, a, b);
    mpz_mod(rop, rop, pp->x0);
}

void
clt_elem_mul_ui(clt_elem_t rop, const clt_pp *pp, const clt_elem_t a, unsigned int b)
{
    mpz_mul_ui(rop, a, b);
    mpz_mod(rop, rop, pp->x0);
}

clt_state *
clt_state_read(const char *const dir)
{
    clt_state *s;
    char *fname;
    int len;

    s = calloc(1, sizeof(clt_state));
    if (s == NULL)
        return NULL;

    len = strlen(dir) + 13;
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
    s->rngs  = malloc(sizeof(aes_randstate_t) * MAX(s->n, s->nzs));

    mpz_inits(s->x0, s->pzt, NULL);
    for (ulong i = 0; i < s->nzs; i++)
        mpz_init(s->zinvs[i]);
    for (ulong i = 0; i < s->n; i++)
        mpz_init(s->gs[i]);

    snprintf(fname, len, "%s/rngs", dir);
    for (ulong i = 0; i < MAX(s->n, s->nzs); ++i) {
        aes_randstate_read(s->rngs[i], fname);
    }

    snprintf(fname, len, "%s/x0", dir);
    clt_elem_read(s->x0, fname);
    snprintf(fname, len, "%s/pzt", dir);
    clt_elem_read(s->pzt, fname);
    snprintf(fname, len, "%s/gs", dir);
    clt_vector_read(s->gs, s->n, fname);
    snprintf(fname, len, "%s/zinvs", dir);
    clt_vector_read(s->zinvs, s->nzs, fname);

    if (s->flags & CLT_FLAG_OPT_CRT_TREE) {
        s->crt = malloc(sizeof(crt_tree));
        snprintf(fname, len, "%s/crt_tree", dir);
        crt_tree_read(fname, s->crt, s->n);
    } else {
        s->crt_coeffs = malloc(sizeof(clt_elem_t) * s->n);
        for (ulong i = 0; i < s->n; i++)
            mpz_init(s->crt_coeffs[i]);
        snprintf(fname, len, "%s/crt_coeffs", dir);
        clt_vector_read(s->crt_coeffs, s->n, fname);
    }
    free(fname);

    return s;
}


int
clt_state_write(clt_state *const s, const char *const dir)
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
    snprintf(fname, len, "%s/rngs", dir);
    for (ulong i = 0; i < MAX(s->n, s->nzs); ++i) {
        aes_randstate_write(s->rngs[i], fname);
    }
    snprintf(fname, len, "%s/rho", dir);
    ulong_save(fname, s->rho);
    snprintf(fname, len, "%s/nu", dir);
    ulong_save(fname, s->nu);
    snprintf(fname, len, "%s/x0", dir);
    clt_elem_write(s->x0, fname);
    snprintf(fname, len, "%s/pzt", dir);
    clt_elem_write(s->pzt, fname);
    snprintf(fname, len, "%s/gs", dir);
    clt_vector_write(s->gs, s->n, fname);
    snprintf(fname, len, "%s/zinvs", dir);
    clt_vector_write(s->zinvs, s->nzs, fname);
    if (s->flags & CLT_FLAG_OPT_CRT_TREE) {
        snprintf(fname, len, "%s/crt_tree", dir);
        crt_tree_save(fname, s->crt, s->n);
    } else {
        snprintf(fname, len, "%s/crt_coeffs", dir);
        clt_vector_write(s->crt_coeffs, s->n, fname);
    }
    free(fname);
    return CLT_OK;
}

clt_state *
clt_state_fread(FILE *const fp)
{
    clt_state *s;
    int ret = 1;

    s = calloc(1, sizeof(clt_state));
    if (s == NULL)
        return NULL;

    if (ulong_fread(fp, &s->flags) || GET_NEWLINE(fp)) {
        fprintf(stderr, "[clt_state_fread] couldn't read flags!\n");
        goto cleanup;
    }
    if (ulong_fread(fp, &s->n) || GET_NEWLINE(fp)) {
        fprintf(stderr, "[clt_state_fread] couldn't read n!\n");
        goto cleanup;
    }
    if (ulong_fread(fp, &s->nzs) || GET_NEWLINE(fp)) {
        fprintf(stderr, "[clt_state_fread] couldn't read nzs!\n");
        goto cleanup;
    }
    if (ulong_fread(fp, &s->rho) || GET_NEWLINE(fp)) {
        fprintf(stderr, "[clt_state_fread] couldn't read rho!\n");
        goto cleanup;
    }
    if (ulong_fread(fp, &s->nu) || GET_NEWLINE(fp)) {
        fprintf(stderr, "[clt_state_fread] couldn't read nu!\n");
        goto cleanup;
    }

    mpz_init(s->x0);
    if (clt_elem_fread(s->x0, fp) || GET_NEWLINE(fp)) {
        fprintf(stderr, "[clt_state_fread] couldn't read x0!\n");
        goto cleanup;
    }
    mpz_init(s->pzt);
    if (clt_elem_fread(s->pzt, fp) || GET_NEWLINE(fp)) {
        fprintf(stderr, "[clt_state_fread] couldn't read pzt!\n");
        goto cleanup;
    }

    s->gs = malloc(sizeof(clt_elem_t) * s->n);
    for (ulong i = 0; i < s->n; ++i)
        mpz_init(s->gs[i]);
    if (clt_vector_fread(s->gs, s->n, fp) || GET_NEWLINE(fp)) {
        fprintf(stderr, "[clt_state_fread] couldn't read gs!\n");
        goto cleanup;
    }

    s->zinvs = malloc(sizeof(clt_elem_t) * s->nzs);
    for (ulong i = 0; i < s->nzs; i++)
        mpz_init(s->zinvs[i]);
    if (clt_vector_fread(s->zinvs, s->nzs, fp) || GET_NEWLINE(fp)) {
        fprintf(stderr, "[clt_state_fread] couldn't read zinvs!\n");
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
        if (clt_vector_fread(s->crt_coeffs, s->n, fp) != 0) {
            fprintf(stderr, "[clt_state_fread] couldn't read crt_coeffs!\n");
            goto cleanup;
        }
    }

    s->rngs = malloc(sizeof(aes_randstate_t) * MAX(s->n, s->nzs));
    for (ulong i = 0; i < MAX(s->n, s->nzs); ++i) {
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
clt_state_fwrite(clt_state *const s, FILE *const fp)
{
    int ret = CLT_ERR;

    if (ulong_fsave(fp, s->flags) || PUT_NEWLINE(fp)) {
        fprintf(stderr, "[clt_state_fsave] failed to save flags!\n");
        goto cleanup;
    }
    if (ulong_fsave(fp, s->n) || PUT_NEWLINE(fp)) {
        fprintf(stderr, "[clt_state_fsave] failed to save n!\n");
        goto cleanup;
    }
    if (ulong_fsave(fp, s->nzs) || PUT_NEWLINE(fp)) {
        fprintf(stderr, "[clt_state_fsave] failed to save n!\n");
        goto cleanup;
    }
    if (ulong_fsave(fp, s->rho) || PUT_NEWLINE(fp)) {
        fprintf(stderr, "[clt_state_fsave] failed to save rho!\n");
        goto cleanup;
    }
    if (ulong_fsave(fp, s->nu) || PUT_NEWLINE(fp)) {
        fprintf(stderr, "[clt_state_fsave] failed to save nu!\n");
        goto cleanup;
    }
    if (clt_elem_fwrite(s->x0, fp) || PUT_NEWLINE(fp)) {
        fprintf(stderr, "[clt_state_fsave] failed to save x0!\n");
        goto cleanup;
    }
    if (clt_elem_fwrite(s->pzt, fp) || PUT_NEWLINE(fp)) {
        fprintf(stderr, "[clt_state_fsave] failed to save pzt!\n");
        goto cleanup;
    }
    if (clt_vector_fwrite(s->gs, s->n, fp) || PUT_NEWLINE(fp)) {
        fprintf(stderr, "[clt_state_fsave] failed to save gs!\n");
        goto cleanup;
    }
    if (clt_vector_fwrite(s->zinvs, s->nzs, fp) || PUT_NEWLINE(fp)) {
        fprintf(stderr, "[clt_state_fsave] failed to save zinvs!\n");
        goto cleanup;
    }
    if (s->flags & CLT_FLAG_OPT_CRT_TREE) {
        if (crt_tree_fsave(fp, s->crt, s->n) != 0) {
            fprintf(stderr, "[clt_state_fsave] failed to save crt_tree!\n");
            goto cleanup;
        }
    } else {
        if (clt_vector_fwrite(s->crt_coeffs, s->n, fp) != 0) {
            fprintf(stderr, "[clt_state_fsave] failed to save crt_coefs!\n");
            goto cleanup;
        }
    }

    for (ulong i = 0; i < MAX(s->n, s->nzs); ++i) {
        aes_randstate_fwrite(s->rngs[i], fp);
    }

    ret = CLT_OK;
cleanup:
    return ret;
}

////////////////////////////////////////////////////////////////////////////////
// public parameters

clt_pp *
clt_pp_new(const clt_state *mmap)
{
    clt_pp *pp;

    pp = calloc(1, sizeof(clt_pp));
    if (pp == NULL)
        return NULL;
    mpz_inits(pp->x0, pp->pzt, NULL);
    mpz_set(pp->x0, mmap->x0);
    mpz_set(pp->pzt, mmap->pzt);
    pp->nu = mmap->nu;
    return pp;
}

void
clt_pp_delete(clt_pp *pp)
{
    mpz_clears(pp->x0, pp->pzt, NULL);
    free(pp);
}

clt_pp *
clt_pp_read(const char *const dir)
{
    clt_pp *pp;
    char *fname;
    int len, ret = CLT_ERR;

    if ((pp = calloc(1, sizeof(clt_pp))) == NULL)
        return NULL;

    len = strlen(dir) + 10;
    fname = malloc(sizeof(char) + len);

    mpz_inits(pp->x0, pp->pzt, NULL);

    snprintf(fname, len, "%s/nu", dir);
    if (ulong_read(fname, &pp->nu) != 0)
        goto cleanup;
    snprintf(fname, len, "%s/x0", dir);
    if (clt_elem_read(pp->x0, fname) != 0)
        goto cleanup;
    snprintf(fname, len, "%s/pzt", dir);
    if (clt_elem_read(pp->pzt, fname) != 0)
        goto cleanup;
    ret = CLT_OK;
cleanup:
    free(fname);
    if (ret == CLT_OK) {
        return pp;
    } else {
        clt_pp_delete(pp);
        return NULL;
    }
}

int
clt_pp_write(clt_pp *const pp, const char *const dir)
{
    char *fname;
    int ret = CLT_ERR;
    int len = strlen(dir) + 10;
    fname = malloc(sizeof(char) * len);

    snprintf(fname, len, "%s/nu", dir);
    if (ulong_save(fname, pp->nu) != 0)
        goto cleanup;
    snprintf(fname, len, "%s/x0", dir);
    if (clt_elem_write(pp->x0, fname) != 0)
        goto cleanup;
    snprintf(fname, len, "%s/pzt", dir);
    if (clt_elem_write(pp->pzt, fname) != 0)
        goto cleanup;
    ret = CLT_OK;
cleanup:
    free(fname);
    return ret;
}

clt_pp *
clt_pp_fread(FILE *const fp)
{
    clt_pp *pp;
    int ret = CLT_ERR;

    if ((pp = calloc(1, sizeof(clt_pp))) == NULL)
        return NULL;
    mpz_inits(pp->x0, pp->pzt, NULL);

    if (ulong_fread(fp, &pp->nu) || GET_NEWLINE(fp))
        goto cleanup;
    if (clt_elem_fread(pp->x0, fp) || GET_NEWLINE(fp))
        goto cleanup;
    if (clt_elem_fread(pp->pzt, fp) || GET_NEWLINE(fp))
        goto cleanup;
    ret = CLT_OK;
cleanup:
    if (ret == CLT_OK) {
        return pp;
    } else {
        clt_pp_delete(pp);
        return NULL;
    }
}

int
clt_pp_fwrite(clt_pp *const pp, FILE *const fp)
{
    int ret = CLT_ERR;

    if (ulong_fsave(fp, pp->nu) || PUT_NEWLINE(fp))
        goto cleanup;
    if (clt_elem_fwrite(pp->x0, fp) || PUT_NEWLINE(fp))
        goto cleanup;
    if (clt_elem_fwrite(pp->pzt, fp) || PUT_NEWLINE(fp))
        goto cleanup;
    ret = CLT_OK;
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
clt_elem_read(clt_elem_t x, const char *fname)
{
    FILE *f;
    if ((f = fopen(fname, "r")) == NULL) {
        perror(fname);
        return 1;
    }
    clt_elem_fread(x, f);
    fclose(f);
    return 0;
}

int
clt_elem_write(clt_elem_t x, const char *fname)
{
    FILE *f;
    if ((f = fopen(fname, "w")) == NULL) {
        perror(fname);
        return 1;
    }
    if (clt_elem_fwrite(x, f) == 0) {
        fclose(f);
        return 1;
    }
    fclose(f);
    return 0;
}

int
clt_elem_fread(clt_elem_t x, FILE *const fp)
{
    return !(mpz_inp_raw(x, fp) > 0);
}

int
clt_elem_fwrite(clt_elem_t x, FILE *const fp)
{
    return !(mpz_out_raw(fp, x) > 0);
}

int
clt_vector_read(clt_elem_t *m, ulong len, const char *fname)
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
clt_vector_write(clt_elem_t *m, ulong len, const char *fname)
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
clt_vector_fread(clt_elem_t *m, ulong len, FILE *const fp)
{
    for (ulong i = 0; i < len; ++i) {
        if (mpz_inp_raw(m[i], fp) == 0)
            return 1;
    }
    return 0;
}

int
clt_vector_fwrite(clt_elem_t *m, ulong len, FILE *const fp)
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
