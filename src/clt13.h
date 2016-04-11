#ifndef __CLT13_H__
#define __CLT13_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <aesrand.h>
#include <gmp.h>

typedef unsigned long ulong;
typedef struct crt_tree crt_tree;

// state

typedef struct {
    ulong n;
    ulong nzs;
    ulong rho;
    ulong nu;
    mpz_t x0;
    mpz_t pzt;
    mpz_t *gs;
    mpz_t *zinvs;
    union {
        crt_tree *crt;
        mpz_t *crt_coeffs;
    };
    ulong flags;
} clt_state;

#define CLT_FLAG_NONE 0x00
#define CLT_FLAG_VERBOSE 0x01
#define CLT_FLAG_OPT_CRT_TREE 0x02
#define CLT_FLAG_OPT_PARALLEL_ENCODE 0x04
#define CLT_FLAG_OPT_COMPOSITE_PS 0x08 // XXX: unimplemented

#define CLT_FLAG_DEFAULT \
    ( CLT_FLAG_VERBOSE \
    & CLT_FLAG_OPT_CRT_TREE \
    & CLT_FLAG_OPT_COMPOSITE_PS \
    )

void clt_state_init (clt_state *s, ulong kappa, ulong lambda, ulong nzs,
                     const int *pows, ulong flags, aes_randstate_t rng);
void clt_state_clear (clt_state *s);
void clt_state_read  (clt_state *s, const char *dir);
void clt_state_save  (const clt_state *s, const char *dir);
void clt_state_fread (FILE *const fp, clt_state *s);
void clt_state_fsave (FILE *const fp, const clt_state *s);

// public parameters

typedef struct {
    mpz_t x0;
    mpz_t pzt;
    ulong nu;
} clt_pp;

void clt_pp_init  (clt_pp *pp, const clt_state *mmap);
void clt_pp_clear (clt_pp *pp);
void clt_pp_read  (clt_pp *pp, const char *dir);
void clt_pp_save  (const clt_pp *pp, const char *dir);
void clt_pp_fread (FILE *const fp, clt_pp *pp);
void clt_pp_fsave (FILE *const fp, const clt_pp *pp);

// encodings

void clt_encode (mpz_t rop, const clt_state *s, size_t nins, mpz_t *ins,
                 const int *pows, aes_randstate_t rng);

int clt_is_zero (const clt_pp *pp, const mpz_t c);

#ifdef __cplusplus
}
#endif

#endif
