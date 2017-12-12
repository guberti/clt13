#ifndef __CLT13_H__
#define __CLT13_H__

#ifdef __cplusplus
extern "C" {
#endif

#define CLT_OK 0
#define CLT_ERR (-1)

#include <aesrand/aesrand.h>
#include <gmp.h>

typedef mpz_t clt_elem_t;
typedef struct clt_state clt_state;
typedef struct clt_pp clt_pp;

typedef struct {
    /* Security parameter */
    size_t lambda;
    /* Multilinearity */
    size_t kappa;
    /* Number of Zs */
    size_t nzs;
    /* Powers for the Zs */
    int *pows;
} clt_params_t;

typedef struct {
    size_t min_slots;
} clt_params_opt_t;

#define CLT_FLAG_NONE 0x00
/* Be verbose */
#define CLT_FLAG_VERBOSE 0x01
/* Use CRT tree optimization */
#define CLT_FLAG_OPT_CRT_TREE 0x02
/* Parallelize the encoding procedure */
#define CLT_FLAG_OPT_PARALLEL_ENCODE 0x04
/* Use composite p_i's instead of primes */
#define CLT_FLAG_OPT_COMPOSITE_PS 0x08
/* Use improved BKZ algorithm when generating attack estimates */
#define CLT_FLAG_SEC_IMPROVED_BKZ 0x10
/* Be conservative when generating attack estimates */
#define CLT_FLAG_SEC_CONSERVATIVE 0x20
/* Use polylog CLT */
# define CLT_FLAG_POLYLOG 0x40

#define CLT_FLAG_DEFAULT                        \
    ( CLT_FLAG_OPT_CRT_TREE                     \
      | CLT_FLAG_OPT_COMPOSITE_PS               \
        )

clt_state * clt_state_new(const clt_params_t *params,
                          const clt_params_opt_t *opts, size_t ncores,
                          size_t flags, aes_randstate_t rng);
void clt_state_delete(clt_state *s);
clt_state * clt_state_fread(FILE *fp);
int clt_state_fwrite(clt_state *s, FILE *fp);
clt_elem_t * clt_state_moduli(const clt_state *s);
size_t clt_state_nslots(const clt_state *s);
size_t clt_state_nzs(const clt_state *s);

// public parameters

clt_pp * clt_pp_new(const clt_state *mmap);
void clt_pp_delete(clt_pp *pp);
clt_pp * clt_pp_fread(FILE *fp);
int clt_pp_fwrite(clt_pp *pp, FILE *fp);

// encodings

void clt_encode(clt_elem_t rop, const clt_state *s, size_t nins, mpz_t *ins,
                const int *pows);
int clt_is_zero(const clt_elem_t c, const clt_pp *pp);

// elements

void clt_elem_init(clt_elem_t rop);
void clt_elem_clear(clt_elem_t rop);
void clt_elem_set(clt_elem_t a, const clt_elem_t b);
void clt_elem_add(clt_elem_t rop, const clt_pp *pp, const clt_elem_t a, const clt_elem_t b);
void clt_elem_sub(clt_elem_t rop, const clt_pp *pp, const clt_elem_t a, const clt_elem_t b);
void clt_elem_mul(clt_elem_t rop, const clt_pp *pp, const clt_elem_t a, const clt_elem_t b);
void clt_elem_mul_ui(clt_elem_t rop, const clt_pp *pp, const clt_elem_t a, unsigned int b);
void clt_elem_print(clt_elem_t a);

int clt_elem_fread(clt_elem_t x, FILE *fp);
int clt_elem_fwrite(clt_elem_t x, FILE *fp);
int clt_vector_fread(clt_elem_t *m, size_t len, FILE *fp);
int clt_vector_fwrite(clt_elem_t *m, size_t len, FILE *fp);

#ifdef __cplusplus
}
#endif

#endif
