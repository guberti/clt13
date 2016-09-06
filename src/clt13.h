#ifndef __CLT13_H__
#define __CLT13_H__

#ifdef __cplusplus
extern "C" {
#endif

#define CLT_OK 0
#define CLT_ERR (-1)

#include <aesrand.h>
#include <gmp.h>

typedef mpz_t clt_elem_t;
typedef struct clt_state clt_state;
typedef struct clt_pp clt_pp;

#define CLT_FLAG_NONE 0x00
#define CLT_FLAG_VERBOSE 0x01
#define CLT_FLAG_OPT_CRT_TREE 0x02
#define CLT_FLAG_OPT_PARALLEL_ENCODE 0x04
#define CLT_FLAG_OPT_COMPOSITE_PS 0x08
#define CLT_FLAG_SEC_IMPROVED_BKZ 0x10
#define CLT_FLAG_SEC_CONSERVATIVE 0x20

#define CLT_FLAG_DEFAULT \
    ( CLT_FLAG_OPT_CRT_TREE \
    | CLT_FLAG_OPT_COMPOSITE_PS \
    )

clt_state *
clt_state_new(size_t kappa, size_t lambda, size_t nzs, const int *const pows,
              size_t ncores, size_t flags, aes_randstate_t rng);
void
clt_state_delete(clt_state *s);
clt_state *
clt_state_read(const char *const dir);
int
clt_state_write(clt_state *const s, const char *const dir);
clt_state *
clt_state_fread(FILE *const fp);
int
clt_state_fwrite(clt_state *const s, FILE *const fp);
clt_elem_t *
clt_state_moduli(const clt_state *const s);
size_t
clt_state_nslots(const clt_state *const s);
size_t
clt_state_nzs(const clt_state *const s);

// public parameters

clt_pp *
clt_pp_new(const clt_state *const mmap);
void
clt_pp_delete(clt_pp *pp);
clt_pp *
clt_pp_read(const char *const dir);
int
clt_pp_write(clt_pp *const pp, const char *const dir);
clt_pp *
clt_pp_fread(FILE *const fp);
int
clt_pp_fwrite(clt_pp *const pp, FILE *const fp);

// encodings

void
clt_encode(clt_elem_t rop, const clt_state *const s, size_t nins, mpz_t *ins,
           const int *const pows);
int
clt_is_zero(const clt_elem_t c, const clt_pp *const pp);

// elements

void clt_elem_init(clt_elem_t rop);
void clt_elem_clear(clt_elem_t rop);
void clt_elem_set(clt_elem_t a, const clt_elem_t b);
void clt_elem_add(clt_elem_t rop, const clt_pp *pp, const clt_elem_t a, const clt_elem_t b);
void clt_elem_sub(clt_elem_t rop, const clt_pp *pp, const clt_elem_t a, const clt_elem_t b);

void clt_elem_mul(clt_elem_t rop, const clt_pp *pp, const clt_elem_t a, const clt_elem_t b);
void clt_elem_mul_ui(clt_elem_t rop, const clt_pp *pp, const clt_elem_t a, unsigned int b);

int clt_elem_read(clt_elem_t x, const char *fname);
int clt_elem_write(clt_elem_t x, const char *fname);
int clt_elem_fread(clt_elem_t x, FILE *const fp);
int clt_elem_fwrite(clt_elem_t x, FILE *const fp);
int clt_vector_read(clt_elem_t *m, size_t len, const char *fname);
int clt_vector_write(clt_elem_t *m, size_t len, const char *fname);
int clt_vector_fread(clt_elem_t *m, size_t len, FILE *const fp);
int clt_vector_fwrite(clt_elem_t *m, size_t len, FILE *const fp);

#ifdef __cplusplus
}
#endif

#endif
