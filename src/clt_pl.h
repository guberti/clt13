#ifndef __CLT_POLYLOG_H__
#define __CLT_POLYLOG_H__

#include "clt13.h"

#include <aesrand/aesrand.h>
#include <stdlib.h>

typedef struct clt_pl_state_t clt_pl_state_t;
typedef struct clt_pl_pp_t clt_pl_pp_t;
typedef struct switch_state_t switch_state_t;

typedef struct {
    size_t level;
    int *ix;
} switch_params_t;

/* Required parameters to clt_pl_state_new */
typedef struct {
    /* security parameter */
    size_t lambda;
    /* number of levels */
    size_t nlevels;
    /* switch parameters of each multiplication */
    switch_params_t *sparams;
    /* number of multiplications */
    size_t nmuls;
    /* number of z's */
    size_t nzs;
    /* powers for the z's */
    int *pows;
} clt_pl_params_t;

/* Optional parameters to clt_pl_state_new */
typedef struct {
    /* number of slots needed */
    size_t slots;
    /* plaintext moduli */
    mpz_t *moduli;
    /* number of plaintext moduli given */
    size_t nmoduli;
    /* word size */
    size_t wordsize;
} clt_pl_opt_params_t;

#define CLT_PL_FLAG_NONE 0x00
/* Be verbose */
#define CLT_PL_FLAG_VERBOSE 0x01

clt_pl_state_t * clt_pl_state_new(const clt_pl_params_t *params,
                                  const clt_pl_opt_params_t *opts,
                                  size_t nthreads, size_t flags,
                                  aes_randstate_t rng);
void             clt_pl_state_free(clt_pl_state_t *s);
clt_pl_state_t * clt_pl_state_fread(FILE *fp);
int              clt_pl_state_fwrite(clt_pl_state_t *s, FILE *fp);
mpz_t *          clt_pl_state_moduli(const clt_pl_state_t *s);
size_t           clt_pl_state_nslots(const clt_pl_state_t *s);
size_t           clt_pl_state_nzs(const clt_pl_state_t *s);

clt_pl_pp_t * clt_pl_pp_new(const clt_pl_state_t *s);
void          clt_pl_pp_free(clt_pl_pp_t *pp);
clt_pl_pp_t * clt_pl_pp_fread(FILE *fp);
int           clt_pl_pp_fwrite(clt_pl_pp_t *pp, FILE *fp);

int clt_pl_encode(clt_elem_t *rop, const clt_pl_state_t *s, size_t n, mpz_t *xs, const int *ix);
int clt_pl_elem_add(clt_elem_t *rop, const clt_pl_pp_t *pp, const clt_elem_t *a,
                     const clt_elem_t *b, size_t level);
int clt_pl_elem_mul(clt_elem_t *rop, const clt_pl_pp_t *s, const clt_elem_t *a,
                     const clt_elem_t *b, size_t idx, bool verbose);
int clt_pl_elem_switch(clt_elem_t *rop, const clt_pl_pp_t *pp, const clt_elem_t *x_,
                       const switch_state_t *sstate, bool verbose);
int clt_pl_elem_decrypt(clt_elem_t *rop, const clt_pl_state_t *s, size_t nzs, const int ix[nzs],
                         size_t level);

int clt_pl_is_zero(const clt_elem_t *c, const clt_pl_pp_t *pp);


#endif  /* __CLT_POLYLOG_H__ */
