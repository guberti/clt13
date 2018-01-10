#include "_clt13.h"
#include "utils.h"

clt_elem_t *
clt_elem_new(void)
{
    clt_elem_t *e = calloc(1, sizeof e[0]);
    mpz_init(e->elem);
    e->level = 0;
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
    a->level = b->level;
}

int
clt_elem_add_level(clt_elem_t *rop, const clt_pp_t *pp, const clt_elem_t *a,
                   const clt_elem_t *b, size_t level)
{
    if (a->level != b->level) {
        fprintf(stderr, "error: levels unequal (%lu ≠ %lu), unable to add\n",
                a->level, b->level);
        return CLT_ERR;
    } else if (level != a->level) {
        fprintf(stderr, "error: adding elements at incorrect level\n");
        return CLT_ERR;
    }
    mpz_add(rop->elem, a->elem, b->elem);
    mpz_mod(rop->elem, rop->elem, pp->x0);
    if (pp->is_polylog)
        rop->level = a->level;
    return CLT_OK;
}

int
clt_elem_add(clt_elem_t *rop, const clt_pp_t *pp, const clt_elem_t *a, const clt_elem_t *b)
{
    return clt_elem_add_level(rop, pp, a, b, 0);
}

int
clt_elem_sub(clt_elem_t *rop, const clt_pp_t *pp, const clt_elem_t *a, const clt_elem_t *b)
{
    if (a->level != b->level) {
        fprintf(stderr, "error: levels unequal (%lu ≠ %lu), unable to subtract",
                a->level, b->level);
        return CLT_ERR;
    }
    mpz_sub(rop->elem, a->elem, b->elem);
    mpz_mod(rop->elem, rop->elem, pp->x0);
    if (pp->is_polylog)
        rop->level = a->level;
    return CLT_OK;
}

int
clt_elem_mul(clt_elem_t *rop, const clt_pp_t *pp, const clt_elem_t *a, const clt_elem_t *b)
{
    if (a->level != b->level) {
        fprintf(stderr, "error: levels unequal (%lu ≠ %lu), unable to multiply",
                a->level, b->level);
        return CLT_ERR;
    }
    mpz_mul(rop->elem, a->elem, b->elem);
    mpz_mod(rop->elem, rop->elem, pp->x0);
    if (pp->is_polylog)
        rop->level = a->level + 1;
    return CLT_OK;
}

int
clt_elem_mul_ui(clt_elem_t *rop, const clt_pp_t *pp, const clt_elem_t *a, unsigned int b)
{
    mpz_mul_ui(rop->elem, a->elem, b);
    mpz_mod(rop->elem, rop->elem, pp->x0);
    if (pp->is_polylog)
        rop->level = a->level;
    return CLT_OK;
}

void
clt_elem_print(const clt_elem_t *a)
{
    gmp_printf("%Zd | %lu", a->elem, a->level);
}

int
clt_elem_fread(clt_elem_t *x, FILE *fp)
{
    if (mpz_fread(x->elem, fp) == CLT_ERR)
        return CLT_ERR;
    if (size_t_fread(fp, &x->level) == CLT_ERR)
        return CLT_ERR;
    return CLT_OK;
}

int
clt_elem_fwrite(clt_elem_t *x, FILE *fp)
{
    if (mpz_fwrite(x->elem, fp) == CLT_ERR)
        return CLT_ERR;
    if (size_t_fwrite(fp, x->level) == CLT_ERR)
        return CLT_ERR;
    return CLT_OK;
}
