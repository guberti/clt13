#include "_clt13.h"
#include "utils.h"

#include <omp.h>

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

static inline size_t
_slot(size_t i, size_t n, size_t maxslots)
{
    if (i == maxslots - 1)
        return n - 1;
    else
        return i / (maxslots / n);
}

int
clt_encode(clt_elem_t *rop, const clt_state_t *s, size_t n, mpz_t *xs,
           const int *ix, clt_encode_opt_params_t *opt)
{
    size_t rho;

    if (rop == NULL || s == NULL || n == 0 || xs == NULL)
        return CLT_ERR;

    if (!(s->flags & CLT_FLAG_OPT_PARALLEL_ENCODE))
        omp_set_num_threads(1);

    if (s->flags & CLT_FLAG_POLYLOG)
        return polylog_encode(rop, s, n, xs, ix, 0);

    rho = opt && opt->rho > 0 ? opt->rho : s->rho;

    /* slots[i] = m[i] + r·g[i] */
    if (s->flags & CLT_FLAG_OPT_CRT_TREE) {
        mpz_t *slots = mpz_vector_new(s->n);
#pragma omp parallel for
        for (size_t i = 0; i < s->n; i++) {
            mpz_random_(slots[i], s->rngs[i], rho);
            mpz_mul(slots[i], slots[i], s->gs[i]);
            mpz_add(slots[i], slots[i], xs[_slot(i, n, s->n)]);
        }
        crt_tree_do_crt(rop->elem, s->crt, slots);
        mpz_vector_free(slots, s->n);
    } else {
        mpz_set_ui(rop->elem, 0);
#pragma omp parallel for
        for (size_t i = 0; i < s->n; ++i) {
            mpz_t tmp;
            mpz_init(tmp);
            mpz_random_(tmp, s->rngs[i], rho);
            mpz_mul(tmp, tmp, s->gs[i]);
            mpz_add(tmp, tmp, xs[_slot(i, n, s->n)]);
            mpz_mul(tmp, tmp, s->crt_coeffs[i]);
#pragma omp critical
            {
                mpz_add(rop->elem, rop->elem, tmp);
            }
            mpz_clear(tmp);
        }
    }
    rop->level = 0;
    if (ix) {
        mpz_t tmp;
        mpz_init(tmp);
        /* multiply by appropriate zinvs */
        rop->ix = calloc(s->nzs, sizeof rop->ix[0]);
        for (unsigned long i = 0; i < s->nzs; ++i) {
            if (ix[i] <= 0)
                continue;
            rop->ix[i] = ix[i];
            mpz_powm_ui(tmp, s->zinvs[i], ix[i], s->x0);
            mpz_mul_mod(rop->elem, rop->elem, tmp, s->x0);
        }
        mpz_clear(tmp);
    }
    return CLT_OK;
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

