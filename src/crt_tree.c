#include "crt_tree.h"

#include <assert.h>
#include <stdbool.h>

static crt_tree *
_crt_tree_new(clt_elem_t *const ps, size_t n)
{
    crt_tree *crt;

    if (ps == NULL || n == 0)
        return NULL;

    crt = calloc(1, sizeof(crt_tree));
    crt->n = n;
    mpz_inits(crt->mod, crt->crt_left, crt->crt_right, NULL);
    if (crt->n == 1) {
        crt->left = NULL;
        crt->right = NULL;
        mpz_set(crt->mod, ps[0]);
    } else {
        clt_elem_t g;

        {
#pragma omp task
        {
            crt->left = _crt_tree_new(ps, crt->n / 2);
        }
#pragma omp task
        {
            crt->right = _crt_tree_new(ps + crt->n / 2, crt->n  - crt->n / 2);
        }
#pragma omp taskwait
        }

        if (!crt->left || !crt->right) {
            crt_tree_free(crt);
            return NULL;
        }

        mpz_init_set_ui(g, 0);
        mpz_gcdext(g, crt->crt_right, crt->crt_left, crt->left->mod, crt->right->mod);
        if (mpz_cmp_ui(g, 1) != 0) {
            crt_tree_free(crt);
            mpz_clear(g);
            return NULL;
        }
        mpz_clear(g);

        mpz_mul(crt->crt_left,  crt->crt_left,  crt->right->mod);
        mpz_mul(crt->crt_right, crt->crt_right, crt->left->mod);
        mpz_mul(crt->mod, crt->left->mod, crt->right->mod);
    }
    return crt;
}

crt_tree *
crt_tree_new(clt_elem_t *const ps, size_t n)
{
    crt_tree *crt;
#pragma omp parallel default(shared)
#pragma omp single
    crt = _crt_tree_new(ps, n);
    return crt;
}

void
crt_tree_free(crt_tree *crt)
{
    if (crt->left)
        crt_tree_free(crt->left);
    if (crt->right)
        crt_tree_free(crt->right);
    mpz_clears(crt->mod, crt->crt_left, crt->crt_right, NULL);
    free(crt);
}

void
crt_tree_do_crt(clt_elem_t rop, const crt_tree *crt, clt_elem_t *cs)
{
    if (crt->n == 1) {
        mpz_set(rop, cs[0]);
    } else {
        clt_elem_t left, right, tmp;
        mpz_inits(left, right, tmp, NULL);

        crt_tree_do_crt(left,  crt->left,  cs);
        crt_tree_do_crt(right, crt->right, cs + crt->n / 2);

        mpz_mul(rop, left,  crt->crt_left);
        mpz_mul(tmp, right, crt->crt_right);
        mpz_add(rop, rop, tmp);
        mpz_mod(rop, rop, crt->mod);

        mpz_clears(left, right, tmp, NULL);
    }
}

static void
_crt_tree_get_leafs(clt_elem_t *leafs, int *i, const crt_tree *const crt)
{
    if (crt->n == 1) {
        mpz_set(leafs[(*i)++], crt->mod);
    } else {
        _crt_tree_get_leafs(leafs, i, crt->left);
        _crt_tree_get_leafs(leafs, i, crt->right);
    }
}

crt_tree *
crt_tree_read(const char *fname, size_t n)
{
    crt_tree *crt;
    clt_elem_t *ps = calloc(n, sizeof(clt_elem_t));
    for (ulong i = 0; i < n; i++)
        mpz_init(ps[i]);

    clt_vector_read(ps, n, fname);
    crt = crt_tree_new(ps, n);

    for (ulong i = 0; i < n; i++)
        mpz_clear(ps[i]);
    free(ps);

    return crt;
}

crt_tree *
crt_tree_fread(FILE *const fp, size_t n)
{
    crt_tree *crt = NULL;

    clt_elem_t *ps = calloc(n, sizeof(clt_elem_t));
    for (ulong i = 0; i < n; i++)
        mpz_init(ps[i]);

    if (clt_vector_fread(ps, n, fp) != 0) {
        fprintf(stderr, "[%s] couldn't read ps!\n", __func__);
        goto cleanup;
    }

    if ((crt = crt_tree_new(ps, n)) == NULL) {
        fprintf(stderr, "[%s] couldn't initialize crt_tree!\n", __func__);
        goto cleanup;
    }

cleanup:
    for (ulong i = 0; i < n; i++)
        mpz_clear(ps[i]);
    free(ps);

    return crt;
}

void
crt_tree_write(const char *fname, const crt_tree *const crt, size_t n)
{
    clt_elem_t *ps = calloc(n, sizeof(clt_elem_t));
    for (ulong i = 0; i < n; i++)
        mpz_init(ps[i]);
    int ctr = 0;

    _crt_tree_get_leafs(ps, &ctr, crt);
    clt_vector_write(ps, n, fname);

    for (ulong i = 0; i < n; i++)
        mpz_clear(ps[i]);
    free(ps);
}



int
crt_tree_fwrite(FILE *const fp, const crt_tree *const crt, size_t n)
{
    int ret = 1;

    clt_elem_t *ps = calloc(n, sizeof(clt_elem_t));
    for (ulong i = 0; i < n; i++)
        mpz_init(ps[i]);
    int ctr = 0;

    _crt_tree_get_leafs(ps, &ctr, crt);
    if (clt_vector_fwrite(ps, n, fp) != 0)
        goto cleanup;

    for (ulong i = 0; i < n; i++)
        mpz_clear(ps[i]);
    free(ps);

    ret = 0;
cleanup:
    return ret;
}
