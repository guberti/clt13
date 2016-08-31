#include "crt_tree.h"

#include <assert.h>

int
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

void
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

void
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

void
crt_tree_save(const char *fname, crt_tree *crt, size_t n)
{
    clt_elem_t *ps = malloc(n * sizeof(clt_elem_t));
    for (ulong i = 0; i < n; i++)
        mpz_init(ps[i]);
    int ctr = 0;

    _crt_tree_get_leafs(ps, &ctr, crt);
    clt_vector_write(ps, n, fname);

    for (ulong i = 0; i < n; i++)
        mpz_clear(ps[i]);
    free(ps);
}

void
crt_tree_read(const char *fname, crt_tree *crt, size_t n)
{
    clt_elem_t *ps = malloc(n * sizeof(clt_elem_t));
    for (ulong i = 0; i < n; i++)
        mpz_init(ps[i]);

    clt_vector_read(ps, n, fname);
    crt_tree_init(crt, ps, n);

    for (ulong i = 0; i < n; i++)
        mpz_clear(ps[i]);
    free(ps);
}

int
crt_tree_fread (FILE *const fp, crt_tree *crt, size_t n)
{
    int ret = 1;

    clt_elem_t *ps = malloc(n * sizeof(clt_elem_t));
    for (ulong i = 0; i < n; i++)
        mpz_init(ps[i]);

    if (clt_vector_fread(ps, n, fp) != 0) {
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

int
crt_tree_fsave(FILE *const fp, crt_tree *crt, size_t n)
{
    int ret = 1;

    clt_elem_t *ps = malloc(n * sizeof(clt_elem_t));
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
