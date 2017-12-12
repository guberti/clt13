#include "clt13.h"
#include "utils.h"

int
mpz_fread(mpz_t x, FILE *fp)
{
    if (mpz_inp_raw(x, fp) == 0)
        return CLT_ERR;
    return CLT_OK;
}

int
mpz_fwrite(mpz_t x, FILE *fp)
{
    if (mpz_out_raw(fp, x) == 0)
        return CLT_ERR;
    return CLT_OK;
}

mpz_t *
mpz_vector_new(size_t n)
{
    mpz_t *v;
    if ((v = calloc(n, sizeof v[0])) == NULL)
        return NULL;
    for (size_t i = 0; i < n; ++i)
        mpz_init(v[i]);
    return v;
}

void
mpz_vector_free(mpz_t *v, size_t n)
{
    for (size_t i = 0; i < n; ++i)
        mpz_clear(v[i]);
    free(v);
}

int
mpz_vector_fread(mpz_t *m, size_t len, FILE *fp)
{
    for (size_t i = 0; i < len; ++i) {
        if (mpz_fread(m[i], fp) == CLT_ERR)
            return CLT_ERR;
    }
    return CLT_OK;
}

int
mpz_vector_fwrite(mpz_t *m, size_t len, FILE *fp)
{
    for (size_t i = 0; i < len; ++i) {
        if (mpz_fwrite(m[i], fp) == CLT_ERR)
            return CLT_ERR;
    }
    return CLT_OK;
}
