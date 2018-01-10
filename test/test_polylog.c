#include <clt13.h>
#include <stdbool.h>

/* static int */
/* expect(char *desc, int expected, int recieved) */
/* { */
/*     if (expected != recieved) { */
/*         printf("\033[1;41m"); */
/*     } */
/*     printf("%s = %d", desc, recieved); */
/*     if (expected != recieved) { */
/*         printf("\033[0m"); */
/*     } */
/*     puts(""); */
/*     return expected == recieved; */
/* } */

static int
test(size_t lambda, size_t kappa, size_t nzs, bool polylog)
{
    srand(time(NULL));
    /* srand(0); */

    size_t flags = CLT_FLAG_VERBOSE;
    if (polylog)
        flags |= CLT_FLAG_POLYLOG;
    clt_state_t *mmap;
    /* clt_pp_t *pp; */
    aes_randstate_t rng;
    int top[nzs];
    int ok = 1;
    clt_elem_t *x0, *x1, *x2, *x3, *x4, *x5, *out;
    mpz_t zero, one;

    for (size_t i = 0; i < nzs; i++)
        top[i] = 0;
    aes_randinit(rng);

    clt_params_t params = {
        .lambda = lambda,
        .kappa = kappa,
        .nzs = nzs,
        .pows = top,
    };
    size_t levels[] = {0, 0, 1};
    clt_opt_params_t opts = {
        .slots = 0,
        .moduli = NULL,
        .nmoduli = 0,
        .nlevels = 2,
        .levels = levels,
        .nops = 3,
    };

    mmap = clt_state_new(&params, &opts, 0, flags, rng);
    /* pp = clt_pp_new(mmap); */

    x0 = clt_elem_new();
    x1 = clt_elem_new();
    x2 = clt_elem_new();
    x3 = clt_elem_new();
    x4 = clt_elem_new();
    x5 = clt_elem_new();
    out = clt_elem_new();
    mpz_init_set_ui(zero, 0);
    mpz_init_set_ui(one, 1);

    clt_encode(x0, mmap, 1, &one, top, 0);
    clt_encode(x1, mmap, 1, &one, top, 0);
    clt_encode(x2, mmap, 1, &one, top, 0);
    clt_encode(x3, mmap, 1, &one, top, 0);
    polylog_elem_decrypt(x0, mmap, 0);
    polylog_elem_decrypt(x1, mmap, 0);
    polylog_elem_decrypt(x2, mmap, 0);
    polylog_elem_decrypt(x3, mmap, 0);
    polylog_elem_mul(x4, mmap, x0, x1, 0);
    polylog_elem_decrypt(x4, mmap, 1);
    polylog_elem_mul(x5, mmap, x2, x3, 1);
    polylog_elem_decrypt(x5, mmap, 1);
    polylog_elem_mul(out, mmap, x4, x5, 2);
    polylog_elem_decrypt(out, mmap, 2);
    /* ok &= expect("is_zero(0 * 1 * 1 * 1)", 1, clt_is_zero(out, pp)); */

    return !ok;
}

int
main(int arg, char **argv)
{
    (void) arg; (void) argv;

    size_t lambda = 16;
    size_t kappa = 5;
    size_t nzs = 10;

    /* if (test(lambda, kappa, nzs, false) == 1) */
    /*     return 1; */
    if (test(lambda, kappa, nzs, true) == 1)
        return 1;
    return 0;
}
