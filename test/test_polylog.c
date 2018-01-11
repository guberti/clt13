#include <clt13.h>
#include <stdbool.h>

static int
expect(char *desc, int expected, int recieved)
{
    if (expected != recieved) {
        printf("\033[1;41m");
    }
    printf("%s = %d", desc, recieved);
    if (expected != recieved) {
        printf("\033[0m");
    }
    puts("");
    return expected == recieved;
}

static int
test(size_t lambda, size_t kappa, bool polylog)
{
    srand(time(NULL));
    /* srand(0); */

    const size_t nzs = 4;
    size_t flags = CLT_FLAG_VERBOSE;
    if (polylog)
        flags |= CLT_FLAG_POLYLOG;
    clt_state_t *mmap;
    clt_pp_t *pp;
    aes_randstate_t rng;
    int top[] = {1, 1, 1, 1};
    int ok = 1;
    clt_elem_t *x0, *x1, *x2, *x3, *x4, *x5, *out;
    mpz_t zero, one;

    clt_params_t params = {
        .lambda = lambda,
        .kappa = kappa,
        .nzs = nzs,
        .pows = top,
    };
    int ix0[] = {1, 0, 0, 0};
    int ix1[] = {0, 1, 0, 0};
    int ix2[] = {0, 0, 1, 0};
    int ix3[] = {0, 0, 0, 1};
    int ix1100[] = {1, 1, 0, 0};
    int ix0011[] = {0, 0, 1, 1};
    switch_params_t ss[] = {
        { .level = 0, .ix = ix1100 },
        { .level = 0, .ix = ix0011 },
        { .level = 1, .ix = top },
    };
    clt_opt_params_t opts = {
        .slots = 1,
        .moduli = NULL,
        .nmoduli = 0,
        .nlevels = 2,
        .sparams = ss,
        .nmuls = 3,
    };

    aes_randinit(rng);

    mmap = clt_state_new(&params, &opts, 0, flags, rng);
    pp = clt_pp_new(mmap);

    x0 = clt_elem_new();
    x1 = clt_elem_new();
    x2 = clt_elem_new();
    x3 = clt_elem_new();
    x4 = clt_elem_new();
    x5 = clt_elem_new();
    out = clt_elem_new();
    mpz_init_set_ui(zero, 0);
    mpz_init_set_ui(one, 1);

    clt_encode(x0, mmap, 1, &zero, ix0);
    clt_encode(x1, mmap, 1, &one, ix1);
    clt_encode(x2, mmap, 1, &one, ix2);
    clt_encode(x3, mmap, 1, &one, ix3);
    polylog_elem_decrypt(x0, mmap, ix0, nzs, 0);
    polylog_elem_decrypt(x1, mmap, ix1, nzs, 0);
    polylog_elem_decrypt(x2, mmap, ix2, nzs, 0);
    polylog_elem_decrypt(x3, mmap, ix3, nzs, 0);
    /* polylog_elem_add(x4, pp, x0, x0); */
    polylog_elem_mul(x4, pp, x0, x1, 0, true);
    polylog_elem_decrypt(x4, mmap, ix1100, nzs, 1);
    polylog_elem_mul(x5, pp, x2, x3, 1, true);
    polylog_elem_decrypt(x5, mmap, ix0011, nzs, 1);
    polylog_elem_mul(out, pp, x4, x5, 2, true);
    polylog_elem_decrypt(out, mmap, top, nzs, 2);
    ok &= expect("is_zero(0 * 1 * 1 * 1)", 1, polylog_is_zero(out, pp));

    mpz_clears(zero, one, NULL);
    clt_elem_free(x0);
    clt_elem_free(x1);
    clt_elem_free(x2);
    clt_elem_free(x3);
    clt_elem_free(x4);
    clt_elem_free(x5);
    clt_elem_free(out);
    return !ok;
}

int
main(int arg, char **argv)
{
    (void) arg; (void) argv;

    size_t lambda = 20;
    size_t kappa = 4;

    /* if (test(lambda, kappa, nzs, false) == 1) */
    /*     return 1; */
    if (test(lambda, kappa, true) == 1)
        return 1;
    return 0;
}
