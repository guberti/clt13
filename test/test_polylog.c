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
test(size_t lambda, size_t kappa, size_t nzs, bool polylog)
{
    /* srand(time(NULL)); */
    srand(0);

    size_t flags = CLT_FLAG_VERBOSE;
    if (polylog)
        flags |= CLT_FLAG_POLYLOG;
    clt_state_t *mmap;
    clt_pp_t *pp;
    aes_randstate_t rng;
    int top[nzs], ix0[nzs], ix1[nzs];
    int ok = 1;
    clt_elem_t *x, *y, *out;
    mpz_t zero, one;

    for (size_t i = 0; i < nzs; i++) {
        top[i] = 0;
    }

    aes_randinit(rng);
    for (size_t i = 0; i < nzs; i++) {
        if (i < nzs / 2) {
            ix0[i] = 0;
            ix1[i] = 0;
        } else {
            ix0[i] = 0;
            ix1[i] = 0;
        }
    }

    clt_params_t params = {
        .lambda = lambda,
        .kappa = kappa,
        .nzs = nzs,
        .pows = top,
    };
    clt_params_opt_t opts = {
        .nlayers = 1,
    };

    mmap = clt_state_new(&params, &opts, 0, flags, rng);
    pp = clt_pp_new(mmap);

    x = clt_elem_new();
    y = clt_elem_new();
    out = clt_elem_new();
    mpz_init_set_ui(zero, 0);
    mpz_init_set_ui(one, 1);

    clt_encode(x, mmap, 1, &zero, ix0);
    clt_encode(y, mmap, 1, &one, ix1);
    /* ok &= expect("is_zero(0)", 1, clt_is_zero(x, pp)); */
    /* ok &= expect("is_zero(1)", 0, clt_is_zero(y, pp)); */
    if (polylog)
        clt_elem_mul_(out, mmap, x, y);
    else
        clt_elem_mul(out, pp, x, y);
    ok &= expect("is_zero(0 * 1)", 1, clt_is_zero(out, pp));

    return !ok;
}

int
main(int arg, char **argv)
{
    (void) arg; (void) argv;

    size_t lambda = 16;
    size_t kappa = 2;
    size_t nzs = 10;

    if (test(lambda, kappa, nzs, false) == 1)
        return 1;
    if (test(lambda, kappa, nzs, true) == 1)
        return 1;
    return 0;
}
