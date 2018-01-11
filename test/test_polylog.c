#include <clt_pl.h>
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
test(size_t lambda)
{
    srand(time(NULL));
    /* srand(0); */

    const size_t nzs = 4;
    size_t flags = CLT_PL_FLAG_VERBOSE;
    clt_pl_state_t *mmap;
    clt_pl_pp_t *pp;
    aes_randstate_t rng;
    int top[] = {1, 1, 1, 1};
    int ok = 1;
    clt_elem_t *x0, *x1, *x2, *x3, *x4, *x5, *out;
    mpz_t zero, one;

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
    clt_pl_params_t params = {
        .lambda = lambda,
        .nlevels = 2,
        .sparams = ss,
        .nmuls = 3,
        .nzs = nzs,
        .pows = top,
    };
    clt_pl_opt_params_t opts = {
        .slots = 1,
        .moduli = NULL,
        .nmoduli = 0,
        .wordsize = 64,
    };

    aes_randinit(rng);

    mmap = clt_pl_state_new(&params, &opts, 0, flags, rng);
    pp = clt_pl_pp_new(mmap);

    x0 = clt_elem_new();
    x1 = clt_elem_new();
    x2 = clt_elem_new();
    x3 = clt_elem_new();
    x4 = clt_elem_new();
    x5 = clt_elem_new();
    out = clt_elem_new();
    mpz_init_set_ui(zero, 0);
    mpz_init_set_ui(one, 1);

    clt_pl_encode(x0, mmap, 1, &zero, ix0);
    clt_pl_encode(x1, mmap, 1, &one, ix1);
    clt_pl_encode(x2, mmap, 1, &one, ix2);
    clt_pl_encode(x3, mmap, 1, &one, ix3);
    clt_pl_elem_decrypt(x0, mmap, nzs, ix0, 0);
    clt_pl_elem_decrypt(x1, mmap, nzs, ix1, 0);
    clt_pl_elem_decrypt(x2, mmap, nzs, ix2, 0);
    clt_pl_elem_decrypt(x3, mmap, nzs, ix3, 0);
    clt_pl_elem_mul(x4, pp, x0, x1, 0, true);
    clt_pl_elem_decrypt(x4, mmap, nzs, ix1100, 1);
    clt_pl_elem_mul(x5, pp, x2, x3, 1, true);
    clt_pl_elem_decrypt(x5, mmap, nzs, ix0011, 1);
    clt_pl_elem_mul(out, pp, x4, x5, 2, true);
    clt_pl_elem_decrypt(out, mmap, nzs, top, 2);
    ok &= expect("is_zero(0 * 1 * 1 * 1)", 1, clt_pl_is_zero(out, pp));

    mpz_clears(zero, one, NULL);
    clt_elem_free(x0);
    clt_elem_free(x1);
    clt_elem_free(x2);
    clt_elem_free(x3);
    clt_elem_free(x4);
    clt_elem_free(x5);
    clt_elem_free(out);

    clt_pl_pp_free(pp);
    clt_pl_state_free(mmap);
    return !ok;
}

int
main(int arg, char **argv)
{
    (void) arg; (void) argv;

    size_t lambda = 20;

    if (test(lambda) == 1)
        return 1;
    return 0;
}
