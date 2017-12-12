#include <clt13.h>
#include <aesrand/aesrand.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

static size_t
ram_usage(void)
{
    FILE *fp;
    size_t rss = 0;
    if ((fp = fopen("/proc/self/statm", "r")) == NULL)
        return 0;
    if (fscanf(fp, "%*s%lu", &rss) != 1) {
        fclose(fp);
        return 0;
    }
    fclose(fp);
    return rss * sysconf(_SC_PAGESIZE) / 1024;
}

static int
expect(char * desc, int expected, int recieved)
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
test(ulong flags, ulong nzs, ulong lambda, ulong kappa)
{
    srand(time(NULL));

    clt_state_t *mmap;
    clt_pp_t *pp;
    clt_elem_t *moduli;
    aes_randstate_t rng;
    int pows[nzs];

    int ok = 1;

    aes_randinit(rng);
    for (ulong i = 0; i < nzs; i++) pows[i] = 1;

    clt_params_t params = {
        .lambda = lambda,
        .kappa = kappa,
        .nzs = nzs,
        .pows = pows
    };
    mmap = clt_state_new(&params, NULL, 0, flags, rng);
    pp = clt_pp_new(mmap);

    /* Test read/write */

    {
        FILE *mmap_f;

        mmap_f = tmpfile();
        if (mmap_f == NULL) {
            fprintf(stderr, "Couldn't open test.map!\n");
            exit(1);
        }

        if (clt_state_fwrite(mmap, mmap_f)) {
            fprintf(stderr, "clt_state_fsave failed!\n");
            exit(1);
        }
        clt_state_free(mmap);
        rewind(mmap_f);
        if ((mmap = clt_state_fread(mmap_f)) == NULL) {
            fprintf(stderr, "clt_state_fread failed for mmap!\n");
            exit(1);
        }
        fclose(mmap_f);
    }

    {
        FILE *pp_f;

        pp_f = tmpfile();
        if (pp_f == NULL) {
            fprintf(stderr, "Couldn't open test.pp!\n");
            exit(1);
        }

        if (clt_pp_fwrite(pp, pp_f) != 0) {
            fprintf(stderr, "clt_pp_fsave failed!\n");
            exit(1);
        }
        clt_pp_free(pp);
        rewind(pp_f);
        if ((pp = clt_pp_fread(pp_f)) == NULL) {
            fprintf(stderr, "clt_pp_fread failed for pp!\n");
            exit(1);
        }
        fclose(pp_f);
    }

    moduli = clt_state_moduli(mmap);

    mpz_t x[1], zero[1], one[1], two[1], three[1];
    int top_level[nzs];

    mpz_init_set_ui(x[0], 0);
    while (mpz_cmp_ui(x[0], 0) <= 0) {
        mpz_set_ui(x[0], rand());
        mpz_mod(x[0], x[0], moduli[0]);
    }
    gmp_printf("x = %Zd\n", x[0]);

    mpz_init_set_ui(zero[0], 0);
    mpz_init_set_ui(one[0], 1);
    mpz_init_set_ui(two[0], 2);
    mpz_init_set_ui(three[0], 3);

    for (ulong i = 0; i < nzs; i++) {
        top_level[i] = 1;
    }

    mpz_t x0, x1, xp;
    mpz_inits(x0, x1, xp, NULL);

    clt_encode(x0, mmap, 1, zero, top_level);
    clt_encode(x1, mmap, 1, zero, top_level);
    clt_elem_add(xp, pp, x0, x1);
    ok &= expect("is_zero(0 + 0)", 1, clt_is_zero(xp, pp));

    clt_encode(x0, mmap, 1, zero, top_level);
    clt_encode(x1, mmap, 1, one,  top_level);
    clt_elem_add(xp, pp, x0, x1);
    ok &= expect("is_zero(0 + 1)", 0, clt_is_zero(xp, pp));

    clt_encode(x0, mmap, 1, one, top_level);
    clt_encode(x1, mmap, 1, two, top_level);
    clt_elem_mul_ui(x0, pp, x0, 2);
    clt_elem_sub(xp, pp, x1, x0);
    ok &= expect("is_zero(2 - 2[1])", 1, clt_is_zero(xp, pp));

    clt_encode(x0, mmap, 1, zero, top_level);
    clt_encode(x1, mmap, 1, x,    top_level);
    clt_elem_add(xp, pp, x0, x1);
    ok &= expect("is_zero(0 + x)", 0, clt_is_zero(xp, pp));

    clt_encode(x0, mmap, 1, x, top_level);
    clt_encode(x1, mmap, 1, x, top_level);
    clt_elem_sub(xp, pp, x0, x1);
    ok &= expect("is_zero(x - x)", 1, clt_is_zero(xp, pp));

    clt_encode(x0, mmap, 1, zero, top_level);
    clt_encode(x1, mmap, 1, x,    top_level);
    clt_elem_sub(xp, pp, x0, x1);
    ok &= expect("is_zero(0 - x)", 0, clt_is_zero(xp, pp));

    clt_encode(x0, mmap, 1, one,  top_level);
    clt_encode(x1, mmap, 1, zero, top_level);
    clt_elem_sub(xp, pp, x0, x1);
    ok &= expect("is_zero(1 - 0)", 0, clt_is_zero(xp, pp));

    clt_encode(x0, mmap, 1, one,  top_level);
    clt_encode(x1, mmap, 1, three, top_level);
    clt_elem_mul_ui(x0, pp, x0, 3);
    clt_elem_sub(xp, pp, x0, x1);
    ok &= expect("is_zero(3*[1] - [3])", 1, clt_is_zero(xp, pp));

    int ix0[nzs], ix1[nzs];
    for (ulong i = 0; i < nzs; i++) {
        if (i < nzs / 2) {
            ix0[i] = 1;
            ix1[i] = 0;
        } else {
            ix0[i] = 0;
            ix1[i] = 1;
        }
    }
    clt_encode(x0, mmap, 1, x   , ix0);
    clt_encode(x1, mmap, 1, zero, ix1);
    clt_elem_mul(xp, pp, x0, x1);
    ok &= expect("is_zero(x * 0)", 1, clt_is_zero(xp, pp));

    clt_encode(x0, mmap, 1, x  , ix0);
    clt_encode(x1, mmap, 1, one, ix1);
    clt_elem_mul(xp, pp, x0, x1);
    ok &= expect("is_zero(x * 1)", 0, clt_is_zero(xp, pp));

    clt_encode(x0, mmap, 1, x, ix0);
    clt_encode(x1, mmap, 1, x, ix1);
    clt_elem_mul(xp, pp, x0, x1);
    ok &= expect("is_zero(x * x)", 0, clt_is_zero(xp, pp));

    // zimmerman-like test

    mpz_t c, in0[2], in1[2], cin[2];

    mpz_inits(c, in0[0], in0[1], in1[0], in1[1], cin[0], cin[1], NULL);

    mpz_urandomb_aes(in1[0], rng, lambda);
    mpz_mod(in1[0], in1[0], moduli[0]);

    mpz_set_ui(in0[0], 0);
    mpz_set_ui(cin[0], 0);

    mpz_urandomb_aes(in0[1], rng, 16);
    mpz_urandomb_aes(in1[1], rng, 16);
    mpz_mul(cin[1], in0[1], in1[1]);

    clt_encode(x0, mmap, 2, in0, ix0);
    clt_encode(x1, mmap, 2, in1, ix1);
    clt_encode(c,  mmap, 2, cin, top_level);

    clt_elem_mul(xp, pp, x0, x1);
    clt_elem_sub(xp, pp, xp, c);

    ok &= expect("[Z] is_zero(0 * x)", 1, clt_is_zero(xp, pp));

    mpz_set_ui(in0[0], 1);
    mpz_set_ui(in1[0], 1);
    mpz_set_ui(cin[0], 0);

    mpz_urandomb_aes(in0[0], rng, lambda);
    mpz_mod(in0[0], in0[0], moduli[0]);

    mpz_urandomb_aes(in1[0], rng, lambda);
    mpz_mod(in1[0], in1[0], moduli[0]);

    mpz_urandomb_aes(in0[1], rng, 16);
    mpz_urandomb_aes(in1[1], rng, 16);
    mpz_mul(cin[1], in0[1], in1[1]);

    clt_encode(x0, mmap, 2, in0, ix0);
    clt_encode(x1, mmap, 2, in1, ix1);
    clt_encode(c,  mmap, 2, cin, top_level);

    clt_elem_mul(xp, pp, x0, x1);
    clt_elem_sub(xp, pp, xp, c);

    ok &= expect("[Z] is_zero(x * y)", 0, clt_is_zero(xp, pp));

    clt_pp_free(pp);
    clt_state_free(mmap);
    mpz_clears(c, x0, x1, xp, x[0], zero[0], one[0], two[0], three[0],
               in0[0], in0[1], in1[0], in1[1], cin[0], cin[1], NULL);
    aes_randclear(rng);

    {
        size_t ram = ram_usage();
        printf("RAM: %lu Kb\n", ram);
    }

    return !ok;
}

static int
test_levels(ulong flags, ulong kappa, ulong lambda)
{
    int pows[kappa], top_level[kappa];
    clt_state_t *s;
    clt_pp_t *pp;
    aes_randstate_t rng;
    mpz_t zero, one, value, result, top_one, top_zero;
    int ok = 1;

    clt_params_t params = {
        .lambda = lambda,
        .kappa = kappa,
        .nzs = kappa,
        .pows = top_level
    };

    printf("Testing levels: λ = %lu, κ = %lu\n", lambda, kappa);

    aes_randinit(rng);
    mpz_init_set_ui(zero, 0);
    mpz_init_set_ui(one, 1);
    mpz_inits(value, result, top_one, top_zero, NULL);

    for (ulong i = 0; i < kappa; ++i)
        top_level[i] = 1;

    s = clt_state_new(&params, NULL, 0, flags, rng);
    pp = clt_pp_new(s);

    clt_encode(top_one, s, 1, &one, top_level);
    clt_encode(top_zero, s, 0, &zero, top_level);

    for (ulong i = 0; i < kappa; ++i) {
        for (ulong j = 0; j < kappa; ++j) {
            if (j != i)
                pows[j] = 0;
            else
                pows[j] = 1;
        }
        clt_encode(value, s, 1, &one, pows);
        if (i == 0)
            clt_elem_set(result, value);
        else {
            clt_elem_mul(result, pp, result, value);
        }
    }

    ok &= expect("is_zero(1 * ... * 1)", 0, clt_is_zero(result, pp));

    clt_elem_sub(result, pp, result, top_one);

    ok &= expect("is_zero(1 * ... * 1 - 1)", 1, clt_is_zero(result, pp));

    for (ulong i = 0; i < kappa; ++i) {
        for (ulong j = 0; j < kappa; ++j) {
            if (j != i)
                pows[j] = 0;
            else
                pows[j] = 1;
        }
        if (i == 0) {
            clt_encode(value, s, 0, &one, pows);
            clt_elem_set(result, value);
        } else {
            clt_encode(value, s, 1, &one, pows);
            clt_elem_mul(result, pp, result, value);
        }
    }

    ok &= expect("is_zero(0 * 1 *  ... * 1)", 1, clt_is_zero(result, pp));

    clt_elem_add(result, pp, result, top_one);

    ok &= expect("is_zero(0 * 1 *  ... * 1 + 1)", 0, clt_is_zero(result, pp));

    {
        size_t ram = ram_usage();
        printf("RAM: %lu Kb\n", ram);
    }

    clt_pp_free(pp);
    clt_state_free(s);

    mpz_clears(value, result, top_one, top_zero, zero, one, NULL);
    aes_randclear(rng);

    return !ok;
}

int
main(int argc, char *argv[])
{
    ulong default_flags = CLT_FLAG_NONE | CLT_FLAG_VERBOSE;
    ulong flags;
    ulong kappa;
    ulong lambda = 40;
    ulong nzs = 10;

    if (argc == 2) {
        lambda = atoi(argv[1]);
    }

    if (test_levels(default_flags, 32, 10))
        return 1;

    printf("* No optimizations\n");
    flags = default_flags;
    if (test(flags, nzs, 16, 2) == 1)
        return 1;

    kappa = 15;

    printf("* No optimizations\n");
    flags = default_flags;
    if (test(flags, nzs, lambda, kappa) == 1)
        return 1;

    printf("* CRT tree\n");
    flags = default_flags | CLT_FLAG_OPT_CRT_TREE;
    if (test(flags, nzs, lambda, kappa) == 1)
        return 1;

    printf("* CRT tree + parallel encode\n");
    flags = default_flags | CLT_FLAG_OPT_CRT_TREE | CLT_FLAG_OPT_PARALLEL_ENCODE;
    if (test(flags, nzs, lambda, kappa) == 1)
        return 1;

    printf("* CRT tree + composite ps\n");
    flags = default_flags | CLT_FLAG_OPT_CRT_TREE | CLT_FLAG_OPT_COMPOSITE_PS;
    if (test(flags, nzs, lambda, kappa) == 1)
        return 1;

    printf("* CRT tree + parallel encode + composite ps\n");
    flags = default_flags | CLT_FLAG_OPT_CRT_TREE | CLT_FLAG_OPT_PARALLEL_ENCODE | CLT_FLAG_OPT_COMPOSITE_PS;
    if (test(flags, nzs, lambda, kappa) == 1)
        return 1;

    kappa = 12;
    printf("* CRT tree + parallel encode + composite ps\n");
    flags = default_flags | CLT_FLAG_OPT_CRT_TREE | CLT_FLAG_OPT_PARALLEL_ENCODE | CLT_FLAG_OPT_COMPOSITE_PS;
    if (test(flags, nzs, lambda, kappa) == 1)
        return 1;

    printf("* polylog\n");
    flags = default_flags | CLT_FLAG_POLYLOG;
    if (test(flags, nzs, lambda, kappa) == 1)
        return 1;

    return 0;
}
