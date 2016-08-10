#include "clt13.h"
#include <aesrand.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <time.h>

int expect(char * desc, int expected, int recieved);

static int test(ulong flags, ulong nzs, ulong lambda, ulong kappa)
{
    srand(time(NULL));

    clt_state mmap, mmap_;
    clt_pp pp_, pp;

    aes_randstate_t rng;
    aes_randinit(rng);

    int pows[nzs];
    for (ulong i = 0; i < nzs; i++) pows[i] = 1;

    /* Test read/write */
    {
        FILE *mmap_f = tmpfile();
        if (mmap_f == NULL) {
            fprintf(stderr, "Couldn't open test.map!\n");
            exit(1);
        }

        FILE *pp_f = tmpfile();
        if (pp_f == NULL) {
            fprintf(stderr, "Couldn't open test.pp!\n");
            exit(1);
        }

        // test initialization & serialization
        clt_state_init(&mmap_, kappa, lambda, nzs, pows, 0, flags, rng);

        if (clt_state_fsave(mmap_f, &mmap_) != 0) {
            fprintf(stderr, "clt_state_fsave failed!\n");
            exit(1);
        }
        rewind(mmap_f);
        clt_state_clear(&mmap_);
        if (clt_state_fread(mmap_f, &mmap) != 0) {
            fprintf(stderr, "clt_state_fread failed for mmap!\n");
            exit(1);
        }

        clt_pp_init(&pp_, &mmap);

        if (clt_pp_fsave(pp_f, &pp_) != 0) {
            fprintf(stderr, "clt_pp_fsave failed!\n");
            exit(1);
        }
        rewind(pp_f);
        clt_pp_clear(&pp_);
        if (clt_pp_fread(pp_f, &pp) != 0) {
            fprintf(stderr, "clt_pp_fread failed for pp!\n");
            exit(1);
        }
    }

    mpz_t x [1];
    mpz_init_set_ui(x[0], 0);
    while (mpz_cmp_ui(x[0], 0) <= 0) {
        mpz_set_ui(x[0], rand());
        mpz_mod(x[0], x[0], mmap.gs[0]);
    }
    gmp_printf("x = %Zd\n", x[0]);

    mpz_t zero [1];
    mpz_init_set_ui(zero[0], 0);
    mpz_t one [1];
    mpz_init_set_ui(one[0], 1);
    mpz_t two[1];
    mpz_init_set_ui(two[0], 2);
    mpz_t three[1];
    mpz_init_set_ui(three[0], 3);

    int top_level [nzs];
    for (ulong i = 0; i < nzs; i++) {
        top_level[i] = 1;
    }

    mpz_t x0, x1, xp;
    mpz_inits(x0, x1, xp, NULL);

    int ok = 1;

    clt_encode(x0, &mmap, 1, zero, top_level);
    clt_encode(x1, &mmap, 1, zero, top_level);
    clt_elem_add(xp, &pp, x0, x1);
    ok &= expect("is_zero(0 + 0)", 1, clt_is_zero(&pp, xp));

    clt_encode(x0, &mmap, 1, zero, top_level);
    clt_encode(x1, &mmap, 1, one,  top_level);
    clt_elem_add(xp, &pp, x0, x1);
    ok &= expect("is_zero(0 + 1)", 0, clt_is_zero(&pp, xp));

    clt_encode(x0, &mmap, 1, one, top_level);
    clt_encode(x1, &mmap, 1, two, top_level);
    clt_elem_mul_ui(x0, &pp, x0, 2);
    clt_elem_sub(xp, &pp, x1, x0);
    ok &= expect("is_zero(2 - 2[1])", 1, clt_is_zero(&pp, xp));
    mpz_clear(two[0]);

    clt_encode(x0, &mmap, 1, zero, top_level);
    clt_encode(x1, &mmap, 1, x,    top_level);
    clt_elem_add(xp, &pp, x0, x1);
    ok &= expect("is_zero(0 + x)", 0, clt_is_zero(&pp, xp));

    clt_encode(x0, &mmap, 1, x, top_level);
    clt_encode(x1, &mmap, 1, x, top_level);
    clt_elem_sub(xp, &pp, x0, x1);
    ok &= expect("is_zero(x - x)", 1, clt_is_zero(&pp, xp));

    clt_encode(x0, &mmap, 1, zero, top_level);
    clt_encode(x1, &mmap, 1, x,    top_level);
    clt_elem_sub(xp, &pp, x0, x1);
    ok &= expect("is_zero(0 - x)", 0, clt_is_zero(&pp, xp));

    clt_encode(x0, &mmap, 1, one,  top_level);
    clt_encode(x1, &mmap, 1, zero, top_level);
    clt_elem_sub(xp, &pp, x0, x1);
    ok &= expect("is_zero(1 - 0)", 0, clt_is_zero(&pp, xp));

    clt_encode(x0, &mmap, 1, one,  top_level);
    clt_encode(x1, &mmap, 1, three, top_level);
    clt_elem_mul_ui(x0, &pp, x0, 3);
    clt_elem_sub(xp, &pp, x0, x1);
    ok &= expect("is_zero(3*[1] - [3])", 1, clt_is_zero(&pp, xp));

    int ix0 [nzs];
    int ix1 [nzs];
    for (ulong i = 0; i < nzs; i++) {
        if (i < nzs / 2) {
            ix0[i] = 1;
            ix1[i] = 0;
        } else {
            ix0[i] = 0;
            ix1[i] = 1;
        }
    }
    clt_encode(x0, &mmap, 1, x   , ix0);
    clt_encode(x1, &mmap, 1, zero, ix1);
    clt_elem_mul(xp, &pp, x0, x1);
    ok &= expect("is_zero(x * 0)", 1, clt_is_zero(&pp, xp));

    clt_encode(x0, &mmap, 1, x  , ix0);
    clt_encode(x1, &mmap, 1, one, ix1);
    clt_elem_mul(xp, &pp, x0, x1);
    ok &= expect("is_zero(x * 1)", 0, clt_is_zero(&pp, xp));

    clt_encode(x0, &mmap, 1, x, ix0);
    clt_encode(x1, &mmap, 1, x, ix1);
    clt_elem_mul(xp, &pp, x0, x1);
    ok &= expect("is_zero(x * x)", 0, clt_is_zero(&pp, xp));

    // zimmerman-like test

    mpz_t c;
    mpz_t in0 [2];
    mpz_t in1 [2];
    mpz_t cin [2];

    mpz_inits(c, in0[0], in0[1], in1[0], in1[1], cin[0], cin[1], NULL);

    mpz_urandomb_aes(in1[0], rng, lambda);
    mpz_mod(in1[0], in1[0], mmap.gs[0]);

    mpz_set_ui(in0[0], 0);
    mpz_set_ui(cin[0], 0);

    mpz_urandomb_aes(in0[1], rng, 16);
    mpz_urandomb_aes(in1[1], rng, 16);
    mpz_mul(cin[1], in0[1], in1[1]);

    clt_encode(x0, &mmap, 2, in0, ix0);
    clt_encode(x1, &mmap, 2, in1, ix1);
    clt_encode(c,  &mmap, 2, cin, top_level);

    clt_elem_mul(xp, &pp, x0, x1);
    clt_elem_sub(xp, &pp, xp, c);

    ok &= expect("[Z] is_zero(0 * x)", 1, clt_is_zero(&pp, xp));

    mpz_set_ui(in0[0], 1);
    mpz_set_ui(in1[0], 1);
    mpz_set_ui(cin[0], 0);

    mpz_urandomb_aes(in0[0], rng, lambda);
    mpz_mod(in0[0], in0[0], mmap.gs[0]);

    mpz_urandomb_aes(in1[0], rng, lambda);
    mpz_mod(in1[0], in1[0], mmap.gs[0]);

    mpz_urandomb_aes(in0[1], rng, 16);
    mpz_urandomb_aes(in1[1], rng, 16);
    mpz_mul(cin[1], in0[1], in1[1]);

    clt_encode(x0, &mmap, 2, in0, ix0);
    clt_encode(x1, &mmap, 2, in1, ix1);
    clt_encode(c,  &mmap, 2, cin, top_level);

    clt_elem_mul(xp, &pp, x0, x1);
    clt_elem_sub(xp, &pp, xp, c);

    ok &= expect("[Z] is_zero(x * y)", 0, clt_is_zero(&pp, xp));
    clt_state_clear(&mmap);
    clt_pp_clear(&pp);
    mpz_clears(c, x0, x1, xp, x[0], zero[0], one[0], in0[0], in0[1], in1[0], in1[1], cin[0], cin[1], NULL);
    return !ok;
}

static int
test_levels(ulong flags, ulong kappa, ulong lambda)
{
    int *pows, *top_level;
    clt_state s;
    clt_pp pp;
    aes_randstate_t rng;
    mpz_t zero, one, value, result, top_one, top_zero;
    int ok = 1;

    printf("Testing levels: λ = %lu, κ = %lu\n", lambda, kappa);

    aes_randinit(rng);
    mpz_init_set_ui(zero, 0);
    mpz_init_set_ui(one, 1);
    mpz_inits(value, result, top_one, top_zero, NULL);

    pows = calloc(kappa, sizeof(int));
    top_level = calloc(kappa, sizeof(int));
    for (ulong i = 0; i < kappa; ++i)
        top_level[i] = 1;

    clt_state_init(&s, kappa, lambda, kappa, top_level, 0, flags, rng);
    clt_pp_init(&pp, &s);

    clt_encode(top_one, &s, 1, &one, top_level);
    clt_encode(top_zero, &s, 0, &zero, top_level);

    for (ulong i = 0; i < kappa; ++i) {
        for (ulong j = 0; j < kappa; ++j) {
            if (j != i)
                pows[j] = 0;
            else
                pows[j] = 1;
        }
        clt_encode(value, &s, 1, &one, pows);
        if (i == 0)
            clt_elem_set(result, value);
        else {
            clt_elem_mul(result, &pp, result, value);
        }
    }

    ok &= expect("is_zero(1 * ... * 1)", 0, clt_is_zero(&pp, result));

    clt_elem_sub(result, &pp, result, top_one);

    ok &= expect("is_zero(1 * ... * 1 - 1)", 1, clt_is_zero(&pp, result));

    for (ulong i = 0; i < kappa; ++i) {
        for (ulong j = 0; j < kappa; ++j) {
            if (j != i)
                pows[j] = 0;
            else
                pows[j] = 1;
        }
        if (i == 0) {
            clt_encode(value, &s, 0, &one, pows);
            clt_elem_set(result, value);
        } else {
            clt_encode(value, &s, 1, &one, pows);
            clt_elem_mul(result, &pp, result, value);
        }
    }

    ok &= expect("is_zero(0 * 1 *  ... * 1)", 1, clt_is_zero(&pp, result));

    clt_elem_add(result, &pp, result, top_one);

    ok &= expect("is_zero(0 * 1 *  ... * 1 + 1)", 0, clt_is_zero(&pp, result));

    return !ok;
}

int main(void)
{
    ulong default_flags = CLT_FLAG_NONE | CLT_FLAG_VERBOSE;
    ulong flags;
    ulong kappa;
    ulong lambda = 40;
    ulong nzs = 10;

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

    return 0;
}

int expect(char * desc, int expected, int recieved)
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
