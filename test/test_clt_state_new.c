#include <clt13.h>

static int lambda = 60;
static int kappa = 21;
static int nzs = 17;
static size_t flags = CLT_FLAG_VERBOSE | CLT_FLAG_OPT_CRT_TREE | CLT_FLAG_OPT_COMPOSITE_PS;

int
main(void)
{
    clt_state *s;
    int pows[nzs];
    aes_randstate_t rng;
    int ret = 1;

    aes_randinit(rng);
    for (int i = 0; i < nzs; ++i)
        pows[i] = 1;

    s = clt_state_new(kappa, lambda, nzs, pows, 0, 0, flags, rng);
    if (s == NULL)
        goto cleanup;

    clt_state_delete(s);
    ret = 0;
cleanup:
    aes_randclear(rng);
    return ret;
}
