CLT13
=====
This is an Implementation of the [CLT13 multilinear map](https://eprint.iacr.org/2013/183) specialized for
obfuscation purposes. 

General Intuition
-----------------
Our CLT13 implementation provides the following functionality. One party can
create a secret key that allows them to create *encodings* of messages and to
create a public key. The public key allows another party to add and multiply
encodings.

The public key also provides *zero-testing* of encodings who are at the
top-level index. This "top-level index" means something specific: every encoded
message has some *index*.  The encoding that results from multiplying two
messages will be at the union of the two multiplicand's indices. Two encodings
can be added only if they share the same index. Zero-testing will fail unless
an encoding is at some predefined top-level index.

Assymetric Modification
-----------------------
This implementation has been modified to support *asymmetric* index sets. That
is, where the original design operates over *levels* 1 through kappa, our
design uses distinct *index sets*. In the original CLT13, there is a single
*z* whose powers represent the encoding levels. In our modified version, we
produce multiple *z*s, one for each distinct index.

The top-level index can be any combination of powers of indices. This top-level
index must be given as the `pows` argument to `clt_state_init`. It is used to
create the zero-testing parameter. Each slot `i` in the `pows` array represents
the power of that `z_i` in the top-level index set.

Usage Overview
==============
Use `clt_state_init` to create a secret key. Parameters here: kappa is the
maximum multiplicative degree allowed (used to determine the size of the
noise), lambda is the security parameter, nzs is the number of distinct
indices, pows is the top-level index. Using a `clt_state`, can create
a `clt_pp` public key using `clt_pp_init`, or create encodings using
`clt_encode`. In addition, there are are number of optimizations and options
you can set using flags. These are documented in the code.

Example Usage
-------------
```
    unsigned long kappa = 2;
    unsigned long lambda = 40; 

    // initialize the rng
    aes_randstate_t rng;
    aes_randinit(rng);

    // create the top-level index
    int top_level [nzs];
    for (ulong i = 0; i < nzs; i++) 
        top_level[i] = 1;

    // initialize the secret key
    clt_state mmap;
    clt_state_init(&mmap, kappa, lambda, nzs, pows, NULL, rng);

    // create the public key from the secret key
    clt_pp pp;
    clt_pp_init(&pp, &mmap);

    // initialize the plaintexts
    mpz_t x [1];
    mpz_init_set_ui(x[0], 1);
    mpz_t y [1];
    mpz_init_set_ui(y[0], 1);

    // create encodings
    clt_elem x0, x1, xp;
    clt_elem_init(x0);
    clt_elem_init(x1)
    clt_elem_init(xp);
    clt_encode(x0, &mmap, 1, x, top_level, rng);
    clt_encode(x1, &mmap, 1, y, top_level, rng);

    // add the encodings
    clt_elem_add(xp, &pp, x0, x1);
    int ok = expect("is_zero(1 + 1)", 0, clt_is_zero(&pp, xp));
```

See `test/test_clt.c` for more examples.

License
=======
Licenced under GPLv2.

Copyright 2016 Brent Carmer & Alex Malozemoff.
