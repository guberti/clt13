#pragma once

#include <gmp.h>
#include <stdio.h>

#ifndef LOCAL
#define LOCAL __attribute__ ((visibility ("hidden")))
#endif

LOCAL int mpz_fread(mpz_t x, FILE *fp);
LOCAL int mpz_fwrite(mpz_t x, FILE *fp);
LOCAL mpz_t * mpz_vector_new(size_t n);
LOCAL void mpz_vector_free(mpz_t *v, size_t n);
LOCAL int mpz_vector_fread(mpz_t *m, size_t len, FILE *fp);
LOCAL int mpz_vector_fwrite(mpz_t *m, size_t len, FILE *fp);

LOCAL double current_time(void);
LOCAL void print_progress(size_t cur, size_t total);

LOCAL int size_t_fread(FILE *fp, size_t *x);
LOCAL int size_t_fwrite(FILE *fp, size_t x);
