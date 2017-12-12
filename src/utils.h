#pragma once

#include <gmp.h>
#include <stdio.h>

int mpz_fread(mpz_t x, FILE *fp);
int mpz_fwrite(mpz_t x, FILE *fp);
mpz_t * mpz_vector_new(size_t n);
void mpz_vector_free(mpz_t *v, size_t n);
int mpz_vector_fread(mpz_t *m, size_t len, FILE *fp);
int mpz_vector_fwrite(mpz_t *m, size_t len, FILE *fp);
