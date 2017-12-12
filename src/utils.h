#pragma once

#include <gmp.h>
#include <stdio.h>

int mpz_fread(mpz_t x, FILE *fp);
int mpz_fwrite(mpz_t x, FILE *fp);
mpz_t * mpz_vector_new(size_t n);
void mpz_vector_free(mpz_t *v, size_t n);
int mpz_vector_fread(mpz_t *m, size_t len, FILE *fp);
int mpz_vector_fwrite(mpz_t *m, size_t len, FILE *fp);

double current_time(void);
void print_progress(size_t cur, size_t total);

int size_t_fread(FILE *fp, size_t *x);
int size_t_fwrite(FILE *fp, size_t x);
