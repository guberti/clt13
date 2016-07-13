#ifndef _CRT_TREE_H_
#define _CRT_TREE_H_

#include "clt13.h"

typedef struct crt_tree {
    ulong n, n2;
    clt_elem_t mod;
    clt_elem_t crt_left;
    clt_elem_t crt_right;
    struct crt_tree *left;
    struct crt_tree *right;
} crt_tree;

int  crt_tree_init   (crt_tree *crt, clt_elem_t *ps, size_t nps);
void crt_tree_clear  (crt_tree *crt);
void crt_tree_do_crt (clt_elem_t rop, const crt_tree *crt, clt_elem_t *cs);
void crt_tree_read   (const char *fname, crt_tree *crt, size_t n);
void crt_tree_save   (const char *fname, crt_tree *crt, size_t n);
int  crt_tree_fread  (FILE *const fp, crt_tree *crt, size_t n);
int  crt_tree_fsave  (FILE *const fp, crt_tree *crt, size_t n);

#endif
