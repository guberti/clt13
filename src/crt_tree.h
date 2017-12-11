#ifndef _CRT_TREE_H_
#define _CRT_TREE_H_

#include "clt13.h"

typedef struct crt_tree {
    size_t n;
    clt_elem_t mod;
    clt_elem_t crt_left;
    clt_elem_t crt_right;
    struct crt_tree *left;
    struct crt_tree *right;
} crt_tree;

crt_tree * crt_tree_new(clt_elem_t *const ps, size_t n);
void crt_tree_free(crt_tree *crt);

void crt_tree_do_crt(clt_elem_t rop, const crt_tree *crt, clt_elem_t *cs);
crt_tree * crt_tree_fread(FILE *const fp, size_t n);
void crt_tree_write(const char *fname, const crt_tree *const crt, size_t n);
int crt_tree_fwrite(FILE *const fp, const crt_tree *const crt, size_t n);

#endif
