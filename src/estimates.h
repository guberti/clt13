#pragma once

#include "clt13.h"

#ifndef LOCAL
#define LOCAL __attribute__ ((visibility ("hidden")))
#endif

LOCAL ulong estimate_n(ulong lambda, ulong eta, ulong flags);
