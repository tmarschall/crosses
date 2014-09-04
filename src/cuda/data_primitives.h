/*  data_primitives.h
 *
 *  exclusive_scan (prefix sum) :
 *
 *
 *
 */

#ifndef DATA_PRIMITIVES_H
#define DATA_PRIMITIVES_H

// Arrays pnIn and pnOut are device arrays
void exclusive_scan(int *pnIn, int *pnOut, int nSize);

void ordered_array(int *pnArray, int nSize, int gridSize = 0, int blockSize = 0);

#endif
