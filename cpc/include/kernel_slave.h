#ifndef KERNEL_SLAVE_H
#define KERNEL_SLAVE_H

#include "macro.h"
#include "dataSet.h"
#include "callback.h"
#include <stdio.h>

#define THREAD_SIZE 64
#define MAX_STORAGE_SIZE 50 * 1024
#define INNER_SIZE 16
#define EXTERNAL_SIZE 1024 //1kb
#define REST_INNER_SIZE 8

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct
    {
        int offset;
        int size;
        int nleft;
        label colBegin;
        label colEnd;
    } Index;

    typedef struct
    {
        label *row;
        label *col;

        DataSet *dataSet_edge;
        DataSet *dataSet_vertex;

        Index *indexTable;
        int cellSize;
        int faceSize;
        int maxRowPerCycle;
        int **localEdge;
        int *localSize;

        FunPtr funPtr;

    } SlavePara;

    typedef struct
    {
        label *row;
        label *col;

        DataSet *dataSet_edge;
        DataSet *dataSet_vertex;

        int cellSize;
        int faceSize;
        int maxRowPerCycle;
        int* couldComputed;
        int computeSize;

        FunPtr funPtr;

    } RestPara;

#ifdef __cplusplus
}
#endif

#endif
