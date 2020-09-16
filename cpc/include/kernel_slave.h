#ifndef KERNEL_SLAVE_H
#define KERNEL_SLAVE_H

#include "macro.h"
#include "dataSet.h"
#include "callback.h"
#include <stdio.h>

#define THREAD_SIZE 64
#define MAX_STORAGE_SIZE 48 * 1024  //50kb
#define EDGE_BUFFER_SIZE 10*1024 //10kb for edge buffer
#define INNER_SIZE 48
#define REST_INNER_SIZE 8

#ifdef __cplusplus
extern "C"
{
#endif
    typedef scalar** EdgeBuffer;
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
        int maxEdgeBufSize;
        EdgeBuffer* edgeBuf;
        int** rows;
        int** cols;
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
