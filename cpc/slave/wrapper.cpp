
#include "kernel_slave.h"
// #include "wrapper.hpp"
#include <map>
using namespace std;
#ifdef __cplusplus
extern "C"
{
#endif
#define CLOCKRATE 1.45e9
#include <athread.h>
#include "callback.h"
    void slave_computation_kernel(SlavePara *para);
    void slave_computation_rest_kernel(RestPara *para);
    int ceiling(int num, int den)
    {
        return (num - 1) / den + 1;
    }
    static inline unsigned long rpcc()
    {
        unsigned long time;
        asm("rtc %0"
            : "=r"(time)
            :);
        return time;
    }

    static FunPtr _funs[4]={integration,spMV,compVisflux,calcLudsFcc};
    static int _funIndex=0;

#ifdef __cplusplus
}
#endif

#define BITS_PER_WORD 32
#define MASK 0x1f
#define SHIFT 5

#define BITMAP_SET(data, i)                    \
    do                                         \
    {                                          \
        data[i >> SHIFT] |= (1 << (i & MASK)); \
    } while (0)

#define BITMAP_TEST(data, i) (data[i >> SHIFT] & (1 << (i & MASK)))
#define CLR_ALL(data, len)                  \
    do                                      \
    {                                       \
        memset(data, 0, sizeof(int) * len); \
    } while (0)

#define SWAP_POINTER(a, b) \
    do                     \
    {                      \
        void *tmp = a;     \
        a = b;             \
        b = tmp;           \
    } while (0)

#define distribute_array_pointer(alldata, anum, allsize, dim, lpointer) \
    do                                                                  \
    {                                                                   \
        int _offset = 0;                                                \
        for (int _i = 0; _i < anum; ++_i)                               \
        {                                                               \
            lpointer[_i] = alldata + _offset;                           \
            _offset += (allsize)*dim[_i];                               \
        }                                                               \
    } while (0)

int getUnitSize(const DataSet *ds) //byte
{
    int num = getArrayNum(ds);
    int byteSize = 0;
    for (int i = 0; i < num; ++i)
    {
        byteSize += getArrayDim(ds, i) * sizeof(scalar);
    }
    return byteSize;
}

static map<FunPtr,FunPtr> _slaveToMain;

void slave_computation(DataSet *dataSet_edge, DataSet *dataSet_vertex,
                       label *row, label *col, FunPtr funPtr)
{
    if(_slaveToMain.find(funPtr)==_slaveToMain.end())
    {
        _slaveToMain[funPtr]=_funs[_funIndex];
        _funIndex=(_funIndex+1)%4;
    }
    unsigned long starttime, endtime, alltime = 0;
    const int blockNum = THREAD_SIZE * 2;
    Index indexTable[blockNum];
    //calculate index table
    int offset = 0;
    int end;
    int nedge = getArraySize(dataSet_edge);
    int nvertex = getArraySize(dataSet_vertex);
    int aveEdge = nedge / blockNum;
    for (int i = 0; i < blockNum - 1; ++i)
    {
        indexTable[i].offset = offset;
        end = offset + aveEdge;
        int pre = row[end - 1];
        while (row[end] == pre)
        {
            end++;
        }
        indexTable[i].size = end - offset;
        indexTable[i].nleft = indexTable[i].size;
        offset = end;
    }
    indexTable[blockNum - 1].offset = end;
    indexTable[blockNum - 1].size = nedge - end;
    indexTable[blockNum - 1].nleft = nedge - end;
    indexTable[blockNum - 1].colBegin = row[end];
    indexTable[blockNum - 1].colEnd = row[nedge - 1];
    for (int i = 0; i < blockNum - 1; ++i)
    {
        indexTable[i].colBegin = row[indexTable[i].offset];
        indexTable[i].colEnd = row[indexTable[i + 1].offset + indexTable[i + 1].size - 1];
    }
    //**********************************************/

    int **localEdge = (int **)malloc(sizeof(int *) * blockNum);
    for (int i = 0; i < blockNum; ++i)
    {
        localEdge[i] = (int *)malloc(sizeof(int) * (aveEdge));
    }
    int localSize[blockNum];

    // for(int i=0;i<blockNum;++i)
    // {
    //     printf("block:%d offset:%d size:%d colbeg:%d colend:%d \n",i,indexTable[i].offset,indexTable[i].size,indexTable[i].colBegin,indexTable[i].colEnd);
    // }

    //**********************************************/
    //calculate cell and face size
    int cellSize = getUnitSize(dataSet_vertex);
    int faceSize = getUnitSize(dataSet_edge);
    int maxRowPerCycle = (MAX_STORAGE_SIZE - (INNER_SIZE + 1) * cellSize) / (sizeof(label) * 2 + cellSize + faceSize);
    int **rows = (int **)malloc(sizeof(int *) * blockNum);
    int **cols = (int **)malloc(sizeof(int *) * blockNum);
    EdgeBuffer *ebuf = (EdgeBuffer *)malloc(sizeof(EdgeBuffer) * blockNum);
    for (int i = 0; i < blockNum; ++i)
    {
        ebuf[i] = (scalar **)malloc(sizeof(scalar *) * dataSet_edge->fArrayNum);
        for (int j = 0; j < dataSet_edge->fArrayNum; ++j)
        {
            ebuf[i][j] = (scalar *)malloc(aveEdge * sizeof(scalar) * dataSet_edge->fArrayDims[j]);
        }
        rows[i] = (int *)malloc(sizeof(int) * aveEdge);
        cols[i] = (int *)malloc(sizeof(int) * aveEdge);
    }
    int maxEdgeBufSize = EDGE_BUFFER_SIZE / (faceSize + sizeof(int) * 3); //index row col

    SlavePara para = {
        row,
        col,

        dataSet_edge,
        dataSet_vertex,

        indexTable,
        cellSize,
        faceSize,
        maxRowPerCycle,
        localEdge,
        localSize,
        maxEdgeBufSize,
        ebuf,
        rows,
        cols,
        funPtr};

    __real_athread_spawn((void *)slave_computation_kernel, &para);
    athread_join();

    // const int bitmapLen = 1 + nvertex / BITS_PER_WORD;
    // int *edgeBitmap = (int *)malloc(sizeof(int) * bitmapLen);
    // CLR_ALL(edgeBitmap, bitmapLen);
    int curSize = 0;
    //int *curEdge = (int *)malloc(sizeof(int) * nedge);
    //int *nextEdge = (int *)malloc(sizeof(int) * nedge);
    //int *couldComputeEdge = (int *)malloc(sizeof(int) * nedge);

    //label *restRow = (label *)malloc(sizeof(label) * nedge);
    //label *restCol = (label *)malloc(sizeof(label) * nedge);
    // for (int i = 0; i < blockNum; ++i)
    // {
    //     for (int j = 0; j < localSize[i]; ++j)
    //     {
    //         int edgeIndex = localEdge[i][j];
    //         restRow[curSize] = row[edgeIndex];
    //         restCol[curSize] = col[edgeIndex];
    //         curEdge[curSize++] = edgeIndex;
    //     }
    // }

    int ncomputeSize = 0;
    int nextSize = 0;

    // int allRestEdgeSize = faceSize * curSize;
    // scalar *allRestEdgeData = (scalar *)malloc(allRestEdgeSize);
    // scalar *edgeData[20];
    // distribute_array_pointer(allRestEdgeData, dataSet_edge->fArrayNum, allRestEdgeSize, dataSet_edge->fArrayDims, edgeData);
    // //copy data

    // for (int i = 0; i < dataSet_edge->fArrayNum; ++i)
    // {
    //     if (dataSet_edge->fArrayInOut[i] == COPYIN || dataSet_edge->fArrayInOut[i] == COPYINOUT)
    //     {
    //         scalar *data = edgeData[i];
    //         scalar *redgeData = dataSet_edge->floatArrays[i];
    //         int dim = dataSet_edge->fArrayDims[i];
    //         int copySize = dim * sizeof(scalar);
    //         for (int j = 0; j < curSize; ++j)
    //         {
    //             memcpy(data, redgeData + curEdge[j] * dim, copySize);
    //             data += dim;
    //         }
    //     }
    // }

    for (int i = 0; i < blockNum; ++i)
    {

        DataSet restEdgeSet;
        restEdgeSet.fArrayDims = dataSet_edge->fArrayDims;
        restEdgeSet.fArrayInOut = dataSet_edge->fArrayInOut;
        restEdgeSet.fArrayNum = dataSet_edge->fArrayNum;
        restEdgeSet.fArraySize = localSize[i];
        restEdgeSet.floatArrays = ebuf[i];

        _slaveToMain[funPtr](&restEdgeSet, dataSet_vertex, rows[i], cols[i]);
        //copy out

        for (int j = 0; j < dataSet_edge->fArrayNum; ++j)
        {
            if (dataSet_edge->fArrayInOut[j] == COPYOUT || dataSet_edge->fArrayInOut[j] == COPYINOUT)
            {
                scalar *data = ebuf[i][j];
                scalar *redgeData = dataSet_edge->floatArrays[j];
                int dim = dataSet_edge->fArrayDims[j];
                int copySize = dim * sizeof(scalar);
                for (int u = 0; u < localSize[i]; ++u)
                {
                    memcpy(redgeData + localEdge[i][u] * dim, data, copySize);
                    data += dim;
                }
            }
        }
    }

    // int ncycle = 0;
    // while (curSize)
    // {
    //     ncycle++;
    //     for (int i = 0; i < curSize; ++i)
    //     {
    //         int edgeIndex = curEdge[i];
    //         if (BITMAP_TEST(edgeBitmap, row[edgeIndex]) || BITMAP_TEST(edgeBitmap, col[edgeIndex]))
    //         {
    //             nextEdge[nextSize++] = edgeIndex;
    //         }
    //         else
    //         {
    //             couldComputeEdge[ncomputeSize++] = edgeIndex;
    //         }
    //         BITMAP_SET(edgeBitmap, row[edgeIndex]);
    //         BITMAP_SET(edgeBitmap, col[edgeIndex]);
    //     }
    //     RestPara rpara = {
    //         row,
    //         col,

    //         dataSet_edge,
    //         dataSet_vertex,

    //         cellSize,
    //         faceSize,
    //         maxRowPerCycle,
    //         couldComputeEdge,
    //         ncomputeSize,

    //         funPtr};
    //     starttime = rpcc();
    //     __real_athread_spawn((void *)slave_computation_rest_kernel, &rpara);
    //     CLR_ALL(edgeBitmap, bitmapLen);
    //     athread_join();
    //     endtime = rpcc();
    //     alltime += (endtime - starttime);
    //     int *tmp;
    //     tmp = curEdge;
    //     curEdge = nextEdge;
    //     nextEdge = tmp;
    //     curSize = nextSize;
    //     nextSize = 0;
    //     ncomputeSize = 0;
    // }
    for (int p = 0; p < blockNum; ++p)
    {
        free(localEdge[p]);
        free(rows[p]);
        free(cols[p]);
        for (int j = 0; j < dataSet_edge->fArrayNum; ++j)
        {
            free(ebuf[p][j]);
        }
        free(ebuf[p]);
    }
    free(localEdge);
    free(rows);
    free(cols);
    free(ebuf);
    //free(curEdge);
    //free(nextEdge);
    //free(couldComputeEdge);
    //free(edgeBitmap);
    //free(allRestEdgeData);
    //free(restCol);
    //free(restRow);
}
