#include "macro.h"
#include "slave.h"
// #include "slaveUtils.h"
#include "kernel_slave.h"
#include <stdio.h>
#include <string.h> //memset
//#include <simd.h>
#define CLOCKRATE 1.45e9

static inline unsigned long rpcc()
{
    unsigned long time;
    asm("rcsr %0,4"
        : "=r"(time)
        :);
    return time;
}

static inline unsigned long hash_long(unsigned long val, unsigned int bits) //windows
{
    unsigned long hash = val * 0x9e370001UL;
    return hash >> (32 - bits);
}

#define HASH_INDEX(_i) (((unsigned)(_i)) * 0x9e370001U >> (32 - 4)) //'unsigned long' for 64 bit windows,'unsigned int' for 64 linux

__thread_local volatile unsigned long long get_reply, put_reply;
__thread_local volatile unsigned long long my_id;
__thread_local volatile unsigned long long pr;
__thread_local volatile unsigned long long gr;

#define PRINT_DEBUG                        \
    do                                     \
    {                                      \
        if (my_id == 0)                    \
            printf("here %d\n", __LINE__); \
    } while (0)

void wait_reply(volatile unsigned long long *reply, int m)
{
    while (*reply < m)
    {
    };
}

#define get_wait_reply(src, dst, size)                             \
    do                                                             \
    {                                                              \
        athread_get(PE_MODE, src, dst, size, &get_reply, 0, 0, 0); \
        wait_reply(&get_reply, ++gr);                              \
    } while (0)

#define get_without_reply(src, dst, size)                          \
    do                                                             \
    {                                                              \
        athread_get(PE_MODE, src, dst, size, &get_reply, 0, 0, 0); \
        gr++;                                                      \
    } while (0)

#define put_wait_reply(src, dst, size)                             \
    do                                                             \
    {                                                              \
        athread_put(PE_MODE, src, dst, size, &put_reply, 0, 0, 0); \
        wait_reply(&put_reply, ++pr);                              \
    } while (0)

#define put_without_reply(src, dst, size)                       \
    do                                                          \
    {                                                           \
        athread_put(PE_MODE, src, dst, size, &put_reply, 0, 0); \
        pr++;                                                   \
    } while (0)
#define wait_get_reply              \
    do                              \
    {                               \
        wait_reply(&get_reply, gr); \
    } while (0)
#define wait_put_reply              \
    do                              \
    {                               \
        wait_reply(&put_reply, pr); \
    } while (0)

#define get_data_without_reply(rfloat, lfloat, anum, size, dim, _roffset, _loffset, inout)              \
    do                                                                                                  \
    {                                                                                                   \
        for (int _i = 0; _i < anum; ++_i)                                                               \
        {                                                                                               \
            if (inout[_i] == COPYINOUT || inout[_i] == COPYIN)                                          \
            {                                                                                           \
                get_without_reply(rfloat[_i] + dim[_i] * (_roffset), lfloat[_i] + dim[_i] * (_loffset), \
                                  sizeof(scalar) * dim[_i] * (size));                                   \
            }                                                                                           \
        }                                                                                               \
    } while (0);

#define put_data_without_reply(rfloat, lfloat, anum, size, dim, _roffset, _loffset, inout)              \
    do                                                                                                  \
    {                                                                                                   \
        for (int _i = 0; _i < anum; ++_i)                                                               \
        {                                                                                               \
            if (inout[_i] == COPYINOUT || inout[_i] == COPYOUT)                                         \
            {                                                                                           \
                put_without_reply(lfloat[_i] + dim[_i] * (_loffset), rfloat[_i] + dim[_i] * (_roffset), \
                                  sizeof(scalar) * dim[_i] * (size));                                   \
            }                                                                                           \
        }                                                                                               \
    } while (0);

#define put_edge_data_without_reply(rfloat, lfloat, anum, size, dim, _roffset, _loffset, inout)         \
    do                                                                                                  \
    {                                                                                                   \
        for (int _i = 0; _i < anum; ++_i)                                                               \
        {                                                                                               \
            if (inout[_i] == COPYINOUT || inout[_i] == COPYIN)                                          \
            {                                                                                           \
                put_without_reply(lfloat[_i] + dim[_i] * (_loffset), rfloat[_i] + dim[_i] * (_roffset), \
                                  sizeof(scalar) * dim[_i] * (size));                                   \
            }                                                                                           \
        }                                                                                               \
    } while (0);

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

#define pointer_shift(lpointer, anum, shift, dim) \
    do                                            \
    {                                             \
        for (int _i = 0; _i < anum; ++_i)         \
        {                                         \
            lpointer[_i] += (shift)*dim[_i];      \
        }                                         \
    } while (0)
#define copy_data(dpointer, spointer, soffset, doffset, size, anum, dim, inout)                                                \
    do                                                                                                                         \
    {                                                                                                                          \
        for (int _i = 0; _i < anum; ++_i)                                                                                      \
        {                                                                                                                      \
            if (inout[_i] == COPYIN || inout[_i] == COPYINOUT)                                                                 \
                memcpy(dpointer[_i] + (doffset)*dim[_i], spointer[_i] + (soffset)*dim[_i], sizeof(scalar) * dim[_i] * (size)); \
        }                                                                                                                      \
    } while (0)

int ceiling(int num, int den)
{
    return (num - 1) / den + 1;
}

void computation_kernel(SlavePara *para)
{
    gr = 0;
    pr = 0;
    get_reply = 0;
    put_reply = 0;

    unsigned long start, end, alltime = 0;

    my_id = athread_get_id(-1);
    if (my_id >= THREAD_SIZE)
    {
        athread_syn(ARRAY_SCOPE, 0xffff);
        return;
    }

    scalar *rEdgeFloatArrays[20];
    scalar *rVertexFloatArrays[20];
    label rEdgeSetDim[20];
    label rVertexSetDim[20];
    InOut rEdgeSetInOut[20];
    InOut rVertexSetInOut[20];
    scalar *lEdgeFloatArrays[20];
    scalar *lVertexFloatArrays[20];
    scalar *lEdgeBufferFloatArrays[20];
    scalar *rEdgeBufferFloatArrays[20];
    int *rlocalEdge;
    int nexternal = 0;
    int exEdgeOffset = 0;

    SlavePara sp;
    get_wait_reply(para, &sp, sizeof(SlavePara));
    label *row = ldm_malloc(sizeof(label) * sp.maxRowPerCycle);
    label *col = ldm_malloc(sizeof(label) * sp.maxRowPerCycle);

    DataSet rEdgeSet, rVertexSet;
    get_without_reply(sp.dataSet_edge, &rEdgeSet, sizeof(DataSet));
    get_without_reply(sp.dataSet_vertex, &rVertexSet, sizeof(DataSet));
    wait_get_reply;

    get_without_reply(rEdgeSet.floatArrays, rEdgeFloatArrays, sizeof(scalar *) * rEdgeSet.fArrayNum);
    get_without_reply(rVertexSet.floatArrays, rVertexFloatArrays, sizeof(scalar *) * rVertexSet.fArrayNum);

    get_without_reply(rEdgeSet.fArrayDims, rEdgeSetDim, sizeof(label) * rEdgeSet.fArrayNum);
    get_without_reply(rVertexSet.fArrayDims, rVertexSetDim, sizeof(label) * rVertexSet.fArrayNum);

    get_without_reply(rEdgeSet.fArrayInOut, rEdgeSetInOut, sizeof(InOut) * rEdgeSet.fArrayNum);
    get_without_reply(rVertexSet.fArrayInOut, rVertexSetInOut, sizeof(InOut) * rVertexSet.fArrayNum);
    wait_get_reply;
    int allVertexDataSize = (sp.cellSize / sizeof(scalar)) * (sp.maxRowPerCycle + INNER_SIZE + 1) * sizeof(scalar);
    int allEdgeDataSize = (sp.faceSize / sizeof(scalar)) * (sp.maxRowPerCycle) * sizeof(scalar);
    scalar *allVertexData = ldm_malloc(allVertexDataSize);
    scalar *allEdgeData = ldm_malloc(allEdgeDataSize);
    distribute_array_pointer(allVertexData, rVertexSet.fArrayNum, sp.maxRowPerCycle + INNER_SIZE + 1, rVertexSetDim, lVertexFloatArrays);
    distribute_array_pointer(allEdgeData, rEdgeSet.fArrayNum, sp.maxRowPerCycle, rEdgeSetDim, lEdgeFloatArrays);
    DataSet lEdgeSet, lVertexSet;
    lEdgeSet.fArrayDims = rEdgeSetDim;
    lEdgeSet.fArrayInOut = rEdgeSetInOut;
    lEdgeSet.fArrayNum = rEdgeSet.fArrayNum;
    lEdgeSet.floatArrays = lEdgeFloatArrays;

    lVertexSet.fArrayDims = rVertexSetDim;
    lVertexSet.fArrayInOut = rVertexSetInOut;
    lVertexSet.fArrayNum = rVertexSet.fArrayNum;
    lVertexSet.floatArrays = lVertexFloatArrays;

    const int maxExternal = sp.maxEdgeBufSize;
    int *externalEdge = ldm_malloc(sizeof(int) * maxExternal);
    int allEdgeBufSize = maxExternal * sp.faceSize;
    scalar *allEdgeBuf = ldm_malloc(allEdgeBufSize);
    int *lrowBuf = ldm_malloc(sizeof(int) * maxExternal);
    int *lcolBuf = ldm_malloc(sizeof(int) * maxExternal);
    distribute_array_pointer(allEdgeBuf, lEdgeSet.fArrayNum, maxExternal, lEdgeSet.fArrayDims, lEdgeBufferFloatArrays);
    for (int ncycle = 0; ncycle < 2; ++ncycle)
    {
        Index itable;
        get_wait_reply(sp.indexTable + my_id * 2 + ncycle, &itable, sizeof(Index));
        int exoffset = itable.offset;
        int roffset = itable.offset;
        label exBegin = itable.colBegin, exEnd = itable.colEnd;
        get_wait_reply(sp.localEdge + my_id * 2 + ncycle, &rlocalEdge, sizeof(int *));
        EdgeBuffer redgeBuf;
        get_wait_reply(sp.edgeBuf + my_id * 2 + ncycle, &redgeBuf, sizeof(EdgeBuffer));
        get_wait_reply(redgeBuf, rEdgeBufferFloatArrays, sizeof(scalar *) * lEdgeSet.fArrayNum);
        int *rrowBuf;
        int *rcolBuf;
        get_without_reply(sp.rows + my_id * 2 + ncycle, &rrowBuf, sizeof(int *));
        get_wait_reply(sp.cols + my_id * 2 + ncycle, &rcolBuf, sizeof(int *));
        while (1)
        {
            int nactualRowSize = itable.nleft >= sp.maxRowPerCycle ? sp.maxRowPerCycle : itable.nleft;
            itable.nleft -= nactualRowSize;
            if (nactualRowSize == 0)
                break;
            get_without_reply(sp.row + roffset, row, sizeof(label) * nactualRowSize);
            get_without_reply(sp.col + roffset, col, sizeof(label) * nactualRowSize);
            wait_get_reply;
            int cellBegin = row[0], cellEnd = row[nactualRowSize - 1];
            int nactualVertexSize = cellEnd - cellBegin + 1;
            lVertexSet.fArraySize = nactualVertexSize;

#define get_edge_data_without_reply(_roffset, _loffset, _size) get_data_without_reply(rEdgeFloatArrays, lEdgeFloatArrays, lEdgeSet.fArrayNum, \
                                                                                      _size, lEdgeSet.fArrayDims, _roffset, _loffset, lEdgeSet.fArrayInOut)
#define get_vertex_data_without_reply(_roffset, _loffset, _size) get_data_without_reply(rVertexFloatArrays, lVertexFloatArrays, lVertexSet.fArrayNum, _size, \
                                                                                        lVertexSet.fArrayDims, _roffset, _loffset, lVertexSet.fArrayInOut)
            get_vertex_data_without_reply(cellBegin, 0, nactualVertexSize);
            get_edge_data_without_reply(roffset, 0, nactualRowSize);
            wait_get_reply;
            label map[INNER_SIZE];
            int remain = nactualRowSize % INNER_SIZE;

            for (int i = 0; i < nactualRowSize; ++i) //todo simd
            {
                row[i] -= cellBegin;
            }

            if (nactualRowSize < INNER_SIZE)
            {
                nactualRowSize = 0;
            }
            else
            {
                nactualRowSize -= remain;
            }
            //!simd is trival
            // intv8 rowv, cellBeginv;
            // cellBeginv = simd_set_intv8(cellBegin, cellBegin, cellBegin, cellBegin,
            //                             cellBegin, cellBegin, cellBegin, cellBegin);
            // for (int i = 0; i < nactualRowSize; i += 8)
            // {
            //     simd_load(rowv, row + i);
            //     rowv = rowv - cellBeginv;
            //     simd_store(rowv, row + i);
            // }
            // for (int i = nactualRowSize; i < (nactualRowSize + remain); ++i)
            // {
            //     row[i] -= cellBegin;
            // }
            lEdgeSet.fArraySize = INNER_SIZE;
            int nactualInnerSize = 0;
            int continualFlag = 1;
            for (int i = 0; i < nactualRowSize; i += INNER_SIZE)
            {
                continualFlag = 1;
                for (int j = 0; j < INNER_SIZE; ++j)
                {
                    if (col[j] < exBegin || col[j] > exEnd)
                    {
                        wait_put_reply;
                        externalEdge[nexternal] = i + j + roffset; //edge index
                        continualFlag = 0;
                        copy_data(lEdgeBufferFloatArrays, lEdgeFloatArrays, j, nexternal, 1, lEdgeSet.fArrayNum,
                                  lEdgeSet.fArrayDims, lEdgeSet.fArrayInOut);
                        lrowBuf[nexternal] = row[j] + cellBegin;
                        lcolBuf[nexternal] = col[j];
                        nexternal++;
                        //dummy delete
                        row[j] = nactualVertexSize;
                        col[j] = nactualVertexSize;
                        if (nexternal == maxExternal)
                        {
                            put_without_reply(externalEdge, rlocalEdge + exEdgeOffset, sizeof(int) * nexternal);
                            put_edge_data_without_reply(rEdgeBufferFloatArrays, lEdgeBufferFloatArrays, lEdgeSet.fArrayNum, maxExternal,
                                                        lEdgeSet.fArrayDims, exEdgeOffset, 0, lEdgeSet.fArrayInOut);
                            put_without_reply(lrowBuf, rrowBuf + exEdgeOffset, sizeof(int) * nexternal);
                            put_without_reply(lcolBuf, rcolBuf + exEdgeOffset, sizeof(int) * nexternal);
                            exEdgeOffset += nexternal;
                            nexternal = 0;
                        }
                    }
                    else if (col[j] < cellBegin || col[j] > cellEnd)
                    {
                        int duplicateFlag = 0;
                        for (int k = 0; k < nactualInnerSize; ++k)
                        {
                            if (map[k] == col[j])
                            {
                                col[j] = k + nactualVertexSize + 1;
                                duplicateFlag = 1;
                                break;
                            }
                        }
                        if (duplicateFlag == 0)
                        {
                            get_vertex_data_without_reply(col[j], nactualVertexSize + nactualInnerSize + 1, 1);
                            map[nactualInnerSize] = col[j];
                            col[j] = nactualInnerSize + nactualVertexSize + 1;
                            nactualInnerSize++;
                        }
                    }
                    else
                    {
                        col[j] -= cellBegin;
                    }
                }

                wait_get_reply;

                //start = rpcc();
                sp.funPtr(&lEdgeSet, &lVertexSet, row, col); //todo double bufferring
                //end = rpcc();
                //alltime += (end - start);
                for (int j = 0; j < nactualInnerSize; ++j)
                {
                    put_data_without_reply(rVertexFloatArrays, lVertexFloatArrays, lVertexSet.fArrayNum, 1, lVertexSet.fArrayDims, map[j], nactualVertexSize + 1 + j, lVertexSet.fArrayInOut);
                }
                int putVertexPr = pr;
                if (continualFlag)
                {
                    put_data_without_reply(rEdgeFloatArrays, lEdgeFloatArrays, lEdgeSet.fArrayNum, INNER_SIZE, lEdgeSet.fArrayDims,
                                           roffset + i, 0, lEdgeSet.fArrayInOut);
                }
                else
                {
                    for (int j = 0; j < INNER_SIZE; ++j)
                    {
                        if (row[j] != nactualVertexSize)
                        {
                            put_data_without_reply(rEdgeFloatArrays, lEdgeFloatArrays, lEdgeSet.fArrayNum, 1, lEdgeSet.fArrayDims,
                                                   roffset + i + j, j, lEdgeSet.fArrayInOut);
                        }
                    }
                }

                wait_reply(&put_reply, putVertexPr);

                pointer_shift(lEdgeSet.floatArrays, lEdgeSet.fArrayNum, INNER_SIZE, lEdgeSet.fArrayDims);
                row += INNER_SIZE;
                col += INNER_SIZE;
                nactualInnerSize = 0;
            }
            wait_put_reply;
            //*************************/ //remain
            if (remain != 0)
            {
                continualFlag = 1;
                for (int i = 0; i < remain; ++i)
                {
                    if (col[i] < exBegin || col[i] > exEnd)
                    {
                        wait_put_reply;
                        externalEdge[nexternal] = i + roffset + nactualRowSize; //edge index
                        continualFlag = 0;
                        copy_data(lEdgeBufferFloatArrays, lEdgeFloatArrays, i, nexternal, 1, lEdgeSet.fArrayNum,
                                  lEdgeSet.fArrayDims, lEdgeSet.fArrayInOut);
                        lrowBuf[nexternal] = row[i] + cellBegin;
                        lcolBuf[nexternal] = col[i];
                        nexternal++;
                        //dummy delete
                        row[i] = nactualVertexSize;
                        col[i] = nactualVertexSize;
                        if (nexternal == maxExternal)
                        {
                            put_without_reply(externalEdge, rlocalEdge + exEdgeOffset, sizeof(int) * nexternal);
                            put_edge_data_without_reply(rEdgeBufferFloatArrays, lEdgeBufferFloatArrays, lEdgeSet.fArrayNum, maxExternal,
                                                        lEdgeSet.fArrayDims, exEdgeOffset, 0, lEdgeSet.fArrayInOut);
                            put_without_reply(lrowBuf, rrowBuf + exEdgeOffset, sizeof(int) * nexternal);
                            put_without_reply(lcolBuf, rcolBuf + exEdgeOffset, sizeof(int) * nexternal);
                            exEdgeOffset += nexternal;
                            nexternal = 0;
                        }
                    }
                    else if (col[i] < cellBegin || col[i] > cellEnd)
                    {
                        int duplicateFlag = 0;
                        for (int k = 0; k < nactualInnerSize; ++k)
                        {
                            if (map[k] == col[i])
                            {
                                col[i] = k + nactualVertexSize + 1;
                                duplicateFlag = 1;
                                break;
                            }
                        }
                        if (duplicateFlag == 0)
                        {
                            get_vertex_data_without_reply(col[i], nactualVertexSize + nactualInnerSize + 1, 1);
                            map[nactualInnerSize] = col[i];
                            col[i] = nactualInnerSize + nactualVertexSize + 1;
                            nactualInnerSize++;
                        }
                    }
                    else
                    {
                        col[i] -= cellBegin;
                    }
                }
                wait_get_reply;
                lEdgeSet.fArraySize = remain;
                sp.funPtr(&lEdgeSet, &lVertexSet, row, col);
                if (continualFlag)
                {
                    put_data_without_reply(rEdgeFloatArrays, lEdgeFloatArrays, lEdgeSet.fArrayNum, remain, lEdgeSet.fArrayDims,
                                           roffset + nactualRowSize, 0, lEdgeSet.fArrayInOut);
                }
                else
                {
                    for (int j = 0; j < remain; ++j)
                    {
                        if (row[j] != nactualVertexSize)
                        {
                            put_data_without_reply(rEdgeFloatArrays, lEdgeFloatArrays, lEdgeSet.fArrayNum, 1, lEdgeSet.fArrayDims,
                                                   roffset + nactualRowSize + j, j, lEdgeSet.fArrayInOut);
                        }
                    }
                }

                for (int j = 0; j < nactualInnerSize; ++j)
                {
                    put_data_without_reply(rVertexFloatArrays, lVertexFloatArrays, lVertexSet.fArrayNum, 1, lVertexSet.fArrayDims, map[j], nactualVertexSize + 1 + j, lVertexSet.fArrayInOut);
                }
                wait_put_reply;

                pointer_shift(lEdgeSet.floatArrays, lEdgeSet.fArrayNum, remain, lEdgeSet.fArrayDims);
                row += remain;
                col += remain;
                nactualInnerSize = 0;
            }
            //**************************/
            put_data_without_reply(rVertexFloatArrays, lVertexFloatArrays, lVertexSet.fArrayNum, nactualVertexSize, lVertexSet.fArrayDims, cellBegin, 0, lVertexSet.fArrayInOut);
            wait_put_reply;

            roffset += (nactualRowSize + remain);

            // //reset pointer
            row -= (remain + nactualRowSize);
            col -= (remain + nactualRowSize);
            pointer_shift(lEdgeSet.floatArrays, lEdgeSet.fArrayNum, (-1) * (remain + nactualRowSize), lEdgeSet.fArrayDims);
        }
        if (nexternal) //flush
        {
            put_without_reply(externalEdge, rlocalEdge + exEdgeOffset, nexternal * sizeof(int));
            put_edge_data_without_reply(rEdgeBufferFloatArrays, lEdgeBufferFloatArrays, lEdgeSet.fArrayNum, nexternal,
                                        lEdgeSet.fArrayDims, exEdgeOffset, 0, lEdgeSet.fArrayInOut);
            put_without_reply(lrowBuf, rrowBuf + exEdgeOffset, sizeof(int) * nexternal);
            put_without_reply(lcolBuf, rcolBuf + exEdgeOffset, sizeof(int) * nexternal);
        }
        exEdgeOffset += nexternal;
        put_without_reply(&exEdgeOffset, sp.localSize + my_id * 2 + ncycle, sizeof(int));
        wait_put_reply;
        exEdgeOffset = 0;
        nexternal = 0;
        if (ncycle == 1)
            continue;
        athread_syn(ARRAY_SCOPE, 0xffff);
    }

    // if (my_id == 0)
    //     printf("all time:%.5lfms\n", (double)(alltime)*1000 / CLOCKRATE);
    ldm_free(row, sizeof(label) * sp.maxRowPerCycle);
    ldm_free(col, sizeof(label) * sp.maxRowPerCycle);
    ldm_free(allVertexData, allVertexDataSize);
    ldm_free(allEdgeData, allEdgeDataSize);
    ldm_free(externalEdge, maxExternal * sizeof(int));
    ldm_free(allEdgeBuf, allEdgeBufSize);
    ldm_free(lrowBuf, sizeof(int) * maxExternal);
    ldm_free(lcolBuf, sizeof(int) * maxExternal);
}

void computation_rest_kernel(RestPara *para)
{
    gr = 0;
    pr = 0;
    get_reply = 0;
    put_reply = 0;

    my_id = athread_get_id(-1);
    RestPara sp;
    get_wait_reply(para, &sp, sizeof(RestPara));

    scalar *rEdgeFloatArrays[20];
    scalar *rVertexFloatArrays[20];
    label rEdgeSetDim[20];
    label rVertexSetDim[20];
    InOut rEdgeSetInOut[20];
    InOut rVertexSetInOut[20];
    scalar *lEdgeFloatArrays[20];
    scalar *lVertexFloatArrays[20];

    int edgeiSize, edgeiOffset;
    int aveSize = ceiling(sp.computeSize, 64);
    int remain = sp.computeSize % 64;
    if (remain != 0)
    {
        if (my_id < remain)
        {
            edgeiSize = aveSize;
            edgeiOffset = aveSize * my_id;
        }
        else
        {
            edgeiSize = aveSize - 1;
            edgeiOffset = aveSize * remain + (aveSize - 1) * (my_id - remain);
        }
    }
    else
    {
        edgeiSize = aveSize;
        edgeiOffset = aveSize * my_id;
    }

    DataSet lEdgeSet, lVertexSet;
    DataSet rEdgeSet, rVertexSet;

    get_without_reply(sp.dataSet_edge, &rEdgeSet, sizeof(DataSet));
    get_without_reply(sp.dataSet_vertex, &rVertexSet, sizeof(DataSet));
    wait_get_reply;

    get_without_reply(rEdgeSet.floatArrays, rEdgeFloatArrays, sizeof(scalar *) * rEdgeSet.fArrayNum);
    get_without_reply(rVertexSet.floatArrays, rVertexFloatArrays, sizeof(scalar *) * rVertexSet.fArrayNum);

    get_without_reply(rEdgeSet.fArrayDims, rEdgeSetDim, sizeof(label) * rEdgeSet.fArrayNum);
    get_without_reply(rVertexSet.fArrayDims, rVertexSetDim, sizeof(label) * rVertexSet.fArrayNum);

    get_without_reply(rEdgeSet.fArrayInOut, rEdgeSetInOut, sizeof(InOut) * rEdgeSet.fArrayNum);
    get_without_reply(rVertexSet.fArrayInOut, rVertexSetInOut, sizeof(InOut) * rVertexSet.fArrayNum);
    wait_get_reply;

    int allEdgeSize = (sp.faceSize / sizeof(scalar)) * (REST_INNER_SIZE + 1) * sizeof(scalar);
    int allVertexSize = (sp.cellSize / sizeof(scalar)) * (REST_INNER_SIZE + 1) * 2 * sizeof(scalar);
    scalar *allEdgeData = ldm_malloc(allEdgeSize);
    scalar *allVertexData = ldm_malloc(allVertexSize);
    distribute_array_pointer(allEdgeData, rEdgeSet.fArrayNum, REST_INNER_SIZE + 1, rEdgeSetDim, lEdgeFloatArrays);
    distribute_array_pointer(allVertexData, rVertexSet.fArrayNum, (REST_INNER_SIZE + 1) * 2, rVertexSetDim, lVertexFloatArrays);
    lEdgeSet.fArrayDims = rEdgeSetDim;
    lEdgeSet.fArrayNum = rEdgeSet.fArrayNum;
    lEdgeSet.fArrayInOut = rEdgeSetInOut;
    lEdgeSet.floatArrays = lEdgeFloatArrays;

    lVertexSet.fArrayDims = rVertexSetDim;
    lVertexSet.fArrayNum = rVertexSet.fArrayNum;
    lVertexSet.fArrayInOut = rVertexSetInOut;
    lVertexSet.fArraySize = REST_INNER_SIZE * 2;
    lVertexSet.floatArrays = lVertexFloatArrays;

    label row[REST_INNER_SIZE];
    label col[REST_INNER_SIZE];
    label mapRow[REST_INNER_SIZE];
    label mapCol[REST_INNER_SIZE];
    const int blockSize = 1000;
    label couldComputeEdge[1000];

    for (int i = 0; i < REST_INNER_SIZE; ++i)
    {
        row[i] = i * 2;
        col[i] = i * 2 + 1;
    }

    while (1)
    {
        int nEdgeSize = edgeiSize >= blockSize ? blockSize : edgeiSize;
        if (nEdgeSize == 0)
            break;
        edgeiSize -= nEdgeSize;
        get_wait_reply(sp.couldComputed + edgeiOffset, couldComputeEdge, sizeof(label) * nEdgeSize);
        int remain = nEdgeSize % REST_INNER_SIZE;
        if (nEdgeSize < REST_INNER_SIZE)
        {
            nEdgeSize = 0;
        }
        else
        {
            nEdgeSize -= remain;
        }
        lEdgeSet.fArraySize = REST_INNER_SIZE;
        for (int i = 0; i < nEdgeSize; i += REST_INNER_SIZE)
        {
            for (int j = 0; j < REST_INNER_SIZE; ++j)
            {
                get_without_reply(sp.row + couldComputeEdge[i + j], mapRow + j, sizeof(label));
                get_without_reply(sp.col + couldComputeEdge[i + j], mapCol + j, sizeof(label));
            }
            wait_get_reply;
            for (int j = 0; j < REST_INNER_SIZE; ++j)
            {
                get_data_without_reply(rEdgeFloatArrays, lEdgeFloatArrays, lEdgeSet.fArrayNum, 1, lEdgeSet.fArrayDims,
                                       couldComputeEdge[i + j], j, lEdgeSet.fArrayInOut);
                get_data_without_reply(rVertexFloatArrays, lVertexFloatArrays, lVertexSet.fArrayNum,
                                       1, lVertexSet.fArrayDims, mapRow[j], j * 2, lVertexSet.fArrayInOut);
                get_data_without_reply(rVertexFloatArrays, lVertexFloatArrays, lVertexSet.fArrayNum,
                                       1, lVertexSet.fArrayDims, mapCol[j], j * 2 + 1, lVertexSet.fArrayInOut);
            }
            wait_get_reply;
            sp.funPtr(&lEdgeSet, &lVertexSet, row, col);
            for (int j = 0; j < REST_INNER_SIZE; ++j)
            {
                put_data_without_reply(rEdgeFloatArrays, lEdgeFloatArrays, lEdgeSet.fArrayNum, 1, lEdgeSet.fArrayDims,
                                       couldComputeEdge[i + j], j, lEdgeSet.fArrayInOut);
                put_data_without_reply(rVertexFloatArrays, lVertexFloatArrays, lVertexSet.fArrayNum, 1,
                                       lVertexSet.fArrayDims, mapRow[j], j * 2, lVertexSet.fArrayInOut);
                put_data_without_reply(rVertexFloatArrays, lVertexFloatArrays, lVertexSet.fArrayNum, 1,
                                       lVertexSet.fArrayDims, mapCol[j], j * 2 + 1, lVertexSet.fArrayInOut);
            }
            wait_put_reply;
        }
        //remain
        for (int i = 0; i < remain; ++i)
        {
            get_without_reply(sp.row + couldComputeEdge[i + nEdgeSize], mapRow + i, sizeof(label));
            get_without_reply(sp.col + couldComputeEdge[i + nEdgeSize], mapCol + i, sizeof(label));
        }
        wait_get_reply;
        for (int i = 0; i < remain; ++i)
        {
            get_data_without_reply(rEdgeFloatArrays, lEdgeFloatArrays, lEdgeSet.fArrayNum, 1, lEdgeSet.fArrayDims,
                                   couldComputeEdge[i + nEdgeSize], i, lEdgeSet.fArrayInOut);
            get_data_without_reply(rVertexFloatArrays, lVertexFloatArrays, lVertexSet.fArrayNum,
                                   1, lVertexSet.fArrayDims, mapRow[i], i * 2, lVertexSet.fArrayInOut);
            get_data_without_reply(rVertexFloatArrays, lVertexFloatArrays, lVertexSet.fArrayNum,
                                   1, lVertexSet.fArrayDims, mapCol[i], i * 2 + 1, lVertexSet.fArrayInOut);
        }
        wait_get_reply;
        lEdgeSet.fArraySize = remain;
        sp.funPtr(&lEdgeSet, &lVertexSet, row, col);
        for (int i = 0; i < remain; ++i)
        {
            put_data_without_reply(rEdgeFloatArrays, lEdgeFloatArrays, lEdgeSet.fArrayNum, 1, lEdgeSet.fArrayDims,
                                   couldComputeEdge[i + nEdgeSize], i, lEdgeSet.fArrayInOut);
            put_data_without_reply(rVertexFloatArrays, lVertexFloatArrays, lVertexSet.fArrayNum, 1,
                                   lVertexSet.fArrayDims, mapRow[i], i * 2, lVertexSet.fArrayInOut);
            put_data_without_reply(rVertexFloatArrays, lVertexFloatArrays, lVertexSet.fArrayNum, 1,
                                   lVertexSet.fArrayDims, mapCol[i], i * 2 + 1, lVertexSet.fArrayInOut);
        }
        wait_put_reply;
        edgeiOffset += (nEdgeSize + remain);
    }
    ldm_free(allEdgeData, allEdgeSize);
    ldm_free(allVertexData, allVertexSize);
}
