#include "spMVTest.h"

extern "C"
{
    void slave_spMV(DataSet *dataSet_edge, DataSet *dataSet_vertex,
                    label *row, label *col);
}

#define CLOCKRATE 1.45e9
static inline unsigned long rpcc()
{
    unsigned long time;
    asm("rtc %0"
        : "=r"(time)
        :);
    return time;
}

void spMV_test(label *row, label *col, label n_edge, label n_vertex)
{
    DataSet dataSet_edge_s, dataSet_vertex_s;
    DataSet dataSet_edge_m, dataSet_vertex_m;
    ArrayArray A, x, b, b_test;

    srand((int)time(0));
    A.initRandom(1, n_edge);
    x.initRandom(1, n_vertex);
    b.initRandom(1, n_vertex);
    b_test.clone(b);

    initDataSet(&dataSet_edge_s);
    initDataSet(&dataSet_vertex_s);

    initDataSet(&dataSet_edge_m);
    initDataSet(&dataSet_vertex_m);

    addSingleArray(dataSet_edge_s, A, COPYIN);
    addSingleArray(dataSet_vertex_s, x, COPYIN);
    addSingleArray(dataSet_vertex_s, b, COPYINOUT);

    addSingleArray(dataSet_edge_m, A, COPYIN);
    addSingleArray(dataSet_vertex_m, x, COPYIN);
    addSingleArray(dataSet_vertex_m, b_test, COPYINOUT);

    // 从核计算部分
    unsigned long long start, end;
    start = rpcc();
    slave_computation(&dataSet_edge_s, &dataSet_vertex_s, row, col, slave_spMV);
    end = rpcc();
    printf("slave:%.5lf ms\n", (double)(end - start) * 1000 / CLOCKRATE);
    // 主核计算部分
    start=rpcc();
    spMV(&dataSet_edge_m, &dataSet_vertex_m, row, col);
    end=rpcc();
    printf("single:%.5lf ms\n", (double)(end - start) * 1000 / CLOCKRATE);
    // 校验结果
    checkResult(b.data, b_test.data, n_vertex);
}
