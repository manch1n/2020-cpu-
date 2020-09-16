#include "integrationTest.h"

extern "C"
{
    void slave_integration(DataSet *dataSet_edge, DataSet *dataSet_vertex,
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

void integration_test(label *row, label *col, label n_edge, label n_vertex)
{
    // 声明数据结构
    DataSet dataSet_edge_s, dataSet_vertex_s;
    DataSet dataSet_edge_m, dataSet_vertex_m;
    ArrayArray U, flux, U_test;

    srand((int)time(0));
    // 初始化边状态和点状态
    U.initRandom(3, n_vertex);
    flux.initRandom(3, n_edge);
    U_test.clone(U);

    // 初始化状态集
    initDataSet(&dataSet_edge_s);
    initDataSet(&dataSet_edge_m);
    initDataSet(&dataSet_vertex_s);
    initDataSet(&dataSet_vertex_m);

    // 将边状态、点状态封装进状态集
    addSingleArray(dataSet_edge_s, flux, COPYIN);
    addSingleArray(dataSet_vertex_s, U, COPYINOUT);

    addSingleArray(dataSet_edge_m, flux, COPYIN);
    addSingleArray(dataSet_vertex_m, U_test, COPYINOUT);

    // 从核计算部分
    unsigned long long start, end;
    start = rpcc();
    slave_computation(&dataSet_edge_s, &dataSet_vertex_s, row, col, slave_integration);
    end = rpcc();
    printf("slave:%.5lf ms\n", (double)(end - start) * 1000 / CLOCKRATE);
    // 主核计算部分
    start = rpcc();
    integration(&dataSet_edge_m, &dataSet_vertex_m, row, col);
    end = rpcc();
    printf("single:%.5lf ms\n", (double)(end - start) * 1000 / CLOCKRATE);
    // 校验结果
    checkResult(U.data, U_test.data, n_vertex * 3);
}
