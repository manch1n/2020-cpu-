#include <cstdlib>
#include <cstdio>
#include <iostream>

#include "macro.h"
#include "arrayArray.hpp"
#include "dataSet.h"
#include "readFile.hpp"
#include "callback.h"
#include "wrapper.hpp"
#include "athread_switch.h"
#include "callbackTest.h"


// #define NONZERONUM 19800

int main(int argc, char * argv[])
{
    // 读取图/拓扑信息
    label n_edge, n_vertex;
    label* row = readFile("./data/row3.txt", &n_edge);
    label* col = readFile("./data/col3.txt", &n_edge);

    n_vertex = getVertexNum(row, col, n_edge);
    printf("n_edge: %d, n_vertex: %d\n", n_edge, n_vertex);
    // 调用测试函数
    integration_test(row, col, n_edge, n_vertex);

    spMV_test(row, col, n_edge, n_vertex);

    compVisFlux_test(row, col, n_edge, n_vertex);

    calcLudsFcc_test(row, col, n_edge, n_vertex);

    return 0;
}
