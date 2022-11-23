#include "common.hpp"

int main()
{
	if (want_kernel == -1)
		MatMatMul_CPU___openmp();
	if (want_kernel == 0)
		MatMatMul_GPU___data_0();
	if (want_kernel == 1)
		MatMatMul_GPU___data_1();
	//MatMatMul_GPU___openmp();
}

