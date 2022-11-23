#include "common.hpp"

int main()
{
	if (want_kernel == -1)
		MatMatMul_CPU___openmp();
	if (want_kernel == 0)
		MatMatMul_GPU___data_0();
	if (want_kernel == 1)
		MatMatMul_GPU___data_1();
	if (want_kernel == 2)
		MatMatMul_GPU___data_2();
	if (want_kernel == 3)
		MatMatMul_GPU___data_3<>();
	//MatMatMul_GPU___openmp();
}

