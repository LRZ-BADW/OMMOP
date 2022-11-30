#include "common.hpp"

#if CPU_DEFAULTS
#define PARMSK 2,2,4
#define PARMSD 512,256,128,4,4,2
#else
#define PARMSK 16,16,256
#define PARMSD 64,64,64,4,2,4
#endif

int main()
{
	if (want_kernel == -3 && want_serial_check != 1)
		MatMatMul_CPU___serial();
	if (want_kernel == -2 && want_serial_check != 2)
		MatMatMul_CPU___openmp();
	if (want_kernel == -1)
		MatMatMul_GPU___openmp();
	if (want_kernel == 0)
		MatMatMul_GPU___data_0();
	if (want_kernel == 1)
		MatMatMul_GPU___data_1();
	if (want_kernel == 2)
		MatMatMul_GPU___data_2<PARMSD>();
	if (want_kernel == 3)
		MatMatMul_GPU___data_3<PARMSD>();
	if (want_kernel == 4)
		MatMatMul_GPU___data_4<PARMSD>();
	if (want_kernel == 5)
		MatMatMul_GPU___data_5<PARMSK>();
}

