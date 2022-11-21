#include <omp.h>
#include <iostream>
int main()
{
#pragma omp parallel
	std::cout << " using " << omp_get_num_threads() << " threads" << std::endl;
}
