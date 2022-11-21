#include <omp.h>
#include <iostream>
int main()
{
	#pragma omp target teams
	#pragma omp parallel for
	for (int i = 0; i < omp_get_num_threads() ; ++ i)
	{
		const int team = omp_get_team_num();
		const int tid = omp_get_thread_num();
		if ( i == 0 )
			printf("i=%d team=%d/%d tid=%d/%d\n",i,team,omp_get_num_teams(),tid,omp_get_num_threads());
	}
}
