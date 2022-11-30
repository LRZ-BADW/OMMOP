#include <omp.h>
#include <iostream>
int main()
{
	#pragma omp target teams
	#pragma omp parallel
	{
		const int team = omp_get_team_num();
		const int tid = omp_get_thread_num();
		if ( tid == 0 )
		if ( team == 0 )
			printf("team=%d/%d tid=%d/%d\n",team,omp_get_num_teams(),tid,omp_get_num_threads());
	}
}
