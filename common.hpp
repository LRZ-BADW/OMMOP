#include <cassert>
#include <random>
#include <vector>
#include <omp.h>
#include <iostream>
#include <iomanip>
#include <functional>
#include <algorithm>

const int N = getenv("N") ? atoi(getenv("N")) : 1024;
const int want_serial_check = getenv("CHECK") ? atoi(getenv("CHECK")) : 2;
using n_t = double;
using a_t = std::vector<n_t>;
const n_t v = 1;
const n_t alpha = v;
const int env_omp_num_teams = getenv("OMP_NUM_TEAMS") ? atoi(getenv("OMP_NUM_TEAMS")) : 1;
const int want_verbose = getenv("VERBOSE") ? atoi(getenv("VERBOSE")) : 0;
const int want_randomized = getenv("RANDOMIZED") ? atoi(getenv("RANDOMIZED")) : 1;
const int want_kernel = getenv("KERNEL") ? atoi(getenv("KERNEL")) : 1;
const auto const_seed = std::random_device()();
using t_t = double;

a_t gen_mtx(void)
{
	std::mt19937 gen(const_seed);
	std::uniform_int_distribution<> distrib(1, 4);

	a_t A(N*N,v);
	if (want_randomized)
		std::generate_n(A.begin(), N*N, std::bind(distrib,gen) );
	return A;
}

void print_performance(const char*fn, t_t dt_s, t_t dt_i=0, t_t dt_o=0, t_t dt_t=0)
{
	const char * tsep = "";
	const char * pmsg = "";
	const char * fmsg = "  ";
	std::cout << std::scientific << std::setprecision(2);
	if (dt_s)
		std::cout <<   fn << pmsg << "     MMM-" << N << " at " << (2ULL*N*N*N              )/(dt_s*1e9) << " GF/s" << tsep;
	if (dt_s)
		std::cout <<  "  took " << dt_s << " s" << tsep;
	if (dt_i)
		std::cout << fmsg << pmsg << "   input at " << (2ULL*N*N   * sizeof(n_t))/(dt_i*1e9) << " GB/s" << tsep;
	if (dt_o)
		std::cout << fmsg << pmsg << "  output at " << (1ULL*N*N   * sizeof(n_t))/(dt_o*1e9) << " GB/s" << tsep;
	if (dt_t)
		std::cout << fmsg << pmsg << " overall at " << (2ULL*N*N*N              )/(dt_t*1e9) << " GF/s";
	if ((dt_i + dt_o + dt_t ) || !*tsep)
		std::cout << std::endl;
	std::cout << std::setprecision(0);
}

a_t MatMatMul_CPU___serial(void)
{
	// IKJ loop
	const a_t A{gen_mtx()}, B(gen_mtx());
	const n_t *a = A.data(), *b = B.data();
	a_t C(N*N,v);
	n_t *c = C.data();
	const auto t0 = omp_get_wtime();
	for(int i=0;i<N;++i)
		for(int k=0;k<N;++k)
			for(int j=0;j<N;++j)
				c[N * i + j] += alpha * a[N * i + k] * b[N * k + j];
	const auto t1 = omp_get_wtime();
	const auto dt_s = t1 - t0;
	print_performance(__FUNCTION__, dt_s);
	return C;
}

a_t MatMatMul_CPU___openmp(void);
const a_t R = want_serial_check ? (want_serial_check == 2 ? MatMatMul_CPU___serial() : MatMatMul_CPU___serial()) : gen_mtx();

a_t MatMatMul_CPU___openmp(void)
{
	const a_t A{gen_mtx()}, B(gen_mtx());
	const n_t *a = A.data(), *b = B.data();
	a_t C(N*N,v);
	n_t *c = C.data();
	const auto t0 = omp_get_wtime();
	#pragma omp parallel for
	for(int i=0;i<N;++i)
		for(int k=0;k<N;++k)
			for(int j=0;j<N;++j)
				c[N * i + j] += alpha * a[N * i + k] * b[N * k + j];
	const auto t1 = omp_get_wtime();
	const auto dt_s = t1 - t0;
	print_performance(__FUNCTION__, dt_s);

	if ( want_serial_check != 2 )
	if ( want_serial_check )
		assert ( R == C );
	return C;
}

void MatMatMul_GPU___openmp(void)
{
	const a_t A{gen_mtx()}, B(gen_mtx());
	const n_t *a = A.data(), *b = B.data();
	a_t C(N*N,v);
	n_t *c = C.data();
	const auto t0 = omp_get_wtime();
#pragma omp target map(to:alpha,a[:N*N],b[:N*N]) map(from:c[:N*N])
#pragma omp teams default(shared)
#pragma omp distribute parallel for
	for(int i=0;i<N;++i)
		for(int k=0;k<N;++k)
			for(int j=0;j<N;++j)
				c[N * i + j] += alpha * a[N * i + k] * b[N * k + j];
	const auto t1 = omp_get_wtime();
	const auto dt_s = t1 - t0;
	print_performance(__FUNCTION__, dt_s);
	if ( want_serial_check )
		assert ( R == C );
}

void MatMatMul_GPU___data_0(void)
{
	const a_t A{gen_mtx()}, B(gen_mtx());
	const n_t *a = A.data(), *b = B.data();
	a_t C(N*N,v);
	n_t *c = C.data();
	const int M = 1;

	const double t0 = omp_get_wtime();
#pragma omp target enter data map(to:a[:N*N],b[:N*N])
#pragma omp target enter data map(to:c[:N*N])
	const double t1 = omp_get_wtime();
	for(int l=0;l<M;++l)
#pragma omp target teams distribute parallel for
	for(int i=0;i<N;++i)
		for(int k=0;k<N;++k)
			for(int j=0;j<N;++j)
				c[N * i + j] += alpha * a[N * i + k] * b[N * k + j];
	const double t2 = omp_get_wtime();
#pragma omp target exit data map(from:c[:N*N])
	const double t3 = omp_get_wtime();

	const auto dt_i = (t1 - t0) / M;
	const auto dt_s = (t2 - t1) / M;
	const auto dt_o = (t3 - t2) / M;
	const auto dt_t = (t3 - t0) / M;
	print_performance(__FUNCTION__, dt_s, dt_i, dt_o, dt_t);
	if ( want_serial_check )
		assert ( R == C );
}

template <int IBS=8, int JBS=8, int KBS=8>
void MatMatMul_GPU___data_1(void)
{
	// 1-level (access) blocking and 1-level parallelism: IJKijk
	// each thread computes a block row. it multiplies a block of rows by a block of columns, proceeding by smaller rectangles
	const int ibs = IBS;
	const int kbs = KBS;
	const int jbs = JBS;
	const a_t A{gen_mtx()}, B{gen_mtx()};
	const n_t *a = A.data(), *b = B.data();
	a_t C(N*N,v);
	n_t *c = C.data();
	const int M = 1;

	const double t0 = omp_get_wtime();
#pragma omp target enter data map(to:a[:N*N],b[:N*N])
#pragma omp target enter data map(to:c[:N*N])
	const double t1 = omp_get_wtime();
	assert ( N % ibs == 0 );
	assert ( N % jbs == 0 );
	assert ( N % kbs == 0 );
	for(int l=0;l<M;++l)
#pragma omp target teams distribute parallel for
	for(int bi=0;bi<N/ibs;++bi)
	for(int bj=0;bj<N/jbs;++bj)
	for(int bk=0;bk<N/kbs;++bk)
	for(int ii=0;ii<ibs;++ii)
	for(int jj=0;jj<jbs;++jj)
	{
		const int i = bi * ibs + ii;
		const int j = bj * jbs + jj;
		n_t acc = 0;
		for(int kk=0;kk<kbs;++kk)
		{
			const int k = bk * kbs + kk;
			acc += alpha * a[N * i + k] * b[N * k + j];
		}
		c[N * i + j] += acc;
	}
	const double t2 = omp_get_wtime();
#pragma omp target exit data map(from:c[:N*N])
	const double t3 = omp_get_wtime();

	const auto dt_i = (t1 - t0) / M;
	const auto dt_s = (t2 - t1) / M;
	const auto dt_o = (t3 - t2) / M;
	const auto dt_t = (t3 - t0) / M;
	print_performance(__FUNCTION__, dt_s, dt_i, dt_o, dt_t);
	if ( want_serial_check )
		assert ( R == C );
}
