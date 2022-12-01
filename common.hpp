#include <cassert>
#include <random>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
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
const std::vector<int> want_kernels = [](){
	std::vector<int> kv;
	std::string ks (getenv("KERNEL"));
	if (!ks.empty()) {
		int i;
		std::replace(ks.begin(),ks.end(),',',' ');
		std::istringstream ins(ks,std::ios::in);
		while( ins >> i )
			kv.push_back(i);
	}
	else
		kv.push_back(1);
	return kv;
} ();
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

void results_vs_reference(const a_t & R, const a_t & C)
{
	for (int i = 0; i < N; ++i)
	{
		std::cout << "results row " << i << ": ";
		for (int j = 0; j < N; ++j)
			std::cout << C[N*i+j] << " ";
		std::cout << "  vs reference: ";
		for (int j = 0; j < N; ++j)
			std::cout << R[N*i+j] << " ";
		std::cout << std::endl;
	}
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
const a_t R = want_serial_check ? (want_serial_check == 2 ? MatMatMul_CPU___openmp() : MatMatMul_CPU___serial()) : gen_mtx();

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
	#pragma omp target map(to:alpha,a[:N*N],b[:N*N]) map(tofrom:c[:N*N])
	#pragma omp teams default(shared)
	#pragma omp distribute parallel for
	for(int i=0;i<N;++i)
		for(int k=0;k<N;++k)
			for(int j=0;j<N;++j)
				c[N * i + j] += alpha * a[N * i + k] * b[N * k + j];
	const auto t1 = omp_get_wtime();
	const auto dt_s = t1 - t0;
	print_performance(__FUNCTION__, dt_s);
	if ( R != C && N < 5 )
		results_vs_reference(R, C);
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
	#pragma omp target teams distribute
	for(int i=0;i<N;++i)
		for(int k=0;k<N;++k)
			#pragma omp parallel for
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
	#pragma omp target teams distribute
	for(int bi=0;bi<N/ibs;++bi)
	#pragma omp parallel for
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

template <class N_T,int N_R,int N_C>
void um2v(std::array<N_T,N_R*N_C> & dm, const N_T *sm, const int lda, const int aro, const int aco)
{
	// untransposed rectangular contiguous submatrix to vector copy
	for(int ii=0;ii<N_R;++ii)
		for(int kk=0;kk<N_C;++kk)
			dm[N_C * ii + kk] = sm[lda * (aro + ii) + (aco + kk)];
}

template <int IBS=256, int JBS=128, int KBS=64, int ISS=4, int JSS=2, int KSS=4>
void MatMatMul_GPU___data_2(void)
{
	// 1-level (cache) blocking and 2-level parallelism: IKJijk
	// each thread computes a rectangular block. it multiplies a block of rows by a block of columns, proceeding by smaller rectangles, computed with a 2-level tiling and explicit cache blocking of operands
	const int ibs = IBS;
	const int kbs = KBS;
	const int jbs = JBS;
	const int iss = ISS;
	const int jss = JSS;
	const int kss = KSS;
	const a_t A{gen_mtx()}, B{gen_mtx()};
	const n_t *__restrict__ a = A.data(), *__restrict__ b = B.data();
	a_t C(N*N,v);
	n_t * __restrict__ c = C.data();
	const int M = 1;

	const double t0 = omp_get_wtime();
	#pragma omp target enter data map(to:a[:N*N],b[:N*N])
	#pragma omp target enter data map(to:c[:N*N])
	const double t1 = omp_get_wtime();
	assert ( N % ibs == 0 );
	assert ( N % jbs == 0 );
	assert ( N % kbs == 0 );
	assert ( N % iss == 0 );
	assert ( N % jss == 0 );
	assert ( N % kss == 0 );
	const int mnt = (N / ibs) * (N / jbs); // max num threads
	const int its = std::gcd(N/ibs,N/jbs);
	const int jts = mnt / its;
	assert ( N % its == 0 );
	assert ( N % jts == 0 );
	assert ( N * N % ( its * jts ) == 0 );
	const int itb = N/(its*ibs);
	const int jtb = N/(jts*jbs);
	assert ( itb > 0 );
	assert ( jtb > 0 );
	for(int l=0;l<M;++l)
	#pragma omp target teams distribute
	for(int ti=0;ti<its;++ti)
	#pragma omp parallel for
	for(int tj=0;tj<jts;++tj)
	for(int bi=ti*itb;bi<(ti+1)*itb;++bi)
	for(int bk=0;bk<N/kbs;++bk)
	{
		std::array<n_t,ibs*kbs> aa;
		um2v<n_t,ibs,kbs>(aa, a, N, bi*ibs, bk*kbs);

		for(int bj=tj*jtb;bj<(tj+1)*jtb;++bj)
		{
			std::array<n_t,kbs*jbs> bb;
			um2v<n_t,kbs,jbs>(bb, b, N, bk*kbs, bj*jbs);

			for(int ii=0;ii<ibs;++ii)
			for(int jj=0;jj<jbs;++jj)
			{
				const int i = bi * ibs + ii;
				const int j = bj * jbs + jj;
				n_t acc = 0;

				for(int kk=0;kk<kbs;++kk)
				{
					acc += alpha * aa[kbs * ii + kk] * bb[jbs * kk + jj];
				}
				c[N * i + j] += acc;
			}
		}
	}
	const double t2 = omp_get_wtime();
	#pragma omp target exit data map(from:c[:N*N])
	const double t3 = omp_get_wtime();

	const auto dt_i = (t1 - t0) / M;
	const auto dt_s = (t2 - t1) / M;
	const auto dt_o = (t3 - t2) / M;
	const auto dt_t = (t3 - t0) / M;
	print_performance(__FUNCTION__, dt_s, dt_i, dt_o, dt_t);

	const auto uvc = std::count_if(C.begin(),C.end(),[] (n_t vv) {return vv == v;});
	if (uvc)
		std::cout << "Found " << uvc << " uninitialized elements out of " << (N*N) << " !" << std::endl;
	if ( want_serial_check )
		assert ( R == C );
}

template <class N_T,int N_R,int N_C>
void tm2v(std::array<N_T,N_R*N_C> & dm, const N_T *sm, const int lda, const int aro, const int aco)
{
	// transposed rectangular contiguous submatrix to vector copy
	for(int ii=0;ii<N_C;++ii)
		for(int kk=0;kk<N_R;++kk)
			dm[N_C * kk + ii] = sm[lda * (aro + ii) + (aco + kk)];
}

template <class N_T,int D_R,int S_R,int S_C,int V_L>
inline
void umr2v(std::array<N_T,V_L> dv[D_R], const std::array<N_T,S_R*S_C> & sm, const int sro, const int sco)
{
	// untransposed matrix' rectangular submatrix to vector rows copy
	for(int ii=0;ii<D_R;++ii)
		for(int vj=0;vj<V_L;++vj)
			dv[ii][vj] = sm[S_C * ( sro + ii ) + (sco + vj)];
}

template <class N_T,int D_R,int S_R,int S_C,int V_L>
inline
void umc2v(std::array<N_T,V_L> dv[D_R], const std::array<N_T,S_R*S_C> & sm, const int sro, const int sco)
{
	// untransposed matrix' rectangular submatrix to vector columns copy (submatrix gets transposed)
	for(int ii=0;ii<D_R;++ii)
		for(int vj=0;vj<V_L;++vj)
			dv[ii][vj] = sm[S_C * ( sro + vj ) + (sco + ii)];
}

template <int IBS=256, int JBS=128, int KBS=64, int ISS=4, int JSS=2, int KSS=4>
void MatMatMul_GPU___data_3(void)
{
	// 2-level (cache+register) blocking and 2-level parallelism: IKJijkijk
	// each thread computes a rectangular block. it multiplies a block of rows by a block of columns, proceeding by smaller rectangles, computed with an explicit 2-level tiling and cache blocking of operands and result vector
	const a_t A (gen_mtx()), B{gen_mtx()};
	const n_t *__restrict__ a = A.data(), *__restrict__ b = B.data();
	a_t C(N*N,v);
	n_t * __restrict__ c = C.data();
	const int M = 1;

	const double t0 = omp_get_wtime();
	#pragma omp target enter data map(to:a[:N*N],b[:N*N])
	#pragma omp target enter data map(to:c[:N*N])
	const double t1 = omp_get_wtime();
	const int ibs = IBS;
	const int kbs = KBS;
	const int jbs = JBS;
	assert ( N % ibs == 0 );
	assert ( N % jbs == 0 );
	assert ( N % kbs == 0 );
	const int iss = ISS;
	const int jss = JSS;
	const int kss = KSS;
	assert ( N % iss == 0 );
	assert ( N % jss == 0 );
	assert ( N % kss == 0 );
	const int mnt = (N / ibs) * (N / jbs); // max num threads
	const int its = std::gcd(N/ibs,N/jbs);
	const int jts = mnt / its;
	assert ( N % its == 0 );
	assert ( N % jts == 0 );
	assert ( N * N % ( its * jts ) == 0 );
	const int itb = N/(its*ibs);
	const int jtb = N/(jts*jbs);
	assert ( itb > 0 );
	assert ( jtb > 0 );
	for(int l=0;l<M;++l)
	#pragma omp target teams distribute
	for(int ti=0;ti<its;++ti)
	#pragma omp parallel for
	for(int tj=0;tj<jts;++tj)
	for(int bi=ti*itb;bi<(ti+1)*itb;++bi)
	for(int bk=0;bk<N/kbs;++bk)
	{
		std::array<n_t,ibs*kbs> aa;
		um2v<n_t,ibs,kbs>(aa, a, N, bi*ibs, bk*kbs);

		for(int tbj=tj*jtb;tbj<(tj+1)*jtb;++tbj)
		{
			const int bj = tbj;
			std::array<n_t,jbs*kbs> bbt;
			tm2v<n_t,jbs,kbs>(bbt, b, N, bk*kbs, bj*jbs);

			for(int ii=0;ii+iss-1<ibs;ii+=iss)
			for(int jj=0;jj+jss-1<jbs;jj+=jss)
			{
				std::array<n_t,iss*jss> cc;
				const int io = bi * ibs + ii;
				const int jo = bj * jbs + jj;

				for(int vi=0;vi<iss;++vi)
					for(int vj=0;vj<jss;++vj)
					{
						const int i = vi + io;
						const int j = vj + jo;

						cc[jss * vi + vj] = c[N * i + j];
					}

				for(int kk=0;kk+kss-1<kbs;kk+=kss)
				{
					std::array<n_t,kss> av[iss];
					umr2v<n_t,iss,ibs,kbs,kss>(av, aa, ii, kk);
					std::array<n_t,kss> bvt[jss];
					umr2v<n_t,jss,jbs,kbs,kss>(bvt, bbt, jj, kk);

					for(int vi=0;vi<iss;++vi)
						for(int vj=0;vj<jss;++vj)
							for(int vk=0;vk<kss;++vk)
								cc[jss * vi + vj] += alpha * av[vi][vk] * bvt[vj][vk];
				}

				for(int vi=0;vi<iss;++vi)
					for(int vj=0;vj<jss;++vj)
					{
						const int i = vi + io;
						const int j = vj + jo;

						c[N * i + j] = cc[jss * vi + vj];
					}
			}
		}
	}

	const double t2 = omp_get_wtime();
	#pragma omp target exit data map(from:c[:N*N])
	const double t3 = omp_get_wtime();

	const auto dt_i = (t1 - t0) / M;
	const auto dt_s = (t2 - t1) / M;
	const auto dt_o = (t3 - t2) / M;
	const auto dt_t = (t3 - t0) / M;
	print_performance(__FUNCTION__, dt_s, dt_i, dt_o, dt_t);

	const auto uvc = std::count_if(C.begin(),C.end(),[] (n_t vv) {return vv == v;});
	if (uvc)
		std::cout << "Found " << uvc << " uninitialized elements out of " << (N*N) << " !" << std::endl;
	if ( want_serial_check )
		assert ( R == C );
}

template <int IBS=256, int JBS=128, int KBS=64, int ISS=4, int JSS=2, int KSS=4>
void MatMatMul_GPU___data_4(void)
{
	// 2-level (cache+register) blocking and 2-level parallelism: IJKijkijk
	// each thread computes a rectangular block. it multiplies a block of rows by a block of columns, proceeding by smaller rectangles, computed with an explicit 2-level tiling and cache+register blocking of operands and result vector
	const int ibs = IBS;
	const int kbs = KBS;
	const int jbs = JBS;
	const int iss = ISS;
	const int jss = JSS;
	const int kss = KSS;

	if (N%IBS || N%JBS || IBS%ISS || JBS%JSS || KBS%KSS )
	{
		std::cout << __func__ << " skip incompatibly parameterized kernel while using cache blocks " << "  ibs x jbs x kbs = " << ibs << " x " << jbs << " x " << kbs << "  and register blocks  " << "iss x jss x kss = " << iss << " x " << jss << " x " << kss << std::endl;
		return;
	}
	const int mnt = (N / ibs) * (N / jbs); // max num threads
	const int its = std::gcd(N/ibs,N/jbs);
	const int jts = mnt / its;
	assert ( N % ibs == 0 );
	assert ( N % jbs == 0 );
	assert ( N % kbs == 0 );
	assert ( N % iss == 0 );
	assert ( N % jss == 0 );
	assert ( N % kss == 0 );
	assert ( ibs % iss == 0 );
	assert ( jbs % jss == 0 );
	assert ( kbs % kss == 0 );
	const a_t A{gen_mtx()}, B{gen_mtx()};
	const n_t *__restrict__ a = A.data(), *__restrict__ b = B.data();
	a_t C(N*N,v);
	n_t * __restrict__ c = C.data();
	const int M = 1;

	if (want_verbose > 0)
		std::cout << __func__ << ": max coarse parallelism is  " << mnt << " = its x jts x 1 = " << its << " x " << jts << " x 1  using cache blocks " << "  ibs x jbs x kbs = " << ibs << " x " << jbs << " x " << kbs << "  and register blocks  " << "iss x jss x kss = " << iss << " x " << jss << " x " << kss << std::endl;
	if (want_verbose > 0)
		std::cout << __func__ << ": omp_get_max_threads=" << omp_get_max_threads() << std::endl;
	if (want_verbose > 0)
		std::cout << __func__ << ": env_omp_num_teams=" << env_omp_num_teams << std::endl;
	assert ( N % its == 0 );
	assert ( N % jts == 0 );
	assert ( N * N % ( its * jts ) == 0 );
	const int itb = N/(its*ibs);
	const int jtb = N/(jts*jbs);
	assert ( itb > 0 );
	assert ( jtb > 0 );

	const double t0 = omp_get_wtime();
	#pragma omp target enter data map(to:a[:N*N],b[:N*N])
	#pragma omp target enter data map(to:c[:N*N])
	const double t1 = omp_get_wtime();
	for(int l=0;l<M;++l)
	#pragma omp target teams distribute
	for(int ti=0;ti<its;++ti)
	#pragma omp parallel for
	for(int tj=0;tj<jts;++tj)
	for(int bi=ti*itb;bi<(ti+1)*itb;++bi)
	for(int bj=tj*jtb;bj<(tj+1)*jtb;++bj)
	{
		std::array<n_t,ibs*jbs> cc;
		std::array<n_t,jbs*kbs> bbt;
		std::array<n_t,kss> av[iss];
		std::array<n_t,kss> bvt[jss];
		std::array<n_t,ibs*kbs> aa;
		std::array<n_t,jss> cv[iss];

		for(int ii=0;ii<ibs;ii++)
		for(int jj=0;jj<jbs;jj++)
			cc[jbs * ii + jj] = 0;

		for(int sbk=0;sbk<N/kbs;++sbk)
		{
			const int bk = sbk;

			tm2v<n_t,jbs,kbs>(bbt, b, N, bk*kbs, bj*jbs);
			um2v<n_t,ibs,kbs>(aa, a, N, bi*ibs, bk*kbs);

			for(int ii=0;ii+iss-1<ibs;ii+=iss)
			for(int jj=0;jj+jss-1<jbs;jj+=jss)
			for(int kk=0;kk+kss-1<kbs;kk+=kss)
			{
				umr2v<n_t,iss,ibs,kbs,kss>(av, aa, ii, kk);
				umr2v<n_t,jss,jbs,kbs,kss>(bvt, bbt, jj, kk);

				for(int vi=0;vi<iss;++vi)
					for(int vj=0;vj<jss;++vj)
					{
						cv[vi][vj] = 0;
						for(int vk=0;vk<kss;++vk)
							cv[vi][vj] += av[vi][vk] * bvt[vj][vk];
					}
				for(int vi=0;vi<iss;++vi)
					for(int vj=0;vj<jss;++vj)
					{
						const int i = ii + vi;
						const int j = jj + vj;
						cc[jbs * i + j] += cv[vi][vj];
					}
			}
		}
		for(int ii=0;ii<ibs;ii++)
		for(int jj=0;jj<jbs;jj++)
		{
			const int i = bi * ibs + ii;
			const int j = bj * jbs + jj;

			c[N * i + j] += alpha * cc[jbs * ii + jj];
		}
	}
	const double t2 = omp_get_wtime();
	#pragma omp target exit data map(from:c[:N*N])
	const double t3 = omp_get_wtime();

	const auto dt_i = (t1 - t0) / M;
	const auto dt_s = (t2 - t1) / M;
	const auto dt_o = (t3 - t2) / M;
	const auto dt_t = (t3 - t0) / M;
	print_performance(__FUNCTION__, dt_s, dt_i, dt_o, dt_t);

	const auto uvc = std::count_if(C.begin(),C.end(),[] (n_t vv) {return vv == v;});
	if (uvc)
		std::cout << "Found " << uvc << " uninitialized elements out of " << (N*N) << " !" << std::endl;
	if ( R != C )
		std::cout << "Warning: results probably wrong!" << std::endl;
	if ( R != C && N < 5 )
		results_vs_reference(R, C);
	if ( want_serial_check )
		assert ( R == C );
}

template <int ISS=4, int JSS=2, int KSS=4>
void MatMatMul_GPU___data_5(void)
{
	const int iss = ISS;
	const int jss = JSS;
	const int kss = KSS;

	const a_t A{gen_mtx()}, B{gen_mtx()};
	const n_t *__restrict__ a = A.data(), *__restrict__ b = B.data();
	a_t C(N*N,v);
	n_t * __restrict__ c = C.data();
	const int M = 1;

	const double t0 = omp_get_wtime();
	#pragma omp target enter data map(to:a[:N*N],b[:N*N])
	#pragma omp target enter data map(to:c[:N*N])
	const double t1 = omp_get_wtime();
	omp_set_num_threads(kss);
	const int blockDim_x = iss;
	const int blockDim_y = jss;
	const int BlockSize = blockDim_y;
	assert ( kss == blockDim_x * blockDim_y );
	const int gridDim_x = N / blockDim_x;
	const int gridDim_y = N / blockDim_y;
	if (want_verbose > 0)
		printf("gridDim %d x %d ",gridDim_x,gridDim_y),
		printf("blockSize:  %d\n",BlockSize);
	assert ( N == blockDim_x * gridDim_x );
	assert ( N == blockDim_y * gridDim_y );
	for(int l=0;l<M;++l)
	#pragma omp target teams distribute collapse(2)
	for (int blockIdx_x =0; blockIdx_x <gridDim_x; ++blockIdx_x )
	for (int blockIdx_y =0; blockIdx_y <gridDim_y; ++blockIdx_y )
	{
		const int bx = blockIdx_x;
		const int by = blockIdx_y;
		const int te = omp_get_team_num();

		n_t a_values[BlockSize][BlockSize];
		n_t b_values[BlockSize][BlockSize];
		#pragma omp parallel
		{
			const int a_cols = N;
			const int tn = omp_get_thread_num();
			const int tx = tn % blockDim_y;
			const int ty = tn / blockDim_y;

			const int b_cols = blockDim_x * gridDim_x;
			const int steps = a_cols / BlockSize;
			n_t thread_result = 0.0;
			for(int step = 0; step < steps; step++)
			{
				const int a_idx = BlockSize * (a_cols * by + step);
				const int b_idx = BlockSize * (b_cols * step + bx);

				a_values[ty][tx] = a[a_idx + a_cols * ty + tx];
				b_values[ty][tx] = b[b_idx + b_cols * ty + tx];
				#pragma omp barrier
				for(int i = 0; i < BlockSize; i++)
				{
					thread_result += a_values[ty][i] * b_values[i][tx];
				}
				#pragma omp barrier
			}
			const unsigned block_offset = b_cols * BlockSize * by + BlockSize * bx;
			c[block_offset + b_cols * ty + tx] += alpha * thread_result;
		}
	}
	const double t2 = omp_get_wtime();
	#pragma omp target exit data map(from:c[:N*N])
	const double t3 = omp_get_wtime();

	const auto dt_i = (t1 - t0) / M;
	const auto dt_s = (t2 - t1) / M;
	const auto dt_o = (t3 - t2) / M;
	const auto dt_t = (t3 - t0) / M;
	print_performance(__FUNCTION__, dt_s, dt_i, dt_o, dt_t);

	const auto uvc = std::count_if(C.begin(),C.end(),[] (n_t vv) {return vv == v;});
	if (uvc)
		std::cout << "Found " << uvc << " uninitialized elements out of " << (N*N) << " !" << std::endl;
	if ( R != C )
	if ( N < 9 )
	{
		std::cout << "Warning: results probably wrong!" << std::endl;
		for (int i = 0; i < N; ++i)
		{
			std::cout << "results row " << i << ": ";
			for (int j = 0; j < N; ++j)
				std::cout << C[N*i+j] << " ";
			std::cout << "  vs reference: ";
			for (int j = 0; j < N; ++j)
				std::cout << R[N*i+j] << " ";
			std::cout << std::endl;
		}

	}
	if ( want_serial_check )
		assert ( R == C );
}
