// avx_info.cpp

#include <cstdlib>
#include <cstdint>
#include <vector>
#include <bitset>
#include <array>
#ifdef _WIN32
#include <intrin.h>
#else
#include <string.h>
#include <cpuid.h>
#endif

class AvxSupport
{
public:
	AvxSupport() : n_ids(0), f_1_ecx(0), f_7_ecx(0)
	{
		std::array<int, 4> cpui;

#ifdef _WIN32
		__cpuid(cpui.data(), 0);
		n_ids = cpui[0];
#else
		n_ids = __get_cpuid_max(0, nullptr);
#endif

		for (int i = 0; i <= n_ids; ++i)
		{
#ifdef _WIN32
			__cpuidex(cpui.data(), i, 0);
#else
			uint32_t eax = 0, ebx = 0, ecx = 0, edx = 0;
			__get_cpuid(i, &eax, &ebx, &ecx, &edx);
			cpui = { (int)eax, (int)ebx, (int)ecx, (int)edx };
#endif
			data.push_back(cpui);
		}

		// load bitset with flags for function 0x00000001
		if (n_ids >= 1)
			f_1_ecx = data[1][2];

		// load bitset with flags for function 0x00000007
		if (n_ids >= 7)
			f_7_ecx = data[7][1];

		set_flags();
	}

private:
	int n_ids;
	std::bitset<32> f_1_ecx; // for avx flag
	std::bitset<32> f_7_ecx; // for avx2 and avx512 flags
	std::vector<std::array<int, 4>> data;

public:
	bool avx = false;
	bool avx2 = false;
	bool avx512f = false;
	bool avx512pf = false;
	bool avx512er = false;
	bool avx512cd = false;

private:
#ifdef __linux__
	void cpuid(int32_t out[4], int32_t eax, int32_t ecx) {
#ifdef _WIN32
		__cpuidex(out, eax, ecx);
#else
		__cpuid_count(eax, ecx, out[0], out[1], out[2], out[3]);
#endif // WIN32

	}
#endif // __linux__
	void set_avx_flags()
	{
		avx = f_1_ecx[28];
		avx2 = f_7_ecx[5];
		avx512f = f_7_ecx[16];
		avx512pf = f_7_ecx[26];
		avx512er = f_7_ecx[27];
		avx512cd = f_7_ecx[28];
	}

	void set_flags()
	{
		set_avx_flags();
#ifdef __linux__
		// some flags are set incorrectly on Linux so re-read them again
		int32_t info[4];
		cpuid(info, 0, 0);
		int nIds = info[0];

		cpuid(info, 0x80000000, 0);
		uint32_t nExIds = info[0];

		//  Detect Features
		if (nIds >= 0x00000001) {
			cpuid(info, 0x00000001, 0);
			avx = (info[2] & ((int)1 << 28)) != 0;
		}
		if (nIds >= 0x00000007) {
			cpuid(info, 0x00000007, 0);
			avx2 = (info[1] & ((int)1 << 5)) != 0;

			avx512f = (info[1] & ((int)1 << 16)) != 0;
			avx512cd = (info[1] & ((int)1 << 28)) != 0;
			avx512pf = (info[1] & ((int)1 << 26)) != 0;
			avx512er = (info[1] & ((int)1 << 27)) != 0;

			cpuid(info, 0x00000007, 1);
		}
#endif

	}
};

int main()
{
    const AvxSupport cpu_info;
	const int avx = 1;
	const int avx2 = 2;
	const int avx512f = 4;
	const int avx512pf = 8;
	const int avx512er = 16;
	const int avx512cd = 32;

	return
		(cpu_info.avx ? avx : 0)
		+ (cpu_info.avx2 ? avx2 : 0)
		+ (cpu_info.avx512f ? avx512f : 0)
		+ (cpu_info.avx512pf ? avx512pf : 0)
		+ (cpu_info.avx512er ? avx512er : 0)
		+ (cpu_info.avx512cd ? avx512cd : 0);

}
