#include "hemi_test.h"
#include "hemi/array.h"
#include "hemi/parallel_for.h"
#include <algorithm>

TEST(ArrayTest, CreatesAndFillsArrayOnHost) 
{
	const int n = 1000;
	const float val = 3.14159f;
	hemi::Array<float> data(n);

	ASSERT_EQ(data.size(), n);

	float *ptr = data.writeOnlyHostPtr();
	std::fill(ptr, ptr+n, val);

	for(int i = 0; i < n; i++) {
		ASSERT_EQ(val, data.readOnlyHostPtr()[i]);
	}
}

template <typename T>
void fillOnDevice(T* ptr, int n, T val) {
	hemi::parallel_for(0, n, [=] HEMI_LAMBDA (int i) { 
		ptr[i] = val; 
	});
}

TEST(ArrayTest, CreatesAndFillsArrayOnDevice)
{
	const int n = 1000;
	const float val = 3.14159f;
	hemi::Array<float> data(n);

	ASSERT_EQ(data.size(), n);

	fillOnDevice(data.writeOnlyPtr(), n, val);

	for(int i = 0; i < n; i++) {
		ASSERT_EQ(val, data.readOnlyPtr(hemi::host)[i]);
	}	

	ASSERT_SUCCESS(hemi::deviceSynchronize());
}

template <typename T>
void squareOnDevice(T* ptr, int n) {
	hemi::parallel_for(0, n, [=] HEMI_LAMBDA (int i) { 
		ptr[i] = ptr[i]*ptr[i]; 
	});
}

TEST(ArrayTest, FillsOnHostModifiesOnDevice) 
{
	const int n = 1000;
	const float val = 3.14159f;
	hemi::Array<float> data(n);

	ASSERT_EQ(data.size(), n);

	float *ptr = data.writeOnlyHostPtr();
	std::fill(ptr, ptr+n, val);

	squareOnDevice(data.ptr(), n);

	for(int i = 0; i < n; i++) {
		ASSERT_EQ(val*val, data.readOnlyPtr(hemi::host)[i]);
	}	

	ASSERT_SUCCESS(hemi::deviceSynchronize());
}