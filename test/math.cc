#include <hemi/hemi.h>
#include <hemi/array.h>
#include <hemi/math.h>
#include <cuda.h>


HEMI_DEFINE_CONSTANT(unsigned int N,6);
// Test harness for the math functions in hemi/math.h 
// Uses hemi::Array

HEMI_KERNEL(PowF)(const float *in, float *out) {
	out[0] = hemi::pow(in[0],in[1]);
	out[1] = hemi::pow(in[1],in[2]);
	out[2] = hemi::pow(in[2],in[3]);
	assert(out[0] == 0.0f);
	assert(out[1] == 1.0f);
	assert(out[2] == 8.0f);
}
HEMI_KERNEL(PowD)(const double *in, double *out) {
	out[0] = hemi::pow(in[0],in[1]);
	out[1] = hemi::pow(in[1],in[2]);
	out[2] = hemi::pow(in[2],in[3]);
	printf("%g^(%g) = %e\n", in[0], in[1], hemi::round(out[0]));
	out[0] = hemi::round(out[0]);
	assert(hemi::round(out[0]) == 0);
	assert(out[1] == 1.0);
	assert(out[2] == 8.0);
	
}
HEMI_KERNEL(Round)() {
	printf("Testing rounding functions.\n");
	assert(hemi::round(0.5f) == 1.0f);
	assert(hemi::round(0.4f) == 0.0f);
	assert(hemi::round(0.4) == 0.0);

}

int main() {
	hemi::Array<double> dbl_test_in(HEMI_CONSTANT(N), false);
	hemi::Array<double> dbl_test_out(HEMI_CONSTANT(N), false);
	hemi::Array<float> flt_test_in(HEMI_CONSTANT(N), false);
	hemi::Array<float> flt_test_out(HEMI_CONSTANT(N), false);

	printf("%s: Initializing memory for %d-length arrays\n", HEMI_LOC_STRING, HEMI_CONSTANT(N));
	for (int i = 0; i < HEMI_CONSTANT(N); i++) {
		dbl_test_in.writeOnlyHostPtr()[i] = i * 1.0;
		dbl_test_out.writeOnlyHostPtr()[i] = 0;
		flt_test_in.writeOnlyHostPtr()[i] = i * 1.0f;
		flt_test_out.writeOnlyHostPtr()[i] = 0;
	}

	int gridDim = 1, blockDim = 1;
    printf("Running %s Version...\n", HEMI_LOC_STRING);

	HEMI_KERNEL_LAUNCH(Round, gridDim, blockDim, 0,0);
	HEMI_KERNEL_LAUNCH(PowF, gridDim, blockDim, 0,0, flt_test_in.readOnlyPtr(), flt_test_out.writeOnlyPtr());
	HEMI_KERNEL_LAUNCH(PowD, gridDim, blockDim, 0,0, dbl_test_in.readOnlyPtr(), dbl_test_out.writeOnlyPtr());

}
