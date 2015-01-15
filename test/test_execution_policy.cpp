#include "gtest/gtest.h"
#include "hemi/execution_policy.h"

TEST(ExecutionPolicyTest, TestDefaults) {
	hemi::ExecutionPolicy p;

	ASSERT_EQ(hemi::ExecutionPolicy::Automatic, p.getConfigState());
	ASSERT_EQ(0, p.getGridSize());
	ASSERT_EQ(0, p.getBlockSize());
	ASSERT_EQ(0, p.getSharedMemBytes());
}

// Ensure we are setting appropriate level of configuration
// Automatic to Full Manual spectrum
TEST(ExecutionPolicyTest, TestExplicitConfiguration) {
	const int gridSize = 64;
	const int blockSize = 256;
	const int sharedMemBytes = 4096;

	hemi::ExecutionPolicy p1(gridSize, blockSize, sharedMemBytes);
	hemi::ExecutionPolicy p2(gridSize, blockSize, 0);

	EXPECT_EQ(gridSize, p1.getGridSize());
	EXPECT_EQ(blockSize, p1.getBlockSize());
	EXPECT_EQ(sharedMemBytes, p1.getSharedMemBytes());
	ASSERT_EQ(hemi::ExecutionPolicy::FullManual, p1.getConfigState());

	EXPECT_EQ(gridSize, p2.getGridSize());
	EXPECT_EQ(blockSize, p2.getBlockSize());
	EXPECT_EQ(0, p2.getSharedMemBytes());
	ASSERT_EQ(hemi::ExecutionPolicy::FullManual, p2.getConfigState());

	hemi::ExecutionPolicy p3;
	p3.setBlockSize(blockSize);

	EXPECT_EQ(blockSize, p3.getBlockSize());
}
