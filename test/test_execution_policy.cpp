#include "gtest/gtest.h"
#include "hemi/execution_policy.h"

using namespace hemi;

// Ensure we are setting appropriate level of configuration
// Automatic to Full Manual spectrum
TEST(ExecutionPolicyTest, StateReflectsConfiguration) {
	// Defaults: fully automatic
	ExecutionPolicy p; 
    int configState = p.getConfigState();
    EXPECT_EQ (ExecutionPolicy::Automatic,  configState);
    EXPECT_NE (ExecutionPolicy::FullManual, configState);
    EXPECT_EQ (0, configState & ExecutionPolicy::SharedMem);
    EXPECT_EQ (0, configState & ExecutionPolicy::BlockSize);
    EXPECT_EQ (0, configState & ExecutionPolicy::GridSize);
	EXPECT_EQ (0, p.getGridSize());
	EXPECT_EQ (0, p.getBlockSize());
	EXPECT_EQ (0, p.getSharedMemBytes());

	// shared memory only
    p = ExecutionPolicy(); // reset
    p.setSharedMemBytes( 1024 );
    configState = p.getConfigState();
    EXPECT_NE (ExecutionPolicy::FullManual, configState);
    EXPECT_NE (ExecutionPolicy::Automatic,  configState);
    EXPECT_NE (0, configState & ExecutionPolicy::SharedMem);
    EXPECT_EQ (0, configState & ExecutionPolicy::BlockSize);
    EXPECT_EQ (0, configState & ExecutionPolicy::GridSize);
	EXPECT_EQ (0, p.getGridSize());
	EXPECT_EQ (0, p.getBlockSize());
	EXPECT_EQ (1024, p.getSharedMemBytes());

    // Block Size Only
    p = ExecutionPolicy(); // reset
    p.setBlockSize( 512 );
    configState = p.getConfigState();
    EXPECT_NE (ExecutionPolicy::FullManual, configState);
    EXPECT_NE (ExecutionPolicy::Automatic,  configState);
    EXPECT_EQ (0, configState & ExecutionPolicy::SharedMem);
    EXPECT_NE (0, configState & ExecutionPolicy::BlockSize);
    EXPECT_EQ (0, configState & ExecutionPolicy::GridSize);
    EXPECT_EQ (0, p.getGridSize());
	EXPECT_EQ (512, p.getBlockSize());
	EXPECT_EQ (0, p.getSharedMemBytes());

    // Block Size and Shared Memory Only
    p = ExecutionPolicy(); // reset
    p.setBlockSize( 512 );
    p.setSharedMemBytes( 1024 );
    configState = p.getConfigState();
    EXPECT_NE (ExecutionPolicy::FullManual, configState);
    EXPECT_NE (ExecutionPolicy::Automatic,  configState);
    EXPECT_NE (0, configState & ExecutionPolicy::SharedMem);
    EXPECT_NE (0, configState & ExecutionPolicy::BlockSize);
    EXPECT_EQ (0, configState & ExecutionPolicy::GridSize);
    EXPECT_EQ (0, p.getGridSize());
	EXPECT_EQ (512, p.getBlockSize());
	EXPECT_EQ (1024, p.getSharedMemBytes());

    // Grid Size Only
    p = ExecutionPolicy(); // reset
    p.setGridSize( 100 );
    configState = p.getConfigState();
    EXPECT_NE (ExecutionPolicy::FullManual, configState);
    EXPECT_NE (ExecutionPolicy::Automatic, configState);
    EXPECT_EQ (0, configState & ExecutionPolicy::SharedMem);
    EXPECT_EQ (0, configState & ExecutionPolicy::BlockSize);
    EXPECT_NE (0, configState & ExecutionPolicy::GridSize);
    EXPECT_EQ (100, p.getGridSize());
	EXPECT_EQ (0, p.getBlockSize());
	EXPECT_EQ (0, p.getSharedMemBytes());

    // Grid Size and Shared Memory Only
    p = ExecutionPolicy(); // reset
    p.setGridSize( 100 );
    p.setSharedMemBytes( 1024 );
    configState = p.getConfigState();
    EXPECT_NE (ExecutionPolicy::FullManual, configState);
    EXPECT_NE (ExecutionPolicy::Automatic, configState);
    EXPECT_NE (0, configState & ExecutionPolicy::SharedMem);
    EXPECT_EQ (0, configState & ExecutionPolicy::BlockSize);
    EXPECT_NE (0, configState & ExecutionPolicy::GridSize);
	EXPECT_EQ (100, p.getGridSize());
	EXPECT_EQ (0, p.getBlockSize());
	EXPECT_EQ (1024, p.getSharedMemBytes());

    // Grid Size and Block Size Only
    p = ExecutionPolicy(); // reset
    p.setGridSize( 100 );
    p.setBlockSize( 512 );
    configState = p.getConfigState();
    EXPECT_NE (ExecutionPolicy::FullManual, configState);
    EXPECT_NE (ExecutionPolicy::Automatic, configState);
    EXPECT_EQ (0, configState & ExecutionPolicy::SharedMem);
    EXPECT_NE (0, configState & ExecutionPolicy::BlockSize);
    EXPECT_NE (0, configState & ExecutionPolicy::GridSize);
	EXPECT_EQ (100, p.getGridSize());
	EXPECT_EQ (512, p.getBlockSize());
	EXPECT_EQ (0, p.getSharedMemBytes());

    // Full Manual Configuration
    p = ExecutionPolicy{1, 256, 10};
    configState = p.getConfigState();
    EXPECT_EQ (ExecutionPolicy::FullManual, configState);
    EXPECT_NE (ExecutionPolicy::Automatic, configState);
    EXPECT_NE (0, configState & ExecutionPolicy::SharedMem);
    EXPECT_NE (0, configState & ExecutionPolicy::BlockSize);
    EXPECT_NE (0, configState & ExecutionPolicy::GridSize);
   	EXPECT_EQ (1, p.getGridSize());
	EXPECT_EQ (256, p.getBlockSize());
	EXPECT_EQ (10, p.getSharedMemBytes());

	// Full Manual Configuration With Separate Calls
    p = ExecutionPolicy(); // reset
    p.setGridSize( 100 );
    p.setBlockSize( 512 );
    p.setSharedMemBytes( 1024);
    configState = p.getConfigState();
    EXPECT_EQ (ExecutionPolicy::FullManual, configState);
    EXPECT_NE (ExecutionPolicy::Automatic, configState);
    EXPECT_NE (0, configState & ExecutionPolicy::SharedMem);
    EXPECT_NE (0, configState & ExecutionPolicy::BlockSize);
    EXPECT_NE (0, configState & ExecutionPolicy::GridSize);
   	EXPECT_EQ (100, p.getGridSize());
	EXPECT_EQ (512, p.getBlockSize());
	EXPECT_EQ (1024, p.getSharedMemBytes());

	p = ExecutionPolicy(); // reset
    p.setGridSize( 0 );
    p.setBlockSize( 0 );
    p.setSharedMemBytes( 0 );
    configState = p.getConfigState();

	// Setting Zero Shared Memory makes it *Manual*
    EXPECT_NE (ExecutionPolicy::FullManual, configState);
    EXPECT_NE (ExecutionPolicy::Automatic, configState);
    EXPECT_NE (0, configState & ExecutionPolicy::SharedMem);
    
    // Setting 0 grid or block size makes them *Automatic*
    EXPECT_EQ (0, configState & ExecutionPolicy::BlockSize);
    EXPECT_EQ (0, configState & ExecutionPolicy::GridSize);
   	EXPECT_EQ (0, p.getGridSize());
	EXPECT_EQ (0, p.getBlockSize());
	EXPECT_EQ (0, p.getSharedMemBytes());

	// Re-setting block or grid size to >0 should set them to manual
	p.setGridSize( 100 );
	p.setBlockSize( 512 );
	configState = p.getConfigState();
	EXPECT_NE(0, configState & ExecutionPolicy::GridSize);
	EXPECT_NE(0, configState & ExecutionPolicy::BlockSize);

	// Re-setting block or grid size to zero should set them to automatic
	p.setGridSize( 0 );
	p.setBlockSize( 0 );
	configState = p.getConfigState();
	EXPECT_EQ(0, configState & ExecutionPolicy::GridSize);
	EXPECT_EQ(0, configState & ExecutionPolicy::BlockSize);

    // Setting stream (trivial)
    p.setStream(1);
    EXPECT_EQ(1, p.getStream());
}
