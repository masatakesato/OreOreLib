#include	<crtdbg.h>


#include	<oreore/common/TString.h>
#include	<oreore/mathlib/MathLib.h>
#include	<oreore/memory/PoolAllocator.h>

#include	<oreore/os/OSAllocator.h>






int main()
{
	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );

	OreOreLib::PoolAllocator	memblock( 65536, 32768 );//memblock( 4100, 4080 );//
	memblock.Display();

	while( 1 )
	{
		tcout << "//=============================== memblock( 4100, 4080 )... ===============================//\n";
		//OreOreLib::PoolAllocator	memblock;
		//OreOreLib::PoolAllocator	memblock( 4100, 4080 );//memblock( 99, 33 );//memblock( 8192, 8150 );//memblock( 4096, 32 );//memblock( 4098, 16 );//memblock( 4096, 4076 );// memblock( 100, 33 );//
		
		memblock.Init( 65536, 32768, 2 );
		memblock.Display();
		
		//tcout << "//================ memblock = OreOreLib::PoolAllocator( 99, 33 )... ================//\n";
		//memblock = OreOreLib::PoolAllocator( 4100, 4080 );// 
		//memblock.Display();
		//tcout << tendl;

		


		tcout << "//==== pIntArray = memblock.Allocate()... ====//\n";
		int* pIntArray = static_cast<int*>( memblock.Allocate() );
		memblock.Display();
		tcout << (unsigned*)pIntArray << tendl;


		tcout << "//==== pIntArray2 = memblock.Allocate()... ====//\n";
		int* pIntArray2 = static_cast<int*>( memblock.Allocate() );
		memblock.Display();
		tcout << (unsigned*)pIntArray2 << tendl;


		//tcout << "//==== pIntArray3 = memblock.Allocate()... ====//\n";
		//int* pIntArray3 = static_cast<int*>( memblock.Allocate() );
		//tcout << (unsigned*)pIntArray3 << tendl;

		tcout << "//==== pIntArray4 = memblock.Allocate()... ====//\n";
		int* pIntArray4 = static_cast<int*>( memblock.Allocate() );
		memblock.Display();
		tcout << (unsigned*)pIntArray4 << tendl;


		tcout << "//==== memblock.Free( (void*&)pIntArray4 )... ====//\n";
		memblock.Free( (void*&)pIntArray4[1] );
		

		memblock.Cleanup();
		memblock.Display();

		tcout << "//==== Free_( pIntArray2 )... ====//\n";
		memblock.Free( (void*&)pIntArray2[1] );
		memblock.Display();

		

		memblock.Display();

		tcout << tendl << tendl;
	}

	return 0;
}




//int main()
//{
//	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );
//
//
//	while( 1 )
//	{
//
//	tcout << "//================ memblock( 100, 33 )... ================//\n";
//	//OreOreLib::PoolAllocator	memblock;
//	OreOreLib::PoolAllocator	memblock( 100, 33 );
//	memblock.Display();
//	tcout << tendl;
//	
//
//	tcout << "//================ memblock = OreOreLib::PoolAllocator( 99, 33 )... ================//\n";
//	memblock = OreOreLib::PoolAllocator( 99, 33 );// OreOreLib::PoolAllocator( 4096, 32 );//
//	memblock.Display();
//	tcout << tendl;
//	
//
//	tcout << "//================ pIntArray = memblock.Allocate()... ================//\n";
//	int* pIntArray = static_cast<int*>( memblock.Allocate() );
//
//	pIntArray[0] = 1;
//	pIntArray[1] = 1;
//	pIntArray[2] = 1;
//	pIntArray[3] = 1;
//	pIntArray[4] = 1;
//	pIntArray[5] = 1;
//	pIntArray[6] = 1;
//	pIntArray[7] = 1;
//
//	//for( int i=0; i<8; ++i )
//	//	tcout << i << ": " << pIntArray[i] << tendl;
//
//	memblock.Display();
//	tcout << tendl;
//	
//
//	tcout << "//================ pIntArray2 = memblock.Allocate()... ================//\n";
//	int* pIntArray2 = nullptr;
//	pIntArray2 = static_cast<int*>( memblock.Allocate() );
//
//	pIntArray2[0] = -2;
//	pIntArray2[1] = -2;
//	pIntArray2[2] = -2;
//	pIntArray2[3] = -2;
//	pIntArray2[4] = -2;
//	pIntArray2[5] = -2;
//	pIntArray2[6] = -2;
//	pIntArray2[7] = -2;
//
//	memblock.Display();
//	tcout << tendl;
//
//	
//	tcout << "//================ pIntArray3 = memblock.Allocate()... ================//\n";
//	int* pIntArray3 = nullptr;
//	pIntArray3 = static_cast<int*>( memblock.Allocate() );
//
//	pIntArray3[0] = 3;
//	pIntArray3[1] = 3;
//	pIntArray3[2] = 3;
//	pIntArray3[3] = -3;
//	pIntArray3[4] = 3;
//	pIntArray3[5] = 3;
//	pIntArray3[6] = 3;
//	pIntArray3[7] = 3;
//
//	memblock.Display();
//	tcout << tendl;
//
//	
//	tcout << "//================ pIntArray4 = memblock.Allocate()... ================//\n";
//	int* pIntArray4 = nullptr;
//	pIntArray4 = static_cast<int*>( memblock.Allocate() );
//
//	pIntArray4[0] = 4;
//	pIntArray4[1] = 4;
//	pIntArray4[2] = 4;
//	pIntArray4[3] = -4;
//	pIntArray4[4] = 4;
//	pIntArray4[5] = 4;
//	pIntArray4[6] = 4;
//	pIntArray4[7] = 4;
//
//	memblock.Display();
//	tcout << tendl;
//
//
//	tcout << "//================ memblock.Free( pIntArray )... ================//\n";
//	memblock.Free( (void* &)pIntArray );//memblock.SafeFree( (void* &)pIntArray );
//	memblock.Display();
//
//
//	tcout << "//================ pIntArray5 = memblock.Allocate()... ================//\n";
//	int* pIntArray5 = nullptr;
//	pIntArray5 = static_cast<int*>( memblock.Allocate() );
//	//for( int i=0; i<8; ++i )	tcout << i << ": " << pIntArray5[i] << tendl;
//	memblock.Display();
//	tcout << tendl;
//
//
//	tcout << "//================ memblock.SafeFree( pIntArray2 )... ================//\n";
//	memblock.SafeFree( (void* &)pIntArray2 );
//	memblock.Display();
//	tcout << tendl;
//
//
//	tcout << "//================ memblock.SafeFree( pIntArray5 )... ================//\n";
//	memblock.SafeFree( (void* &)pIntArray5 );
//	memblock.Display();
//	tcout << tendl;
//
//
//	tcout << "//================ memblock.SafeFree( (uint32*)malloc( sizeof(uint32) * 4096 ) )... ================//\n";
//	uint32 *a = (uint32*)malloc( sizeof(uint32) * 4096 );//new uint32[4096];	
//	memblock.SafeFree( (void*&)a );// OK. but cannot free.
//	free(a);//delete [] a;
//	tcout << tendl;
//
//
//	tcout << "//================ memblock.SafeFree( new int[4096] )... ================//\n";
//	int *c = new int[4096];
//	memblock.SafeFree( (void*&)c );// OK. but cannot free
//	//OreOreLib::Free__( (void*&)c  );// NG
//	delete [] c;
//	tcout << tendl;
//
//
//	}
//
//
//	return 0;
//}





/*
	{
		tcout << _T( "//============= Tag and Memory Block capacity calculation... ==============//\n" );

		int blockSize = 112;//32
		int allocSize = 4096;
		auto tagBitLength = allocSize / blockSize;

		tcout << "blockSize: " << blockSize << tendl;
		tcout << "allocSize: " << allocSize << tendl;
		tcout << "tagBitLength: " << tagBitLength << tendl;

		tcout << tendl;

		{
			tcout << _T( "//==== 8 bit aligned tag block version =====//\n" );

			auto tagSize = DivUp( tagBitLength, BitSize::uInt8 );
			auto numActiveBlocks = ( allocSize - tagSize ) / blockSize;
			auto dataSize = numActiveBlocks * blockSize;

			tcout << "numActiveBlocks: " << numActiveBlocks << " (" << dataSize << "[bytes])\n";
			tcout << "tagSize: " << tagSize << "[bytes]\n";
			tcout << "Active size: " << tagSize + dataSize << "[bytes]\n";

			// check data
			uint8* data = new uint8[ allocSize ];
			std::fill( &data[ 0 ], &data[ dataSize ], 0 );
			std::fill( &data[ dataSize ], &data[ allocSize ], 0xff );

			DisplayBitArray( &data[ dataSize ], tagBitLength );

			delete[] data;
		}

		tcout << tendl;

		//{
		//	tcout << "//==== blockSize (" << blockSize << "[bytes]) aligned tag block version =====//\n";

		//	auto numTagBlocks = DivUp( tagBitLength, blockSize*BitSize::uInt8 );
		//	auto tagSize = numTagBlocks * blockSize;
		//	auto numActiveBlocks = ( allocSize - tagSize ) / blockSize;
		//	auto dataSize = numActiveBlocks * blockSize;

		//	tcout << "numActiveBlocks: " << numActiveBlocks << " (" << dataSize << "[bytes])\n";
		//	tcout << "tagSize: " << tagSize << "[bytes] (" << numTagBlocks << "[blocks] )\n";
		//	tcout << "Active size: " << tagSize + dataSize << "[bytes]\n";

		//	uint8* data = new uint8[ allocSize ];
		//	std::fill( &data[ 0 ], &data[ dataSize ], 0 );
		//	std::fill( &data[ dataSize ], &data[ allocSize ], 0xff );

		//	DisplayBitArray( &data[ dataSize ], tagBitLength );

		//	delete[] data;
		//}

		//tcout << tendl;


	}
*/
