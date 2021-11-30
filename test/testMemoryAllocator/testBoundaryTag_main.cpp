#include	<chrono>
#include	<crtdbg.h>

#include	<oreore/common/TString.h>
#include	<oreore/memory/BoundaryTagBlock.h>

using namespace OreOreLib;


const int	g_MemorySize = 1024;

int main()
{
	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );
	
	// Allocate Memory
	uint8 *memory_block = new uint8[ g_MemorySize ];

	
	BoundaryTagBlock* block1 = new( memory_block ) BoundaryTagBlock( memory_block, 4, true );// assign 4bytes
	BoundaryTagBlock* block2 = new( memory_block + block1->TotalSize() ) BoundaryTagBlock( memory_block + block1->TotalSize() + sizeof( BoundaryTagBlock ), 1, true );// assgin nect 1byte 


	*(uint32 *)block1->Data()	= 0xffffffff;

	*(char *)block2->Data()	= 'V';


	SafeDeleteArray( memory_block );
	

	/*
	// new連発と一括の速度比較
	std::chrono::system_clock::time_point  start, end; // 型は auto で可


	start = std::chrono::system_clock::now(); // 計測開始時間
	{
		for( int i=0; i<1000000; ++i )
		{
			__int64 *a = new __int64[1];
			delete [] a;
		}
	}
	end = std::chrono::system_clock::now();  // 計測終了時間
	double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>( end-start ).count(); //処理に要した時間をミリ秒に変換

	tcout << "time: " << elapsed << "[ms].\n";


	start = std::chrono::system_clock::now(); // 計測開始時間
	{
		__int64 *a = new __int64[1000000];
		delete [] a;
	}
	end = std::chrono::system_clock::now();  // 計測終了時間
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>( end-start ).count(); //処理に要した時間をミリ秒に変換
	tcout << "time: " << elapsed << "[ms].\n";
	*/


	return 0;
}