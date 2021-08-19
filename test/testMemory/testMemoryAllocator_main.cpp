#include	<chrono>
#include	<crtdbg.h>


#include	<oreore/common/TString.h>
#include	<oreore/MathLib.h>


#include	<oreore/memory/MemoryAllocator.h>
using namespace OreOreLib;


const uint32 g_MemorySize = 4294967295/2;//300;//65536;//


uint8* byte_array = nullptr;
MemoryAllocator	mem;



//int main()
//{
//	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );
//
//
//	tcout << _T( "//################### testMemoryAllocator:: Performance Comparison ####################//\n" );
//
//
//	
//	std::chrono::system_clock::time_point	start, end;
//	
//	start = std::chrono::system_clock::now();
//
//	for( int i=0; i<1000000; ++i )
//	{
//		byte_array = new uint8[g_MemorySize];
//		delete [] byte_array;
//	}
//
//	end = std::chrono::system_clock::now();
//
//	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>( end - start ).count();
//	tcout << _T( " new/delete performance: " ) << elapsed << _T( " [ms].\n" );
//	
//
//	start = std::chrono::system_clock::now();
//	
//	mem.Init( g_MemorySize, N );
//
//	for( int i=0; i<1000000; ++i )
//	{
//		byte_array = mem.Allocate( g_MemorySize );
//		mem.Free( byte_array );
//	}
//
//	mem.Release();
//
//	end = std::chrono::system_clock::now();
//
//	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>( end - start ).count();
//	tcout << _T( "Performance: " )  << elapsed << _T( " [ms].\n" );
//
//
//
//	return 0;
//}





int main()
{
	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );


	uint8* byte_array = nullptr;

	
	mem.Init( g_MemorySize );

	mem.Info();

	byte_array = (uint8*)mem.Allocate( 1000 );
	
//	byte_array = (uint8*)mem.AlignedAllocate( 1000, 8 );


	mem.Info();


	mem.Free( byte_array );

	mem.Release();

	return 0;
}










//
//int main()
//{
//	//Tインスタンスを確保. 
//	MemoryAllocator* mem	= new MemoryAllocator( 4294967295/2, 4096 );
//	Vec4f	*pData, *pData2;
//
//	//pにメモリを割り当て
////	pData	= mem->GetMemory<Vec4f>();
////	pData2	= mem->GetMemory<Vec4f>();
//
//	tcout << sizeof(Vec4f) << tendl;
//	tcout << sizeof(Vec2f) << tendl;
//	//以下pを通して通常と同じようにアクセスできる
////	InitVec( *pData, 1.0f, 2.0f, 3.0f, 4.0f );
////	InitVec( *pData2, 4.0f, 5.0f, 6.0f, 7.0f );
//	const int NumVec = 1000;
//	Vec4f *pool	= mem->Allocate<Vec4f>( sizeof(Vec4f) * NumVec );//  GetMemory<Vec4f>( sizeof(Vec4f) * NumVec );
//
//	// https://thinkingeek.com/2017/11/19/simple-memory-pool/
//	for( int i=0; i<NumVec; ++i )
//	{
//		Vec4f* a = new (&pool[i]) Vec4f();
//		InitVec( *a, float(i), float(i), float(i), float(i) );
//		InitVec( pool[i], float(i), float(i), float(i), float(i) );
//	}
//
//
//
//	//*pData	= 123;
//
//	for( int i=0; i<NumVec; ++i )
//		tcout << pool[i](0) << ", " << pool[i](1) << ", " << pool[i](2) << ", " << pool[i](3) << tendl;
//	//tcout << pData2->x << ", " << pData2->y << ", " << pData2->z << ", " << pData2->w << tendl;
//
//
//
//	//解放
//	//mem->ReleaseMemory( (uint8 *)pData );
//	mem->Free<Vec4f>( pool );//DeleteMemory<Vec4f>( pool );
//
//
//	delete mem;
//
//	return 0;
//}
