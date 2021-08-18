#include	<crtdbg.h>

#ifdef _WIN64
#include	<windows.h>
#endif


#include	<oreore/common/TString.h>
#include	<oreore/common/Utility.h>
#include	<oreore/memory/MemoryManager.h>
using namespace OreOreLib;



struct data
{
	double a[32];

};


struct /*alignas(16)*/ Struct
{
	uint8 a;
	int b;
	uint8 c;
	uint8 d;


	void Display()
	{
		tcout << "Struct: " << (unsigned*)this << tendl;
		tcout << " a: " << a << tendl;
		tcout << " b: " << b << tendl;
		tcout << " c: " << c << tendl;
		tcout << " d: " << d << tendl << tendl;


	}
};



// Equivalent to Roundup
size_t AlignForward( size_t base, size_t alignment )
{
	return ( base + alignment - 1 ) & ~(alignment - 1);
}



size_t AlignForwardAdjustment( size_t base, size_t alignment )
{
	size_t adjustment = alignment - base & (alignment - 1);
	return adjustment==alignment ? 0 : adjustment;
}



//int main()
//{
//	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );
//
//
////	MemoryManager manager;
//
//
//	size_t size = 157;
//	size_t alignment = 64;
//
//
//	//tcout << RoundUp( size, alignment ) << tendl;
//	//tcout << AlignForward( size, alignment ) << tendl;
//	//tcout << AlignForwardAdjustment( size, alignment ) << tendl;
//
//
//	// 先頭アドレスを取得する
//	size_t requiredSize = size + alignment;
//	uint8* bytes = new uint8[requiredSize];
//
//
//	size_t first = AlignForwardAdjustment( size_t(bytes), alignment );
//
//	bytes[ first + size - 1 ] = 112;
//	tcout << bytes[ first + size-1 ] << tendl;
//
//
//	// アライメントをかける
//	uint8* aligned_ptr = (uint8*)RoundUp( size_t(bytes), alignment );
//	tcout << "original: "<< (unsigned*)bytes << " (" << size_t(bytes) % alignment << ")\n";
//	tcout << "  adjustment: " << AlignForwardAdjustment( size_t(bytes), alignment ) << tendl;
//	tcout << "aligned: "<< (unsigned*)aligned_ptr << " (" << size_t(aligned_ptr) % alignment << ")\n";
//
//
//
//	tcout << aligned_ptr[ size-1 ] << tendl;
//
//
//
////	tcout << sizeof(PageTag) << tendl;
//
//
//	return 0;
//}





int main()
{
	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );


	MemoryManager manager;

	size_t size = 33000;//16;
	size_t alignment = 128;

	void* ptr = manager.Allocate( size, alignment );

	tcout << "  Aligned address: " << (uint8*) ptr << " ( modulo(" << alignment << ")=" << size_t(ptr)%alignment << " )\n\n";

	manager.Free( ptr );


	return 0;
}