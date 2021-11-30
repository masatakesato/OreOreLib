#include	<crtdbg.h>


#include	<oreore/memory/MemoryManager.h>
#include	<oreore/common/TString.h>
using namespace OreOreLib;


int main()
{
	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );

	MemoryManager manager;


	while(1 )
	{
		float32* ptr = (float32*)manager.Allocate( 8 );
		ptr[1] = -6666.6f;
		//manager.Free( (void*&)ptr );
		tcout << (size_t)ptr % ByteSize::DefaultAlignment << tendl;

		manager.Cleanup();

		uint8* ptr2 = (uint8*)manager.Allocate(32769);
		ptr2[3] = 'U';
		manager.Free( (void*&)ptr2 );


		float32* ptr3 = (float32*)manager.Reallocate( (void*&)ptr, 8, 8 );
		manager.Display();

		//manager.Free( (void*&)ptr3 );
		//return 0;
	}

	manager.Display();

	return 0;
}