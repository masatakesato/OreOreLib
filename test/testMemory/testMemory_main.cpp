#include	<crtdbg.h>

#include	<oreore/common/TString.h>
#include	<oreore/memory/Memory.h>
#include	<oreore/container/Array.h>


using namespace OreOreLib;



int main()
{
	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );


	Array<float32> f32Array(16);

	for( int i=0; i<f32Array.Length(); ++i )
	{
		f32Array[i] = float32(i) + 0.001f;
	//	tcout << f32Array[i] << tendl;
	}

	f32Array.Display();

	tcout << f32Array.AllocatedSize() << tendl;
	tcout << f32Array.Length() << tendl;
	tcout << f32Array.ElementSize() << tendl;

	f32Array.Remove(0);
	f32Array.Remove(2);
	f32Array.Remove(4);
	f32Array.Remove(6);

	f32Array.Display();


	f32Array.FastRemove(0);


	f32Array.Display();

	f32Array.AddToFront( -111.1f );

	f32Array.Display();


	f32Array.AddToTail(-666.2584f);

	f32Array.Display();

	//f32Array.Init(32);
	f32Array.Release();

	return 0;
}