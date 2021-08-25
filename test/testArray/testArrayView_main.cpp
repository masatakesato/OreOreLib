// https://stackoverflow.com/questions/20086206/is-there-a-cleaner-way-to-replicate-an-unordered-map-with-multi-type-values-in-c

// http://libclaw.sourceforge.net/multi_type_map.html


#include <crtdbg.h>
#include <vector>

#include	<oreore/common/TString.h>
#include	<oreore/container/ArrayView.h>
#include	<oreore/container/Array.h>




using fArray = OreOreLib::Array<float>;


int main()
{
	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );



	fArray arr = { 0.1f, 0.2f, 0.3f, 0.4f };



	OreOreLib::ArrayView<float> arrview( arr.begin(), 4 );


	tcout << arrview[1] << tendl;

	arrview[1] *= -1.0f;

	tcout << arr[1] << tendl;



//	memview.Resize(4);

	
	return 0;

}
