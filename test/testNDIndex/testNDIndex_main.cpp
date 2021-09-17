#include	<oreore/common/TString.h>
#include	<oreore/container/NDIndex.h>
using namespace OreOreLib;



int main()
{
	NDIndex<4> idx( (size_t)2, (size_t)2, (size_t)2, (size_t)6 );

	idx.Init( 3, 3, 6, 9 );

	idx.Disiplay();



	return 0;
}