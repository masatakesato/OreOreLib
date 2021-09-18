#include	<oreore/common/TString.h>
#include	<oreore/container/NDShape.h>
using namespace OreOreLib;



int main()
{
	NDShape<3> idx( (size_t)4, (size_t)4, (size_t)2 );//, (size_t)6 );

	idx.Init( 3, 3, 3 );
	idx.Disiplay();


	for( int z=0; z<idx.Dim(2); ++z )
	{
		for( int y=0; y<idx.Dim( 1 ); ++y )
		{
			for( int x=0; x<idx.Dim(0); ++x )
			{
				auto idx1d = idx.To1D( x, y, z );
				tcout << "(" << x << ", " << y << ", " << z << ") -> (" << idx1d << ")\n";
			}
		}
	}


	for(int i=0; i<27; ++i )
	{
		int idx_[3];
		idx.ToND( idx_, i );

		tcout << "(" << i << ") -> (" << idx_[0] << ", " << idx_[1] << ", " << idx_[2] << ")\n";
	}


	return 0;
}