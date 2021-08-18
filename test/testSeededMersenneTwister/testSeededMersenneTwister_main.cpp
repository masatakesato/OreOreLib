#include	<oreore/common/TString.h>
#include	<oreore/mathlib/Random.h>
using namespace OreOreLib;


int main()
{
	tcout << "set seed" << tendl;
	init_genrand( 56 );

	tcout << "gen random..." << tendl;
	for( int i=0; i<10; i++ )
	{
		printf( "%lf\n", genrand_real1() );
	}


//	tcout << "set seed" << tendl;
	init_genrand( 56 );

	tcout << "gen random..." << tendl;

	for( int i=0; i<10; i++ )
	{
		printf( "%lf\n", genrand_real1() );
	}


	return 0;
}






