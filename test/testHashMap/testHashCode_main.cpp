#include	<oreore/common/TString.h>
#include	<oreore/common/HashCode.h>
using namespace OreOreLib;



int main()
{

//	tcout << OreOreLib::HashCode( (float)4 ) << tendl;

	tcout << OreOreLib::HashCode( 4.4f ) << tendl;
	tcout << OreOreLib::HashCode( 4.4 ) << tendl;
	tcout << OreOreLib::HashCode( 4.5 ) << tendl;

	tcout << OreOreLib::HashCode( tstring(_T("value1")) ) << tendl; 
	tcout << OreOreLib::HashCode( tstring(_T("value2")) ) << tendl; 


	int *a = new int(10);
	int *b = new int(10);
	tcout << OreOreLib::HashCode( a ) << tendl;
	tcout << OreOreLib::HashCode( b ) << tendl;


//	tcout << OreOreLib::hashCode( 4.4f ) << tendl;
//	tcout << OreOreLib::hashCode( 4.4 ) << tendl;


	return 0;
}