#include	<oreore/datamanager/ObjectPage.h>
using namespace OreOreLib;


#include	<Windows.h>



int main()
{

	ObjectPage<int, 5> page;

	tcout << sizeof(page) << tendl;

	int* vals[5]={0};


	for( auto& val : vals )
		val = page.Reserve();

//	int* val = page.Reserve();
	tcout << page.IsUsedup() << tendl;

//	int* val2 = page.Reserve();
//	tcout << page.IsFull() << tendl;


	page.Free( vals[0] );
	page.Free( vals[2] );
	page.Free( vals[4] );
	tcout << page.IsUsedup() << tendl;

	vals[0] = page.Reserve();

	return 0;
}