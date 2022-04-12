#include	<oreore/memory/DebugNew.h>
#include	<oreore/common/TypeID.h>



int main()
{
	_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );

	tcout << OreOreLib::TypeID<int>::value << tendl;
	tcout << OreOreLib::TypeID<float>::value << tendl;
	tcout << OreOreLib::TypeID<double>::value << tendl;
	tcout << OreOreLib::TypeID<int>::value << tendl;

	return 0;
}