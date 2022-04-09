#include <vector>

#include	<oreore/common/TString.h>
#include	<oreore/memory/DebugNew.h>
#include	<oreore/memory/Memory.h>
#include	<oreore/container/Array.h>
using namespace OreOreLib;



struct Val
{
	Val() : value( new float(0) )
	{
		tcout << "Val()\n";
	}

	Val( float val )
		: value( new float(val) )
	{
		tcout << "Val( float val )\n";
	}

	~Val()
	{
		tcout << "~Val(): " << tendl;
		SafeDelete( value );
	}

	Val( const Val& obj )
		: value( new float(*obj.value)  )
	{
		tcout << "Val( const Val& obj ) : " << *value << tendl;
	}

	Val( Val&& obj )
		: value( obj.value )
	{
		tcout << "Val( Val&& obj )\n";
		 obj.value = nullptr;
	}


	float* value;

};



int main()
{
	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );

	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );

	tcout << _T("//============== Checking vector behavior ==============//\n");
	{
		std::vector<Val> vec;
		tcout << _T("vec.reserve(6);...\n");
		vec.reserve(6);

		tcout << _T("vec.resize(4);...\n");
		vec.resize(4);

		tcout << _T("vec.resize(8, 999);...\n");
		vec.resize(8, 999 );

		tcout << _T("vec.resize(6);...\n");
		vec.resize(6);
	
		tcout << _T("vec.reserve(1);...\n");
		vec.reserve(1);

		tcout << _T("end...\n");
	}
	tcout << _T("return...\n\n");
	

	tcout << _T("//============== Checking Memory behavior ================//\n");
	{
		OreOreLib::Memory<Val> vec;
		tcout << _T("vec.Reserve(6);...\n");
		vec.Reserve(6);

		tcout << _T("vec.Resize(4);...\n");
		vec.Resize(4);

		tcout << _T("vec.Resize(8);...\n");
		vec.Resize(8, 999);

		tcout << _T("vec.Resize(6);...\n");
		vec.Resize(6);

		tcout << _T("vec.Reserve(1);...\n");
		vec.Reserve(1);

		tcout << _T("end...\n");
	}
	tcout << _T("return...\n\n");


	tcout << _T("//============== Checking Array behavior ================//\n");
	{
		OreOreLib::Array<Val> vec;
		tcout << _T("vec.Reserve(6);...\n");
		vec.Reserve(6);

		tcout << _T("vec.Resize(4);...\n");
		vec.Resize(4);

		tcout << _T("vec.Resize(8, 999);...\n");
		vec.Resize(8, 999);

		tcout << _T("vec.Resize(6);...\n");
		vec.Resize(6);

		tcout << _T("vec.Reserve(1);...\n");
		vec.Reserve(1);

		tcout << _T("end...\n");
	}
	tcout << _T("return...\n\n");



	return 0;

}
