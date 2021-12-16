// https://stackoverflow.com/questions/20086206/is-there-a-cleaner-way-to-replicate-an-unordered-map-with-multi-type-values-in-c

// http://libclaw.sourceforge.net/multi_type_map.html


#include <crtdbg.h>
#include <vector>

#include	<oreore/common/TString.h>
#include	<oreore/container/Array.h>
#include	<oreore/container/ArrayView.h>



//struct Data
//{
//	float value;
//
//};
//
//
//
//using strArray = OreOreLib::Array<Data>;
//
//
//int main()
//{
//	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );
//
//	
//	tcout << _T("//=================== Struct Array FindIf test ====================//\n" );
//
//	strArray	arr1{ {0.5f}, {0.1f}, {0.3f}, {0.6f}, {0.8f}, {0.9f}, {1.1f}, {-5.5f}, {9.6f}, {0.0f} };
//	
//
//	float x = -5.5f;
//	tcout << "FintIf(" << x << "): " << OreOreLib::FindIf( arr1, [&]( const Data& d ){ return d.value==x; } ) << tendl;
//
//
//	auto* refd = &arr1[4];
//	tcout << "FintIf(" << refd << "): " << OreOreLib::FindIf( arr1, [&]( const Data& d ){	return (Data*)&d==refd; } ) << tendl;
//
//
//
//	return 0;
//
//}



float a=0.066f;

/*struct*/class Data
{
public:

	Data()
		: value( a )
	{
		tcout << "Data()\n";
	}

	Data( float val ) : value(val)
	{
		tcout << "Data( float val )\n";
	}

	~Data()
	{
		tcout << "~Data()\n";
	}

	//Data( const Data& ) = delete;


	const float& value;
	//const float& fff;


};





int main()
{
	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );

	Data d;

	OreOreLib::Array<Data>	arr1(6, d);
	arr1.Init(4, d);
	arr1.Reserve(4);//Resize(4);//
	arr1.Clear();
	arr1.Release();

	//std::vector<Data> vecarr1;
	//vecarr1.resize(6, d);
	//vecarr1.reserve(4);
	//vecarr1.clear();

	//std::fill( vecarr1.begin(), vecarr1.end(), d );

	return 0;

}
