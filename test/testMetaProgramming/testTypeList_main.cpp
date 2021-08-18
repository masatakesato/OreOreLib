#include	<oreore/meta/TypeList.h>
using namespace OreOreLib::TypeList;


class AAA{};


int main()
{
	tcout << _T( "//================== Test TypeList ===================//\n" );

	// Create typelist
	using my_types = MakeTypeList_t< char, short, int, AAA, float, double >;

	// Print types
	tcout << _T("my_types...\n");
	Print< my_types >::print();

	// Get Length of my_types
	tcout << "Length = " << Length< my_types >::value << tendl;

	tcout << tendl;


	tcout << _T( "//===================== IndexOf ======================//\n" );

	// Get index using specified type
	tcout << _T("IndexOf<char> = ") << IndexOf< my_types, char >::value << tendl;
	tcout << _T("IndexOf<short> = ") << IndexOf< my_types, short >::value << tendl;
	tcout << _T("IndexOf<int> = ") << IndexOf< my_types, int >::value << tendl;
	tcout << _T("IndexOf<float> = ") << IndexOf< my_types, float >::value << tendl;
	tcout << _T("IndexOf<double> = ") << IndexOf< my_types, double >::value << tendl;
	tcout << _T("IndexOf<unsigned int> = ") << IndexOf< my_types, unsigned int >::value << tendl;// querying unregistered type.(returns -1)

	tcout << tendl;


	tcout << _T( "//===================== TypeAt ======================//\n" );
	
	// Get type using specified index
	tcout << _T("TypeAt<0> = ") << typeid( TypeAt< my_types, 0 >::type ).name() << tendl;
	tcout << _T("TypeAt<1> = ") << typeid( TypeAt< my_types, 1 >::type ).name() << tendl;
	tcout << _T("TypeAt<2> = ") << typeid( TypeAt< my_types, 2 >::type ).name() << tendl;
	tcout << _T("TypeAt<3> = ") << typeid( TypeAt< my_types, 3 >::type ).name() << tendl;
	tcout << _T("TypeAt<4> = ") << typeid( TypeAt< my_types, 4 >::type ).name() << tendl;

	tcout << tendl;


	tcout << _T( "//===================== Contains ======================//\n" );

	tcout << _T("Contains<char> = ") << Contains< my_types, char >::value << tendl;
	tcout << _T("Contains<short> = ") << Contains< my_types, short >::value << tendl;
	tcout << _T("Contains<int> = ") << Contains< my_types, int >::value << tendl;

	tcout << _T("Contains<unsigned int> = ") << Contains< my_types, unsigned int >::value << tendl;

	tcout << tendl;


	return 0;
}
