#include <iostream>
using namespace std;

#include	<oreore/meta/TypeList.h>
using namespace OreOreLib::TypeList;


// https://medium.com/@igorpener/arrays-of-c-function-pointers-lambdas-29189deffac4



class Base
{
public:

	Base()
	{
		//cout << "Base()...\n";
	}

};



template< typename T >
class Derived : public Base
{
public:

	Derived()
	{
		cout << "Derived<" << typeid(T).name() << ">()...\n";
	}

};





using g_Typelist = MakeTypeList_t< int, float, char, double, unsigned short, unsigned int >;

Base *(* /*const*/ createBaseFuncs[ Length<g_Typelist>::value ] )( )
{
//	[]()->Base* { return new Derived<int>(); },
//	...
};



template < typename TList >
struct BaseFuncInitializer
{
	static void Execute( int i=0 )
	{
		tcout << "[" << i << "]: BaseFuncInitializer::Execute< " << typeid( typename TList::head ).name() << " >...\n";
		createBaseFuncs[i] = []()->Base* { return new Derived< typename TList::head >(); };
		BaseFuncInitializer< typename TList::tail >::Execute( i+1 );
	}

};


// typelist end
template <>
struct BaseFuncInitializer< NullType  >
{
	static void Execute( int i ){}// Do nothing
};






int main()
{
	
	BaseFuncInitializer< g_Typelist >::Execute();



//    createBaseFuncs[2] = []()->Base* { return new Derived<char>(); };
	Base* pderived = ( *createBaseFuncs[5] )( );

	return 0;
}
