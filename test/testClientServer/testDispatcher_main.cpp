#include	<oreore/memory/Memory.h>
#include	<oreore/network/Dispatcher.h>


#include	<tuple>



////template<typename T>
//struct Point
//{
//    //template<typename U>
//    //std::enable_if_t<std::is_same<U, int>::value>
//    //MyFunction()
//    //{
//    //    std::cout << "T is int." << std::endl;
//    //}
//
//    //template<typename U>
//    //std::enable_if_t<std::is_same<U, float>::value>
//    //MyFunction()
//    //{
//    //    std::cout << "T is not int." << std::endl;
//    //}
//
//	//template<typename U>std::enable_if_t<std::is_same<U, int>::value, void>
//	//MyFunction();
//
//	//template<typename U>std::enable_if_t<std::is_same<U, float>::value, void>
//	//MyFunction();
//
//	template<typename U>
//	void MyFunction();
//
//};
//
//    template<typename U>
//    std::enable_if_t<std::is_same<U, int>::value, void>
//    Point::MyFunction()
//    {
//        std::cout << "T is int." << std::endl;
//    }
//
//    template<typename U>
//    std::enable_if_t<std::is_same<U, float>::value, void>
//    Point::MyFunction()
//    {
//        std::cout << "T is not int." << std::endl;
//    }
//







void foo()
{
	std::cout << "foo was called!" << std::endl;
}


void bar( int a )
{
	std::cout << "bar was called! " << a << std::endl;
}


int baz()
{
	std::cout << "baz was called! " << std::endl;
	return -123456;
}


int qux( int a )
{
	std::cout << "qux was called! " << a << std::endl;
	return a;
}




int add_int( int a, int b )
{
	std::cout << "add_int: " << a << ", " << b << "\n";
	return a + b;
}


double add_double( double a, double b, double c )
{
	return a + b + c;
}



class A
{
public:
	float func( int a, int b, float c )
	{
		std::cout << "//========== A::func =========//" << std::endl;
		std::cout << "a: " << a << std::endl;
		std::cout << "b: " << b << std::endl;
		std::cout << "c: " << c << std::endl;

		return c;
	}

};





Dispatcher g_Dispatcher;



template < typename IndexType >
auto Dispatch( OreOreLib::MemoryBase<char, IndexType>& data )
{
	return g_Dispatcher.Dispatch( data );
}



template < typename T, typename... Args >
auto Call_client( const std::string& proc_name, Args ...args )
{
	auto msg = std::make_tuple( proc_name, std::make_tuple( args... ) );

	msgpack::sbuffer sbuf;
	msgpack::pack( &sbuf, msg );




	OreOreLib::MemoryBase<char, int> mem( /*(OreOreLib::MemSizeType)sbuf.size(),*/ sbuf.data(), sbuf.data() + sbuf.size() ); 
	auto result = g_Dispatcher.Dispatch( mem );//Dispatch( mem );//


	return (*result)->as<T>();
}








int main()
{
	//g_Dispatcher.BindFunc();

	A a;


	g_Dispatcher.BindFunc( "add", []( int a, int b ) { return a + b; } );
	g_Dispatcher.BindFunc( "foo", &foo );
	g_Dispatcher.BindFunc( "bar", &bar );
	g_Dispatcher.BindFunc( "baz", &baz );
	g_Dispatcher.BindFunc( "qux", &qux );
	g_Dispatcher.BindFunc( "a.func", [&a]( int v1, int v2, float v3 ) { return a.func( v1, v2, v3 ); } );



	auto result = Call_client<int>( "add", 4, 14 );



//	result = Call_client<float>("a.func", 4, 4, -9999.056f );


	std::cout << result << std::endl;

}