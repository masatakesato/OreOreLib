#include	<oreore/network/Server.h>
#include	<oreore/network/ServerThreading.h>
#include	<oreore/network/ServerPrethreading.h>



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




int add_int( int& a, int& b )
{
	std::cout << "add_int: " << a << ", " << b << "\n";
	return a + b;
}



double add( double a, double b, double c )
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



int main()
{
	A a;

//	Server server;
//	ServerThreading server;
	ServerPrethreading<2> server;

	server.BindFunc( "add", []( int a, int b ) { Sleep(5000); return a + b; } );
	server.BindFunc( "add_int", &add_int );
	server.BindFunc( "foo", &foo );
	server.BindFunc( "bar", &bar );
	server.BindFunc( "baz", &baz );
	server.BindFunc( "qux", &qux );
	server.BindFunc( "a.func", [&a]( int v1, int v2, float v3 ) { return a.func( v1, v2, v3 ); } );


	server.Listen( "127.0.0.1", 5007, 1 );

	server.Run();

	


}