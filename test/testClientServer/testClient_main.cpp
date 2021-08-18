#include	<iostream>

#include	<oreore/network/Client.h>


int main()
{
	Client client( "127.0.0.1", 5007, 10000, 5 );
	tcout << "!!" << tendl;
	//auto result = client.Call( "add", -5555, 5, 66 );
	auto result = client.Call( "add", 5, 5 );
	tcout << result->as<int>() << tendl;

//	result = client.Call( "add", 5, 5.6, 3 );
	result = client.Call( "add_int", -5, -5 );
	tcout << result->as<int>() << tendl;

}

