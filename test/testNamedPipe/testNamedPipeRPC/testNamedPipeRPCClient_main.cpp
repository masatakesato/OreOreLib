#include    <oreore/network/namedpipe/NamedPipeRPC.h>



const TCHAR* g_PipeName = _T( "\\\\.\\pipe\\Foo" );



int main()
{
	PipeClientRPC client = PipeClientRPC();
	client.Connect( g_PipeName );


	//client.Call( _T( "NoReturn" ) );
	//client.Call( _T( "Test" ) );

	//client.Call( _T("Ahgfdd"), 4, 6 );


	int a = 4, b = 6;
	//for( int i=0; i<100; ++i )
	//{
		auto result = client.Call( _T( "Add" ), a, b );
		tcout << _T( "Add(" ) << a << _T( ", " ) << b << _T( ") -> " ) << result->as<int>() << tendl;
	//}

	tcout << "...\n";

	return 0;
}