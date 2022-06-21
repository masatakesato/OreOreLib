#include    <oreore/network/namedpipe/NamedPipeRPC.h>


const tstring pipe_prefix = _T( "\\\\.\\pipe\\" );
const tstring pipe_name = _T( "Foo" );
//print( pipe_prefix + pipe_name )



void NoReturn()
{
	tcout << _T( "Procedure::NoReturn()...\n" );
}



tstring Test()
{
	tcout << _T( "Procedure::Test()...\n" );
	return _T( "OK..." );
}



int Add( int a, int b )
{
	tcout << _T( "Procedure::Add()...\n" );
	return a + b;
}




int main()
{
	PipeServerRPC server( _T( "\\\\.\\pipe\\Foo" ) );//#pipe_prefix + pipe_name )#

	server.BindFunc( _T( "NoReturn" ), &NoReturn );
	server.BindFunc( _T( "Test" ), &Test );
	server.BindFunc( _T( "Add" ), &Add );

	server.Run();

	return 0;
}
