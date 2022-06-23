#include    <oreore/network/namedpipe/NamedPipeRPC.h>

#include	<oreore/memory/Memory.h>


const tstring pipe_prefix = _T( "\\\\.\\pipe\\" );
const tstring pipe_name = _T( "Foo" );
//print( pipe_prefix + pipe_name )



void NoReturn()
{
	tcout << _T( "Procedure::NoReturn()...\n" );
}



charstring Test()
{
	tcout << _T( "Procedure::Test()...\n" );
	return "OK...";
}



int Add( int a, int b )
{
	tcout << _T( "Procedure::Add()...\n" );
	return a + b;
}



//int TestSum( std::vector<int>& vec1 )//int* a, int numelm )
//{
//	int result = 0;
//	for( const auto& val : vec1 )
//		result += val;
//	return result;
//}

int TestSum( std::vector<int>& vec1 )//int* a, int numelm )
{
	int result = 0;
	for( const auto& val : vec1 )
		result += val;
	return result;
}




int main()
{
	PipeServerRPC server( "\\\\.\\pipe\\Foo" );//#pipe_prefix + pipe_name )#

	server.BindFunc( "NoReturn", &NoReturn );
	server.BindFunc( "Test", &Test );
	server.BindFunc( "Add", &Add );
	server.BindFunc( "TestSum", &TestSum );

	server.Run();

	return 0;
}
