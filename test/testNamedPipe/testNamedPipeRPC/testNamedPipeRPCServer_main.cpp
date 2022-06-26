#include    <oreore/network/namedpipe/NamedPipeRPC.h>

#include	<oreore/memory/Memory.h>
#include	<oreore/extra/MsgpackAdaptor.h>



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


template< typename T, typename IndexType >
void TestMemoryTransfer( const OreOreExtra::MemoryMsgpkImpl<T, IndexType>& arr )//const OreOreExtra::ArrayMsgpkImpl<T, IndexType>& arr )
{
	tcout << _T( "TestMemoryTransfer()...\n" );

	for( auto& v : arr )
		tcout << v << tendl;
}



template< typename T, typename IndexType >
void TestArrayTransfer( const OreOreExtra::ArrayMsgpkImpl<T, IndexType>& arr )
{
	tcout << _T( "TestArrayTransfer()...\n" );

	for( auto& v : arr )
		tcout << v << tendl;
}



template< typename T, sizeType Size, typename IndexType >
void TestStaticArrayTransfer( const OreOreExtra::StaticArrayMsgpkImpl<T, Size, IndexType>& arr )
{
	tcout << _T( "TestStaticArrayTransfer()...\n" );

	for( auto& v : arr )
		tcout << v << tendl;
}




int main()
{
	PipeServerRPC server( "\\\\.\\pipe\\Foo" );//#pipe_prefix + pipe_name )#

	server.BindFunc( "NoReturn", &NoReturn );
	server.BindFunc( "Test", &Test );
	server.BindFunc( "Add", &Add );
	server.BindFunc( "TestSum", &TestSum );
	server.BindFunc( "TestMemoryTransfer", &TestMemoryTransfer<int, uint32> );
	server.BindFunc( "TestMemoryTransfer", &TestMemoryTransfer<int, int32> );
	server.BindFunc( "TestArrayTransfer", &TestArrayTransfer<int, int64> );

	server.Run();

	return 0;
}
