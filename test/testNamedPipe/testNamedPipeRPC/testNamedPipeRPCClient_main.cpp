#include    <oreore/network/namedpipe/NamedPipeRPC.h>

#include	<oreore/extra/MsgpackAdaptor.h>



const charstring g_PipeName = "\\\\.\\pipe\\Foo";



int main()
{
	PipeClientRPC client = PipeClientRPC();
	client.Connect( g_PipeName );


	//client.Call( _T( "NoReturn" ) );
	//client.Call( _T( "Test" ) );

	//client.Call( _T("Ahgfdd"), 4, 6 );

	{
		int a = 4, b = 6;
		//for( int i=0; i<100; ++i )
		//{
		auto result = client.Call( "Add", a, b );
		tcout << _T( "Add(" ) << a << _T( ", " ) << b << _T( ") : " ) << result->as<int>() << tendl;
		//}
	}

	{
		std::vector<int> vec ={ 1, 2, 3, 4 };
		auto result = client.Call( "TestSum", vec );
		tcout << _T( "TestSum : " ) << result->as<int>() << tendl;
	}

	{
		OreOreLib::Array<int> arr ={ 1, 2, 3, 4 };
		//OreOreExtra::ArrayMsgpk<int> arr = { 1, 2, 3, 4 };
		auto p = static_cast<OreOreExtra::ArrayMsgpk<int>*>( &arr );//( OreOreExtra::ArrayMsgpk<int>* )( &arr);//
		arr[0]= 6666;
		/*auto result = */client.Call( "TestArrayTransfer", *p );
//		tcout << _T( "TestSum : " ) << result->as<int>() << tendl;
	}


	return 0;
}