#include	<iostream>

#include    <oreore/network/namedpipe/HalfDuplexRPCNode.h>
#include	<oreore/extra/MsgpackAdaptor.h>

const charstring g_InPipeName = "\\\\.\\pipe\\Bar";
const charstring g_OutPipeName = "\\\\.\\pipe\\Foo";




template< typename T, typename IndexType >
void TestMemoryTransfer( const OreOreExtra::MemoryMsgpkImpl<T, IndexType>& arr )//const OreOreExtra::ArrayMsgpkImpl<T, IndexType>& arr )
{
	tcout << _T( "RemoteProcedure::TestMemoryTransfer()...\n" );

	for( auto& v : arr )
		tcout << v << tendl;
}



template< typename T, typename IndexType >
OreOreExtra::ArrayMsgpkImpl<T, IndexType> TestArrayTransfer( OreOreExtra::ArrayMsgpkImpl<T, IndexType>& arr )
{
	tcout << _T( "RemoteProcedure::TestArrayTransfer()...\n" );

	OreOreExtra::ArrayMsgpkImpl<T, IndexType> out;
	out.Init( 5 );
	//for( auto& v : arr )
	//	tcout << v << tendl;

	return out;
}



template< typename T, sizeType Size, typename IndexType >
void TestStaticArrayTransfer( const OreOreExtra::StaticArrayMsgpkImpl<T, Size, IndexType>& arr )
{
	tcout << _T( "RemoteProcedure::TestStaticArrayTransfer()...\n" );

	for( auto& v : arr )
		tcout << v << tendl;
}







class RemoteProcedure : public RemoteProcedureBase
{
public:

	RemoteProcedure( HalfDuplexRPCNode& node )
		: RemoteProcedureBase( node )
	{
		node.BindFunc( "NoReturn", [this]{ NoReturn(); } );
		node.BindFunc( "Test", [this]{ return Test(); } );
		node.BindFunc( "Add", [this]( int a, int b ){ return Add( a, b ); } );
		node.BindFunc( "Add64", [this]( uint64 a, uint64 b ){ return Add64( a, b ); } );
		node.BindFunc( "TestArrayTransfer", &TestArrayTransfer<int, uint32> );
		//node.BindFunc( "Connect", [this]( const charstring& out_pipe_name ){ return Connect( out_pipe_name ); } );
		//node.BindFunc( "Disconnect", [this]{ return Disconnect(); } );
	}


	void NoReturn()
	{
		tcout << _T( "RemoteProcedure::NoReturn()...\n" );
	}


	charstring Test()
	{
		tcout << _T( "RemoteProcedure::Test()...\n" );
		return "OK...";
	}


	int Add( int a, int b )
	{
		tcout << _T( "RemoteProcedure::Add( " ) << a << _T( ", " ) << b << _T( ")\n" );
		return a + b;
	}


	uint64 Add64( uint64 a, uint64 b )
	{
		tcout << _T( "RemoteProcedure::Add64( " ) << a << _T( ", " ) << b << _T( ")\n" );
		return a + b;
	}

};






int main()
{
	SetConsoleTitleA( g_InPipeName.c_str() );

	auto node = HalfDuplexRPCNode( g_InPipeName );

	auto proc = RemoteProcedure( node );
	//node.BindFunc( "NoReturn", [&proc]{ proc.NoReturn(); } );
	//node.BindFunc( "Test", [&proc]{ return proc.Test(); } );
	//node.BindFunc( "Add", [&proc]( int a, int b ){ return proc.Add( a, b ); } );
	//node.BindFunc( "TestArrayTransfer", &TestArrayTransfer<int, uint32> );
	//node.BindFunc( "Connect", [&proc]( const charstring& out_pipe_name ){ return proc.Connect( out_pipe_name ); } );
	//node.BindFunc( "Disconnect", [&proc]{ return proc.Disconnect(); } );

	if( !node.StartListen() )
		return 0;


	std::string input_text;

	while( true )
	{
		tcout << ">";
		std::cin >> input_text;

		if( input_text == "quit" )
			break;

		else if( input_text=="connectto" )
			node.Connect( g_OutPipeName );

		else if( input_text=="disconnectto" )
			node.Disconnect();

		else if( input_text=="connectfrom" )
			node.Call( "Connect", g_InPipeName );// Connect from

		else if( input_text=="disconnectfrom" )
			node.Call( "Disconnect" );

		else if( input_text=="startlisten" )
			node.StartListen();

		else if( input_text=="stoplisten" )
			node.StopListen();

		else if( input_text=="testrpc" )
		{
			//node.Call( "Str", "Key" );
			//tcout << node.Call( "Str", "Key" )->as<charstring>().c_str() << tendl;

			try
			{
				auto result = node.Call( "Add", 4, 6 );
				result->type != msgpack::type::object_type::NIL
					? (tcout << result->as<int>() << tendl)
					: (tcout << "None\n");
			}
			catch( TCHAR *e )
			{
				tcout << e << tendl;
			}

		}
	}


	return 0;
}







//
//
//
//if __name__=="__main__":
//
//    proc = Procedure()
//    node = HalfDuplexRPCNode( g_InPipeName )
//
//    node.BindProcInstance( proc )
//    node.StartListen()
//
//   
//    input_text = ""
//
//    while( True ):
//
//        input_text = compat.Input(">")
//
//        if( input_text == "quit" ):
//            break
//
//        elif( input_text=="disconnect" ):
//            node.Disconnect()
//
//        elif( input_text=="connect" ):
//            node.Connect( g_OutPipeName )
//
//        elif( input_text=="startlisten" ):
//            node.StartListen()
//
//        elif( input_text=="stoplisten" ):
//            node.StopListen()
//
//        elif( input_text=="testrpc" ):
//            print( node.Call( compat.ToUnicode("Str"), u"Key" ) )
//            #print( node.Call( "Add", 4, 6 ) )
//
//    del node