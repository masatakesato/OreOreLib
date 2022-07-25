#include	<iostream>


#include    <oreore/network/namedpipe/HalfDuplexRPCNode.h>
#include	<oreore/extra/MsgpackAdaptor.h>

const charstring g_InPipeName = "\\\\.\\pipe\\Foo";
const charstring g_OutPipeName = "\\\\.\\pipe\\Bar";



class RemoteProcedure : public RemoteProcedureBase
{
public:

	RemoteProcedure( const HalfDuplexRPCNode& node )
		: RemoteProcedureBase( node )
	{
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


	void Str( const charstring& string )
	{
		Sleep( 1000 );

		std::unordered_map<charstring, int> d{ {"Key", 55555} };
		tcout << d[string] << tendl;
	}

};




int main()
{
	SetConsoleTitleA( g_InPipeName.c_str() );

	auto node = HalfDuplexRPCNode( g_InPipeName );
	
	auto proc = RemoteProcedure( node );
	node.BindFunc( "NoReturn", [&proc]{ proc.NoReturn(); } );
	node.BindFunc( "Test", [&proc]{ return proc.Test(); } );
	node.BindFunc( "Add", [&proc]( int a, int b ){ return proc.Add( a, b ); } );
	node.BindFunc( "Str", [&proc]( const charstring& string ){ return proc.Str( string ); } );
	node.BindFunc( "Connect", [&proc]( const charstring& out_pipe_name ){ return proc.Connect( out_pipe_name ); } );
	node.BindFunc( "Disconnect", [&proc]{ return proc.Disconnect(); } );

	if( !node.StartListen() )
		return 0;


	//node.StopListen();
	//
	//node.StartListen();
	//node.StopListen();


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
			//try
			//{
			//	auto result = node.Call( "Add", 4, 6 );
			//	result->type != msgpack::type::object_type::NIL
			//		? (tcout << result->as<int>() << tendl)
			//		: (tcout << "None\n");
			//}
			//catch( TCHAR *e )
			//{
			//	tcout << e << tendl;
			//}

			OreOreLib::Array<int> arr ={ 1, 2, 3, 4 };

			auto result = node.Call( "TestArrayTransfer", OreOreExtra::CastToMsgpk( arr ) );

			if( result->type != msgpack::type::object_type::NIL )
			{
				auto ret = result->as<OreOreExtra::ArrayMsgpk<int>>();
				
			}
				//tcout << result->as<int>() << tendl;

		}
			
			//tcout << node.Call( "Add", 4, 6 )->as<int>() << tendl;
			//else:
			//    node.Send( input_text.encode() )
	}

	return 0;
}




//import time
//

//
//
//
//class Procedure:
//
//    def NoReturn( self ):
//        print( "Procedure::NoReturn()..." )
//
//
//    def Test( self ):
//        print( "Procedure::Test()..." )
//        return "OK..."
//
//
//    def Add( self, a, b ):
//        print( "Procedure::Add( %d, %d )..." % (a, b) )
//        return a + b
//
//
//    def Str( self, string ):
//        time.sleep(1)
//        d = { "Key": 55555 }
//
//        print( d[string] )
//
//        
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
//            print( node.Call( "Add", 4, 6 ) )
//        #else:
//        #    node.Send( input_text.encode() )
//
//    del node