#include	<iostream>

#include    <oreore/network/namedpipe/HalfDuplexRPCNode.h>
#include	<oreore/extra/MsgpackAdaptor.h>

const charstring g_InPipeName = "\\\\.\\pipe\\Foo2";
const charstring g_OutPipeName = "\\\\.\\pipe\\Foo1";



class Procedure
{
public:

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
		tcout << _T( "Procedure::Add( " ) << a << _T( ", " ) << b << _T( ")\n" );
		return a + b;
	}

};




template< typename T, typename IndexType >
void TestMemoryTransfer( const OreOreExtra::MemoryMsgpkImpl<T, IndexType>& arr )//const OreOreExtra::ArrayMsgpkImpl<T, IndexType>& arr )
{
	tcout << _T( "TestMemoryTransfer()...\n" );

	for( auto& v : arr )
		tcout << v << tendl;
}



template< typename T, typename IndexType >
OreOreExtra::ArrayMsgpkImpl<T, IndexType> TestArrayTransfer( OreOreExtra::ArrayMsgpkImpl<T, IndexType>& arr )
{
	tcout << _T( "TestArrayTransfer()...\n" );

	OreOreExtra::ArrayMsgpkImpl<T, IndexType> out;
	out.Init(5);
	//for( auto& v : arr )
	//	tcout << v << tendl;

	return out;
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
	SetConsoleTitleA( g_InPipeName.c_str() );

	auto proc = Procedure();
	auto node = HalfDuplexRPCNode( g_InPipeName );

	node.BindFunc( "NoReturn", [&proc]{ proc.NoReturn(); } );
	node.BindFunc( "Test", [&proc]{ return proc.Test(); } );
	node.BindFunc( "Add", [&proc]( int a, int b ){ return proc.Add( a, b ); } );
	node.BindFunc( "TestArrayTransfer", &TestArrayTransfer<int, uint32> );

	if( !node.StartListen() )
		return 0;


	std::string input_text;

	while( true )
	{
		tcout << ">";
		std::cin >> input_text;

		if( input_text == "quit" )
			break;

		else if( input_text=="disconnect" )
			node.Disconnect();

		else if( input_text=="connect" )
			node.Connect( g_OutPipeName );

		else if( input_text=="startlisten" )
			node.StartListen();

		else if( input_text=="stoplisten" )
			node.StopListen();

		else if( input_text=="testrpc" )
			node.Call( "Str", "Key" );
			//tcout << node.Call( "Str", "Key" )->as<charstring>().c_str() << tendl;
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