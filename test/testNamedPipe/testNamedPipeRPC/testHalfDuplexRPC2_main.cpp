#include	<iostream>

#include    <oreore/network/namedpipe/HalfDuplexRPCNode.h>

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




int main()
{
	SetConsoleTitleA( g_InPipeName.c_str() );

	auto proc = Procedure();
	auto node = HalfDuplexRPCNode( g_InPipeName );

	node.BindFunc( "NoReturn", [&proc]{ proc.NoReturn(); } );
	node.BindFunc( "Test", [&proc]{ return proc.Test(); } );
	node.BindFunc( "Add", [&proc]( int a, int b ){ return proc.Add( a, b ); } );

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