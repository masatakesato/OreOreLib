#include	<iostream>

#include    <oreore/network/namedpipe/HalfDuplexRPCNode.h>

const charstring g_InPipeName = "\\\\.\\pipe\\Foo1";
const charstring g_OutPipeName = "\\\\.\\pipe\\Foo2";



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

	//def Str( self, string ) :
	//	time.sleep( 1 )
	//	d ={ "Key": 55555 }

	//	print( d[string] )
};




int main()
{
	auto proc = Procedure();
	auto node = HalfDuplexRPCNode( g_InPipeName );

	//node.BindProcInstance( proc );
	node.BindFunc( "NoReturn", [&proc]{ proc.NoReturn(); } );
	node.BindFunc( "Test", [&proc]{ return proc.Test(); } );
	node.BindFunc( "Add", [&proc]( int a, int b ){ return proc.Add( a, b ); } );

	node.StartListen();

	//std::cin.get();
	node.StopListen();
	
	node.StartListen();

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