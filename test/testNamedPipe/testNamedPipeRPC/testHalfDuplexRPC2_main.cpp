//#include    <oreore/network/namedpipe/halfduplexrpcnode.h>

#include <stdio.h>

int main()
{

	printf( "Hello World" );

	return 0;
}




//import oreorepylib.utils.compat as compat
//
//from halfduplexrpcnode import *
//
//
//g_InPipeName = r"\\.\pipe\Foo2"
//g_OutPipeName = r"\\.\pipe\Foo1"
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