from namedpiperpc import *


import threading
import time


#pipe_prefix = "\\\\.\\pipe\\"
#pipe_name = "Foo"
#print( pipe_prefix + pipe_name )


class Procedure:

    def NoReturn( self ):
        print( "Procedure::NoReturn()..." )


    def Test( self ):
        print( "Procedure::Test()..." )
        return "OK..."


    def Add( self, a, b ):
        print( "Procedure::Add( %d, %d )..." % (a, b) )
        return a + b



proc = Procedure()


server = PipeServerRPC( r"\\.\pipe\Foo" )#pipe_prefix + pipe_name )#

server.BindProcInstance( proc )
server.Run()
