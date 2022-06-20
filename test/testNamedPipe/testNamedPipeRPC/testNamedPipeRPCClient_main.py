import oreorepylib.utils.compat as compat
from namedpiperpc import *

import threading
import time



g_PipeName = r"\\.\pipe\Foo"



if __name__=="__main__":

    client = PipeClientRPC()# r"\\.\pipe\Foo" )
    client.Connect( g_PipeName )


    print( client.Call( u"NoReturn" ) )
    print( client.Call( u"Test" ) )

    print( client.Call( u"Ahgfdd", 4, 6 ) )

    print( client.Call( u"Add", 4, 6 ) )
