import os
import argparse



parser = argparse.ArgumentParser()
parser.add_argument( "-ppid", type=int, default=None )
args = parser.parse_args()
print( args.ppid )

#ppid = os.getppid()
#if( args.ppid ):
#    print( args.ppid )
#    ppid = args.ppid

while( True ):
    print( "!!!", args.ppid )
