import random

def func():
	a = [ 111, 222, 333, 444, 555 ]
	b = random.choice(a)
	print( "py_test1.func..." )
	return b


count = 0


def func2( input ):
	print( "py_test1.func2" )
	a = input
	print( a )
	list_data = [ 3, 7, 9, 11 ]
	list_data.append( a )
	b = random.choice( list_data )

	global count
	count += 1
	print( count )

	return b



def func3( array ):
	print( "py_test1.func3" )
	print( array )