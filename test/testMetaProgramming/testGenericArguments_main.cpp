#include	<iostream>
#include	<tuple>
using namespace std;

#include	<oreore/meta/FuncTraits.h>



// http://nsb248.hatenablog.com/entry/2016/05/02/190405




//##################### Composing tuple from arguments  #########################//
// https://stackoverflow.com/questions/8224648/retrieve-function-arguments-as-a-tuple-in-c

static float static_func( int a, int b, float c )
{
	cout << "//========== static_func =========//" << endl;
	cout << "a: " << a << endl;
	cout << "b: " << b << endl;
	cout << "c: " << c << endl;

	return c;
}




class A
{
public:
	float func( int a, int b, float c )
	{
		cout << "//========== A::func =========//" << endl;
		cout << "a: " << a << endl;
		cout << "b: " << b << endl;
		cout << "c: " << c << endl;

		return c;
	}

};









int main()
{
	A a;


	{
		cout << "//============== Create tuple from function arguments =============//\n";

		// non-member function
		auto func_arg_tuple = CreateTupleFromFuncion( &static_func );

		// class method
		auto method_arg_tuple = CreateTupleFromFuncion( &A::func );

		//for_each_tuple( );
		for_each_tuple( func_arg_tuple, []( auto it ){ std::cout << it << std::endl; } );
	}

	


	//auto args = function_args( &static_func );// &A::func );
	auto params = ToTuple( 55, 3, -55.454f );

	{
		cout << "//=================== Scatic function callback test ================//\n";
	
		cout << "std::apply() (C++17 or above)\n";
		auto result = std::apply( static_func, params );
		
		cout << endl;

		cout << "My own Call() version (below C++14)\n";
		result = Call( static_func, params );
	}

	cout << endl;

	{
		cout << "//=================== Member function callback test ================//\n";

		// https://stackoverflow.com/questions/44776927/call-member-function-with-expanded-tuple-parms-stdinvoke-vs-stdapply
		cout << "std::apply() (C++17 or above)\n";
		//auto result = std::apply( &A::func, tuple_cat(make_tuple(&a), params) );
		//auto result = std::apply( &A::func, tuple_cat( forward_as_tuple(&a), params ) );
		auto result = std::apply( &A::func, tuple_cat( tie(a), params ) );

		cout << endl;

		// https://www.tutorialspoint.com/function-pointer-to-member-function-in-cplusplus
		// https://qiita.com/_EnumHack/items/677363eec054d70b298d
		cout << "My own Call() version (below C++14)\n";
		Call( &A::func,  &a, params );
	}

	cout << endl;


//	Call( 55, 3, -55.454f );

	return 0;
}




/*

template< typename ...Args >
void Callback( Args ...args )
{
// TODO: call function
func();
}

*/