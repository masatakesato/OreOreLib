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


// for non-member functions
template< typename R, typename... T >
tuple<T...> function_args( R( *)(T...) )
{
	return tuple<T...>();
}


// for class member functions
template< typename RR, typename R, typename... T >
tuple<T...> function_args( R(RR::*)(T...) )
{
	return tuple<T...>();
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




static void Call( int a, int b, float c )
{
	// TODO: compoise a, b, c to parameter pack
	auto params = ToTuple( a, b, c );

	for_each(params, [](auto it)
	{
		std::cout << it << std::endl;
	});

}




int main()
{
	A a;

	//auto args = function_args( &static_func );// &A::func );
	auto params = ToTuple( 55, 3, -55.454f );


	//=================== 静的関数の呼び出し ================//
	auto result = std::apply( static_func, params );


	//================== メンバ関数の呼び出し ===============//
	// https://stackoverflow.com/questions/44776927/call-member-function-with-expanded-tuple-parms-stdinvoke-vs-stdapply
	//auto result = std::apply( &A::func, tuple_cat(make_tuple(&a), params) );
	//auto result = std::apply( &A::func, tuple_cat( forward_as_tuple(&a), params ) );
	result = std::apply( &A::func, tuple_cat( tie(a), params ) );



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