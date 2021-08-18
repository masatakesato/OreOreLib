#include	<tuple>
#include	<iostream>


#include	<oreore/meta/FuncTraits.h>


//##########################  helper function ###########################//

template <int N, typename ... Ts >
struct get;


template <int N, typename T, typename ... Ts>
struct get< N, std::tuple<T, Ts...> >
{
	using type = typename get<N-1, std::tuple<Ts...>>::type;
};


// get N's tuple type
template <typename T, typename ... Ts>
struct get< 0, std::tuple<T, Ts...> >
{
	using type = T;
};




//############################## function checker #################################//

template <typename F>
void CheckFunc( F func )
{
	using result_type = typename func_traits<F>::result_type;
	using args_count = typename func_traits<F>::args_count;
	using args_type = typename func_traits<F>::args_type;

	std::cout << "//========== DetectFuncTypes...";
	DetectFuncTypes( result_trait<result_type>::type(), arg_count_trait<args_count::value>::type() );
	std::cout << " ==========//\n";


	std::cout << "  result_type: " << typeid(result_type).name() << "\n";
	std::cout << "  args_type: " << typeid(args_type).name() << std::endl;
	std::cout << "  args_count( std::integral_constant ): " << args_count::value << "\n";
	//std::cout << "  args_count( std::tuple_size ): " << std::tuple_size<args_type>::value << "\n";

	std::cout << std::endl;
}



void DetectFuncTypes( const result_void&, const args_zero& )
{
	std::cout << "void_result zero_arg";
}


void DetectFuncTypes( const result_void&, const args_nonzero& )
{
	std::cout << "void_result nonzero_arg";
}


void DetectFuncTypes( const result_nonvoid&, const args_zero& )
{
	std::cout << "nonvoid_result zero_arg";
}


void DetectFuncTypes( const result_nonvoid&, const args_nonzero& )
{
	std::cout << "nonvoid_result nonzero_arg";
}






//################################ test function #################################//

float add( float a, float b, int c )
{
	return a + b;
}





class Op
{
public:

	float aaa( int &f ) const { return 9999.0f; }
	int bbb() const { return 5; }
	void ccc() {}

	float add( float a, float b, int c )
	{
		return a + b;
	}
};


int main()
{
	CheckFunc(&add);	
	CheckFunc( &Op::add );
	CheckFunc( &Op::aaa );
	CheckFunc( &Op::bbb );
	CheckFunc( &Op::ccc );
}