#ifndef FUNC_TRAITS_H
#define	FUNC_TRAITS_H

#include	<tuple>


//######################################################################################//
//																						//
//					Function kind ( args/return existence ) detection					//
//																						//
//######################################################################################//


//======================== args/result type detection ======================//

// structs for function kind detection
struct result_nonvoid{};
struct result_void{};
struct args_nonzero{};
struct args_zero{};


// type declaration for return types
template <typename R> struct result_trait	{ using type = typename result_nonvoid; };
template <> struct result_trait<void>		{ using type = typename result_void; };

// type declaration for "existence of arguments"
template <int N> struct arg_count_trait	{ using type = typename args_nonzero; };
template <> struct arg_count_trait<0>	{ using type = typename args_zero; };


//======================== Function kind detection ======================//

template <typename T>
struct func_kind_info : func_kind_info< decltype(&T::operator()) > {};

// non-member function
template < typename F, typename ... Args >
struct func_kind_info< F ( *)(Args...) >
{
	using args_type = typename arg_count_trait<sizeof...(Args)>::type;// args_nonzero if Args exist, else args_zero
	using result_type = typename result_trait<F>::type;// result_nonvoid if F returns anything, else result_void
};

// member function
template < typename Class, typename R, typename... Args >
struct func_kind_info< R( Class::* )(Args...) > : func_kind_info<R ( *)(Args...)> {};

// const member function
template <typename C, typename R, typename... Args>
struct func_kind_info< R( C::* )(Args...) const > : func_kind_info<R ( *)(Args...)> {};




//######################################################################################//
//																						//
//									Function traits										//
//																						//
//######################################################################################//

template < typename T >
struct func_traits : func_traits< decltype(&T::operator()) >{};

// non-member function
template < typename R, typename ... Args >
struct func_traits< R( *)(Args...) >
{
	using result_type = R;
	using args_count = std::integral_constant< std::size_t, sizeof...(Args) >;
	using args_type = std::tuple< typename std::decay<Args>::type... >;
};

// member function
template < typename C, typename R, typename... Args>
struct func_traits< R( C::* )(Args...) > : func_traits<R ( *)(Args...)> {};

// const member function
template < typename C, typename R, typename... Args>
struct func_traits< R( C::* )(Args...) const > : func_traits<R ( *)(Args...)> {};




//######################################################################################//
//																						//
//								Create tuple from arguments								//
//																						//
//######################################################################################//

// parameter pack expansion
template < typename ... Args >
auto ToTuple( Args ...args ) -> std::tuple<Args...>
{
	return { args... };
}




//######################################################################################//
//																						//
//						Create tuple from function arguments							//
//																						//
//######################################################################################//

// non-member variable
template < typename F, typename ... Args >
std::tuple<Args...> CreateTupleFromFuncion( F( *)(Args...) )
{
	return std::tuple<Args...>();
}


// class member variable
template < typename Class, typename F, typename... Args >
std::tuple<Args...> CreateTupleFromFuncion( F(Class::*)(Args...) )
{
	return std::tuple<Args...>();
}




//######################################################################################//
//																						//
//							function callback per tuple element  						//
//																						//
//######################################################################################//

#if __cplusplus >= 201703L

// for_each for tuple. c++17 or later required.
template < typename F, typename ... Args >
void for_each_tuple( std::tuple<Args...> const& t, F f )
{
	std::apply( [&]( auto... args ) constexpr { (f( args ), ...); }, t );
}


#else

// for_each for tuple. below c++14 non-member function

namespace detail
{
	template < typename F, typename Tuple, size_t ... I >
	void for_each_tuple_impl( F&& func, Tuple&& t, std::index_sequence<I ...> )
	{
		( func( std::get<I>( (Tuple&&)t ) ), ... );
	}

}


template < typename F, typename Tuple >
void for_each_tuple( Tuple&& t, F&& func )
{
	static constexpr auto size = std::tuple_size_v< std::decay_t<Tuple> >;
	detail::for_each_tuple_impl( func, t, std::make_index_sequence<size>{} );
}


#endif




//######################################################################################//
//																						//
//						Call function using tuple arguments								//
//																						//
//######################################################################################//

// https://stackoverflow.com/questions/7858817/unpacking-a-tuple-to-call-a-matching-function-pointer

// Below c++14 static function callback
template < typename F, typename Tuple, size_t ... I >
auto CallImpl( F&& func, Tuple&& t, std::index_sequence<I ...> )
{
	return func( std::get<I>(t) ... );
}


template < typename F, typename Tuple >
auto Call( F&& func, Tuple&& t )
{
	static constexpr auto size = std::tuple_size_v< std::decay_t<Tuple> >;
	return CallImpl( func, t, std::make_index_sequence<size>{} );
}



// https://www.tutorialspoint.com/function-pointer-to-member-function-in-cplusplus
// https://qiita.com/_EnumHack/items/677363eec054d70b298d
// Below c++14 member function callback
template < typename F, typename Class, typename Tuple, size_t ... I >
auto CallImpl( F&& mfunc, Class&& obj, Tuple&& t, std::index_sequence<I ...> )
{
	return (obj->*mfunc)( std::get<I>(t) ... );
}


template< typename F, typename Class, typename Tuple >
auto Call( F&& mfunc, Class&& obj, Tuple&& t )
{
	static constexpr auto size = std::tuple_size_v< std::decay_t<Tuple> >;
	return CallImpl( (F&&)mfunc, (Class&&)obj, (Tuple&&)t, std::make_index_sequence<size>{} );
}





#endif // !FUNC_TRAITS_H
