#ifndef FUNC_TRAITS_H
#define	FUNC_TRAITS_H

#include	<tuple>


//######################### result/args type detection ########################//

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



//########################### Function kind detection #######################//

template <typename T>
struct func_kind_info : func_kind_info< decltype(&T::operator()) > {};


template <typename R, typename ... Args>
struct func_kind_info< R ( *)(Args...) >
{
	using args_type = typename arg_count_trait<sizeof...(Args)>::type;
	using result_type = typename result_trait<R>::type;
};


template <typename C, typename R, typename... Args>
struct func_kind_info< R( C::* )(Args...) > : func_kind_info<R ( *)(Args...)> {};


template <typename C, typename R, typename... Args>
struct func_kind_info< R( C::* )(Args...) const > : func_kind_info<R ( *)(Args...)> {};



//############################### Function traits ##############################//

template <typename T>
struct func_traits : func_traits< decltype(&T::operator()) >{};


template <typename R, typename ... Args>
struct func_traits< R( *)(Args...) >
{
	using result_type = R;
	using args_count = std::integral_constant< std::size_t, sizeof...(Args) >;
	using args_type = std::tuple< typename std::decay<Args>::type... >;
};


template <typename C, typename R, typename... Args>
struct func_traits< R( C::* )(Args...) > : func_traits<R ( *)(Args...)> {};


template <typename C, typename R, typename... Args>
struct func_traits< R( C::* )(Args...) const > : func_traits<R ( *)(Args...)> {};




//################################ Helper functions ###########################//

// parameter pack expansion
template <typename ... Args>
auto ToTuple( Args ...args ) -> std::tuple<Args...>
{
	return { args... };
}


// for_each for tuple. c++17 or later required.
template <typename F, typename ... Args>
void for_each( std::tuple<Args...> const& t, F f )
{
	std::apply( [&]( auto... args ) constexpr { (f( args ), ...); }, t );
}



// TODO: ToArgs



#endif // !FUNC_TRAITS_H
