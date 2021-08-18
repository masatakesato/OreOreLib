#include	<future>
#include	<thread>
#include	<string>
#include	<unordered_map>
#include	<iostream>
#include	<functional>

#include	<msgpack.hpp>


#include	<oreore/meta/FuncTraits.h>




// https://stackoverflow.com/questions/51624070/accessing-stdvariant-using-stdgetindex
struct get_visitor
{
	template< typename T >
	T operator() ( const T& value ) const
	{
		return value;//std::cout << value << "\n";
	}   
};





std::unordered_map< std::string, std::function< std::unique_ptr<msgpack::object_handle>( const msgpack::object& )> > g_Funcs;


void foo()
{
	std::cout << "foo was called!" << std::endl;
}


void bar( int a )
{
	std::cout << "bar was called! " << a << std::endl;
}


int baz()
{
	std::cout << "baz was called! " << std::endl;
	return -123456;
}


int qux( int a )
{
	std::cout << "qux was called! " << a << std::endl;
	return a;
}




int add_int( int a, int b )
{
	std::cout << "add_int: " << a << ", " << b << "\n";
	return a + b;
}


double add_double( double a, double b, double c )
{
	return a + b + c;
}



class A
{
public:
	float func( int a, int b, float c )
	{
		std::cout << "//========== A::func =========//" << std::endl;
		std::cout << "a: " << a << std::endl;
		std::cout << "b: " << b << std::endl;
		std::cout << "c: " << c << std::endl;

		return c;
	}

};



//########################### Function binding ##################################//
/*
// https://github.com/rpclib/rpclib/blob/3b00c4ccf480b9f9569b1d064e7a3b43585b8dfd/include/rpc/dispatcher.inl


template <typename Functor, typename Arg>
auto Call( Functor f, Arg &&arg ) -> decltype( f( std::forward<Arg>( arg ) ) )
{
	return f( std::forward<Arg>(arg) );
}



// まだよく理解してない. call.hからコピーしただけ.
template <typename Functor, typename ... Args, std::size_t... I>
decltype(auto) call_helper( Functor func, std::tuple<Args...> &&params, std::index_sequence<I...> )
{
	return func( std::get<I>( params )... );
}

// まだよく理解してない.  call.hからコピーしただけ.
//! \brief Calls a functor with arguments provided as a tuple
template <typename Functor, typename... Args>
decltype(auto) Call( Functor f, std::tuple<Args...> &args )
{
	return std::apply( f, args );
	//return call_helper( f, std::forward<std::tuple<Args...>>( args ), std::index_sequence_for<Args...>{} );
}
*/


// BindFunc entry point
template <typename F>
void BindFunc( std::string name, F func )
{
	BindFunc<F>( name, func, func_kind_info<F>::result_type(), func_kind_info<F>::args_type() );
}




// result/args = void/zero
template <typename F>
void BindFunc( std::string name, F func, const result_void&, const args_zero& )
{
	g_Funcs.insert
	(
		std::make_pair
		(
			name,
			[func]( const msgpack::object& args )
			{
				func();
				return std::make_unique<msgpack::object_handle>();
			}
		)
	);
}


// BindFunc with result/args = void/non-zero
template <typename F>
void BindFunc( std::string name, F func, const result_void&, const args_nonzero& )
{
	using args_type = typename func_traits<F>::args_type;
	//std::cout << typeid(args_type).name() << std::endl;

	g_Funcs.insert
	(
		std::make_pair
		(
			name,
			[func]( const msgpack::object& args )
			{
				//int args_count = std::tuple_size<args_type>::value;
				args_type args_real;
				args.convert( args_real );

				std::apply( func, args_real );//Call( func, args_real );

				return std::make_unique<msgpack::object_handle>();
			}
		)
	);
}


// BindFunc with result/args = non-void/zero
template <typename F>
void BindFunc( std::string name, F func, const result_nonvoid&, const args_zero& )
{
	using args_type = typename func_traits<F>::args_type;

	g_Funcs.insert
	(
		std::make_pair
		(
			name,
			[func]( const msgpack::object& args )
			{
				auto z = std::make_unique<msgpack::zone>();
				auto result = msgpack::object( func(), *z );

				return std::make_unique<msgpack::object_handle>( result, std::move( z ) );
			}
		)
	);
}


// BindFunc with result/args = non-void/non-zero
template <typename F>
void BindFunc( std::string name, F func, const result_nonvoid&, const args_nonzero& )
{
	using args_type = typename func_traits<F>::args_type;
	//std::cout << typeid(args_type).name() << std::endl;

	g_Funcs.insert
	(
		std::make_pair
		(
			name,
			[func]( const msgpack::object& args )
			{
				args_type args_real;
				args.convert( args_real );

				auto z = std::make_unique<msgpack::zone>();
				auto result = msgpack::object( std::apply( func, args_real ), *z );

				return std::make_unique<msgpack::object_handle>( result, std::move( z ) );
			}
		)
	);


}




auto dispatch_call( const msgpack::sbuffer& sbuf )
{
	msgpack::unpacked msg;
	msgpack::unpack( msg, sbuf.data(), sbuf.size() );
	auto obj_array = msg.get().via.array;

	auto&& proc_name = (obj_array.ptr[0]).as<std::string>();
	auto&& arg_obj  = obj_array.ptr[1];

	auto it_func = g_Funcs.find( proc_name );

	if( it_func != end( g_Funcs ) )
	{
		auto result = (it_func->second)(arg_obj);
		return result;
		/*
		TODO: パラメータパック展開が必要だけど、どうやって？
		-> dispatcher.cc::dispatch_callに参考実装あり https://github.com/rpclib/rpclib/blob/3b00c4ccf480b9f9569b1d064e7a3b43585b8dfd/lib/rpc/dispatcher.cc

			using call_t = std::tuple<int8_t, uint32_t, std::string, msgpack::object>;// dispatcher.hに定義あり

			call_t the_call;
			msg.convert(the_call);// object::convertメソッドがどこかにある!! objectからtupleに変換している

		*/
	}
	return std::make_unique<msgpack::object_handle>();
}



// プロシージャ名と引数タプルをひとまとめにしたsbufferを作っておくこと!
template <typename... Args>
auto Call( const std::string& proc_name, Args ...args )
{
	auto msg = std::make_tuple( proc_name, std::make_tuple( args... ) );

	msgpack::sbuffer sbuf;
	msgpack::pack( &sbuf, msg );

	return dispatch_call( sbuf );
}




int main()
{
	A a;


	BindFunc( "add_int", [](int a, int b) { return a + b; } );
	BindFunc( "foo", &foo );
	BindFunc( "bar", &bar );
	BindFunc( "baz", &baz );
	BindFunc( "qux", &qux );
	BindFunc( "a.func", [&a]( int v1, int v2, float v3 ) { return a.func( v1, v2, v3 ); } );


	// foo
	{
		Call( "foo" );
	}
	
	// bar
	{
		int a = 666669;
		Call( "bar", a );
	}
	
	// baz
	{
		Call( "baz" );
	}
	
	// qux
	{	
		int a = -362546;
		Call( "qux", a );
	}
	
	// a.func
	{
		Call( "a.func", 1, 2, 3.0f );	
	}
	
	auto result = (*Call( "add_int", 3, 3 ))->as<int>();


	// rpclibのコードの流れ
//	auto result = (it_func->second)(args);// 関数実行して、std::unique_ptr<msgpack::object_handle>型の戻り値を受け取る.
//	return response::make_result(id, std::move(result));// resultの所有権を仮引数に移す.
//	 ↓
//	response::make_result(uint32_t id, std::unique_ptr<RPCLIB_MSGPACK::object_handle> &&r);  なんでこんな実装?
//	 responseはobject_handle型メンバ変数result_を持ってる. object_handleはがあって、引数で持ってきたrを直接割り当てる

// dispatcher::dispatchメソッド.
//  -> responseインスタンスを返す
//  -> server_session::do_read()メソッド内で呼び出される: auto resp = disp_->dispatch(msg, suppress_exceptions_);

// server_session::do_read()メソッド
//  ->write_strand_.post(
//	[=]() { write(resp.get_data()); }); responseの中身をごにょごにょやってる


}