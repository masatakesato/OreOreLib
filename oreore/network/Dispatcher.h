#ifndef DISPATCHER_H
#define	DISPATCHER_H


#include	<functional>
#include	<unordered_map>

#include	"../common/TString.h"
#include	"../memory/Memory.h"
#include	"../meta/FuncTraits.h"

#include	<msgpack.hpp>




class Dispatcher
{
public:
	
	template < typename F >
	std::enable_if_t<
		std::is_same_v< typename func_kind_info<F>::result_type, result_void > &&
		std::is_same_v< typename func_kind_info<F>::args_type, args_zero >,
		void >
	BindFunc( const charstring& name, F func )
	{
		m_Funcs.insert
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
	template < typename F >
	std::enable_if_t<
		std::is_same_v< typename func_kind_info<F>::result_type, result_void > &&
		std::is_same_v< typename func_kind_info<F>::args_type, args_nonzero >,
		void >
	BindFunc( const charstring& name, F func )
	{
		using args_type = typename func_traits<F>::args_type;

		m_Funcs.insert
		(
			std::make_pair
			(
				name,
				[func]( const msgpack::object& args )
				{
					//int args_count = std::tuple_size<args_type>::value;
					// Extract arguments tuple
					args_type args_real;
					args.convert( args_real );
					std::apply( func, args_real );//Call( func, args_real );

					return std::make_unique<msgpack::object_handle>();
				}
			)
		);
	}


	// BindFunc with result/args = non-void/zero
	template < typename F >
	std::enable_if_t<
		std::is_same_v< typename func_kind_info<F>::result_type, result_nonvoid > &&
		std::is_same_v< typename func_kind_info<F>::args_type, args_zero >,
		void >
	BindFunc( const charstring& name, F func )
	{
		using args_type = typename func_traits<F>::args_type;

		m_Funcs.insert
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
	template < typename F >
	std::enable_if_t<
		std::is_same_v< typename func_kind_info<F>::result_type, result_nonvoid > &&
		std::is_same_v< typename func_kind_info<F>::args_type, args_nonzero >,
		void >
	BindFunc( const charstring& name, F func )
	{
		using args_type = typename func_traits<F>::args_type;

		m_Funcs.insert
		(
			std::make_pair
			(
				name,
				[func]( const msgpack::object& args )
				{
					// Extract arguments tuple
					args_type args_real;
					args.convert( args_real );

					auto z = std::make_unique<msgpack::zone>();
					auto result = msgpack::object( std::apply( func, args_real ), *z );

					return std::make_unique<msgpack::object_handle>( result, std::move( z ) );
				}
			)
		);
	}


	// Old BindFunc implementation. deprecated. 2022.05.04
	//template < typename F >
	//void BindFunc( const charstring& name, F func )
	//{
	//	//BindFunc<F>( name, func, func_kind_info<F>::result_type(), func_kind_info<F>::args_type() );
	//	BindFunc<F>( name, func );
	//}



	template < typename IndexType >
	auto Dispatch( OreOreLib::MemoryBase<char, IndexType>& sbuf )//const msgpack::sbuffer& sbuf )
	{
		//msgpack::unpacked msg;
		//msgpack::unpack( msg, sbuf.begin(), sbuf.Length() );//sbuf.data(), sbuf.size() );
		//auto obj_array = msg.get().via.array;
		auto oh = msgpack::unpack( sbuf.begin(), sbuf.Length() );
		auto obj_array = (*oh).via.array;

		auto&& proc_name = (obj_array.ptr[0]).as<charstring>();
		auto&& arg_obj  = obj_array.ptr[1];

		auto it_func = m_Funcs.find( proc_name );

		if( it_func != end( m_Funcs ) )
		{
			auto result = (it_func->second)(arg_obj);
			return result;
		}
		return std::make_unique<msgpack::object_handle>();
	}
	


private:

	std::unordered_map< charstring, std::function< std::unique_ptr<msgpack::object_handle>( const msgpack::object& ) > >	m_Funcs;


	// Old BindFunc implementation. deprecated. 2022.05.04


	//// result/args = void/zero
	//template < typename F >
	//void BindFunc( const charstring& name, F func, const result_void&, const args_zero& );

	//// BindFunc with result/args = void/non-zero
	//template < typename F >
	//void BindFunc( const charstring& name, F func, const result_void&, const args_nonzero& );

	//// BindFunc with result/args = non-void/zero
	//template < typename F >
	//void BindFunc( const charstring& name, F func, const result_nonvoid&, const args_zero& );

	//// BindFunc with result/args = non-void/non-zero
	//template < typename F >
	//void BindFunc( const charstring& name, F func, const result_nonvoid&, const args_nonzero& );


};



// Old BindFunc implementation. deprecated. 2022.05.04

//// result/args = void/zero
//template < typename F >
//void Dispatcher::BindFunc( const charstring& name, F func, const result_void&, const args_zero& )
//{
//	m_Funcs.insert
//	(
//		std::make_pair
//		(
//			name,
//			[func]( const msgpack::object& args )
//			{
//				func();
//				return std::make_unique<msgpack::object_handle>();
//			}
//		)
//	);
//}
//
//
//// BindFunc with result/args = void/non-zero
//template < typename F >
//void Dispatcher::BindFunc( const charstring& name, F func, const result_void&, const args_nonzero& )
//{
//	using args_type = typename func_traits<F>::args_type;
//	//std::cout << typeid(args_type).name() << std::endl;
//
//	m_Funcs.insert
//	(
//		std::make_pair
//		(
//			name,
//			[func]( const msgpack::object& args )
//			{
//				//int args_count = std::tuple_size<args_type>::value;
//				// Extract arguments tuple
//				args_type args_real;
//				args.convert( args_real );
//				std::apply( func, args_real );//Call( func, args_real );
//
//				return std::make_unique<msgpack::object_handle>();
//			}
//		)
//	);
//}
//
//
//// BindFunc with result/args = non-void/zero
//template < typename F >
//void Dispatcher::BindFunc( const charstring& name, F func, const result_nonvoid&, const args_zero& )
//{
//	using args_type = typename func_traits<F>::args_type;
//
//	m_Funcs.insert
//	(
//		std::make_pair
//		(
//			name,
//			[func]( const msgpack::object& args )
//			{
//				auto z = std::make_unique<msgpack::zone>();
//				auto result = msgpack::object( func(), *z );
//
//				return std::make_unique<msgpack::object_handle>( result, std::move( z ) );
//			}
//		)
//	);
//}
//
//
//// BindFunc with result/args = non-void/non-zero
//template <typename F>
//void Dispatcher::BindFunc( const charstring& name, F func, const result_nonvoid&, const args_nonzero& )
//{
//	using args_type = typename func_traits<F>::args_type;
//	//std::cout << typeid(args_type).name() << std::endl;
//
//	m_Funcs.insert
//	(
//		std::make_pair
//		(
//			name,
//			[func]( const msgpack::object& args )
//			{
//				// Extract arguments tuple
//				args_type args_real;
//				args.convert( args_real );
//
//				auto z = std::make_unique<msgpack::zone>();
//				auto result = msgpack::object( std::apply( func, args_real ), *z );
//
//				return std::make_unique<msgpack::object_handle>( result, std::move( z ) );
//			}
//		)
//	);
//
//
//}





#endif // !DISPATCHER_H
