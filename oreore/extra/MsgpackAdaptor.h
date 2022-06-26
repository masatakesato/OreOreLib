#ifndef MSGPACK_ADAPTOR_H
#define	MSGPACK_ADAPTOR_H

#include	<msgpack.hpp>

#include	"../common/Utility.h"
#include	"../container/Array.h"
#include	"../container/StaticArray.h"




namespace OreOreExtra
{
	// http://c.msgpack.org/cpp/classmsgpack_1_1packer.html


	// Packing
	namespace detail
	{
		template < typename Packer, typename T >
		inline static std::enable_if_t< std::is_same_v<T, int8>, void >		PackByType( Packer& pk, const T& data ){ pk.pack_int8( data ); }

		template < typename Packer, typename T >
		inline static std::enable_if_t< std::is_same_v<T, uint8>, void >	PackByType( Packer& pk, const T& data ){ pk.pack_uint8( data ); }

		template < typename Packer, typename T >
		inline static std::enable_if_t< std::is_same_v<T, int16>, void >	PackByType( Packer& pk, const T& data ){ pk.pack_int16( data ); }

		template < typename Packer, typename T >
		inline static std::enable_if_t< std::is_same_v<T, uint16>, void >	PackByType( Packer& pk, const T& data ){ pk.pack_uint16( data ); }

		template < typename Packer, typename T >
		inline static std::enable_if_t< std::is_same_v<T, int32>, void >	PackByType( Packer& pk, const T& data ){ pk.pack_int32( data ); }

		template < typename Packer, typename T >
		inline static std::enable_if_t< std::is_same_v<T, uint32>, void >	PackByType( Packer& pk, const T& data ){ pk.pack_uint32( data ); }

		template < typename Packer, typename T >
		inline static std::enable_if_t< std::is_same_v<T, int64>, void >	PackByType( Packer& pk, const T& data ){ pk.pack_int64( data ); }

		template < typename Packer, typename T >
		inline static std::enable_if_t< std::is_same_v<T, uint64>, void >	PackByType( Packer& pk, const T& data ){ pk.pack_uint64( data ); }

		template < typename Packer, typename T >
		inline static std::enable_if_t< std::is_same_v<T, float32>, void >	PackByType( Packer& pk, const T& data ){ pk.pack_float( data ); }

		template < typename Packer, typename T >
		inline static std::enable_if_t< std::is_same_v<T, float64>, void >	PackByType( Packer& pk, const T& data ){ pk.pack_double( data ); }

	}


	// DynamicArrayクラスを継承してメッセージパック対応させた
	template< typename T, typename IndexType >
	class ArrayMsgpkImpl : public OreOreLib::ArrayImpl<T, IndexType>
	{
	public:

		using OreOreLib::ArrayImpl<T, IndexType>::ArrayImpl;


		template <typename Packer>
		void msgpack_pack( Packer& pk ) const
		{
			if( this->Empty() )
				throw msgpack::parse_error( "parse error" );

			// since it is array of doubles, we can't use direct conversion or copying
			// memory because it would be a machine-dependent representation of floats
			// instead, we converting this POD array to some msgpack array, like this:
			pk.pack_array( this->m_Length );
			for( int i=0; i<this->Length<int>(); ++i )
				detail::PackByType( pk, this->m_pData[i] );
		}


		// this function is looks like de-serializer, taking an msgpack object
		 // and extracting data from it to the current class fields
		void msgpack_unpack( msgpack::object o )
		{
			// check if received structure is an array
			if( o.type != msgpack::type::ARRAY )
				throw msgpack::type_error();

			const size_t size = o.via.array.size;

			// sanity check
			if( size <= 0 ) return;
			this->Init( static_cast<IndexType>( size ) );

			// extract value of second array entry which is array itself:
			for( int i=0; i<this->Length<int>(); ++i )
				this->m_pData[i] = o.via.array.ptr[i].as<T>();
		}

		// destination of this function is unknown - i've never ran into scenary
		// what it was called. some explaination/documentation needed.
		template <typename MSGPACK_OBJECT>
		void msgpack_object( MSGPACK_OBJECT* o, msgpack::zone* z ) const
		{

		}

	};



	template< typename T, sizeType Size, typename IndexType >
	class StaticArrayMsgpkImpl : public OreOreLib::StaticArrayImpl<T, Size, IndexType>
	{
	public:

		using OreOreLib::StaticArrayImpl<T, Size, IndexType>::StaticArrayImpl;

		template <typename Packer>
		void msgpack_pack( Packer& pk ) const
		{
			if( this->Empty() )
				throw msgpack::parse_error( "parse error" );

			// since it is array of doubles, we can't use direct conversion or copying
			// memory because it would be a machine-dependent representation of floats
			// instead, we converting this POD array to some msgpack array, like this:
			pk.pack_array( this->m_Length );
			for( int i=0; i<this->Length<int>(); ++i )
				detail::PackByType( pk, this->m_pData[i] );
		}


		// this function is looks like de-serializer, taking an msgpack object
		 // and extracting data from it to the current class fields
		void msgpack_unpack( msgpack::object o )
		{
			// check if received structure is an array
			if( o.type != msgpack::type::ARRAY )
				throw msgpack::type_error();

			const size_t size = o.via.array.size;

			// sanity check
			if( size <= 0 ) return;

			// extract value of second array entry which is array itself:
			for( int i=0; i<this->Length<int>(); ++i )
				this->m_pData[i] = o.via.array.ptr[i].as<T>();
		}

		// destination of this function is unknown - i've never ran into scenary
		// what it was called. some explaination/documentation needed.
		template <typename MSGPACK_OBJECT>
		void msgpack_object( MSGPACK_OBJECT* o, msgpack::zone* z ) const
		{

		}

	};



	// Dynamic array for msgpack
	template< typename T >
	using ArrayMsgpk = ArrayMsgpkImpl< T, OreOreLib::MemSizeType >;

	// Static array
	template< typename T, sizeType Size >
	using StaticArrayMsgpk = StaticArrayMsgpkImpl< T, Size, OreOreLib::MemSizeType >;



}// end of namespace OreOreLib


#endif // !MSGPACK_ADAPTOR_H
