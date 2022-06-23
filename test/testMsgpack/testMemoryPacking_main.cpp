//https://stackoverflow.com/questions/11694730/msgpack-c-implementation-how-to-pack-binary-data

//https://stackoverflow.com/questions/16498027/how-to-msgpack-a-user-defined-c-class-with-pod-arrays


#include	<msgpack.hpp>

#include	<oreore/common/Utility.h>
//#include	<oreore/container/ArrayBase.h>
#include	<oreore/container/Array.h>
#include	<oreore/container/StaticArray.h>


//template < typename Packer, typename T > void PackByType( Packer& pk, T& data );


// http://c.msgpack.org/cpp/classmsgpack_1_1packer.html

// Packing

template < typename Packer, typename T >
std::enable_if_t< std::is_same_v<T, int8>, void >		PackByType( Packer& pk, const T& data ){ pk.pack_int8( data ); }

template < typename Packer, typename T >
std::enable_if_t< std::is_same_v<T, uint8>, void >		PackByType( Packer& pk, const T& data ){ pk.pack_uint8( data ); }

template < typename Packer, typename T >
std::enable_if_t< std::is_same_v<T, int16>, void >		PackByType( Packer& pk, const T& data ){ pk.pack_int16( data ); }

template < typename Packer, typename T >
std::enable_if_t< std::is_same_v<T, uint16>, void >		PackByType( Packer& pk, const T& data ){ pk.pack_uint16( data ); }

template < typename Packer, typename T >
std::enable_if_t< std::is_same_v<T, int32>, void >		PackByType( Packer& pk, const T& data ){ pk.pack_int32( data ); }

template < typename Packer, typename T >
std::enable_if_t< std::is_same_v<T, uint32>, void >		PackByType( Packer& pk, const T& data ){ pk.pack_uint32( data ); }

template < typename Packer, typename T >
std::enable_if_t< std::is_same_v<T, int64>, void >		PackByType( Packer& pk, const T& data ){ pk.pack_int64( data ); }

template < typename Packer, typename T >
std::enable_if_t< std::is_same_v<T, uint64>, void >		PackByType( Packer& pk, const T& data ){ pk.pack_uint64( data ); }

template < typename Packer, typename T >
std::enable_if_t< std::is_same_v<T, float32>, void >	PackByType( Packer& pk, const T& data ){ pk.pack_float( data ); }

template < typename Packer, typename T >
std::enable_if_t< std::is_same_v<T, float64>, void >	PackByType( Packer& pk, const T& data ){ pk.pack_double( data ); }


// Unpacking
// using object::as<T>()





struct ArrayMsgpack
{
	char name[256];
	double mat[16];


	template <typename Packer>
	void msgpack_pack( Packer& pk ) const {
		// make array of two elements, by the number of class fields
		pk.pack_array( 2 );

		// pack the first field, strightforward
		pk.pack_bin( sizeof( name ) );
		pk.pack_bin_body( name, sizeof( name ) );

		// since it is array of doubles, we can't use direct conversion or copying
		// memory because it would be a machine-dependent representation of floats
		// instead, we converting this POD array to some msgpack array, like this:
		pk.pack_array( 16 );
		for( int i = 0; i < 16; i++ )
		{
			PackByType( pk, mat[i] );//pk.pack_double( mat[i] );//
		}
	}


	// this function is looks like de-serializer, taking an msgpack object
	 // and extracting data from it to the current class fields
	void msgpack_unpack( msgpack::object o ) {
		// check if received structure is an array
		if( o.type != msgpack::type::ARRAY ) { throw msgpack::type_error(); }

		const size_t size = o.via.array.size;

		// sanity check
		if( size <= 0 ) return;
		// extract value of first array entry to a class field
		memcpy( name, o.via.array.ptr[0].via.bin.ptr, o.via.array.ptr[0].via.bin.size );

		// sanity check
		if( size <= 1 ) return;
		// extract value of second array entry which is array itself:
		for( int i = 0; i < 16; i++ )
			mat[i] = o.via.array.ptr[1].via.array.ptr[i].as<double>();//mat[i] = o.via.array.ptr[1].via.array.ptr[i].via.f64;
	}

	// destination of this function is unknown - i've never ran into scenary
	// what it was called. some explaination/documentation needed.
	template <typename MSGPACK_OBJECT>
	void msgpack_object( MSGPACK_OBJECT* o, msgpack::zone* z ) const {

	}

};



template < typename T >
struct CArrayMsgpack
{
	T mat[16];

	//	T* m_refData; = nullptr;
	//	size_t m__Size;


	template <typename Packer>
	void msgpack_pack( Packer& pk ) const
	{
		// since it is array of doubles, we can't use direct conversion or copying
		// memory because it would be a machine-dependent representation of floats
		// instead, we converting this POD array to some msgpack array, like this:
		pk.pack_array( 16 );
		for( int i = 0; i < 16; i++ )
			PackByType( pk, mat[i] );//pk.pack_double( mat[i] );//
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
		for( int i = 0; i<16; i++ )
			mat[i] = o.via.array.ptr[i].as<T>();
	}

	// destination of this function is unknown - i've never ran into scenary
	// what it was called. some explaination/documentation needed.
	template <typename MSGPACK_OBJECT>
	void msgpack_object( MSGPACK_OBJECT* o, msgpack::zone* z ) const
	{

	}

};




// DynamicArrayクラスを継承してメッセージパック対応させた
template< typename T, typename IndexType >
class ArrayMsgpack_ : public OreOreLib::ArrayImpl<T, IndexType>
{
public:

	template <typename Packer>
	void msgpack_pack( Packer& pk ) const
	{
		if( this->Empty() )
			throw msgpack::parse_error("parse error");

		// since it is array of doubles, we can't use direct conversion or copying
		// memory because it would be a machine-dependent representation of floats
		// instead, we converting this POD array to some msgpack array, like this:
		pk.pack_array( this->m_Length );
		for( int i=0; i<this->Length<int>(); ++i )
			PackByType( pk, this->m_pData[i] );
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
		this->Init( size );

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






int main()
{
	{
		ArrayMsgpack  aaa;
		for( auto& c : aaa.name )	c = 'g';
		for( auto& v : aaa.mat )	v = -6.666;

		// pack
		msgpack::sbuffer sbuf;
		msgpack::pack( &sbuf, aaa );// dataがバイト配列だと終端記号が消える.

		// unpack
		msgpack::object_handle oh = msgpack::unpack( sbuf.data(), sbuf.size() );
		auto&& result = oh->as<ArrayMsgpack>();
	}

	{
		CArrayMsgpack<double>  aaa;
		for( auto& v : aaa.mat )	v = -9.999;

		// pack
		msgpack::sbuffer sbuf;
		msgpack::pack( &sbuf, aaa );// dataがバイト配列だと終端記号が消える.

		// unpack
		msgpack::object_handle oh = msgpack::unpack( sbuf.data(), sbuf.size() );
		auto&& result = oh->as<CArrayMsgpack<double>>();
	}



	{
		ArrayMsgpack_<double, uint32>	aaa;
		aaa.Init( 16 );
		for( auto& v : aaa )	v = -9.999;


		// pack
		msgpack::sbuffer sbuf;
		msgpack::pack( &sbuf, aaa );// dataがバイト配列だと終端記号が消える.

		// unpack
		msgpack::object_handle oh = msgpack::unpack( sbuf.data(), sbuf.size() );
		auto&& result = oh->as<ArrayMsgpack_<double, uint32>>();


	}

	return 0;
}