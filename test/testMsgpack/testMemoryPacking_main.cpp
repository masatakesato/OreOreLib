//https://stackoverflow.com/questions/11694730/msgpack-c-implementation-how-to-pack-binary-data

//https://stackoverflow.com/questions/16498027/how-to-msgpack-a-user-defined-c-class-with-pod-arrays


#include	<msgpack.hpp>

#include	<oreore/extra/MsgpackAdaptor.h>




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
			OreOreExtra::detail::PackByType( pk, mat[i] );//pk.pack_double( mat[i] );//
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




int main()
{

	{
		std::vector<int> aaa = { 1, 2, 3 };

		msgpack::sbuffer sbuf;
		msgpack::pack( &sbuf, aaa );
	}

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
		OreOreExtra::ArrayMsgpkImpl<double, uint32>	aaa;
		aaa.Init( 16 );
		for( auto& v : aaa )	v = -9.999;


		// pack
		msgpack::sbuffer sbuf;
		msgpack::pack( &sbuf, aaa );// dataがバイト配列だと終端記号が消える.

		// unpack
		msgpack::object_handle oh = msgpack::unpack( sbuf.data(), sbuf.size() );
		auto&& result = oh->as<OreOreExtra::ArrayMsgpkImpl<double, uint32>>();


	}

	return 0;
}