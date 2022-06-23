#include	<iostream>
#include	<tuple>
#include	<msgpack.hpp>




int main()
{
	auto name = std::string( "name" );
	auto args = std::make_tuple( 1.0, 2, std::string("True") );

	msgpack::sbuffer sbuf_args;
	msgpack::pack( &sbuf_args, args );



	auto data = std::make_tuple( name, args );



	msgpack::sbuffer sbuf;
	msgpack::pack( &sbuf, data );

	msgpack::object_handle oh = msgpack::unpack( sbuf.data(), sbuf.size() );
	msgpack::object obj = oh.get();

	decltype(data) result;
	obj.convert( result );


}