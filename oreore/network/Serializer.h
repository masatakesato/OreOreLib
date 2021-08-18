#ifndef SERIALIZER_H
#define	SERIALIZER_H

#include	<msgpack.hpp>

#include	"../common/TString.h"


//class PackError(Exception): pass
//class UnpackError(Exception): pass



class Serializer
{
public:

	Serializer(){}
	//Serializer( pack_encoding, unpack_encoding );
	virtual ~Serializer(){}

	const char* Pack( void* data, size_t data_size );

	template< typename T >
	T Unpack( const char* data, size_t data_size );


protected:


};


/*
Serializer::Serializer( pack_encoding, unpack_encoding )
{
	//self.__m_Encode = encode
	//m_Pakcer = msgpack.Packer(use_bin_type=True)# encoding=pack_encoding )
	//m_Unpackcer = msgpack.Unpacker(raw=False)# encoding=unpack_encoding )
}
*/


const char* Serializer::Pack( void* data, size_t data_size )
{
	try
	{
		// https://github.com/msgpack/msgpack-c/wiki/v2_0_cpp_packer
		msgpack::sbuffer sbuf;
		msgpack::packer<msgpack::sbuffer> packer(sbuf);
		packer.pack_bin( data_size ); 
		packer.pack_bin_body( (const char *)data, data_size );

		//msgpack::pack( sbuf, (const char *)data );
		tcout << sbuf.data() << tendl;
		tcout << sbuf.size() << tendl;
		return sbuf.data();
	}
	catch( const std::exception& e )
	{
		tcout << "Exception occured at Serializer::Pack" << tendl;
		//raise PackError( traceback.format_exc() )
		return nullptr;
	}
}


template< typename T >
T Serializer::Unpack( const char* data, size_t data_size )// byte array data
{
	try
	{
		// https://stackoverflow.com/questions/42837630/read-binary-file-using-msgpack-in-c
		// https://qiita.com/m_mizutani/items/c40295549c3368a4257d
		msgpack::unpacker pac;
		pac.reserve_buffer( data_size );
		memcpy( pac.buffer(), data, data_size );
		pac.buffer_consumed( data_size );

		char *result = new char[data_size];

		msgpack::object_handle oh;
		while ( pac.next(oh) )
		{
			msgpack::object msg = oh.get();
			//msg.convert(&result[0]);
			std::cout << msg << std::endl;
		}
		//msgpack::unpacked msg;
		//msgpack::unpack( msg, data, data_size );

		//msgpack::object obj = msg.get();
		//tcout << obj << tendl;
		return T();
	}
	catch( const std::bad_cast& e )
	{
		tcout << "Exception occured at Serializer::Unpack" << tendl;
		//raise UnpackError( traceback.format_exc() )
		return T();
	}
}




#endif // !SERIALIZER_H
