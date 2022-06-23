#include	<iostream>
#include	<msgpack.hpp>



// https://gist.github.com/mashiro/5150508
// zeromqとmsgpackの組み合わせ
// 


int main()
{
	char data[] = "abcd";
	
	std::cout << "//============= Original Data =============//" << std::endl;
	std::cout << "size: " << sizeof( data ) << std::endl;
	std::cout << "data: " << data << std::endl;



	std::cout << "//============= Serializing Data =============//" << std::endl;

	// pack data using msgpack::pack
	//msgpack::sbuffer sbuf;
	//msgpack::pack(&sbuf, data);// dataがバイト配列だと終端記号が消える.
	//std::cout << "serialized with msgpack::pack..." << std::endl;

	// pack data using packer
	msgpack::sbuffer sbuf;
	msgpack::packer<msgpack::sbuffer> packer(&sbuf);
	packer.pack_bin(sizeof(data));
	packer.pack_bin_body(data, sizeof(data));
	std::cout << "serialized with msgpack::packer..." << std::endl;


	// send message
	auto send_size = sbuf.size();
	char* send_message = new char[ send_size + sizeof send_size ];
	std::memcpy( send_message,reinterpret_cast<const char*>(&send_size), sizeof send_size );//std::copy( reinterpret_cast<const char*>(&send_size), reinterpret_cast<const char*>(&send_size) + sizeof send_size, send_data );
	std::memcpy( send_message + sizeof send_size, sbuf.data(), sbuf.size() );//std::copy( sbuf.data(), sbuf.data() + sbuf.size(), send_data + sizeof send_size );
	



	std::cout << "//============= Deserializing Data =============//" << std::endl;

	// receive mesage 
	auto recv_size = *reinterpret_cast<size_t*>(send_message);
	char* recv_message = send_message + sizeof recv_size;

	
	//// unpack using msgpack::unpacker
	//msgpack::unpacker unpacker;
	//unpacker.reserve_buffer( recv_size );
	//memcpy( unpacker.buffer(), recv_message, recv_size );
	//unpacker.buffer_consumed( recv_size );
	//
	//int unpacked_size = 0;
	//char *unpacked_data = nullptr;

	//msgpack::object_handle oh;
	//while( unpacker.next(oh) )
	//{
	//	msgpack::object obj = oh.get();
	//	auto data_size = obj.via.bin.size;
	//	auto data = obj.via.bin.ptr;
	//	unpacked_size = data_size;
	//	unpacked_data = const_cast<char *>(data);//new char[data_size];
	//	//std::memcpy( unpacked_data, data, data_size );
	//	break;
	//}

	//std::cout << "size: " << unpacked_size << std::endl;
	//std::cout << "data: " << unpacked_data << std::endl;


	// unpack using msgpack::unpack
	msgpack::object_handle oh = msgpack::unpack( recv_message, recv_size );
	msgpack::object obj = oh.get();

	auto data_size = obj.via.bin.size;
	auto pdata = obj.via.bin.ptr;
	char *unpacked_data = const_cast<char *>(pdata);//new char[data_size];
	//memcpy(unpacked_data, pdata, data_size );

	std::cout << "size: " << data_size << std::endl;
	std::cout << "data: " << unpacked_data << std::endl;


	return 0;
}
