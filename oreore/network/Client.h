#ifndef CLIENT_H
#define	CLIENT_H


#include	<WinSock2.h>
#include	<Ws2tcpip.h>// included for InetPton

#include	<msgpack.hpp>

#include	"../memory/Memory.h"
#include	"../common/TString.h"
#include	"../meta/FuncTraits.h"


#include	"SocketException.h"
#include	"Message_Protocol.h"




class Client
{

public:

	Client();
	Client( const charstring& host, int port, int timeout, int trial );
	virtual ~Client();

	template < typename... Args >
	msgpack::object_handle Call( const charstring& proc_name, Args ...args );
	bool IsReady();
	void Close();


private:

	charstring	__m_host;
	int			__m_port;

	int			__m_timeout;// miliseconds.
	int			__m_trial;
	//void*	__m_Serializer;
	SOCKET		__m_socket;

	OreOreLib::MemoryBase<char, int>	buffer;


	static SOCKET make_connection( const charstring& host_, int port_, int timeout_, int trial_ );

};



Client::Client()
	: __m_host( "localhost" )
	, __m_port( 8080 )
	, __m_timeout( 1000 )
	, __m_trial( 5 )
{
	__m_socket = make_connection( __m_host, __m_port, __m_timeout, __m_trial );
}



Client::Client( const charstring& host, int port, int timeout, int trial )
{
	__m_host = host;
	__m_port = port;
	__m_timeout = timeout;
	__m_trial = trial;

	//__m_Serializer = Serializer( pack_encoding=pack_encoding, unpack_encoding=unpack_encoding )

	__m_socket = make_connection( __m_host, __m_port, __m_timeout, __m_trial );
	tcout << "!!" << tendl;
}



Client::~Client()
{
	Close();
}



template <typename... Args >
msgpack::object_handle Client::Call( const charstring& proc_name, Args ...args )
{
	tcout << _T("client.call...") << tendl; 
	//print( '    args: ', args )
	//print( '    kwargs: ', kwargs )
	int trial = 0;
	int numrcv;
	msgpack::object_handle oh;

	while( true )
	{
		try
		{
			// serialize data
			auto msg = std::make_tuple( proc_name, std::make_tuple( args... ) );
			//for_each_tuple( args_, []( auto it ){ std::cout << it << std::endl; } );
			auto sbuf = std::make_shared<msgpack::sbuffer>();// 別スレッドにstd::moveすることを想定して実装.
			msgpack::pack( *sbuf, msg );

			// send message to server
			send_message( __m_socket, sbuf->data(), (int)sbuf->size() );// buffer.c_str(), buffer.length() );


			// receive data from server
			numrcv = receive_message( __m_socket, buffer );
			if( numrcv ==0 || numrcv ==-1 )
			{
				tcout << "Client::Call()... received data is None!" << tendl;
				break;
			}

			// deserialize and return
			oh = msgpack::unpack( buffer.begin(), buffer.Length() );
			return oh;//__m_Serializer.Unpack( recv_data );
		}
		catch( const SendMessageException& e )
		{
			tcout << "SendMessageException caught at Client::Call()..." << e.what() << tendl;
			// tcout << "Error at Client::call()..." << e.what() << "\n";
			trial++;

			if( trial >= __m_trial )
				break;
			tcout << "   trying to reconnect[" << trial << "]" << tendl;
			__m_socket = make_connection( __m_host, __m_port, __m_timeout, __m_trial );// retry connection
		}
		catch( const RecvMessageException& e )
		{
			tcout << "RecvMessageException caught at Client::Call()..." << e.what() << tendl;
			break;
		}
		catch( const std::exception& e )//catch( ... )
		{
			tcout << "Unknown Exception caught at Client::Call()..." << e.what() << tendl;
			break;
		}

	}
	
	return oh;//nullptr;
}



bool Client::IsReady()
{
	try
	{
		//send_data = __m_Serializer.Pack( ( "echo", (), {} ) );
		//send_message( __m_socket, send_data )
		// tcout << "connection is active.\n";
		return true;
	}
	catch( ... )
	{
		tcout << "connection is NOT active." << tendl;
		return true;
	}
}



SOCKET Client::make_connection( const charstring& host_, int port_, int timeout_, int trial_ )
{
	try
	{
		WSADATA data;
		WSAStartup( MAKEWORD( 2, 0 ), &data );

		// initialize socket_addr
		struct sockaddr_in dstAddr = {};
		memset( &dstAddr, 0, sizeof( dstAddr ) );
		dstAddr.sin_port = htons( port_ );
		dstAddr.sin_family = AF_INET;
		inet_pton( dstAddr.sin_family, host_.c_str(), &dstAddr.sin_addr.S_un.S_addr );//dstAddr.sin_addr.s_addr = inet_addr( destination );

		// create socket
		BOOL yes = 1;
		SOCKET dstSocket = socket( AF_INET, SOCK_STREAM, IPPROTO_TCP );
		setsockopt( dstSocket, SOL_SOCKET, SO_REUSEADDR, (const char*)&yes, sizeof(yes) );

		// set timeout
		// https://stackoverflow.com/questions/1824465/set-timeout-for-winsock-recvfrom
		setsockopt( dstSocket, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char*>(&timeout_), sizeof timeout_ );
		setsockopt( dstSocket, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&timeout_), sizeof timeout_ );


		tcout << "connecting...\n";
		if( connect( dstSocket, ( struct sockaddr * ) &dstAddr, sizeof( dstAddr ) ) )
		{
			tcout << "failed to connect to " << host_.c_str() << tendl;
			throw SocketException();
			//return( -1 );
		}

		tcout << "Client()::make_connection...connected" << tendl;

		return dstSocket;

		//sock = socket.socket( socket.AF_INET, socket.SOCK_STREAM );
		//sock.settimeout( timeout_ );
		//sock.setsockopt( socket.SOL_SOCKET, socket.SO_REUSEADDR, 1 );
		//sock.connect( (host_, port_) );
		//tcout << "Client()::make_connection...connected\n";
		//return sock;
	}
	catch( const SocketException& e )
	{
		tcout << "Error at Client::make_connection... " << e.what() << tendl;
		//Sleep(1000);
		return -1;
	}
}


void Client::Close()
{
	closesocket( __m_socket );
	WSACleanup();
}




#endif // !CLIENT_H