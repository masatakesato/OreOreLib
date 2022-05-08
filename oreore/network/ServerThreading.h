#ifndef SERVER_THREADING_H
#define	SERVER_THREADING_H


#include	<thread>
#include	<WinSock2.h>
#include	<Ws2tcpip.h>// included for InetPton

#include	<msgpack.hpp>

#include	"../memory/Memory.h"
#include	"../common/TString.h"
#include	"../meta/FuncTraits.h"


#include	"SocketException.h"
#include	"Message_Protocol.h"
#include	"Dispatcher.h"



class ServerThreading
{
public:

	ServerThreading();
	virtual ~ServerThreading();

	template <typename F>
	void BindFunc( const tstring& name, F func );

	void Listen( tstring host, int port, int backlog=1 );
	void Run();
	void Close();

	static void Send_Recv( SOCKET sock, void *serializer, std::shared_ptr<Dispatcher>& func );


protected:

	tstring		m_Host;
	int			m_Port;
	int			m_Backlog;
	SOCKET		m_Socket;
	//void*	m_Serializer;// = nullptr;// Serializer( pack_encoding=pack_encoding, unpack_encoding=unpack_encoding ). 後で.
	std::shared_ptr<Dispatcher>	m_Dispatcher;
};



ServerThreading::ServerThreading() :
	m_Host( _T( "localhost" ) ),
	m_Port( 8080 ),
	m_Backlog( 5 ),
	m_Socket()
	// m_Serializer()
{
	m_Dispatcher = std::make_shared<Dispatcher>();
}



ServerThreading::~ServerThreading()
{
	Close();
}



template <typename F>
void ServerThreading::BindFunc( const tstring& name, F func )
{
	m_Dispatcher->BindFunc<F>( name, func );
}



void ServerThreading::Listen( tstring host, int port, int backlog )
{
	try
	{
		m_Host		= host;
		m_Port		= port;
		m_Backlog	= backlog;

		// for windows
		WSADATA data;
		WSAStartup( MAKEWORD( 2, 0 ), &data );

		// initialize socket_addr
		struct sockaddr_in	srcAddr ={};
		srcAddr.sin_port = htons( m_Port );
		srcAddr.sin_family	= AF_INET;
		srcAddr.sin_addr.s_addr	= htonl( INADDR_ANY );
		inet_pton( srcAddr.sin_family, m_Host.c_str(), &srcAddr.sin_addr.S_un.S_addr );

		// Create socket
		BOOL yes = 1;
		m_Socket = socket( AF_INET, SOCK_STREAM, IPPROTO_TCP );
		setsockopt( m_Socket, SOL_SOCKET, SO_REUSEADDR, (const char*)&yes, sizeof( yes ) );

		// Bind socket
		bind( m_Socket, ( struct sockaddr * )&srcAddr, sizeof( srcAddr ) );

		// Listen
		listen( m_Socket, m_Backlog );

		tcout << "Waiting for connection..." << tendl;
	}
	catch( SocketException& e )
	{
		tcout << "Exception caught at ServerThreading::Listen()..." << e.what() << tendl;
	}

}



void ServerThreading::Run()
{
	struct sockaddr_in	dstAddr;
	int dstAddrSize = sizeof( dstAddr );
	SOCKET dstSocket;

	// Wait for connection
	while( true )
	{
		dstSocket = accept( m_Socket, ( struct sockaddr * )&dstAddr, &dstAddrSize );
		tcout << "Established connection." << tendl;

		std::thread( ServerThreading::Send_Recv, dstSocket, nullptr, std::ref(m_Dispatcher) ).detach();
	}

	Close();
	//WSACleanup();
}



void ServerThreading::Close()
{
	closesocket( m_Socket );
	WSACleanup();
}



void ServerThreading::Send_Recv( SOCKET sock, void *serializer, std::shared_ptr<Dispatcher>& func )
{
	int numrcv;
	static OreOreLib::MemoryBase<char, int> raw_message;

	while( true )
	{
		try
		{
			// receive message from client
			numrcv = receive_message( sock, raw_message );
			if( numrcv ==0 || numrcv ==-1 )
				break;

			// do something
			auto oh = func->Dispatch( raw_message );

			// send back result to client
			msgpack::sbuffer sbuf;
			msgpack::pack( &sbuf, oh->get() );
			send_message( sock, sbuf.data(), (int)sbuf.size() );
		}
		catch( const SocketException& e )
		{
			tcout << "SocketException caught at ServerThreading::send_recv()..." << e.what() << tendl;
			break;
		}
		catch( const SendMessageException& e )
		{
			tcout << "SendMessageException caught at ServerThreading::send_recv()..." << e.what() << tendl;
			break;
		}
		catch( const RecvMessageException& e )
		{
			tcout << "RecvMessageException caught at ServerThreading::send_recv()..." << e.what() << tendl;
			break;
		}
		catch( const std::exception& e )
		{
			msgpack::sbuffer sbuf;
			msgpack::pack( &sbuf, e.what() );
			tcout << "Unknown Exception caught at ServerThreading::send_recv()..." << e.what() << tendl;
			send_message( sock, sbuf.data(), (int)sbuf.size() );
		}

	}

	closesocket( sock );

	tcout << "ServerThreading::send_recv exit..." << tendl;
}



#endif // !SERVER_THREADING_H

