#ifndef SERVER_H
#define	SERVER_H


#include	<WinSock2.h>
#include	<Ws2tcpip.h>// included for InetPton

#include	<msgpack.hpp>

#include	"../memory/Memory.h"
#include	"../common/TString.h"
#include	"../meta/FuncTraits.h"


#include	"SocketException.h"
#include	"Message_Protocol.h"
#include	"Dispatcher.h"



class Server
{
public:
	
	Server();// proc=EchoServer(), pack_encoding=None, unpack_encoding=None );
	virtual ~Server();

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



Server::Server() :
	m_Host(_T("localhost")),
	m_Port(8080),
	m_Backlog(5),
	m_Socket()
	// m_Serializer()
{
	m_Dispatcher = std::make_shared<Dispatcher>();
}



Server::~Server()
{
	Close();
}



template <typename F>
void Server::BindFunc( const tstring& name, F func )
{
	m_Dispatcher->BindFunc<F>( name, func );
}



void Server::Listen( tstring host, int port, int backlog )
{
	try
	{
		m_Host	= host;
		m_Port	= port;
		m_Backlog = backlog;

		// for windows
		WSADATA data;
		WSAStartup( MAKEWORD(2, 0), &data );

		// initialize socket_addr
		struct sockaddr_in	srcAddr = {};
		srcAddr.sin_port = htons( m_Port );
		srcAddr.sin_family	= AF_INET;
		srcAddr.sin_addr.s_addr	= htonl( INADDR_ANY );
		inet_pton( srcAddr.sin_family, m_Host.c_str(), &srcAddr.sin_addr.S_un.S_addr );

		// Create socket
		BOOL yes = 1;
		m_Socket = socket( AF_INET, SOCK_STREAM, IPPROTO_TCP );
		setsockopt( m_Socket, SOL_SOCKET, SO_REUSEADDR, (const char*)&yes, sizeof(yes) );

		// Bind socket
		bind( m_Socket, (struct sockaddr *)&srcAddr, sizeof( srcAddr ) );

		// Listen
		listen( m_Socket, m_Backlog );

		tcout << "Waiting for connection..." << tendl;
	}
	catch( SocketException& e )
	{
		tcout << "Exception caught at Server::Listen()..." << e.what() << tendl;
	}

}



void Server::Run()
{
	struct sockaddr_in	dstAddr;
	int dstAddrSize = sizeof( dstAddr );
	SOCKET dstSocket;

	// Wait for connection
	while( true )
	{
		dstSocket = accept( m_Socket, (struct sockaddr *)&dstAddr, &dstAddrSize );
		tcout << "Established connection." << tendl;

		Send_Recv( dstSocket, /*m_Serializer*/nullptr, m_Dispatcher );
	}

	Close();
	//WSACleanup();
}



void Server::Close()
{
	closesocket( m_Socket );
	WSACleanup();
}



void Server::Send_Recv( SOCKET sock, void *serializer, std::shared_ptr<Dispatcher>& func )
{
	int numrcv;
	static OreOreLib::Memory<char, int> raw_message;

	while(true)
	{
		try
		{
			// recieve message from client
			numrcv = recieve_message( sock, raw_message );
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
			tcout << "SocketException caught at Server::send_recv()..." << e.what() << tendl;
			break;
		}
		catch( const SendMessageException& e )
		{
			tcout << "SendMessageException caught at Server::send_recv()..." << e.what() << tendl;
			break;
		}
		catch( const RecvMessageException& e )
		{
			tcout << "RecvMessageException caught at Server::send_recv()..." << e.what() << tendl;
			break;
		}
		catch( const std::exception& e )
		{
			msgpack::sbuffer sbuf;
			msgpack::pack( &sbuf, e.what() );
			tcout << "Unknown Exception caught at Server::send_recv()..." << e.what() << tendl;
			send_message( sock, sbuf.data(), (int)sbuf.size() );
		}

	}

	closesocket( sock );
	
	tcout << "Server::send_recv exit..." << tendl;
}



#endif // !SERVER_H