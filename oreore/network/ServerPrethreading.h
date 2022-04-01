#ifndef SERVER_PRETHREADING_H
#define	SERVER_PRETHREADING_H


#include	<thread>
#include	<mutex>
#include	<WinSock2.h>
#include	<Ws2tcpip.h>// included for InetPton

#include	<msgpack.hpp>

#include	"../memory/Memory.h"
#include	"../common/TString.h"
#include	"../meta/FuncTraits.h"


#include	"SocketException.h"
#include	"Message_Protocol.h"
#include	"Dispatcher.h"




template < int N >
class ServerPrethreading
{
public:

	ServerPrethreading();
	virtual ~ServerPrethreading();

	template <typename F>
	void BindFunc( const tstring& name, F func );

	void Listen( tstring host, int port, int backlog=1 );
	void Run();
	void Accept( SOCKET sock );
	void Close();

	static void Send_Recv( SOCKET sock, void *serializer, std::shared_ptr<Dispatcher>& func );


protected:

	tstring		m_Host;
	int			m_Port;
	int			m_Backlog;
	SOCKET		m_Socket;
	//void*	m_Serializer;// = nullptr;// Serializer( pack_encoding=pack_encoding, unpack_encoding=unpack_encoding ). 後で.
	std::shared_ptr<Dispatcher>	m_Dispatcher;

	std::thread	m_ThreadList[N];
	std::mutex	m_Lock;
};



template <int N>
ServerPrethreading<N>::ServerPrethreading() :
	m_Host( _T( "localhost" ) ),
	m_Port( 8080 ),
	m_Backlog( 5 ),
	m_Socket()
	// m_Serializer()
{
	m_Dispatcher = std::make_shared<Dispatcher>();
}



template <int N>
ServerPrethreading<N>::~ServerPrethreading()
{
	Close();
}



template <int N>
template <typename F> void ServerPrethreading<N>::BindFunc( const tstring& name, F func )
{
	m_Dispatcher->BindFunc<F>( name, func );
}



template <int N>
void ServerPrethreading<N>::Listen( tstring host, int port, int backlog )
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
		tcout << "Exception caught at ServerPrethreading::Listen()..." << e.what() << tendl;
	}

}



template <int N>
void ServerPrethreading<N>::Run()
{
	for( auto& th : m_ThreadList )
	{
		th = std::thread( &ServerPrethreading::Accept, this, m_Socket );
	}

	for( auto& th : m_ThreadList )
	{
		th.join();
	}

	Close();
	//closesocket( m_Socket );
}



template <int N>
void ServerPrethreading<N>::Accept( SOCKET sock )
{
	struct sockaddr_in	dstAddr;
	int dstAddrSize = sizeof( dstAddr );
	SOCKET dstSocket;

	auto ident = std::this_thread::get_id();

	while( true )
	{
		tcout << ident << " Start..." << tendl;

		m_Lock.lock();// avoid multiple thread acceptance against single client connection.
		tcout << ident << " Lock" << tendl;

		dstSocket = accept( m_Socket, ( struct sockaddr * )&dstAddr, &dstAddrSize );
		tcout << "Established connection." << tendl;

		m_Lock.unlock();
		tcout << ident << " Unlock" << tendl;

		Send_Recv( dstSocket, nullptr, std::ref( m_Dispatcher ) );
		//closesocket( dstSocket );
	}

	Close();
//	WSACleanup();
}



template < int N >
void ServerPrethreading<N>::Close()
{
	closesocket( m_Socket );
	WSACleanup();
}



template < int N >
void ServerPrethreading<N>::Send_Recv( SOCKET sock, void *serializer, std::shared_ptr<Dispatcher>& func )
{
	int numrcv;
	static OreOreLib::Memory<char, OreOreLib::MemSizeType> raw_message;

	while( true )
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
			tcout << "SocketException caught at ServerPrethreading::Send_Recv()..." << e.what() << tendl;
			break;
		}
		catch( const SendMessageException& e )
		{
			tcout << "SendMessageException caught at ServerPrethreading::Send_Recv()..." << e.what() << tendl;
			break;
		}
		catch( const RecvMessageException& e )
		{
			tcout << "RecvMessageException caught at ServerPrethreading::Send_Recv()..." << e.what() << tendl;
			break;
		}
		catch( const std::exception& e )
		{
			msgpack::sbuffer sbuf;
			msgpack::pack( &sbuf, e.what() );
			tcout << "Unknown Exception caught at ServerPrethreading::Send_Recv()..." << e.what() << tendl;
			send_message( sock, sbuf.data(), (int)sbuf.size() );
		}

	}

	closesocket( sock );

	tcout << "ServerPrethreading::Send_Recv exit..." << tendl;
}



#endif // !SERVER_PRETHREADING_H

