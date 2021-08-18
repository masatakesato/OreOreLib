#ifndef MESSAGE_PROTOCOL_H
#define	MESSAGE_PROTOCOL_H


#include	"../common/TString.h"
#include	"../memory/Memory.h"

#include	"SocketException.h"


#include	<WinSock2.h>
#include	<Ws2tcpip.h>// included for InetPton




// https://stackoverflow.com/questions/43759231/sending-and-receiving-all-data-c-sockets

static int SendAll( SOCKET sock, const void *data, int data_size )
{
	const char *data_ptr =(const char *)data;
	int bytes_sent;

	while( data_size > 0 )
	{
		bytes_sent = send( sock, data_ptr, data_size, 0 );
		if( bytes_sent==SOCKET_ERROR )
		{
			tcout << "SOCKET_ERROR at SendAll..." << tendl;
			throw SendMessageException();
			return -1;
		}

		data_ptr += bytes_sent;
		data_size -= bytes_sent;
	}// end of while

	return 1;
}



static int RecieveAll( SOCKET sock, void* data, int data_size )
{
	char *data_ptr =(char *)data;
	int bytes_recv;

	while( data_size > 0 )
	{
		bytes_recv = recv( sock, data_ptr, data_size, 0 );
		//tcout << "recieved " << bytes_recv << "[bytes]." <<tendl;
		if( bytes_recv==SOCKET_ERROR )
		{
			tcout << "SOCKET_ERROR at RecieveAll..." << tendl;
			throw RecvMessageException();
			return -1;
		}

		if( bytes_recv <= 0 )
			return bytes_recv;

		data_ptr += bytes_recv;
		data_size -= bytes_recv;
	}// end f while

	return 1;
}






// https://stackoverflow.com/questions/17667903/python-socket-receive-large-amount-of-data

// |-- size(4byes) --|++++++++ data ++++++++|-- size(4bytes) --|+++++ data +++++|-- data ...

static int send_message( SOCKET sock, const char* data, int data_size )
{
	try
	{
		tcout << "sending data_size..." << data_size << tendl;
		// Send data size first
		auto size_nl = htonl( data_size );
		int result = SendAll( sock, &size_nl, sizeof size_nl );//sizeof data_size );// TODO: Need to convert to network byte order??

		// Send data 
		if( result == 1 )
		{
			tcout << "sending data..." << tendl;
			result = SendAll( sock, data, data_size );
		}

		return result;
	}
	catch( ... )
	{
		tcout << "Exception occured at send_message" << tendl;
		//traceback.print_exc()
		throw; //SendMessageError();//raise SendMessageError( traceback.format_exc() )
		return 0;
	}
}



static int recieve_message( SOCKET sock, OreOreLib::Memory<char>& data )//char* data, int data_size  )
{
	try
	{
		// Extract message length first
		u_long size_nl = 0;
		int result = RecieveAll( sock, &size_nl, sizeof size_nl );
		
		if( result == 1 )
		{
			int msg_size = ntohl( size_nl );
			//tcout << "recieve_message::recieving data_size..." << msg_size << tendl;

			// Resize Memory if needed
			if( data.Length() < msg_size )
			{
				//tcout << "Expanding memory..." << tendl;
				data.Extend( msg_size - data.Length() );
			}

			//tcout << "recieving data..." << tendl;
			// Recieve all message data
			result = RecieveAll( sock, data.begin(), msg_size );
			if( result != 1 )
			{
				//tcout << "releasing data..." << result << tendl;
				data.Release();
			}
		}

		return result;
	}
	catch( ... )
	{
		data.Release();
		std::cout << "Exception occured at recieve_message" << std::endl;
		throw;// RecvMessageError();
		return 0;
	}
}


#endif // !MESSAGE_PROTOCOL_H