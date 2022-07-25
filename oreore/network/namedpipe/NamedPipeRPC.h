#ifndef NAMED_PIPE_RPC_H
#define	NAMED_PIPE_RPC_H

#include	<Windows.h>
#include	<mutex>

#include	<msgpack.hpp>

#include    "../../common/TString.h"
#include    "../../memory/Memory.h"
#include	"../../memory/SharedPtr.h"

#include    "../SocketException.h"
#include	"../Dispatcher.h"	




static int send_message( HANDLE pipe_handle, const char* msg, int data_size )
{
	try
	{
		if( !msg )
			return 0;

		// Send data length
		auto size_nl = data_size;//htonl( data_size );
		if( !WriteFile( pipe_handle, &size_nl, sizeof( size_nl ), nullptr, nullptr ) )
		{
			tcout << _T( "SendMessageException at send_message..." ) << tendl;
			throw SendMessageException();
			return -1;
		}

		// Send data
		if( !WriteFile( pipe_handle, msg, data_size, nullptr, nullptr ) )
			throw SendMessageException();

		return 1;

	}
	catch( ... )
	{
		tcout << "Exception occured at send_message" << tendl;
		//traceback.print_exc()
		throw; //SendMessageError();//raise SendMessageError( traceback.format_exc() )
		return 0;
	}

}



template < typename IndexType >
static int receive_message( HANDLE pipe_handle, OreOreLib::MemoryBase<char, IndexType>& data )
{
	try
	{
		// Read buffer size first
		u_long size_nl = 0;
		int result = ReadFile( pipe_handle, &size_nl, sizeof( size_nl ), nullptr, nullptr );

		if( result == 1 )
		{
			int msg_size = ntohl( size_nl );

			// Resise memory if needed
			if( data.Length() < msg_size )
				data.Resize( msg_size );

			// Then read actual buffer
			DWORD  recv_msg_len;
			// https://github.com/ipython/ipython/blob/master/IPython/utils/_process_win32_controller.py
			result = ReadFile( pipe_handle, data.begin(), msg_size, &recv_msg_len, nullptr );
			if( result != 1 )
			{
				//tcout << _T("releasing data...") << result << tendl;
				data.Release();
			}
		}
		
		return result;
	}
	catch( ... )
	{
		data.Release();
		std::cout << "Exception occured at receive_message" << std::endl;
		throw;// RecvMessageError();
		return 0;
	}
}





class PipeServerRPC
{
public:

	PipeServerRPC( const charstring& pipe_name )
		: m_IsListening( false )
		, m_PipeName( pipe_name )
		, m_PipeHandle( INVALID_HANDLE_VALUE )
		//, self.__m_Serializer = Serializer( pack_encoding=None, unpack_encoding=None )
		, m_Dispatcher( new Dispatcher )
		, m_NotifyReady( false )
	{

	}
	

	~PipeServerRPC()
	{
		ReleasePipe();
	}

	
	template <typename F>
	void BindFunc( const charstring& name, F func )
	{
		m_Dispatcher->BindFunc<F>( name, func );
	}


	bool InitPipe()
	{
		tcout << _T( "PipeServerRPC::InitPipe()...\n" );

		// Disconnect existing named pipe
		ReleasePipe();

		m_PipeHandle = CreateNamedPipeA(
			m_PipeName.c_str(), //'\\.\pipe\Foo',
			PIPE_ACCESS_DUPLEX,
			PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
			1, 65536, 65536,
			0,
			NULL );

		// Check error after file creation
		auto err = GetLastError();
		if( err > 0 )
		{
			tcout << _T( "error check after client::CreateNamedPipe(): " ) << err << tendl;
			m_PipeHandle = INVALID_HANDLE_VALUE;
			return false;
		}

		tcout << _T( "Successfully created named pipe: " ) << m_PipeName.c_str() << tendl;
	
		m_IsListening = true;

		return true;
	}


	bool ReleasePipe()
	{
		if( m_PipeHandle == INVALID_HANDLE_VALUE )
			return false;

		tcout << _T( "PipeServerRPC::ReleasePipe()...\n" );
	
		DisconnectNamedPipe( m_PipeHandle );
		CloseHandle( m_PipeHandle );

		m_PipeHandle = INVALID_HANDLE_VALUE;
		m_IsListening = false;
		m_NotifyReady = false;

		return true;
	}


	void SetListen( bool flag )
	{
		m_IsListening = flag;
	}


	bool IsListening()
	{
		return m_IsListening;
	}


	void WaitForStartup()
	{
		//std::mutex mtx;
		//{
			std::unique_lock<std::mutex> lock( m_Mutex );//mtx );// 
			m_CvReady.wait( lock, [&]{ return m_NotifyReady; } );
		//}
	}

	
	void Status()
	{
		tcout << _T( "//============ PipeServer Status ===========//\n" );
		tcout << _T( "PipeName: " ) << m_PipeName.c_str() << tendl;
		tcout << _T( "PipeHandle: " ) << m_PipeHandle << tendl;
		tcout << _T( "IsListening: " ) << m_IsListening << tendl << tendl;
	}

	
	void Run()
	{
		m_IsListening = true;
		m_NotifyReady = false;

		while( m_IsListening ) //true
		{
			tcout << _T( "waiting for client connection...\n" );
			{
				std::unique_lock<std::mutex> lock( m_Mutex );
				m_NotifyReady = true;
				m_CvReady.notify_all();
			}
			bool result = ConnectNamedPipe( m_PipeHandle, nullptr );

			// クライアント側で閉じたらサーバー側でも名前付きパイプの作り直しが必要.
			if( result==0 )
			{
				auto err = GetLastError();
				tcout << _T( "PipeServer::Run()...Error occured while ConnectNamedPipe(): ") << err << tendl;

				if( m_IsListening==false )//err==6 &&
				{
					ReleasePipe();
					return;
				}

				if( InitPipe()==false )
					return;

				continue;
			}

			tcout<< _T( "established connection. starts listening.\n" );
			__Listen();
		

m_NotifyReady = false;
		}// end of while( m_IsListening )

	}


	void __Listen()
	{
		int numrcv;
		static OreOreLib::MemoryBase<char, int> raw_message;

		while( m_IsListening )
		{
			try
			{
				// Receive message
				tcout << _T( "waiting for message...\n" );
				numrcv = receive_message( m_PipeHandle, raw_message );
				if( numrcv ==0 || numrcv ==-1 )
					break;

				// Do something
				auto oh = m_Dispatcher->Dispatch( raw_message );

				// Send back result to client
				msgpack::sbuffer sbuf;
				msgpack::pack( &sbuf, oh->get() );
				send_message( m_PipeHandle, sbuf.data(), (int)sbuf.size() );
			}
			catch( const SendMessageException& e )
			{
				tcout << _T( "SendMessageException caught at PipeServerRPC::__Listen()..." ) << e.what() << tendl;
				break;
			}
			catch( const RecvMessageException& e )
			{
				tcout << _T( "RecvMessageException caught at PipeServerRPC::__Listen()..." ) << e.what() << tendl;
				break;
			}
			catch( const std::exception& e )
			{
				msgpack::sbuffer sbuf;
				msgpack::pack( &sbuf, e.what() );
				tcout << _T( "Unknown Exception caught at PipeServerRPC::__Listen()..." ) << e.what() << tendl;
				send_message( m_PipeHandle, sbuf.data(), (int)sbuf.size() );
			}

		}// end of while( m_IsListening )

	}



private:

	bool		m_IsListening;
	charstring	m_PipeName;
	HANDLE		m_PipeHandle;
	OreOreLib::SharedPtr<Dispatcher>	m_Dispatcher;

std::mutex	m_Mutex;
	std::condition_variable	m_CvReady;
	bool					m_NotifyReady;

};




class PipeClientRPC
{
public:

	PipeClientRPC()
		: m_PipeName()
		, m_PipeHandle( INVALID_HANDLE_VALUE )
	{

	}


	~PipeClientRPC()
	{
		Disconnect();
	}


	void Connect( const charstring& pipe_name )
	{
		//if( m_PipeHandle != INVALID_HANDLE_VALUE )
		//	return;
		Disconnect();

		m_PipeName = pipe_name;
		
		// Establish pipe connection
		uint32 trials = 0;
		while( trials++ < m_MaxTrials )
		{
			// https://programtalk.com/vs4/python/7855/conveyor/src/main/python/conveyor/address.py/
			m_PipeHandle = CreateFileA(
				m_PipeName.c_str(),//r'\\.\pipe\Foo',
				GENERIC_READ | GENERIC_WRITE,
				0,
				nullptr,
				OPEN_EXISTING,
				0,
				nullptr
			);
			if( m_PipeHandle != INVALID_HANDLE_VALUE )
				break;

			Sleep( 15 );// sleep 15ms
		}

		// Check error after file creation
		auto err = GetLastError();
		if( err > 0 )
		{
			tcout << _T( "PipeClientRPC::Connect()...Error occured while CreateFile(): " ) << err << tendl;
			return;
		}
		DWORD lpMode = PIPE_READMODE_BYTE;//Win32Constant.PIPE_READMODE_MESSAGE )
		auto res = SetNamedPipeHandleState( m_PipeHandle, &lpMode, nullptr, nullptr );
		
		if( res == 0 )
		 {
			tcout << _T( "PipeClientRPC::Connect()...Error occured while SetNamedPipeHandleState(): " ) << GetLastError() << tendl;
			return;
		 }

		tcout << _T( "Successfully connected to named pipe: " ) << m_PipeName.c_str() << tendl;
	}


	void Disconnect()
	{
		if( m_PipeHandle != INVALID_HANDLE_VALUE )
		{
			DisconnectNamedPipe( m_PipeHandle );
//WaitForInputIdle( m_PipeHandle, INFINITE );
			CloseHandle( m_PipeHandle );
			
		}

		m_PipeHandle = INVALID_HANDLE_VALUE;
		m_PipeName = "";
	}


	template <typename... Args >
	msgpack::object_handle Call( charstring const& proc_name, Args ...args )
	{
		uint32 trials = 0;
		int numrcv;
		msgpack::object_handle oh;

		while( trials < m_MaxTrials )
		{
			try
			{
				// Serialize data
				auto msg = std::make_tuple( proc_name, std::make_tuple( args... ) );
				//for_each_tuple( args_, []( auto it ){ std::cout << it << std::endl; } );
				auto sbuf = std::make_shared<msgpack::sbuffer>();// 別スレッドにstd::moveすることを想定して実装.
				msgpack::pack( *sbuf, msg );

				// Send message to server
				send_message( m_PipeHandle, sbuf->data(), (int)sbuf->size() );
				
				// Receive data from server
				numrcv = receive_message( m_PipeHandle, buffer );
				if( numrcv ==0 || numrcv ==-1 )
				{
					tcout << _T( "Client::Call()... received data is None!\n" );
					break;
				}

				// Deserialize and return data
				//return self.__m_Serializer.Unpack( char_array.value )
				oh = msgpack::unpack( buffer.begin(), buffer.Length() );
				return oh;//__m_Serializer.Unpack( recv_data );

			}
			catch( const SendMessageException& e )
			{
				tcout << _T( "PipeClientRPC::Call()...SendMessageException occured.... trials: " ) << trials << _T( ", " ) << e.what() << tendl;
				trials++;
			}

		}// end of while( trials < self.__m_MaxTrials )

		return oh;//nullptr;

	}



private:

	charstring	m_PipeName;
	HANDLE	m_PipeHandle;

	uint32	m_MaxTrials = 5;


	OreOreLib::MemoryBase<char, int>	buffer;

};


//class PipeClientRPC:
//

//
//

//
//
//
//    def Connect( self, pipe_name ):
//
//        self.__m_PipeName = pipe_name
//
//        # https://programtalk.com/vs4/python/7855/conveyor/src/main/python/conveyor/address.py/
//        # Establish pipe connection
//        self.__m_PipeHandle = CreateFile(
//            self.__m_PipeName,#r'\\.\pipe\Foo',
//            Win32Constant.GENERIC_READ | Win32Constant.GENERIC_WRITE,
//            0,
//            None,
//            Win32Constant.OPEN_EXISTING,
//            0,
//            None
//        )
//
//        # Check error after file creation
//        err = ctypes.GetLastError()
//        if( err > 0 ):
//            print( "PipeClient::Connect()...Error occured while CreateFile(): %d" % ctypes.GetLastError() )
//            return
//
//        lpMode = DWORD( Win32Constant.PIPE_READMODE_BYTE )#Win32Constant.PIPE_READMODE_MESSAGE )
//        res = Kernel32.SetNamedPipeHandleState( self.__m_PipeHandle, ctypes.byref(lpMode), None, None )
//
//        if( res == 0 ):
//            print( "PipeClient::Connect()...Error occured while SetNamedPipeHandleState(): %d" % ctypes.GetLastError() )
//            return
//
//
//        print( "Successfully connected to named pipe: %s" % self.__m_PipeName )
//
//
//
//    def Disconnect( self ):
//        if( self.__m_PipeHandle ):
//            Kernel32.DisconnectNamedPipe ( self.__m_PipeHandle )
//            Kernel32.CloseHandle( self.__m_PipeHandle )
//        self.__m_PipeHandle = None
//
//        self.__m_PipeName = ""
//
//
//
//    def Call( self, proc_name, *args, **kwargs ):
//        
//        trial = 0
//
//        while( trial < self.__m_MaxTrials ):
//            try:
//
//                send_data = self.__m_Serializer.Pack( ( proc_name, args, kwargs ) )
//
//                # Send message to server
//                send_message( self.__m_PipeHandle, send_data )
//
//                # Receive data from server
//                recv_data = receive_message( self.__m_PipeHandle )
//                char_array = ctypes.cast( recv_data, ctypes.c_char_p )
//                #print( ">>", char_array.value )
//
//                # Deserialize and return data
//                return self.__m_Serializer.Unpack( char_array.value )
//
//
//            except SendMessageError as e:#pywintypes.error as e::
//                print( "Client::Send()...SendMessageError occured.... trial %d" % trial )
//                trial += 1
//


#endif // !NAMED_PIPE_RPC_H