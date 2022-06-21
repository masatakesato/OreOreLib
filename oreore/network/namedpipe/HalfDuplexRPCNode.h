#ifndef HALF_DUPLEX_RPC_NODE_H
#define	HALF_DUPLEX_RPC_NODE_H

#include	<thread>
#include	<future>

#include	"NamedPipeRPC.h"



class HalfDuplexRPCNode
{
public:

	HalfDuplexRPCNode( const charstring& in_pipe_name )
		: m_Receiver( in_pipe_name )
		//, m_Sender()
	{

	}


	~HalfDuplexRPCNode()
	{
		StopListen();
	}


	template <typename F>
	void BindFunc( const charstring& name, F func )
	{
		m_Receiver.BindFunc<F>( name, func );
	}


	
	void Connect( const charstring& out_pipe_name )
	{
		m_Sender.Connect( out_pipe_name );
	}


	void Disconnect()
	{
		m_Sender.Disconnect();
	}


	template <typename... Args >
	msgpack::object_handle Call( const charstring& proc_name, Args ...args )
	{
		return m_Sender.Call( proc_name, args... );
	}


	void StartListen()
	{
		tcout << _T( "HalfDuplexRPCNode::StartListen()...\n" );

		if( m_Receiver.IsListening() )
		{
			tcout << _T( "  Aborting: already listening...\n" );
			return;
		}

		// Init pipe
		m_Receiver.InitPipe();


		// Initialize promise/future
		m_Promise = std::promise<bool>();
		m_Future = m_Promise.get_future();

		// Start listen thread
		//tcout << _T( "StartListen::running thread...\n" );
		m_ListenThread = std::thread(
			[&p=m_Promise, &recv=m_Receiver]
			{
				recv.Run();
				p.set_value( true );
			}
		);// &PipeServerRPC::Run, m_Receiver );
	}


	void StopListen()
	{
		tcout << _T( "HalfDuplexNode::StopListen()...\n" );

		// Set polling flag to false
		//tcout << _T("StopListen::self.__m_Receiver.SetListen(False)...\n");
		m_Receiver.SetListen( false );

Sleep( 100 );
TODO: Atomic SetListen!!!
		if( IsRunning() )
		{
			tcout << _T( "StopListen::m_ListenThread.join()...\n" );
			HANDLE hthread = m_ListenThread.native_handle();//hthread = OpenThread( 0x40000000, false, (DWORD)(m_ListenThread.get_id()) );
			CancelSynchronousIo( hthread );

m_ListenThread.join();

			CloseHandle( hthread );
		}

		//m_ListenThread = std::thread();
		//self.__m_ListenThread = None

		// Release pipe instances
		//tcout << _T("StopListen::m_Receiver.ReleasePipe()...\n")
		m_Receiver.ReleasePipe();

		// Check status
		//m_Receiver.Status();
	}



private:

	PipeServerRPC	m_Receiver;
	PipeClientRPC	m_Sender;

	std::thread		m_ListenThread;

	std::promise<bool>	m_Promise;
	std::future<bool>	m_Future;// スレッド終了したかチェックするオブジェクト



	bool IsRunning()
	{
		if( !m_Future.valid() )
			return false;

#ifdef _DEBUG

		auto result = m_Future.wait_for( std::chrono::seconds( 0 ) );
		if( result == std::future_status::ready )
		{
			tcout << _T( "Thread finished\n;" );
			return false;
		}
		else
		{
			tcout << _T( "Thread still running\n;" );
			return true;
		}

#else

		return  m_Future.wait_for( std::chrono::seconds( 0 ) ) != std::future_status::ready;

#endif
	}
};


#endif // !HALF_DUPLEX_RPC_NODE_H
