#ifndef HALF_DUPLEX_RPC_NODE_H
#define	HALF_DUPLEX_RPC_NODE_H

#include	<thread>
#include	<future>
#include	<mutex>

#include	"NamedPipeRPC.h"

#include	"../../memory/ReferenceWrapper.h"



class HalfDuplexRPCNode
{
public:

	HalfDuplexRPCNode( const charstring& in_pipe_name )
		: m_Receiver( in_pipe_name )
		//, m_Sender()
	{

	}


	virtual ~HalfDuplexRPCNode()
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


	bool StartListen()
	{
		tcout << _T( "HalfDuplexRPCNode::StartListen()...\n" );

		if( m_Receiver.IsListening() )
		{
			tcout << _T( "  Aborting: already listening...\n" );
			return false;
		}

		// Init pipe
		if( !m_Receiver.InitPipe() )
		{
			tcout << _T( "  Aborting: pipe creation failed...\n" );
			return false;
		}

		// Initialize promise/future
		m_Promise = std::promise<bool>();
		m_Future = m_Promise.get_future();

		// Start listen thread
		tcout << _T( "StartListen::running thread...\n" );
		m_ListenThread = std::thread(
			[&p=m_Promise, &recv=m_Receiver]
			{
				recv.Run();
				p.set_value( true );
			}
		);// &PipeServerRPC::Run, m_Receiver );

		m_Receiver.WaitForStartup();

		return true;
	}


	void StopListen()
	{
		tcout << _T( "HalfDuplexNode::StopListen()...\n" );

		if( !m_ListenThread.joinable() )//!IsRunning() )//
			return;

		// Set polling flag to false
		//tcout << _T("StopListen::self.__m_Receiver.SetListen(False)...\n");
		m_Receiver.SetListen( false );

		// Stop listening thread
		//tcout << _T( "  Stop listening thread...\n" );
		HANDLE hthread = m_ListenThread.native_handle();
		CancelSynchronousIo( hthread );
		//CloseHandle( hthread );
		m_ListenThread.join();
		m_ListenThread.~thread();

		//Release pipe instances
		//tcout << _T( "  Release pipe instances...\n" );
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
			tcout << _T( "Thread finished.\n" );
			return false;
		}
		else
		{
			tcout << _T( "Thread still running.\n" );
			return true;
		}

#else

		return  m_Future.wait_for( std::chrono::seconds( 0 ) ) != std::future_status::ready;

#endif
	}
};



class RemoteProcedureBase
{
public:

	RemoteProcedureBase( const HalfDuplexRPCNode& node )
	{
		m_refNode = const_cast<HalfDuplexRPCNode&>( node );
	}


	virtual ~RemoteProcedureBase()
	{
		//m_refNode.Reset();
	}


	void Connect( const charstring& out_pipe_name )
	{
		try
		{
			m_refNode->Connect( out_pipe_name );
		}
		catch( const std::exception& e )
		{
			tcout << e.what() << tendl;
		}
	}


	void Disconnect()
	{
		try
		{
			m_refNode->Disconnect();
		}
		catch( const std::exception& e )
		{
			tcout << e.what() << tendl;
		}
	}



protected:

	OreOreLib::ReferenceWrapper<HalfDuplexRPCNode>	m_refNode;

};



#endif // !HALF_DUPLEX_RPC_NODE_H
