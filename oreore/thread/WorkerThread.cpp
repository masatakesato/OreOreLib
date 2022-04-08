// https://stackoverflow.com/questions/40553609/approach-of-using-an-stdatomic-compared-to-stdcondition-variable-wrt-pausing
#include	"WorkerThread.h"

#include	"../common/TString.h"



namespace OreOreLib
{


	// Default consttuctor
	WorkerThread::WorkerThread()
		: m_bPauseEvent( true )
		, m_bEndEvent( true )
	{

	}



	// Constructor
	WorkerThread::WorkerThread( int numThreads, int queueSize )
		: m_Threads( numThreads )
		, m_Queue( queueSize )
		, m_bPauseEvent( true )
		, m_bEndEvent( true )
	{

	}



	// Destructor
	WorkerThread::~WorkerThread()
	{
		//m_Thread.join();
	}



	void WorkerThread::Init( int numThreads, int queueSize )
	{
		Release();

		m_Threads.Init( numThreads );
		m_Queue.Init( queueSize );
		
		m_bPauseEvent	= true;
		m_bEndEvent		= true;
	}



	void WorkerThread::Release()
	{
		{
			Lock();
			m_bEndEvent = true;
		}

		m_CV.notify_all();

		for( auto& thread : m_Threads )
			thread.join();
	}



	void WorkerThread::Start()
	{
//		m_bEndEvent		= false;
//		m_bPauseEvent	= false;


		for( auto& thread : m_Threads )
			thread = std::thread( &WorkerThread::Process, this );
	}


	
	void WorkerThread::Pause()
	{
		m_bPauseEvent	= true;
	}


	
	void WorkerThread::Stop()
	{
		m_bPauseEvent	= true;
		m_bEndEvent		= true;
	}



	// スレッド処理実行可否の判定
	bool WorkerThread::IsActive()
	{
//		auto l = Lock();
//		return m_hEndEvent == false;

		return m_bEndEvent == false;
	}



	void WorkerThread::Process()
	{
		while( true )
		{
			SharedPtr<IRunnable> runnable;

			// begin lock
			{
				auto lock = Lock();

				m_CV.wait( lock, [&]{ return m_bEndEvent || m_Queue; } );// 終了フラグ検出 or キューに中身詰まるまで待機する

				// 終了フラグかつキューが空ならスレッド終了する.
				if( m_bEndEvent && !m_Queue )
				{
					return;
				}
				else// タスクに残っている処理を実行する
				{
					runnable = m_Queue.Dequeue();
				}
			}
			// end lock

			// Do task
			runnable->Run();

		}// end of while

	}


}// end of namespace