// https://stackoverflow.com/questions/40553609/approach-of-using-an-stdatomic-compared-to-stdcondition-variable-wrt-pausing
#include	"WorkerThread.h"

#include	"../common/TString.h"



namespace OreOreLib
{

	void ThreadFunc( int& num )
	{
		tcout << "ThreadFunc " << num++ << tendl;
	}



	// Default consttuctor
	WorkerThread::WorkerThread()
		: m_pRunnable( nullptr )
		, m_Thread()
		, m_ThreadID()
		, m_bPauseEvent( true )
		, m_bEndEvent( true )
	{

	}



	// Destructor
	WorkerThread::~WorkerThread()
	{
		m_Thread.join();
	}



	void WorkerThread::Init( IRunnable* runnable )
	{
		Release();

		m_pRunnable		= runnable;
	}



	void WorkerThread::Release()
	{
		//std::unique_lock<std::mutex> lock(m_Mutex);

		if( m_Thread.joinable() )
			m_Thread.join();

		m_pRunnable = nullptr;

	}



	void WorkerThread::Start()
	{
		m_bEndEvent		= false;
		m_bPauseEvent	= false;

		m_Thread = std::thread( &IRunnable::Run, m_pRunnable );
	}


	
	void WorkerThread::Pause()
	{
		m_bPauseEvent	= true;
	}


	
	void WorkerThread::Stop()
	{
		m_bPauseEvent	= true;
		m_bEndEvent		= true;
		m_Thread.join();
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
		while( 1 )
		{
			{
				auto l = Lock();
				m_CV.wait( l, [&]{ return m_bPauseEvent || m_bEndEvent; } );
			}

			if( m_bEndEvent )
				break;

			// Do task
			if( m_pRunnable )
				m_pRunnable->Run();
		}
	}


}// end of namespace