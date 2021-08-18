// https://stackoverflow.com/questions/40553609/approach-of-using-an-stdatomic-compared-to-stdcondition-variable-wrt-pausing
#include	"Thread.h"

#include	"../common/TString.h"



// 1回だけ実行して終わりにするケース
// 反復実行するケース

namespace OreOreLib
{

	void ThreadFunc( int& num )
	{
		tcout << "ThreadFunc " << num++ << tendl;
	}



	// Default consttuctor
	Thread::Thread()
		: m_pRunnable( nullptr )
//		: m_Thread( 0 )
//		, m_ThreadID( 0 )
	{

	}



	// Destructor
	Thread::~Thread()
	{
		m_Thread.join();
	}



	void Thread::Init( int& val )
	{
		m_hPauseEvent	= false;
		m_hEndEvent		= false;
		m_Thread		= std::thread( ThreadFunc, std::ref(val) );
	}



	void Thread::Init( IRunnable* runnable )
	{
		m_pRunnable		= runnable;
		m_hPauseEvent	= false;
		m_hEndEvent		= false;
		m_Thread		= std::thread( &IRunnable::Run, m_pRunnable );
	}


	

	void Thread::Play()
	{
		m_hPauseEvent = false;
	}


	
	void Thread::Pause()
	{
		m_hPauseEvent = true;
	}


	
	void Thread::Stop()
	{
		m_hPauseEvent	= true;
		m_hEndEvent		= true;
		m_Thread.join();
	}



	// スレッド処理実行可否の判定
	bool Thread::IsActive()
	{
//		auto l = Lock();
//		return m_hEndEvent == false;

		return false;
	}




	void Thread::Work()
	{
		/*
		while( 1 )
		{
			{
				auto l = Lock();
				m_CV.wait( l, [&]{ return m_hPauseEvent || m_hEndEvent; } );
			}

			if( m_hEndEvent )
				break;

			// Do task
		}
		*/
	}


}// end of namespace