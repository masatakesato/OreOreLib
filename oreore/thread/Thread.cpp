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
		, m_Thread()
		, m_ThreadID()
	{

	}



	// Destructor
	Thread::~Thread()
	{
		Release();
	}



	void Thread::Init( IRunnable* runnable )
	{
		Release();

		m_pRunnable		= runnable;
	}



	void Thread::Release()
	{
		//std::unique_lock<std::mutex> lock(m_Mutex);

		if( IsRunning() )//if( m_Thread.joinable() )
			m_Thread.join();

		m_pRunnable = nullptr;
	}




	void Thread::Start()
	{
		// Initialize promise/future
		m_Promise = std::promise<bool>();
		m_Future = m_Promise.get_future();

		//m_Thread = std::thread( &IRunnable::Run, m_pRunnable );
		//m_hPauseEvent = false;
		
		// Execute
		m_Thread = std::thread(
			[&p=m_Promise, runnabe=m_pRunnable]
			{
				runnabe->Run();
				p.set_value(true);//p.set_value_at_thread_exit(true);//
				//std::cout << "thread function\n";
			}
		);

		//tcout << "IsRunning: " << IsRunning() << tendl;
	}


	
	void Thread::Stop()
	{
		//m_hPauseEvent	= true;
		//m_hEndEvent		= true;

		if( IsRunning() )//if( m_Thread.joinable() )//
			m_Thread.join();
	}



	// スレッド処理実行可否の判定
	bool Thread::IsRunning()
	{
		if( !m_Future.valid() )
			return false;

		#ifdef _DEBUG

			auto result = m_Future.wait_for( std::chrono::seconds(0) );
			if( result == std::future_status::ready )
			{
				tcout << _T("Thread finished\n;");
				return false;
			}
			else
			{
				tcout << _T("Thread still running\n;");
				return true;
			}

		#else

			return  m_Future.wait_for( std::chrono::seconds(0) ) != std::future_status::ready;

		#endif
	}


}// end of namespace