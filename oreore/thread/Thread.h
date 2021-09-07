#ifndef	THREAD_H
#define	THREAD_H

#include	<atomic>
#include	<mutex>
#include	<condition_variable>
#include	<thread>

#include	"IRunnable.h"

#include	"../common/Utility.h"



namespace OreOreLib
{

	class Thread
	{

	public:

		Thread();	// default constructor
		~Thread();	// destructor

		void Init( IRunnable* );
		void Release();

		void Play();	// start/resume thread
		void Pause();	// suspend
		void Stop();	// end thread and joint to main thread

		bool IsActive();// イベントの状態（スレッドが動作中かつ実行中かどうか）を調べる



	private:

		std::thread	m_Thread;	// thread object
		uint32		m_ThreadID;	// Thread ID

		std::mutex	m_Mutex;
		bool		m_hPauseEvent;	// Thread pause flag. false: operating, true: paused
		bool		m_hEndEvent;	// Thread end flab. false: operating, true: stopped


		IRunnable*	m_pRunnable;


		std::condition_variable	m_CV;


		std::unique_lock<std::mutex> Lock()
		{
			return std::unique_lock<std::mutex>( m_Mutex );
		}


		void Work();

	};

}


#endif	// THREAD_H //