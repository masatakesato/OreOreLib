#ifndef	WORKER_THREAD_H
#define	WORKER_THREAD_H

#include	<atomic>
#include	<mutex>
#include	<condition_variable>
#include	<thread>

#include	"IRunnable.h"

#include	"../common/Utility.h"



namespace OreOreLib
{

	class WorkerThread
	{

	public:

		WorkerThread();	// default constructor
		~WorkerThread();	// destructor

		void Init( IRunnable* );
		void Release();

		void Start();	// start/resume thread
		void Pause();	// suspend
		void Stop();	// end thread and joint to main thread

		bool IsActive();// イベントの状態（スレッドが動作中かつ実行中かどうか）を調べる



	private:

		std::thread	m_Thread;	// thread object
		std::thread::id	m_ThreadID;	// Thread ID

		IRunnable*	m_pRunnable;


		std::mutex	m_Mutex;
		bool m_bPauseEvent;// Thread pause flag. false: operating, true: paused
		bool m_bEndEvent;	// Thread end flab. false: operating, true: stopped
		std::condition_variable	m_CV;


		std::unique_lock<std::mutex> Lock()	{ return std::unique_lock<std::mutex>( m_Mutex ); }

		void Process();

	};

}


#endif	// WORKER_THREAD_H //