#ifndef	THREAD_H
#define	THREAD_H

#include	<atomic>
#include	<mutex>
#include	<condition_variable>
#include	<thread>
#include	<future>

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

		void Start();	// start/resume thread
		void Stop();	// end thread and joint to main thread

		bool IsRunning();// イベントの状態（スレッドが動作中かつ実行中かどうか）を調べる



	private:

		std::thread		m_Thread;	// thread object
		std::thread::id	m_ThreadID;	// Thread ID

		IRunnable*	m_pRunnable;


		// thread status check variables
		std::promise<bool>	m_Promise;
		std::future<bool>	m_Future;// スレッド終了したかチェックするオブジェクト

		//std::condition_variable	m_CV;// 自スレッドが終わるまで別スレッド待たせるフラグ


	};

}


#endif	// THREAD_H //