#ifndef	WORKER_THREAD_H
#define	WORKER_THREAD_H

#include	<atomic>
#include	<mutex>
#include	<condition_variable>
#include	<thread>

#include	"IRunnable.h"

#include	"../common/Utility.h"
#include	"../memory/SharedPtr.h"
#include	"../container/RingQueue.h"



namespace OreOreLib
{

	class WorkerThread
	{

	public:

		WorkerThread();	// Default constructor
		WorkerThread( int numThreads, int queueSize );	// Constructor
		~WorkerThread();	// destructor

		WorkerThread( const WorkerThread& ) = delete;
		WorkerThread( WorkerThread&& ) = delete;
		WorkerThread& operator=( const WorkerThread& ) = delete;
		WorkerThread& operator=( WorkerThread&& ) = delete;

		void Init( int numThreads, int queueSize );
		void Release();

		void Start();	// start/resume thread
		void Pause();	// suspend
		void Stop();	// end thread and joint to main thread

		bool IsActive();// イベントの状態（スレッドが動作中かつ実行中かどうか）を調べる



	private:

		Memory< std::thread >				m_Threads;
		RingQueue< SharedPtr<IRunnable> >	m_Queue;
		Dequeueする時に変数返せないの? -> インスタンス作るからコンストラクタ無駄に走ってめんどい -> 参照渡してムーブするDequeueはOK
		-> RingQueueでクラス使うとバグる -> メモリ領域のデストラクタちゃんと呼んでない

		std::mutex	m_Mutex;
		bool m_bPauseEvent;// Thread pause flag. false: operating, true: paused
		bool m_bEndEvent;	// Thread end flab. false: operating, true: stopped
		std::condition_variable	m_CV;




		std::unique_lock<std::mutex> Lock()	{ return std::unique_lock<std::mutex>( m_Mutex ); }

		void Process();

	};

}


#endif	// WORKER_THREAD_H //