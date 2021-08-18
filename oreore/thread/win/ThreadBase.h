#ifndef	THREAD_BASE_H
#define	THREAD_BASE_H


#include	<process.h>
#include	<Windows.h>


namespace OreOreLib
{

class ThreadBase
{
private:

	static unsigned __stdcall func_thread(void* param)// スレッドが呼び出す関数
	{
		ThreadBase* t= reinterpret_cast< ThreadBase* >(param);

		if(t)	t->run();

		return 0;
	}

	virtual void run() = 0;	// スレッド上で行う処理


protected:

	HANDLE		m_hThread;	// スレッハンドル
	unsigned	m_ThreadID;	// スレッドID

	HANDLE	m_hPauseEvent;	// スレッド待機状態のイベント。ノンシグナル：待機中、シグナル：実行中
	HANDLE	m_hEndEvent;	// スレッド停止状態のイベント。ノンシグナル：動作中、シグナル：停止中
	

public:

	ThreadBase();	// default constructor
	~ThreadBase();	// destructor

	void Create();	// スレッド作成
	void Start();	// 開始
	void Pause();	// 一時停止
	void End();		// 終了

	void Join();	// 親スレッドに復帰するまでの終了待ち

	bool IsActive();// イベントの状態（スレッドが動作中かつ実行中かどうか）を調べる
};





}

#endif	// THREAD_BASE_H //