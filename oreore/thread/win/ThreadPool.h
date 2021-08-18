#ifndef THREAD_POOL_H
#define THREAD_POOL_H


#include	<process.h>
#include	<windows.h>
#include	<deque>


#define MAXTHREAD	4	// スレッド最大数
#define MAXQUEUE	16	// 待ち行列タスク最大数




class CWorkThread
{
public:
	unsigned virtual RunProcess(void *Param) = 0;
	
};




class ThreadPool
{
private:

	LONG			nWorkInProgress;	// 待ち行列に入っているスレッド数
	int				m_nMaxNumThreads;	// 同時処理するスレッドの数
	unsigned int	m_Thrdaddr;			// スレッド作成時に使う変数

	HANDLE			*m_threadhandles;	// ジョブ待ち行列（スレッドハンドル）
	CWorkThread*	*m_pQueue;			// ジョブ待ち行列（処理）
	void*			*m_pParamArray;		// 引き数？

	int				m_nTopIndex;		// ジョブ待ち行列（キュー）の先頭インデックス
	int				m_nBottomIndex;		// ジョブ待ち行列（キュー）の末尾インデックス
	
	HANDLE			hEmptySlot;			// 待ち行列の空き状態を保持するセマフォ．空きがある場合はシグナル状態になる
	HANDLE			hWorkToDo;			// 実行可能なスレッドの数を保持するセマフォ．スレッド実行数に余裕がある場合はシグナル状態になる
	HANDLE			hExit;				// インスタンス消去フラグ
	
	CRITICAL_SECTION CriticalSection;

	static unsigned _stdcall ThreadExecute(void *Param);// 


public:
	ThreadPool();
	ThreadPool(int nMaxNumberThreads);
	BOOL SubmitJob(CWorkThread* cWork, void* cParam);// 待ち行列にスレッドを追加する
	//BOOL GetWork(CWorkThread** cWork);
	BOOL GetWork(CWorkThread** cWork, void** cParam);

	virtual ~ThreadPool();	// デストラクタ
	void DoWork();			// 待ち行列のスレッドを実行する
	void DestroyPool();		//

};



#endif // THREAD_POOL_H //