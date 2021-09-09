#include	"ThreadPool.h"


//#include	<iostream>
//using namespace std;
#include	"../../common/TString.h"



// デフォルトコンストラクタ
ThreadPool::ThreadPool()
{

}


// コンストラクタ
ThreadPool::ThreadPool(int nMaxNumberThreads)
{
//tcout << _T("Initializing ThreadPool...") << tendl;

	m_nMaxNumThreads = nMaxNumberThreads;

	hEmptySlot	= CreateSemaphore(NULL, MAXQUEUE, MAXQUEUE, _T("EmptySlot") );// 待ち行列が空いている状態（値MAXQUEUEのシグナル状態）でセマフォ作成
	hWorkToDo	= CreateSemaphore(NULL, 0, MAXQUEUE, _T("WorkToDo") );		// 実行可能スレッド数ゼロ（値ゼロの非シグナル状態）でセマフォ作成
	hExit		= CreateEvent(NULL, TRUE, FALSE, _T("Exit") );				// インスタンス消去フラグを非シグナル状態で初期化
	InitializeCriticalSection(&CriticalSection);// クリティカルセクションの初期化
	
	m_threadhandles = new HANDLE[m_nMaxNumThreads];
	m_pQueue		= new CWorkThread*[MAXQUEUE];
	m_pParamArray	= new void*[MAXQUEUE];
	
	for(int i=0; i<m_nMaxNumThreads; i++)// スレッドプールをm_nMaxNumThreads個作る
	{
		m_threadhandles[i] = (HANDLE) _beginthreadex( NULL, 0, ThreadPool::ThreadExecute, this, 0, &m_Thrdaddr);// スレッドm_nMaxNumThreads個分DoWorkを走らせる
	}
		
	m_nTopIndex = 0;
	m_nBottomIndex = 0;
	nWorkInProgress=0;
}


// デストラクタ
ThreadPool::~ThreadPool()
{
	//SetEvent(hExit);	// インスタンス消去フラグをシグナル状態にする
	//SetEvent(hWorkToDo);// イベントをシグナル状態にする
	
	DestroyPool();

	DeleteCriticalSection(&CriticalSection);
	delete	[] m_threadhandles;
	delete	[] m_pQueue;
	delete	[] m_pParamArray;
}


// スレッド実行時に呼ぶコールバック関数
unsigned _stdcall ThreadPool::ThreadExecute(void *Param)
{
	((ThreadPool*)Param)->DoWork();
	return(0);
}

// スレッドで実行する処理の本体．待ち行列にタスクがある限り，タスクを実行し続ける
void ThreadPool::DoWork()
{
	CWorkThread*	cWork;
	void*			cParam;

	while(GetWork(&cWork, &cParam))// 待ち行列の先頭からタスクを取得できる間，，，，，
	{
		cWork->RunProcess(cParam);
		InterlockedDecrement(&nWorkInProgress);// 待ちタスクの数を減らす
	}
}


// 待ち行列にスレッドを追加する
//Queues up another to work
BOOL ThreadPool::SubmitJob(CWorkThread* cWork, void* cParam)
{
	
	// 待ち行列が満杯（hEmptySlotが非シグナル状態）の場合は処理を終了する
	if(WaitForSingleObject(hEmptySlot, 0/*INFINITE*/) != WAIT_OBJECT_0)
	{
		return 0;
	}

	InterlockedIncrement(&nWorkInProgress);		// 待ち行列のスレッド数をインクリメントする

	EnterCriticalSection(&CriticalSection);
	
	m_pQueue[m_nTopIndex] = cWork;					// 待ち行列に新しいスレッドを追加する
	m_pParamArray[m_nTopIndex] = cParam;			// スレッドで使うパラメータも追加する
	m_nTopIndex = (m_nTopIndex++) % (MAXQUEUE -1);	// 待ち行列の最後尾インデックスを更新する（1要素だけインクリメント）
	ReleaseSemaphore(hWorkToDo, 1, NULL);			// セマフォをインクリメントして，実行可能スレッドの数を増やす

	LeaveCriticalSection(&CriticalSection);
	
	return(1);
}


// 待ち行列からスレッドを取り出す
BOOL ThreadPool::GetWork(CWorkThread** cWork, void **cParam)
{
	HANDLE hWaitHandles[2];
	hWaitHandles[0] = hWorkToDo;
	hWaitHandles[1] = hExit;

	// hWorkToDoがシグナル状態になるのを待ちつつ，hExitのシグナル状態を検出する
	// hExitがシグナル状態の場合：戻り値-WAIT_OBJECT_0が1になる→終了
	// それ以外の場合：hWorkToDoがシグナル状態になった時点で，待ち状態から解放されて処理を続行
	if((WaitForMultipleObjects(2, hWaitHandles, FALSE, INFINITE) - WAIT_OBJECT_0) == 1)
	{
		return(0);
	}
	
	EnterCriticalSection(&CriticalSection);
	
	CWorkThread*	cWorker = m_pQueue[m_nBottomIndex];		// 待ち行列の最後からタスクを取り出す
	void*			cParameter = m_pParamArray[m_nBottomIndex];	// 待ち行列の最後からパラメータを取り出す
	*cWork = cWorker;										// 引き数に代入
	*cParam = cParameter;
	m_nBottomIndex = (m_nBottomIndex++) % (MAXQUEUE -1);	// 先頭インデックスを一つ後ろにずらす
	ReleaseSemaphore(hEmptySlot, 1, NULL);					// 待ち行列セマフォのカウンタをインクリメントする

	LeaveCriticalSection(&CriticalSection);
	
	return(1);	
}


void ThreadPool::DestroyPool()
{
	SetEvent(hExit);	// インスタンス消去フラグをシグナル状態にする

	while(nWorkInProgress > 0)// 実行中のスレッドが全部完了するまで待つ
	{
		Sleep(10);
	}

}