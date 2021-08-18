#include	"ThreadBase.h"

#include	<iostream>
using namespace std;


namespace OreOreLib
{


// デフォルトコンストラクタ
ThreadBase::ThreadBase() : m_hThread(0), m_ThreadID(0)
{
	// 待機状態イベント作成
	m_hPauseEvent	= CreateEvent(
							NULL,
							TRUE,	// 手動リセットする
							FALSE,	// ノンシグナル状態(WaitFor**関数で引っかかる)で初期化
							NULL );


	// 停止状態イベント作成
	m_hEndEvent		= CreateEvent(
							NULL,
							TRUE,	// 手動リセットする
							TRUE,	// シグナル状態(WaitFor**関数で引っかからない)で初期化
							NULL );


}


// デストラクタ
ThreadBase::~ThreadBase()
{
	if( m_hEndEvent)
	{
		CloseHandle( m_hEndEvent );
		m_hEndEvent	= 0;
	}

	if( m_hPauseEvent )
	{
		CloseHandle( m_hPauseEvent );
		m_hPauseEvent	= 0;
	}

	if( m_hThread)
	{
		CloseHandle( m_hThread );
		m_hThread	= 0;
		m_ThreadID	= 0;
	}

}


// スレッド作成
void ThreadBase::Create()
{
	m_hThread	= (HANDLE)_beginthreadex(
					NULL,
					0,
					func_thread,
					this,
					CREATE_SUSPENDED,
					&m_ThreadID );

}


// スレッド開始or再開
void ThreadBase::Start()
{
	if( m_hThread )
	{
		SetEvent( m_hPauseEvent );	// 待機状態イベントをシグナル状態に設定する
		ResetEvent( m_hEndEvent );	// 停止状態イベントをノンシグナルに設定する
		ResumeThread( m_hThread );	// スレッド開始
	}
}


// 一時停止
void ThreadBase::Pause()
{
	if( m_hThread )
	{
		// 待機状態イベントをノンシグナルに設定する(WaitFor**関数に引っかかるようになる)
		ResetEvent( m_hPauseEvent );	
	}
}


// 終了
void ThreadBase::End()
{
	// 停止状態イベントをシグナルに設定する
	SetEvent( m_hEndEvent );

	// スレッドが終了するまで待つ
	DWORD	exitCode;
	do
	{
		GetExitCodeThread( m_hThread, &exitCode );
	}
	while( exitCode == STILL_ACTIVE );
}


// 親スレッド復帰のための待ち
void ThreadBase::Join()
{
	// 停止状態イベントがシグナル状態になるまで待ち続ける
	WaitForSingleObject( m_hEndEvent, INFINITE );
}


// スレッド処理実行可否の判定
bool ThreadBase::IsActive()
{
	// 待機状態がシグナルになるまで待ち続ける
	WaitForSingleObject( m_hPauseEvent, INFINITE );

	// 停止状態がノンシグナルであればtrueを返す
	return WaitForSingleObject( m_hEndEvent, 0 ) != WAIT_OBJECT_0;
}



}// end of namespace