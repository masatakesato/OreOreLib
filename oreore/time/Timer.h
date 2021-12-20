#ifndef TIMER_H
#define	TIMER_H

#include	<chrono>
#include	<thread>

#include	"../common/Utility.h"



namespace OreOreLib
{

	class Timer
	{
	public:

		Timer(){}
		~Timer(){}


		void SetSleepTime( const int64& val )
		{
			m_SleepTime = val;
		}


		void Start()
		{
			m_Start = std::chrono::system_clock::now();//(float)timeGetTime();
		}


		void End()
		{
			m_End = std::chrono::system_clock::now();//(float)timeGetTime();

			int64 timeElapsed = std::chrono::duration_cast<std::chrono::milliseconds>( m_End - m_Start ).count();
			auto wait = m_SleepTime - timeElapsed;
			if( wait > 0 )
			{
				std::this_thread::sleep_for( std::chrono::milliseconds(wait) );//	Sleep( wait );
				timeElapsed += wait;
			}

			m_DetlaTime	= timeElapsed;
			//m_DetlaTime = 0.5f * ( m_DetlaTime + (timeElapsed) );// moving average ver
		}


		int64 DeltaTime()
		{
			return m_DetlaTime;
		}



	private:

		int64	m_DetlaTime;
		int64	m_SleepTime;

		std::chrono::system_clock::time_point	m_Start;
		std::chrono::system_clock::time_point	m_End;

	};


}// end of namespace OreOreLib


#endif // !TIMER_H
