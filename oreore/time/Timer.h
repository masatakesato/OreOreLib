#ifndef TIMER_H
#define	TIMER_H

#include	<Windows.h>

#include	"../common/Utility.h"



namespace OreOreLib
{

	//template < typename T >
	class Timer
	{
	public:


		void Init()
		{
			m_Start = (float)timeGetTime();
			m_End = m_Start;
		}


		void SetSleepTime( const float& val )
		{
			m_SleepTime = val;
		}


		void Start()
		{
			m_Start = (float)timeGetTime();
		}


		void End()
		{
			m_End = (float)timeGetTime();

			auto timeElapsed = m_End - m_Start;
			auto wait = m_SleepTime - timeElapsed;
			if( wait > 0 )
			{
				Sleep( wait );
				timeElapsed += wait;
			}

			m_DetlaTime	= timeElapsed;
			//m_DetlaTime = 0.5f * ( m_DetlaTime + (timeElapsed) );// moving average ver
		}


		float DeltaTime()
		{
			return m_DetlaTime;
		}



	private:

		float	m_DetlaTime;

		float	m_Start;
		float	m_End;

		float	m_SleepTime;
	};


}// end of namespace OreOreLib


#endif // !TIMER_H
