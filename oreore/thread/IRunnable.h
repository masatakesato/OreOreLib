#ifndef	I_RUNNABLE_H
#define	I_RUNNABLE_H


namespace OreOreLib
{
	
	class IRunnable
	{
	public:

		IRunnable()
		{

		}


		virtual ~IRunnable()
		{
		
		}


		virtual void Run() = 0;

	};



}// end of namespace


#endif	//!I_RUNNABLE_H