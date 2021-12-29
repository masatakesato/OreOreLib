#ifndef COORDINATOR_H
#define	COORDINATOR_H

#include	"ComponentManager.h"
#include	"EntityManager.h"
#include	"SystemManager.h"



class Coordinator
{
public:

	void Init()
	{

	}


	// Entity methods
	Entity CreateEntity()
	{
		return mEntityManager.CreateEntity();
	}


	void DestroyEntity( Entity entity )
	{
		mEntityManager.DestroyEntity( entity );
		mComponentManager.EntityDestroyed( entity );
		mSystemManager.EntityDestroyed( entity );
	}


	// Component methods
	template < typename T >
	void RegisterComponent()
	{
		mComponentManager.RegisterComponent<T>();
	}


	template < typename T >
	void AddComponent( Entity entity, T component )
	{

	}


	template < typename T >
	void RemoveComponent( Entity entity )
	{

	}

	
	template < typename T >
	T& GetComponent( Entity entity, T component )
	{
		return mComponentManager.GetComponent<T>( entity );
	}


	template < typename T >
	ComponentType GetComponentType()
	{
		return mComponentManager.GetComponentType<T>();
	}


	// System methods
	template < typename T >
	T* RegisterSystem()
	{

	}


	template < typename T >
	void SetSystemSignature( Signature signature )
	{
		
	}



private:

	ComponentManager	mComponentManager;
	EntityManager		mEntityManager;
	SystemManager		mSystemManager;
};



#endif // !COORDINATOR_H
