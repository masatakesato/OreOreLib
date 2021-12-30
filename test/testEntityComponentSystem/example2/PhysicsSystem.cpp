#include	"PhysicsSystem.h"

#include	"Components.h"
#include	"Coordinator.h"


extern Coordinator gCoordinator;


void PhysicsSystem::Init()
{

}



void PhysicsSystem::Update( float dt )
{
	for( const auto& entity : mEntities )
	{
		auto& rigidBody = gCoordinator.GetComponent<RigidBody>( entity );
		auto& transform = gCoordinator.GetComponent<Transform>( entity );
		const auto& gravity = gCoordinator.GetComponent<Gravity>( entity );

		AddScaled( transform.position, rigidBody.velocity, dt );//transform.position += rigidBody.velocity * dt;
		AddScaled( rigidBody.velocity, gravity.force, dt );//rigidBody.velocity += gravity.force * dt;
	}
}
