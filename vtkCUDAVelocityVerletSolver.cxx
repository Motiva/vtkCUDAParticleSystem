#include "vtkCUDAVelocityVerletSolver.h"

#include "vtkObjectFactory.h"
#include "vtkDoubleArray.h"

#include "vtkCUDAParticleSystem.h"
#include "vtkCUDAParticleSystem.cuh"

vtkCxxRevisionMacro(vtkCUDAVelocityVerletSolver, "$Revision: 0.1 $");
vtkStandardNewMacro(vtkCUDAVelocityVerletSolver);

//----------------------------------------------------------------------------
vtkCUDAVelocityVerletSolver::vtkCUDAVelocityVerletSolver()
{
}

//----------------------------------------------------------------------------
vtkCUDAVelocityVerletSolver::~vtkCUDAVelocityVerletSolver()
{
	//Free CUDA arrays
	freeArray(this->dPos);
	freeArray(this->dVel);
	freeArray(this->dAcc);
}

//----------------------------------------------------------------------------
void vtkCUDAVelocityVerletSolver::Init()
{
	unsigned int memSize = sizeof(float) * 4 * this->NumberOfParticles;

	//Initialize CUDA device
	cudaInit();

	//Allocate device vectors memory
	allocateArray((void **)&this->dPos, memSize);
	allocateArray((void **)&this->dVel, memSize);
	allocateArray((void **)&this->dAcc, memSize);

}

//----------------------------------------------------------------------------
void vtkCUDAVelocityVerletSolver::ComputeNextStep(float *p, float *v, float *a)
{
	double dt05 = this->DeltaTime*0.5;
	double dt = this->DeltaTime;

	// Copy host -> device
	unsigned int memSize = sizeof(float) * 4 * this->NumberOfParticles;
	copyArrayToDevice(this->dPos, p, 0, memSize);
	copyArrayToDevice(this->dVel, v, 0, memSize);
	copyArrayToDevice(this->dAcc, a, 0, memSize);

	//CUDA procedure
	//Xn+1 = Xn + dt*Vn + 1/2*dt*An
	integrateSystem(this->dVel, this->dAcc, dt05, this->NumberOfParticles);
	integrateSystem(this->dPos, this->dVel, dt, this->NumberOfParticles);

	// Copy Device -> host
	copyArrayFromDevice(p, this->dPos, 0, memSize);
	copyArrayFromDevice(v, this->dVel, 0, memSize);

	//Derive An+1
	this->DeformationModel->ComputeForces();
	// Copy host -> device
	//copyArrayToDevice(this->dPos, p, 0, memSize);
	copyArrayToDevice(this->dVel, v, 0, memSize);
	copyArrayToDevice(this->dAcc, a, 0, memSize);

	//Vn+1 = Vn + 1/2*dt*An+1
	integrateSystem(this->dVel, this->dAcc, dt05, this->NumberOfParticles);

	// Copy Device -> host
	copyArrayFromDevice(p, this->dPos, 0, memSize);
	copyArrayFromDevice(v, this->dVel, 0, memSize);

}

//----------------------------------------------------------------------------
void vtkCUDAVelocityVerletSolver::PrintSelf(ostream& os, vtkIndent indent)
{
	this->Superclass::PrintSelf(os,indent);
}
