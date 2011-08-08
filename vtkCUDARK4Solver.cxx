#include "vtkCUDARK4Solver.h"

#include "vtkObjectFactory.h"
#include "vtkDoubleArray.h"

#include "vtkCUDAParticleSystem.h"
#include "vtkCUDAParticleSystem.cuh"

vtkCxxRevisionMacro(vtkCUDARK4Solver, "$Revision: 0.1 $");
vtkStandardNewMacro(vtkCUDARK4Solver);

//----------------------------------------------------------------------------
vtkCUDARK4Solver::vtkCUDARK4Solver()
{
}

//----------------------------------------------------------------------------
vtkCUDARK4Solver::~vtkCUDARK4Solver()
{
	//if(this->dx) this->dx->Delete();
	//if(this->dv) this->dv->Delete();
}

//----------------------------------------------------------------------------
void vtkCUDARK4Solver::Init()
{
	if(!this->Initialized){
		size_t memSize = this->NumberOfParticles*3*sizeof(float);
		// Allocate host memory
		/*this->dx1 = new float[this->NumberOfParticles*3];
		this->dv1 = new float[this->NumberOfParticles*3];
		this->dx2 = new float[this->NumberOfParticles*3];
		this->dv2 = new float[this->NumberOfParticles*3];
		this->dx3 = new float[this->NumberOfParticles*3];
		this->dv3 = new float[this->NumberOfParticles*3];
		this->dx4 = new float[this->NumberOfParticles*3];
		this->dv4 = new float[this->NumberOfParticles*3];
		memset(this->dx1, 0 , memSize);
		memset(this->dv1, 0 , memSize);
		memset(this->dx2, 0 , memSize);
		memset(this->dv2, 0 , memSize);
		memset(this->dx3, 0 , memSize);
		memset(this->dv3, 0 , memSize);
		memset(this->dx4, 0 , memSize);
		memset(this->dv4, 0 , memSize);*/

		this->hAdx = new float[this->NumberOfParticles*4];
		this->hAdv = new float[this->NumberOfParticles*4];
		memset(this->hAdx, 0 , memSize);
		memset(this->hAdv, 0 , memSize);

	}
}

//----------------------------------------------------------------------------
void vtkCUDARK4Solver::ComputeNextStep(float *p, float *v, float *a)
{
	double dt05 = 0.5*this->DeltaTime;
	double dt1_6 = this->DeltaTime/6;

	//copy initial values
	size_t memsize =  this->NumberOfParticles*4*sizeof(float);
	memcpy(this->hAdx, p, memsize);
	memcpy(this->hAdv, v, memsize);

	this->Evaluate(p, v, a, dt05, 1);
	this->Evaluate(p, v, a, dt05, 2);
	this->Evaluate(p, v, a, this->DeltaTime, 3);

	//Last step
	this->Accumulate(p, v, a, dt1_6);

}

//----------------------------------------------------------------------------
void vtkCUDARK4Solver::Evaluate(float *p, float *v, float *a, double deltaT, int order)
{
	double h = this->DeltaTime/6;
	if(order == 2 || order == 3) h = this->DeltaTime/3;

	//Accumulate position and velocity values
	this->Accumulate(p, v, a, h);

	//CUDA procedure
	integrateSystem(p, v, a, deltaT, this->NumberOfParticles);

}

void vtkCUDARK4Solver::Accumulate(float *p, float *v, float *a, double h)
{
	// Compute Deformation Model Forces
	this->DeformationModel->ComputeForces();

	// Update intermediate position and velocity vectors
	for (int i = 0; i < this->NumberOfParticles; i++)
	{
		int id = i*4;
		this->hAdv[id] += h*a[id];
		this->hAdx[id] += h*v[id];
	}
}

//----------------------------------------------------------------------------
void vtkCUDARK4Solver::PrintSelf(ostream& os, vtkIndent indent)
{
	this->Superclass::PrintSelf(os,indent);
	os << indent << "Id: "  << "\n";
}
