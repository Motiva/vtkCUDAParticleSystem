#include "vtkCUDAEulerSolver.h"

#include "vtkObjectFactory.h"
#include "vtkDoubleArray.h"

#include "vtkCUDAParticleSystem.h"
#include "vtkCUDAParticleSystem.cuh"

vtkCxxRevisionMacro(vtkCUDAEulerSolver, "$Revision: 0.1 $");
vtkStandardNewMacro(vtkCUDAEulerSolver);

//----------------------------------------------------------------------------
vtkCUDAEulerSolver::vtkCUDAEulerSolver()
{
}

//----------------------------------------------------------------------------
vtkCUDAEulerSolver::~vtkCUDAEulerSolver()
{
}


//----------------------------------------------------------------------------
void vtkCUDAEulerSolver::Init()
{
}

//----------------------------------------------------------------------------
void vtkCUDAEulerSolver::ComputeNextStep(float *p, float *v, float *a)
{

	this->DeformationModel->ComputeForces();

	//CUDA procedure
	integrateSystem(p, v, a, this->DeltaTime, this->NumberOfParticles);
}

//----------------------------------------------------------------------------
void vtkCUDAEulerSolver::PrintSelf(ostream& os, vtkIndent indent)
{
	this->Superclass::PrintSelf(os,indent);
}
