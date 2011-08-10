#include "vtkRenderer.h"
#include "vtkRenderWindow.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkPolyData.h"
#include "vtkPolyDataMapper.h"
#include "vtkXMLPolyDataReader.h"
#include "vtkDataSetMapper.h"
#include "vtkPoints.h"
#include "vtkCellArray.h"
#include "vtkIdList.h"
#include "vtkActor.h"
#include "vtkProperty.h"
#include "vtkCamera.h"
#include "vtkTimerLog.h"
#include "vtkDoubleArray.h"
#include "vtkPointLocator.h"
#include "vtkCommand.h"

#include "vtkSmartPointer.h"

#include "vtkCUDAParticleSystem.h"
#include "vtkCUDAMotionEquationSolver.h"

class vtkTimerCallback : public vtkCommand
{
public:
	static vtkTimerCallback *New()
	{
		vtkTimerCallback *cb = new vtkTimerCallback;
		cb->FastTimerId = 0;
		cb->FasterTimerId = 0;
		cb->RenderTimerId = 0;
		return cb;
	}

	virtual void Execute(vtkObject *caller, unsigned long eventId, void *callData)
	{
		vtkTimerLog * timer = vtkTimerLog::New();

		if (vtkCommand::TimerEvent == eventId)
		{
			int tid = * static_cast<int *>(callData);

			if (tid == this->FastTimerId)
			{

			}
			else if (tid == this->FasterTimerId)
			{
				timer->StartTimer();
				this->BMM->Modified();
				this->BMM->Update();
				timer->StopTimer();

				std::cout << "[Test] Execution Rate: " << 1/(timer->GetElapsedTime()) << "\n";

			}
			else if (tid == this->RenderTimerId)
			{
				vtkRenderWindowInteractor *iren = vtkRenderWindowInteractor::SafeDownCast(caller);
				if (iren && iren->GetRenderWindow() && iren->GetRenderWindow()->GetRenderers())
				{
					iren->Render();
				}
			}
		}
	}

	void SetFastTimerId(int tid)
	{
		this->FastTimerId = tid;
	}

	void SetFasterTimerId(int tid)
	{
		this->FasterTimerId = tid;
	}

	void SetRenderTimerId(int tid)
	{
		this->RenderTimerId = tid;
	}

	void SetBMM(vtkCUDAParticleSystem * bmm)
	{
		this->BMM = bmm;
	}

	void SetContactIds(vtkIdList * list)
	{
		this->List = list;
	}
private:
	int FastTimerId;
	int RenderTimerId;
	int FasterTimerId;

	vtkIdList * List;

	vtkCUDAParticleSystem * BMM;
};

int main(int argc, char * argv[])
{
	const char * filename = "";

	if (argc > 1)
	{
		filename = argv[1];
	}
	else
	{
		cout << "This test should at least contain 1 argument.\nUsage: Test $inputFile" << endl;
		exit(0);
	}

	vtkXMLPolyDataReader * reader = vtkXMLPolyDataReader::New();
	reader->SetFileName(filename);
	reader->Update();

	vtkPolyData * mesh = reader->GetOutput();

	vtkCUDAParticleSystem* ParticleSpringSystem = vtkCUDAParticleSystem::New();
	ParticleSpringSystem->SetInput(mesh);
	ParticleSpringSystem->SetSolverType(vtkCUDAMotionEquationSolver::RungeKutta4);
	ParticleSpringSystem->SetSpringCoefficient(250);
	ParticleSpringSystem->SetDistanceCoefficient(10);
	ParticleSpringSystem->SetDampingCoefficient(5);//Friction
	ParticleSpringSystem->SetMass(.1);
	ParticleSpringSystem->SetDeltaTime(0.005);//10ms
	ParticleSpringSystem->Init();

	ParticleSpringSystem->Print(cout);

	vtkRenderer * renderer = vtkRenderer::New();

	//Locate contact points
	vtkPointLocator * locator = vtkPointLocator::New();
	double bounds[6];
	mesh->GetBounds(bounds);

	vtkIdList * list = vtkIdList::New();
	double p[3] = {0, bounds[2], 0};

	locator->SetDataSet(mesh);
	locator->FindClosestNPoints(3, p, list);

	//Set Contact
	double dir[3];
	dir[0] = 0;//-0.1;
	dir[1] = 0.2;
	dir[2] = 0;//0.05;

	for(vtkIdType i = 0; i< list->GetNumberOfIds(); i++)
	{
		ParticleSpringSystem->InsertCollision(list->GetId(i), dir);
	}

	vtkRenderWindow * renWin = vtkRenderWindow::New();
	renWin->SetSize(500,500);
	renWin->AddRenderer(renderer);

	vtkRenderWindowInteractor * iren = vtkRenderWindowInteractor::New();
	iren->SetRenderWindow(renWin);

	vtkPolyDataMapper * mapper = vtkPolyDataMapper::New();
	mapper->SetInput(mesh);
	mapper->ScalarVisibilityOff();

	vtkActor * actor = vtkActor::New();
	actor->SetMapper(mapper);
	actor->GetProperty()->SetColor(0,1,0);

	vtkPolyDataMapper * mapper2 = vtkPolyDataMapper::New();
	mapper2->SetInput(ParticleSpringSystem->GetOutput());
	mapper2->ScalarVisibilityOff();

	vtkActor * actor2 = vtkActor::New();
	actor2->SetMapper(mapper2);
	actor2->GetProperty()->SetColor(1,0,0);
	actor2->GetProperty()->SetRepresentationToWireframe();

	//renderer->AddActor(actor);
	renderer->AddActor(actor2);
	renderer->SetBackground(1,1,1);

	renderer->ResetCamera();

	iren->Initialize();

	renWin->Render();

	// Sign up to receive TimerEvent:
	//
	vtkTimerCallback *cb = vtkTimerCallback::New();
	iren->AddObserver(vtkCommand::TimerEvent, cb);
	int tid;

	cb->SetBMM(ParticleSpringSystem);

	//Create a faster timer for BMM update
	tid = iren->CreateRepeatingTimer(10);
	cb->SetFasterTimerId(tid);

	// Create a slower repeating timer to trigger Render calls.
	// (This fires at the rate of approximately 25 frames per second.)
	//
	tid = iren->CreateRepeatingTimer(25);
	cb->SetRenderTimerId(tid);

	iren->Start();

	ParticleSpringSystem->Delete();
	mapper->Delete();
	actor->Delete();
	mapper2->Delete();
	actor2->Delete();
	renderer->Delete();
	renWin->Delete();
	iren->Delete();
	return 0;
}

