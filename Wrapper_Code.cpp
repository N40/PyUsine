#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include <TURunPropagation.h>

class PyRunPropagation: public TURunPropagation{
private:
    FILE *f_log;
public:
    PyRunPropagation():TURunPropagation(){ };

    void PySetLogFile(string const &log_filename){
        f_log = fopen(log_filename.c_str(), "w");
    }

    void PySetClass(string const &usine_initfile, Bool_t is_verbose, string const &output_dir){
            SetClass(usine_initfile, is_verbose, output_dir, f_log);
            UpdateFitParsAndFitData(usine_initfile, is_verbose, False);
        }

    Double_t PyChi2(std::vector<double> const v_pars){
        return Chi2_TOAFluxes(&v_pars[0]);
    }

    string PyGetFreeParNames(){
        return GetFitPars()->GetParNames();
    }

    std::vector<std::vector<double>> PyGetInitVals(){
        std::vector<std::vector<double>> InitVals;
        for (Int_t l = 0; l < GetFitPars()->GetNPars(); ++l){
            std::vector<double> InitPar;
            InitPar.push_back(GetFitPars()->GetParEntry(l)->GetFitInit());
            InitPar.push_back(GetFitPars()->GetParEntry(l)->GetFitInitMin());
            InitPar.push_back(GetFitPars()->GetParEntry(l)->GetFitInitMax());
            InitPar.push_back(GetFitPars()->GetParEntry(l)->GetFitInitSigma());
            InitPar.push_back(GetFitPars()->GetParEntry(l)->GetFitSampling());
            InitVals.push_back(InitPar);
        }

        return InitVals;

    }

};

namespace py = pybind11;
namespace pybind11{

PYBIND11_MODULE(PyProp, m){
    // Interface of the original Usine-Class
    class_<TURunPropagation>(m, "TURunPropagation")
        .def(init<>(), "Initializer, not doing anything")
        .def("SetOutputDir", &TURunPropagation::SetOutputDir)
        .def("GetOutputDir", &TURunPropagation::GetOutputDir)
    // Double_t TURunPropagation::Chi2_TOAFluxes(const Double_t *pars)
        ;

    // Interface for the inheritated class
    class_<PyRunPropagation, TURunPropagation>(m, "PyRunPropagation")
        .def(init<>(), "Initializer, not doing anything")
        .def("PySetClass"   , &PyRunPropagation::PySetClass)
        .def("PySetLogFile" , &PyRunPropagation::PySetLogFile, "log_filename"_a = "run.log")
        .def("PyChi2"       , &PyRunPropagation::PyChi2)
        .def("PyGetFreeParNames", &PyRunPropagation::PyGetFreeParNames)
        .def("PyGetInitVals", &PyRunPropagation::PyGetInitVals)
        ;

    m.def("Null", [](double a){return a; } , "a"_a=5.5);

    }
}
