#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include <TURunPropagation.h>

#include <TUMath.h>

class PyRunPropagation: public TURunPropagation{
private:
    FILE *f_log;
    std::vector<int> FixedIndices;

    std::vector<double> v_pars_all (std::vector<double> const v_pars){
        std::vector<double> v_pars_ (v_pars);
        for (signed int i = 0; i < static_cast<int>( FixedIndices.size() ); i++){
            int FI = FixedIndices[i];
            double val = GetFitPars()->GetParEntry(FI)->GetFitInit() ;
            v_pars_.insert(v_pars_.begin()+ FI, val);
        }
        return v_pars_;
    }

    std::vector<int> PyIndicesPars(int i){
        // 0 : FREE , 1 : FIXED , 2 : NUISANCE
        return GetFitPars()->IndicesPars( gENUM_FREEPARTYPE(i) );
    }

public:
    PyRunPropagation():TURunPropagation(){ };

    void PySetLogFile(string const &log_filename){
        f_log = fopen(log_filename.c_str(), "w");
    }

    void PySetClass(string const &usine_initfile, Bool_t is_verbose){
            string const output_dir = "Usine_Out"; // This is a dummy directory
            SetClass(usine_initfile, is_verbose, output_dir, f_log);
            UpdateFitParsAndFitData(usine_initfile, is_verbose, false);
            FixedIndices = PyIndicesPars(1);
        }

    Double_t PyChi2(std::vector<double> const v_pars){
        std::vector<double> v_input = v_pars_all(v_pars);
        // for (int i = 0; i < v_input.size(); i++){
        //     std::cout << v_input[i] << ', ';
        // }
        // std::cout << '\n';
        return Chi2_TOAFluxes(&v_input[0]);
    }

    std::vector<string> PyGetFreeParNames(){
        std::vector<string> ParNames;
        for (Int_t l = 0; l < GetFitPars()->GetNPars(); ++l){
            if (GetFitPars()->GetParEntry(l)->GetFitType() != gENUM_FREEPARTYPE(1)){
                ParNames.push_back(GetFitPars()->GetParEntry(l)->GetFitName());
            }
        }
        return ParNames;
    }

    std::vector<string> PyGetFixedParNames(){
        std::vector<string> ParNames;
        for (Int_t l = 0; l < GetFitPars()->GetNPars(); ++l){
            if (GetFitPars()->GetParEntry(l)->GetFitType() == gENUM_FREEPARTYPE(1)){
                ParNames.push_back(GetFitPars()->GetParEntry(l)->GetFitName());
            }
        }
        return ParNames;
    }

    std::vector<std::vector<double>> PyGetInitVals(){
        std::vector<std::vector<double>> InitVals;
        for (Int_t l = 0; l < GetFitPars()->GetNPars(); ++l){
            if (GetFitPars()->GetParEntry(l)->GetFitType() != gENUM_FREEPARTYPE(1)){
                std::vector<double> InitPar;
                InitPar.push_back(GetFitPars()->GetParEntry(l)->GetFitInit());
                InitPar.push_back(GetFitPars()->GetParEntry(l)->GetFitInitMin());
                InitPar.push_back(GetFitPars()->GetParEntry(l)->GetFitInitMax());
                InitPar.push_back(GetFitPars()->GetParEntry(l)->GetFitInitSigma());
                // InitPar.push_back(GetFitPars()->GetParEntry(l)->GetFitSampling());
                InitVals.push_back(InitPar);
            }
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
        // .def("SetOutputDir", &TURunPropagation::SetOutputDir)
        // .def("GetOutputDir", &TURunPropagation::GetOutputDir)
    // Double_t TURunPropagation::Chi2_TOAFluxes(const Double_t *pars)
        ;


    // Interface for the inheritated class
    class_<PyRunPropagation, TURunPropagation>(m, "PyRunPropagation")
        .def(init<>(), "Initializer, not doing anything")
        .def("PySetClass"   , &PyRunPropagation::PySetClass)
        .def("PySetLogFile" , &PyRunPropagation::PySetLogFile, "log_filename"_a = "run.log")
        .def("PyChi2"       , &PyRunPropagation::PyChi2)
        .def("PyGetFreeParNames" , &PyRunPropagation::PyGetFreeParNames )
        .def("PyGetFixedParNames", &PyRunPropagation::PyGetFixedParNames)
        .def("PyGetInitVals", &PyRunPropagation::PyGetInitVals)
        // .def("PyIndicesPars", &PyRunPropagation::PyIndicesPars)
        ;


    //m.def("Null", [](double a){return a; } , "a"_a=5.5);
 

    } 
}
