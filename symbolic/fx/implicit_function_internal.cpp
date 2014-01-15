/*
 *    This file is part of CasADi.
 *
 *    CasADi -- A symbolic framework for dynamic optimization.
 *    Copyright (C) 2010 by Joel Andersson, Moritz Diehl, K.U.Leuven. All rights reserved.
 *
 *    CasADi is free software; you can redistribute it and/or
 *    modify it under the terms of the GNU Lesser General Public
 *    License as published by the Free Software Foundation; either
 *    version 3 of the License, or (at your option) any later version.
 *
 *    CasADi is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *    Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public
 *    License along with CasADi; if not, write to the Free Software
 *    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */

#include "implicit_function_internal.hpp"
#include "../mx/mx_node.hpp"
#include "../mx/mx_tools.hpp"
#include <iterator>

#include "../casadi_options.hpp"
#include "../profiling.hpp"

using namespace std;
namespace CasADi{

  ImplicitFunctionInternal::ImplicitFunctionInternal(const FX& f, const FX& jac, const LinearSolver& linsol) : f_(f), jac_(jac), linsol_(linsol){
    addOption("linear_solver",            OT_LINEARSOLVER, GenericType(), "User-defined linear solver class. Needed for sensitivities.");
    addOption("linear_solver_options",    OT_DICTIONARY,   GenericType(), "Options to be passed to the linear solver.");
    addOption("constraints",              OT_INTEGERVECTOR,GenericType(),"Constrain the unknowns. 0 (default): no constraint on ui, 1: ui >= 0.0, -1: ui <= 0.0, 2: ui > 0.0, -2: ui < 0.0.");
    addOption("implicit_input",           OT_INTEGER,      0, "Index of the input that corresponds to the actual root-finding");
    addOption("implicit_output",          OT_INTEGER,      0, "Index of the output that corresponds to the actual root-finding");
  }

  ImplicitFunctionInternal::~ImplicitFunctionInternal(){
  }

  void ImplicitFunctionInternal::deepCopyMembers(std::map<SharedObjectNode*,SharedObject>& already_copied){
    FXInternal::deepCopyMembers(already_copied);
    f_ = deepcopy(f_,already_copied);
    jac_ = deepcopy(jac_,already_copied);
    linsol_ = deepcopy(linsol_,already_copied);
  }

  void ImplicitFunctionInternal::init(){

    // Initialize the residual function
    f_.init(false);

    // Which input/output correspond to the root-finding problem?
    iin_ = getOption("implicit_input");
    iout_ = getOption("implicit_output");
  
    // Get the number of equations and check consistency
    casadi_assert_message(f_.output(iout_).dense() && f_.output(iout_).size2()==1, "Residual must be a dense vector");
    casadi_assert_message(f_.input(iin_).dense() && f_.input(iin_).size2()==1, "Unknown must be a dense vector");
    n_ = f_.output(iout_).size();
    casadi_assert_message(n_ == f_.input(iin_).size(), "Dimension mismatch. Input size is " << f_.input(iin_).size() << ", while output size is " << f_.output(iout_).size());

    // Allocate inputs
    setNumInputs(f_.getNumInputs());
    for(int i=0; i<getNumInputs(); ++i){
      input(i) = f_.input(i);
    }
  
    // Allocate output
    setNumOutputs(f_.getNumOutputs());
    for(int i=0; i<getNumOutputs(); ++i){
      output(i) = f_.output(i);
    }

    // Call the base class initializer
    FXInternal::init();
  
    // Generate Jacobian if not provided
    if(jac_.isNull()) jac_ = f_.jacobian(iin_,iout_);
    jac_.init(false);
  
    // Check for structural singularity in the Jacobian
    casadi_assert_message(!isSingular(jac_.output().sparsity()),"ImplicitFunctionInternal::init: singularity - the jacobian is structurally rank-deficient. sprank(J)=" << sprank(jac_.output()) << " (instead of "<< jac_.output().size1() << ")");
  
    // Get the linear solver creator function
    if(linsol_.isNull()){
      if(hasSetOption("linear_solver")){
        linearSolverCreator linear_solver_creator = getOption("linear_solver");
        
        // Allocate an NLP solver
        linsol_ = linear_solver_creator(jac_.output().sparsity(),1);
        
        // Pass options
        if(hasSetOption("linear_solver_options")){
          const Dictionary& linear_solver_options = getOption("linear_solver_options");
          linsol_.setOption(linear_solver_options);
        }
        
        // Initialize
        linsol_.init();
      }
    } else {
      // Initialize the linear solver, if provided
      linsol_.init(false);
      casadi_assert(linsol_.input().sparsity()==jac_.output().sparsity());
    }
    
    // No factorization yet;
    fact_up_to_date_ = false;
    
    // Constraints
    if (hasSetOption("constraints")) u_c_ = getOption("constraints");
    
    casadi_assert_message(u_c_.size()==n_ || u_c_.empty(),"Constraint vector if supplied, must be of length n, but got " << u_c_.size() << " and n = " << n_);
  }

  void ImplicitFunctionInternal::evaluate(){

    // Set up timers for profiling
    double time_zero;
    double time_start;
    double time_stop;
    if (CasadiOptions::profiling) {
      time_zero = getRealTime();
    }

    // Mark factorization as out-of-date. TODO: make this conditional
    fact_up_to_date_ = false;

    // Get initial guess
    //output(iout_).set(input(iin_));

    // Solve the nonlinear system of equations
    solveNonLinear();
  }

  void ImplicitFunctionInternal::evaluateMX(MXNode* node, const MXPtrV& arg, MXPtrV& res, const MXPtrVV& fseed, MXPtrVV& fsens, const MXPtrVV& aseed, MXPtrVV& asens, bool output_given){
    // Evaluate non-differentiated
    vector<MX> argv = MXNode::getVector(arg);
    if(!output_given){
      vector<MX> resv = callSelf(argv);
      for(int i=0; i<resv.size(); ++i){
        if(res[i]!=0) *res[i] = resv[i];
      }
    }

    // Quick return if no derivatives
    int nfwd = fsens.size();
    int nadj = aseed.size();
    if(nfwd==0 && nadj==0) return;

    // Temporaries
    vector<int> row_offset(1,0);
    vector<MX> rhs;

    // Arguments when calling f/f_der
    vector<MX> v;
    v.reserve(getNumInputs()*(1+nfwd) + nadj);
    v.insert(v.end(),argv.begin(),argv.end());
    v[iin_] = *res[iout_];
    
    // Get an expression for the Jacobian
    MX J = jac_.call(v).front();

    // Directional derivatives of f
    FX f_der = f_.derivative(nfwd,nadj);

    // Forward sensitivities, collect arguments for calling f_der
    for(int d=0; d<nfwd; ++d){
      argv = MXNode::getVector(fseed[d]);
      argv[iin_] = MX::sparse(input(iin_).shape());
      v.insert(v.end(),argv.begin(),argv.end());
    }

    // Adjoint sensitivities, solve to get arguments for calling f_der
    if(nadj>0){
      for(int d=0; d<nadj; ++d){
        rhs.push_back(trans(*aseed[d][iout_]));
        row_offset.push_back(row_offset.back()+1);
        *aseed[d][iout_] = MX();
      }
      rhs = vertsplit(J->getSolve(vertcat(rhs),false,linsol_),row_offset);
      for(int d=0; d<nadj; ++d){
        v.push_back(trans(rhs[d]));
      }
      row_offset.resize(1);
      rhs.clear();
    }
  
    // Propagate through the implicit function
    v = f_der.call(v);
    vector<MX>::const_iterator v_it = v.begin();

    // Discard non-differentiated evaluation (change?)
    v_it += getNumOutputs();

    // Solve for the forward sensitivities
    if(nfwd>0){
      for(int d=0; d<nfwd; ++d){
        rhs.push_back(trans(*v_it++));
        row_offset.push_back(row_offset.back()+1);        
      }
      rhs = vertsplit(J->getSolve(vertcat(rhs),true,linsol_),row_offset);
      for(int d=0; d<nfwd; ++d){
        if(fsens[d][iout_]!=0){
          *fsens[d][iout_] = -trans(rhs[d]);
        }
      }
      row_offset.resize(1);
      rhs.clear();
    }

    // Collect adjoint sensitivities
    for(int d=0; d<nadj; ++d){
      for(int i=0; i<asens[d].size(); ++i, ++v_it){
        if(i!=iin_ && asens[d][i]!=0 && !v_it->isNull()){
          *asens[d][i] += - *v_it;
        }
      }
    }
    casadi_assert(v_it==v.end());
  }

  void ImplicitFunctionInternal::spEvaluate(bool fwd){

    // Get arrays
    bvec_t* z0 = reinterpret_cast<bvec_t*>(input(iin_).ptr());
    bvec_t* z = reinterpret_cast<bvec_t*>(output(iout_).ptr());
    bvec_t* zf = reinterpret_cast<bvec_t*>(f_.input(iin_).ptr());
    bvec_t* rf = reinterpret_cast<bvec_t*>(f_.output(iout_).ptr());

    if(fwd){

      // Pass inputs to function
      fill(zf,zf+n_,0);
      for(int i=0; i<getNumInputs(); ++i){
        if(i!=iin_) f_.input(i).set(input(i));
      }

      // Propagate dependencies through the function
      f_.spInit(true);
      f_.spEvaluate(true);
      
      // "Solve" in order to propagate to z
      fill(z,z+n_,0);
      linsol_.spSolve(z,rf,true);

    } else { 

      // "Solve" in order to get seed
      fill(rf,rf+n_,0);
      linsol_.spSolve(rf,z,false);
      
      // Propagate dependencies through the function
      f_.spInit(false);
      f_.spEvaluate(false);

      // Collect influence on inputs
      fill(z0,z0+n_,0);
      for(int i=0; i<getNumInputs(); ++i){
        if(i!=iin_) f_.input(i).get(input(i));
      }
    }
  }

 
} // namespace CasADi

  


