/*
 *    This file is part of CasADi.
 *
 *    CasADi -- A symbolic framework for dynamic optimization.
 *    Copyright (C) 2010-2014 Joel Andersson, Joris Gillis, Moritz Diehl,
 *                            K.U. Leuven. All rights reserved.
 *    Copyright (C) 2011-2014 Greg Horn
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


#include "piqp_interface.hpp"
#include "casadi/core/casadi_misc.hpp"


using namespace std;
namespace casadi {

  extern "C"
  int CASADI_CONIC_PIQP_EXPORT
  casadi_register_conic_piqp(Conic::Plugin* plugin) {
    plugin->creator = PiqpInterface::creator;
    plugin->name = "piqp";
    plugin->doc = PiqpInterface::meta_doc.c_str();
    plugin->version = CASADI_VERSION;
    plugin->options = &PiqpInterface::options_;
    plugin->deserialize = &PiqpInterface::deserialize;
    return 0;
  }

  extern "C"
  void CASADI_CONIC_PIQP_EXPORT casadi_load_conic_piqp() {
    Conic::registerPlugin(casadi_register_conic_piqp);
  }

  PiqpInterface::PiqpInterface(const std::string& name,
                                   const std::map<std::string, Sparsity>& st)
    : Conic(name, st) {

    has_refcount_ = true;
  }

  PiqpInterface::~PiqpInterface() {
    clear_mem();
  }

  const Options PiqpInterface::options_
  = {{&Conic::options_},
     {{"piqp",
       {OT_DICT,
        "const Options to be passed to piqp."}},
      {"warm_start_primal",
       {OT_BOOL,
        "Use x0 input to warmstart [Default: true]."}},
      {"warm_start_dual",
       {OT_BOOL,
        "Use lam_a0 and lam_x0 input to warmstart [Default: truw]."}}
     }
  };

  void PiqpInterface::init(const Dict& opts) {
    // Initialize the base classes
    Conic::init(opts);

    // Read options
    for (auto&& op : opts) {
      if (op.first=="warm_start_primal") {
      } else if (op.first=="warm_start_dual") {
      } else if (op.first=="piqp") {
        // const Dict& opts = op.second;
        // for (auto&& op : opts) {
        //   if (op.first=="rho") {
        //     settings_.rho = op.second;
        //   } else if (op.first=="sigma") {
        //     settings_.sigma = op.second;
        //   } else if (op.first=="scaling") {
        //     settings_.scaling = op.second;
        //   } else if (op.first=="adaptive_rho") {
        //     settings_.adaptive_rho = op.second;
        //   } else if (op.first=="adaptive_rho_interval") {
        //     settings_.adaptive_rho_interval = op.second;
        //   } else if (op.first=="adaptive_rho_tolerance") {
        //     settings_.adaptive_rho_tolerance = op.second;
        //   //} else if (op.first=="adaptive_rho_fraction") {
        //   //  settings_.adaptive_rho_fraction = op.second;
        //   } else if (op.first=="max_iter") {
        //     settings_.max_iter = op.second;
        //   } else if (op.first=="eps_abs") {
        //     settings_.eps_abs = op.second;
        //   } else if (op.first=="eps_rel") {
        //     settings_.eps_rel = op.second;
        //   } else if (op.first=="eps_prim_inf") {
        //     settings_.eps_prim_inf = op.second;
        //   } else if (op.first=="eps_dual_inf") {
        //     settings_.eps_dual_inf = op.second;
        //   } else if (op.first=="alpha") {
        //     settings_.alpha = op.second;
        //   } else if (op.first=="delta") {
        //     settings_.delta = op.second;
        //   } else if (op.first=="polish") {
        //     settings_.polish = op.second;
        //   } else if (op.first=="polish_refine_iter") {
        //     settings_.polish_refine_iter = op.second;
        //   } else if (op.first=="verbose") {
        //     settings_.verbose = op.second;
        //   } else if (op.first=="scaled_termination") {
        //     settings_.scaled_termination = op.second;
        //   } else if (op.first=="check_termination") {
        //     settings_.check_termination = op.second;
        //   } else if (op.first=="warm_start") {
        //     casadi_error("PIQP's warm_start option is impure and therefore disabled. "
        //                  "Use CasADi options 'warm_start_primal' and 'warm_start_dual' instead.");
        //   //} else if (op.first=="time_limit") {
        //   //  settings_.time_limit = op.second;
        //   } else {
        //     casadi_error("Not recognised");
        //   }
        // }
      }
    }

    nnzHupp_ = H_.nnz_upper();
    nnzA_ = A_.nnz()+nx_;

    alloc_w(nnzHupp_+2*nnzA_, false);
    alloc_w(2*nx_+2*na_, false);
  }

  int PiqpInterface::init_mem(void* mem) const {
    if (Conic::init_mem(mem)) return 1;
    auto m = static_cast<PiqpMemory*>(mem);

    m->tripletList.reserve(2 * H_.nnz());
    m->tripletListEq.reserve(na_);

    m->g_vector.resize(nx_);
    m->uba_vector.resize(na_);
    m->lba_vector.resize(na_);
    m->ubx_vector.resize(na_);
    m->lbx_vector.resize(na_);
    m->ineq_b_vector.resize(na_);
    m->eq_b_vector.resize(na_);

    m->add_stat("preprocessing");
    m->add_stat("solver");
    m->add_stat("postprocessing");
    return 0;
  }

  inline const char* return_status_string(casadi_int status) {
    return "Unknown";
  }

  int PiqpInterface::
  solve(const double** arg, double** res, casadi_int* iw, double* w, void* mem) const {
    typedef Eigen::Triplet<double> TripletT;

    auto m = static_cast<PiqpMemory*>(mem);
    m->fstats.at("preprocessing").tic();

    // Get problem data
    double* g=w; w += nx_;
    casadi_copy(arg[CONIC_G], nx_, g);
    double* lbx=w; w += nx_;
    casadi_copy(arg[CONIC_LBX], nx_, lbx);
    double* ubx=w; w += nx_;
    casadi_copy(arg[CONIC_UBX], nx_, ubx);
    double* lba=w; w += na_;
    casadi_copy(arg[CONIC_LBA], na_, lba);
    double* uba=w; w += na_;
    casadi_copy(arg[CONIC_UBA], na_, uba);
    double* H=w; w += nnz_in(CONIC_H);
    casadi_copy(arg[CONIC_H], nnz_in(CONIC_H), H);
    double* A=w; w += nnz_in(CONIC_A);
    casadi_copy(arg[CONIC_A], nnz_in(CONIC_A), A);

    m->g_vector = Eigen::Map<const Eigen::VectorXd>(g, nx_);
    m->uba_vector = Eigen::Map<Eigen::VectorXd>(uba, na_);
    m->lba_vector = Eigen::Map<Eigen::VectorXd>(lba, na_);
    m->ubx_vector = Eigen::Map<Eigen::VectorXd>(ubx, nx_);
    m->lbx_vector = Eigen::Map<Eigen::VectorXd>(lbx, nx_);

    // Use lhs_equals_rhs_constraint to split double-sided bounds into one-sided
    // bound for equality constraints and double-sided for inequality constraints
    const Eigen::Array<bool, Eigen::Dynamic, 1>
    lhs_equals_rhs_constraint = (m->uba_vector.array() == m->lba_vector.array()).eval();
    const Eigen::Array<bool, Eigen::Dynamic, 1>
    lhs_is_inf = m->lba_vector.array().isInf();
    const Eigen::Array<bool, Eigen::Dynamic, 1>
    rhs_is_inf = m->uba_vector.array().isInf();
    std::vector<unsigned int> number_of_prev_equality(lhs_equals_rhs_constraint.size(), 0);
    std::vector<unsigned int> number_of_prev_lb_inequality(lhs_equals_rhs_constraint.size(), 0);
    std::vector<unsigned int> number_of_prev_ub_inequality(lhs_equals_rhs_constraint.size(), 0);
    std::vector<double> tmp_eq_vector;
    std::vector<double> tmp_ineq_lb_vector;
    std::vector<double> tmp_ineq_ub_vector;
    {

      // number_of_prev_equality and number_of_prev_inequality are two vectors that contains the number of
      // equality and inequality that can be found before the current index
      // number_of_prev_equality[i] = number of equality that can be found before index i
      // number_of_prev_inequality[i] = number of inequality that can be found before index i
      // For instance:
      //     equality and inequality   [i, e, e, e, i, i, i, e]
      //     lhs_equals_rgs_contraint  [f, t, t, t, f, f, f, t]
      //     number_of_prev_equality   [0, 0, 1, 3, 3, 3, 3, 3]
      //     number_of_prev_inequality [0, 1, 1, 1, 1, 2, 3, 4]
      for (std::size_t k=1; k<lhs_equals_rhs_constraint.size(); ++k) {
        if (lhs_equals_rhs_constraint[k-1]) {
          number_of_prev_equality[k] = number_of_prev_equality[k-1] + 1;
          number_of_prev_lb_inequality[k] = number_of_prev_lb_inequality[k-1];
          number_of_prev_ub_inequality[k] = number_of_prev_ub_inequality[k-1];
        } else {
          number_of_prev_equality[k] = number_of_prev_equality[k-1] + 1;
          if (lhs_is_inf[k]) {
              number_of_prev_lb_inequality[k] = number_of_prev_lb_inequality[k-1];
          } else {
              number_of_prev_lb_inequality[k] = number_of_prev_lb_inequality[k-1] +1;
          }
          if (rhs_is_inf[k]) {
              number_of_prev_ub_inequality[k] = number_of_prev_ub_inequality[k-1];
          } else {
              number_of_prev_ub_inequality[k] = number_of_prev_ub_inequality[k-1] +1;
          }
        }
      }

      for (std::size_t k=0; k<lhs_equals_rhs_constraint.size(); ++k) {
        if (lhs_equals_rhs_constraint[k]) {
          tmp_eq_vector.push_back(m->lba_vector[k]);
        } else {
          if (!lhs_is_inf[k]) {
              tmp_ineq_lb_vector.push_back(m->lba_vector[k]);
          }
          if (!rhs_is_inf[k]) {
              tmp_ineq_ub_vector.push_back(m->uba_vector[k]);
          }
        }
      }

      m->eq_b_vector.resize(tmp_eq_vector.size());
      if (tmp_eq_vector.size() > 0) {
        m->eq_b_vector = Eigen::Map<Eigen::VectorXd>(
          get_ptr(tmp_eq_vector), tmp_eq_vector.size());
      }

      m->lba_vector.resize(tmp_ineq_lb_vector.size());
      if (tmp_ineq_lb_vector.size() > 0) {
        m->lba_vector = Eigen::Map<Eigen::VectorXd>(
          get_ptr(tmp_ineq_lb_vector), tmp_ineq_lb_vector.size());
      }

      m->uba_vector.resize(tmp_ineq_ub_vector.size());
      if (tmp_ineq_ub_vector.size() > 0) {
        m->uba_vector = Eigen::Map<Eigen::VectorXd>(
          get_ptr(tmp_ineq_ub_vector), tmp_ineq_ub_vector.size());
      }
    }
    std::size_t n_eq = m->eq_b_vector.size();
    std::size_t n_ineq_lb = m->lba_vector.size();
    std::size_t n_ineq_ub = m->lbx_vector.size();
    std::size_t n_ineq = n_ineq_lb + n_ineq_ub;

    // Convert H_ from casadi::Sparsity to Eigen::SparseMatrix (misuse tripletList)
    H_.get_triplet(m->row, m->col);
    for (int k=0; k<H_.nnz(); ++k) {
      m->tripletList.push_back(PiqpMemory::TripletT(
        static_cast<double>(m->row[k]),
        static_cast<double>(m->col[k]),
        static_cast<double>(H[k])));
    }
    Eigen::SparseMatrix<double> H_spa(H_.size1(), H_.size2());
    H_spa.setFromTriplets(m->tripletList.begin(), m->tripletList.end());
    m->tripletList.clear();

    // Convert A_ from casadi Sparsity to Eigen::SparseMatrix and split
    // in- and equality constraints into different matrices
    m->tripletList.reserve(A_.nnz());
    A_.get_triplet(m->row, m->col);
    for (int k=0; k<A_.nnz(); ++k) {
      // Detect equality constraint
      if (lhs_equals_rhs_constraint[m->row[k]]) {
        // Equality constraint the row[k] is decreased by the number of previous inequality constraints
        m->tripletListEq.push_back(TripletT(
          static_cast<double>(number_of_prev_equality[m->row[k]]),
          static_cast<double>(m->col[k]),
          static_cast<double>(A[k])));
      } else {
        // Inequality constraint the row[k] is decreased by the number of previous equality constraints
		  if (!lhs_is_inf[m->row[k]]) {
			m->tripletList.push_back(TripletT(
			  static_cast<double>(number_of_prev_ub_inequality[m->row[k]]),
			  static_cast<double>(m->col[k]),
			  static_cast<double>(A[k])));
		  }
		  if (!lhs_is_inf[m->row[k]]) {
			m->tripletList.push_back(TripletT(
			  static_cast<double>(n_ineq_ub + number_of_prev_lb_inequality[m->row[k]]),
			  static_cast<double>(m->col[k]),
			  static_cast<double>(-A[k]))); // Reverse sign!
		  }
      }
    }

    // Handle constraints on decision variable x in inequality constraint matrix C
    Eigen::SparseMatrix<double> A_spa(n_eq, nx_);
    A_spa.setFromTriplets(m->tripletListEq.begin(), m->tripletListEq.end());
    m->tripletListEq.clear();

    Eigen::SparseMatrix<double> C_spa(n_ineq, nx_);
    C_spa.setFromTriplets(m->tripletList.begin(), m->tripletList.end());
    m->tripletList.clear();

    // Get stacked lower and upper inequality bounds
    m->ineq_b_vector.resize(n_ineq);
    m->ineq_b_vector << m->uba_vector, -m->lbx_vector;

    m->fstats.at("preprocessing").toc();

    // Solve Problem
    m->fstats.at("solver").tic();
    // TODO: if (m->ubx_vector.size() > 0 || m->lbx_vector.size() > 0) {
	//
    if (sparse_backend) {
		piqp::SparseSolver<double> solver;
		solver.settings() = settings_;

		solver.setup(P, c, A, b, G, h, x_lb, x_ub);
		m->status = solver.solve();

		m->results_x = std::make_unique<Eigen::VectorXd>(solver.results().x);
		m->results_y = std::make_unique<Eigen::VectorXd>(solver.results().y);
		m->results_z = std::make_unique<Eigen::VectorXd>(solver.results().z);
		m->objValue = solver.results().info.primal_obj;
    } else {
		piqp::DenseSolver<double> solver;
		solver.settings() = settings_;
		// TODO
		//  m->dense_solver = proxsuite::proxqp::dense::QP<double> (nx_, n_eq, n_ineq);
		//  m->dense_solver.init(Eigen::MatrixXd(H_spa), m->g_vector,
		//	Eigen::MatrixXd(A_spa), m->b_vector,
		//	Eigen::MatrixXd(C_spa), m->lb_vector, m->ub_vector);
		//  m->dense_solver.settings = settings_;

		//  m->dense_solver.solve();

		m->status = solver.solve();

		m->results_x = std::make_unique<Eigen::VectorXd>(solver.results().x);
		m->results_y = std::make_unique<Eigen::VectorXd>(solver.results().y);
		m->results_z = std::make_unique<Eigen::VectorXd>(solver.results().z);
		m->objValue = solver.results().info.primal_obj;
    }
    m->fstats.at("solver").toc();

    // Outputs
    double *x=res[CONIC_X],
           *cost=res[CONIC_COST],
           *lam_a=res[CONIC_LAM_A],
           *lam_x=res[CONIC_LAM_X];

    int ret;

	//	piqp_update_dense(m->work,
	//                      h, g,
	//                      nullptr, nullptr,
	//                      A, uba,
	//					  lbx, ubx);
	//

    // if (warm_start_primal_) {
    //   ret = piqp_warm_start_x(m->work, arg[CONIC_X0]);
    //   casadi_assert(ret==0, "Problem in piqp_warm_start_x");
    // }

    // if (warm_start_dual_) {
    //   casadi_copy(arg[CONIC_LAM_X0], nx_, w);
    //   casadi_copy(arg[CONIC_LAM_A0], na_, w+nx_);
    //   ret = piqp_warm_start_y(m->work, w);
    //   casadi_assert(ret==0, "Problem in piqp_warm_start_y");
    // }

    // Solve Problem
    // ret = piqp_solve(m->work);
    // casadi_assert(ret==0, "Problem in piqp_solve");

    // casadi_copy(m->work->result->x, nx_, res[CONIC_X]);
	// // TODO: Reconstruct: z_ub, z_lb, y and z!
    // casadi_copy(m->work->result->z_lb, nx_, res[CONIC_LAM_X]);
    // casadi_copy(m->work->result->z, na_, res[CONIC_LAM_A]);
    // if (res[CONIC_COST]) *res[CONIC_COST] = m->work->result->info.primal_obj;
	// // info.runtime

    // m->success = (ret == PIQP_SOLVED);
    // if (m->success) m->unified_return_status = SOLVER_RET_SUCCESS;

    return 0;
  }

  void PiqpInterface::codegen_free_mem(CodeGenerator& g) const {
    g << "piqp_cleanup(" + codegen_mem(g) + ");\n";
  }

  void PiqpInterface::codegen_init_mem(CodeGenerator& g) const {
    // Sparsity Asp = vertcat(Sparsity::diag(nx_), A_);
    // casadi_int dummy_size = max(nx_+na_, max(Asp.nnz(), H_.nnz()));

    // g.local("A", "piqp_csc");
    // g.local("dummy[" + str(dummy_size) + "]", "casadi_real");
    // g << g.clear("dummy", dummy_size) << "\n";

    // g.constant_copy("A_row", Asp.get_row());
    // g.constant_copy("A_colind", Asp.get_colind());
    // g.constant_copy("H_row", H_.get_row());
    // g.constant_copy("H_colind", H_.get_colind());

    // g.local("A", "piqp_csc");
    // g << "A.m = " << nx_ + na_ << ";\n";
    // g << "A.n = " << nx_ << ";\n";
    // g << "A.nz = " << nnzA_ << ";\n";
    // g << "A.nzmax = " << nnzA_ << ";\n";
    // g << "A.x = dummy;\n";
    // g << "A.i = A_row;\n";
    // g << "A.p = A_colind;\n";

    // g.local("H", "piqp_csc");
    // g << "H.m = " << nx_ << ";\n";
    // g << "H.n = " << nx_ << ";\n";
    // g << "H.nz = " << H_.nnz() << ";\n";
    // g << "H.nzmax = " << H_.nnz() << ";\n";
    // g << "H.x = dummy;\n";
    // g << "H.i = H_row;\n";
    // g << "H.p = H_colind;\n";

    // g.local("data", "PPIPData");
    // g << "data.n = " << nx_ << ";\n";
    // g << "data.m = " << nx_ + na_ << ";\n";
    // g << "data.P = &H;\n";
    // g << "data.q = dummy;\n";
    // g << "data.A = &A;\n";
    // g << "data.l = dummy;\n";
    // g << "data.u = dummy;\n";

    // g.local("settings", "piqp_settings");
    // g << "piqp_set_default_settings(&settings);\n";
    // g << "settings.rho = " << settings_.rho << ";\n";
    // g << "settings.sigma = " << settings_.sigma << ";\n";
    // g << "settings.scaling = " << settings_.scaling << ";\n";
    // g << "settings.adaptive_rho = " << settings_.adaptive_rho << ";\n";
    // g << "settings.adaptive_rho_interval = " << settings_.adaptive_rho_interval << ";\n";
    // g << "settings.adaptive_rho_tolerance = " << settings_.adaptive_rho_tolerance << ";\n";
    // //g << "settings.adaptive_rho_fraction = " << settings_.adaptive_rho_fraction << ";\n";
    // g << "settings.max_iter = " << settings_.max_iter << ";\n";
    // g << "settings.eps_abs = " << settings_.eps_abs << ";\n";
    // g << "settings.eps_rel = " << settings_.eps_rel << ";\n";
    // g << "settings.eps_prim_inf = " << settings_.eps_prim_inf << ";\n";
    // g << "settings.eps_dual_inf = " << settings_.eps_dual_inf << ";\n";
    // g << "settings.alpha = " << settings_.alpha << ";\n";
    // g << "settings.delta = " << settings_.delta << ";\n";
    // g << "settings.polish = " << settings_.polish << ";\n";
    // g << "settings.polish_refine_iter = " << settings_.polish_refine_iter << ";\n";
    // g << "settings.verbose = " << settings_.verbose << ";\n";
    // g << "settings.scaled_termination = " << settings_.scaled_termination << ";\n";
    // g << "settings.check_termination = " << settings_.check_termination << ";\n";
    // g << "settings.warm_start = " << settings_.warm_start << ";\n";
    // //g << "settings.time_limit = " << settings_.time_limit << ";\n";

    // g << codegen_mem(g) + " = piqp_setup(&data, &settings);\n";
    // g << "return 0;\n";
  }

  void PiqpInterface::codegen_body(CodeGenerator& g) const {
    // g.add_include("piqp/piqp.h");
    // g.add_auxiliary(CodeGenerator::AUX_INF);

    // g.local("work", "piqp_workspace", "*");
    // g.init_local("work", codegen_mem(g));

    // g.comment("Set objective");
    // g.copy_default(g.arg(CONIC_G), nx_, "w", "0", false);
    // g << "if (piqp_update_lin_cost(work, w)) return 1;\n";

    // g.comment("Set bounds");
    // g.copy_default(g.arg(CONIC_LBX), nx_, "w", "-casadi_inf", false);
    // g.copy_default(g.arg(CONIC_LBA), na_, "w+"+str(nx_), "-casadi_inf", false);
    // g.copy_default(g.arg(CONIC_UBX), nx_, "w+"+str(nx_+na_), "casadi_inf", false);
    // g.copy_default(g.arg(CONIC_UBA), na_, "w+"+str(2*nx_+na_), "casadi_inf", false);
    // g << "if (piqp_update_bounds(work, w, w+" + str(nx_+na_)+ ")) return 1;\n";

    // g.comment("Project Hessian");
    // g << g.tri_project(g.arg(CONIC_H), H_, "w", false);

    // g.comment("Get constraint matrix");
    // std::string A_colind = g.constant(A_.get_colind());
    // g.local("offset", "casadi_int");
    // g.local("n", "casadi_int");
    // g.local("i", "casadi_int");
    // g << "offset = 0;\n";
    // g << "for (i=0; i< " << nx_ << "; ++i) {\n";
    // g << "w[" + str(nnzHupp_) + "+offset] = 1;\n";
    // g << "offset++;\n";
    // g << "n = " + A_colind + "[i+1]-" + A_colind + "[i];\n";
    // g << "casadi_copy(" << g.arg(CONIC_A) << "+" + A_colind + "[i], n, "
    //      "w+offset+" + str(nnzHupp_) + ");\n";
    // g << "offset+= n;\n";
    // g << "}\n";

    // g.comment("Pass Hessian and constraint matrices");
    // g << "if (piqp_update_P_A(work, w, 0, " + str(nnzHupp_) + ", w+" + str(nnzHupp_) +
    //      ", 0, " + str(nnzA_) + ")) return 1;\n";

    // g << "if (piqp_warm_start_x(work, " + g.arg(CONIC_X0) + ")) return 1;\n";
    // g.copy_default(g.arg(CONIC_LAM_X0), nx_, "w", "0", false);
    // g.copy_default(g.arg(CONIC_LAM_A0), na_, "w+"+str(nx_), "0", false);
    // g << "if (piqp_warm_start_y(work, w)) return 1;\n";

    // g << "if (piqp_solve(work)) return 1;\n";

    // g.copy_check("&work->result->obj_val", 1, g.res(CONIC_COST), false, true);
    // g.copy_check("work->result->x", nx_, g.res(CONIC_X), false, true);
    // g.copy_check("work->result->y", nx_, g.res(CONIC_LAM_X), false, true);
    // g.copy_check("work->result->y+" + str(nx_), na_, g.res(CONIC_LAM_A), false, true);

    // g << "if (work->info->status_val != PIQP_SOLVED) return 1;\n";
  }

  Dict PiqpInterface::get_stats(void* mem) const {
    Dict stats = Conic::get_stats(mem);
    auto m = static_cast<PiqpMemory*>(mem);
	// TODO
    //stats["return_status"] = m->work->info->status;
    return stats;
  }

  PiqpMemory::PiqpMemory() {
  }

  PiqpMemory::~PiqpMemory() {
  }

  PiqpInterface::PiqpInterface(DeserializingStream& s) : Conic(s) {
	  // TODO
    // s.version("PiqpInterface", 1);
    // s.unpack("PiqpInterface::nnzHupp", nnzHupp_);
    // s.unpack("PiqpInterface::nnzA", nnzA_);
    // s.unpack("PiqpInterface::warm_start_primal", warm_start_primal_);
    // s.unpack("PiqpInterface::warm_start_dual", warm_start_dual_);

    // piqp_set_default_settings(&settings_);
    // s.unpack("PiqpInterface::settings::rho", settings_.rho);
    // s.unpack("PiqpInterface::settings::sigma", settings_.sigma);
    // s.unpack("PiqpInterface::settings::scaling", settings_.scaling);
    // s.unpack("PiqpInterface::settings::adaptive_rho", settings_.adaptive_rho);
    // s.unpack("PiqpInterface::settings::adaptive_rho_interval", settings_.adaptive_rho_interval);
    // s.unpack("PiqpInterface::settings::adaptive_rho_tolerance", settings_.adaptive_rho_tolerance);
    // //s.unpack("PiqpInterface::settings::adaptive_rho_fraction", settings_.adaptive_rho_fraction);
    // s.unpack("PiqpInterface::settings::max_iter", settings_.max_iter);
    // s.unpack("PiqpInterface::settings::eps_abs", settings_.eps_abs);
    // s.unpack("PiqpInterface::settings::eps_rel", settings_.eps_rel);
    // s.unpack("PiqpInterface::settings::eps_prim_inf", settings_.eps_prim_inf);
    // s.unpack("PiqpInterface::settings::eps_dual_inf", settings_.eps_dual_inf);
    // s.unpack("PiqpInterface::settings::alpha", settings_.alpha);
    // s.unpack("PiqpInterface::settings::delta", settings_.delta);
    // s.unpack("PiqpInterface::settings::polish", settings_.polish);
    // s.unpack("PiqpInterface::settings::polish_refine_iter", settings_.polish_refine_iter);
    // s.unpack("PiqpInterface::settings::verbose", settings_.verbose);
    // s.unpack("PiqpInterface::settings::scaled_termination", settings_.scaled_termination);
    // s.unpack("PiqpInterface::settings::check_termination", settings_.check_termination);
    // s.unpack("PiqpInterface::settings::warm_start", settings_.warm_start);
    // //s.unpack("PiqpInterface::settings::time_limit", settings_.time_limit);
  }

  void PiqpInterface::serialize_body(SerializingStream &s) const {
    // Conic::serialize_body(s);
    // s.version("PiqpInterface", 1);
    // s.pack("PiqpInterface::nnzHupp", nnzHupp_);
    // s.pack("PiqpInterface::nnzA", nnzA_);
    // s.pack("PiqpInterface::warm_start_primal", warm_start_primal_);
    // s.pack("PiqpInterface::warm_start_dual", warm_start_dual_);
    // s.pack("PiqpInterface::settings::rho", settings_.rho);
    // s.pack("PiqpInterface::settings::sigma", settings_.sigma);
    // s.pack("PiqpInterface::settings::scaling", settings_.scaling);
    // s.pack("PiqpInterface::settings::adaptive_rho", settings_.adaptive_rho);
    // s.pack("PiqpInterface::settings::adaptive_rho_interval", settings_.adaptive_rho_interval);
    // s.pack("PiqpInterface::settings::adaptive_rho_tolerance", settings_.adaptive_rho_tolerance);
    // //s.pack("PiqpInterface::settings::adaptive_rho_fraction", settings_.adaptive_rho_fraction);
    // s.pack("PiqpInterface::settings::max_iter", settings_.max_iter);
    // s.pack("PiqpInterface::settings::eps_abs", settings_.eps_abs);
    // s.pack("PiqpInterface::settings::eps_rel", settings_.eps_rel);
    // s.pack("PiqpInterface::settings::eps_prim_inf", settings_.eps_prim_inf);
    // s.pack("PiqpInterface::settings::eps_dual_inf", settings_.eps_dual_inf);
    // s.pack("PiqpInterface::settings::alpha", settings_.alpha);
    // s.pack("PiqpInterface::settings::delta", settings_.delta);
    // s.pack("PiqpInterface::settings::polish", settings_.polish);
    // s.pack("PiqpInterface::settings::polish_refine_iter", settings_.polish_refine_iter);
    // s.pack("PiqpInterface::settings::verbose", settings_.verbose);
    // s.pack("PiqpInterface::settings::scaled_termination", settings_.scaled_termination);
    // s.pack("PiqpInterface::settings::check_termination", settings_.check_termination);
    // s.pack("PiqpInterface::settings::warm_start", settings_.warm_start);
    //s.pack("PPPiInterface::settings::time_limit", settings_.time_limit);
  }

} // namespace casadi
