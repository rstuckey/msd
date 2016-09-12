
/*********************************************************
 * Mass-Spring-Damper Simulation Python Boost Extension. *
 * 29 Jul 2013                                           *
 *                                                       *
 * Roger Stuckey                                         *
 * Defence Science and Technology Organisation           *
 *********************************************************/

#include <boost/array.hpp>
#include <boost/assign.hpp>
#include <boost/numeric/odeint.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/python/class.hpp>
#include <boost/python/def.hpp>
#include <boost/python/dict.hpp>
#include <boost/python/module.hpp>

#include <pyublas/numpy.hpp>

#include <iostream>
#include <map>
#include <string>



namespace { // Avoid cluttering the global namespace.

    // using namespace boost::numeric;

    typedef boost::array<double, 2> state_type;
    typedef pyublas::numpy_vector<double> vector_type;
    typedef pyublas::numpy_matrix<double> matrix_type;
    typedef boost::numeric::odeint::result_of::make_dense_output< boost::numeric::odeint::runge_kutta_dopri5< state_type > >::type dense_stepper_type;

    typedef std::map<std::string, double> CoeffMap;
    typedef std::map<std::string, double>::iterator CoeffMapIt;

    int MAXSIZE = 150;

    class Plant
    {
        public:
            double m;
            // double C_k;
            // double C_b;
            // double C_d;
            CoeffMap C;

            Plant() : m(30.48)
            {
                C = boost::assign::map_list_of ("k", -50.0) ("b", -10.0) ("d", 1.0) ("z", 0.0);
            }
            boost::python::dict get_coeffs() const
            {
                boost::python::dict py_dict;
                for (CoeffMap::const_iterator it = C.begin(); it != C.end(); ++it)
                    py_dict[it->first] = it->second;

                return py_dict;
            }
            void set_coeffs(boost::python::dict& py_dict)
            {
                boost::python::list keys = py_dict.keys();
                for (int i = 0; i < len(keys); ++i) {
                    boost::python::extract<std::string> extracted_key(keys[i]);
                    if ((!extracted_key.check()) || (!C.count(extracted_key)))
                    {
                        std::cout << "Key invalid, map might be incomplete" << std::endl;
                        continue;
                    }
                    std::string key = extracted_key;
                    boost::python::extract<double> extracted_val(py_dict[key]);
                    if (!extracted_val.check()) {
                        std::cout << "Value invalid, map might be incomplete" << std::endl;
                        continue;
                    }
                    double value = extracted_val;
                    C[key] = value;
                }
            }
            state_type get_initial_state() const
            {
                return _x0;
            }
            void set_initial_state(const vector_type& x0)
            {
                for (int i = 0; i < x0.size(); ++i)
                    _x0[i] = x0(i);
            }
            void set_external_forces(const vector_type& T_S, const vector_type& D_S, std::string interp_kind)
            {
                _T_S = T_S;
                _D_S = D_S;
                _N_S = T_S.size();
                _interp_kind = interp_kind;
            }
            double interp1d_zero(double t) const
            {
                int n = 0;

                if (t <= _T_S[0])
                    return _D_S[0];
                else if (_T_S[_N_S- 1] <= t)
                    return _D_S[_N_S - 1];

                while ((n < _N_S - 1) && (_T_S[n] < t))
                    n++;

                return _D_S[n - 1];
            }
            double interp1d_linear(double t) const
            {
                int n = 0;
                double dddt;

                if (t <= _T_S[0])
                {
                    // dddt = (_D_S[1] - _D_S[0])/(_T_S[1] - _T_S[0]);
                    return _D_S[0]; // - dddt*(_T_S[0] - t);
                }
                else if (_T_S[_N_S - 1] <= t)
                {
                    // dddt = (_D_S[_N_S - 1] - _D_S[_N_S - 2])/(_T_S[_N_S - 1] - _T_S[_N_S - 2]);
                    return _D_S[_N_S - 1]; // + dddt*(t - _T_S[_N_S - 1]);
                }

                while ((n < _N_S - 1) && (_T_S[n] < t))
                    n++;

                dddt = (_D_S[n] - _D_S[n - 1])/(_T_S[n] - _T_S[n - 1]);
                // std::cout << _D_S[n - 1] + dddt*(t - _T_S[n - 1]) << std::endl;
                return _D_S[n - 1] + dddt*(t - _T_S[n - 1]);
            }
            double interp1d_linear_uniform(double t) const
            {
                int n = 0;
                double dddt;

                if (t <= _T_S[0])
                {
                    // dddt = (_D_S[1] - _D_S[0])/(_T_S[1] - _T_S[0]);
                    return _D_S[0]; // - dddt*(_T_S[0] - t);
                }
                else if (_T_S[_N_S - 1] <= t)
                {
                    // dddt = (_D_S[_N_S - 1] - _D_S[_N_S - 2])/(_T_S[_N_S - 1] - _T_S[_N_S - 2]);
                    return _D_S[_N_S - 1]; // + dddt*(t - _T_S[_N_S - 1]);
                }

                n = (t - _T_S[0])/(_T_S[1] - _T_S[0]) + 1; // will be cast as int

                dddt = (_D_S[n] - _D_S[n - 1])/(_T_S[n] - _T_S[n - 1]);
                // std::cout << _D_S[n - 1] + dddt*(t - _T_S[n - 1]) << std::endl;
                return _D_S[n - 1] + dddt*(t - _T_S[n - 1]);
            }
            void operator() (const state_type& x, state_type& xdot, double t)
            {
                double d;

                if (_interp_kind == "zero")
                {
                    d = interp1d_zero(t);
                }
                else if (_interp_kind == "linear_uniform")
                {
                    d = interp1d_linear_uniform(t);
                }
                else // Default
                {
                    d = interp1d_linear(t);
                }

                xdot[0] = 1.0/m*(   0.0*x[0] +      m*x[1] +      0.0);
                xdot[1] = 1.0/m*(C["k"]*x[0] + C["b"]*x[1] + C["d"]*d + C["z"]*x[1]*x[1]); // z is a dummy coefficient
            }
            void forces(const state_type& xdot, double& f)
            {
                f = m*xdot[1];
            }

        private:
            state_type _x0;
            vector_type _T_S;
            vector_type _D_S;
            int _N_S;
            std::string _interp_kind;
    };

    class Observer
    {
        public:
            // ublas::vector<double> T
            vector_type T;
            // ublas::matrix<double> X;
            matrix_type X;
            // ublas::vector<double> D;
            vector_type D;

            matrix_type Xdot;
            vector_type F;

            int idx;

            Observer(int N) : T(N), X(N, 2), D(N), Xdot(N, 2), F(N), idx(0) { }

            void operator() (const state_type& x, const double t)
            {
                T(idx) = t;
                X(idx, 0) = x[0];
                X(idx, 1) = x[1];
                idx++;
            }
    };

    // int integrate(Plant& plant, Observer& observer, double t0, double tN, double dt)
    int integrate(Plant& plant, Observer& observer, double t0, double dt, int N)
    {
        // int N;

        dense_stepper_type stepper = boost::numeric::odeint::make_dense_output(1.0e-6, 1.0e-6, boost::numeric::odeint::runge_kutta_dopri5< state_type >());

        // boost::numeric::odeint::integrate_const(stepper, boost::ref(plant), plant.get_initial_state(), t0, tN, dt, boost::ref(observer));
        state_type x0 = plant.get_initial_state(); // Initial conditions

        // for (int i = 0; i < 2; i++)
        // {
        //     observer.X(0, i) = x0[i];
        // }
        // observer.idx = 1;

        observer.idx = 0;

        // boost::numeric::odeint::integrate_const(stepper, boost::ref(plant), x0, t0, tN, dt, boost::ref(observer));
        boost::numeric::odeint::integrate_n_steps(stepper, boost::ref(plant), x0, t0, dt, N - 1, boost::ref(observer));
        // N = observer.idx;

        for (int n = 0; n < N; n++)
        {
            observer.D(n) = plant.interp1d_zero(observer.T(n));
            state_type x;
            state_type xdot;
            double f;
            for (int i = 0; i < 2; i++)
            {
                x[i] = observer.X(n, i);
            }
            plant(x, xdot, observer.T(n));
            for (int i = 0; i < 2; i++)
            {
                observer.Xdot(n, i) = xdot[i];
            }
            plant.forces(xdot, f);
            observer.F(n) = f;
        }

        return N;
    };

    class System
    {
        public:
            Plant plant;
            Observer observer;

            System() : plant(), observer(MAXSIZE) { }
            // System(Plant& p) : plant(p), observer(MAXSIZE) { }

            void set_initial_state(const vector_type& x0)
            {
                plant.set_initial_state(x0);
            }

            void set_external_forces(const vector_type& T_S, const vector_type& D_S, std::string interp_kind)
            {
                plant.set_external_forces(T_S, D_S, interp_kind);
            }

            // void integrate(MSDS &msds, MSDO &msdo, double t0, double tN, double dt)
            int integrate(double t0, double tN, double dt)
            {
                int N;

                dense_stepper_type stepper = boost::numeric::odeint::make_dense_output(1.0e-6, 1.0e-6, boost::numeric::odeint::runge_kutta_dopri5< state_type >());

                state_type x0 = { { 0.0, 0.0 } }; // Initial conditions

                observer.idx = 0;
                boost::numeric::odeint::integrate_const(stepper, plant, x0, t0, tN, dt, observer);
                N = observer.idx;

                // T.reshape(N);
                // X.reshape(N);
                // D.reshape(N);

                for (int n = 0; n < N; n++)
                {
                    // msdo.D(n) = msds.interp1d_zero(msdo.T(n));
                    observer.D(n) = plant.interp1d_zero(observer.T(n));
                }

                return N;
            }
    };
}

BOOST_PYTHON_MODULE(msde)
{
    using namespace boost::python;

    class_<Plant>("Plant")
        .def("get_coeffs", &Plant::get_coeffs)
        .def("set_coeffs", &Plant::set_coeffs)
        .def("get_initial_state", &Plant::get_initial_state)
        .def("set_initial_state", &Plant::set_initial_state)
        .def("set_external_forces", &Plant::set_external_forces)
        ;
    class_<Observer>("Observer", init<int>())
        .def(pyublas::by_value_rw_member("T", &Observer::T))
        .def(pyublas::by_value_rw_member("X", &Observer::X))
        .def(pyublas::by_value_rw_member("D", &Observer::D))
        .def(pyublas::by_value_rw_member("Xdot", &Observer::Xdot))
        .def(pyublas::by_value_rw_member("F", &Observer::F))
        ;
    class_<System>("System")
        .def(pyublas::by_value_rw_member("plant", &System::plant))
        .def(pyublas::by_value_rw_member("observer", &System::observer))
        .def("set_initial_state", &System::set_initial_state)
        .def("set_external_forces", &System::set_external_forces)
        .def("integrate", &System::integrate);
        ;
    def("integrate", integrate);
}
