
/*********************************************************
 * Mass-Spring-Damper Simulation Python Boost Extension. *
 * 01 Dec 2017                                           *
 *                                                       *
 * Roger Stuckey                                         *
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
#include <boost/python/stl_iterator.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <iostream>
#include <map>
#include <string>


namespace msd_boost { // Avoid cluttering the global namespace.

    typedef boost::array<double, 2> state_type;
    typedef std::vector<double> vector_type;
    typedef std::vector<vector_type> matrix_type;
    typedef boost::numeric::odeint::result_of::make_dense_output< boost::numeric::odeint::runge_kutta_dopri5< state_type > >::type dense_stepper_type;

    typedef std::map<std::string, double> CoeffMap;
    typedef std::map<std::string, double>::iterator CoeffMapIt;

    typedef boost::python::list list_type;

    // From here: https://stackoverflow.com/a/19092051/642463
    template< typename T >
    inline
    std::vector< T > to_std_vector( const boost::python::list& iterable )
    {
        return std::vector< T >( boost::python::stl_input_iterator< T >( iterable ),
                                 boost::python::stl_input_iterator< T >( ) );
    };

    int MAXSIZE = 150;

    class Plant
    // The system plant model
    {
        public:
            double m;
            CoeffMap C;

            Plant() : m(30.48)
            {
                // _T_S.resize(N_S);
                // _D_S.resize(N_S);
                _N_S = 0;

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
            void set_initial_state(const boost::python::list& x0)
            {
                // _x0 = to_std_vector<double>(x0);
                for (int i = 0; i < boost::python::len(x0); ++i)
                    _x0[i] = boost::python::extract<double>(x0[i]);
            }
            void set_external_forces(const boost::python::list& T_S, const boost::python::list& D_S, std::string interp_kind)
            {
                // for (int i = 0; i < boost::python::len(T_S); i++)
                // {
                //     _T_S[i] = boost::python::extract<double>(T_S[i]);
                //     _D_S[i] = boost::python::extract<double>(D_S[i]);
                // }
                _T_S = to_std_vector<double>(T_S);
                // This will only work if D_S is a 1d list
                _D_S = to_std_vector<double>(D_S);
                _N_S = _T_S.size();
                _interp_kind = interp_kind;
            }
            double interp1d_zero(double t) const
            // Zero-order interpolation of the external force vector
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
            // First-order interpolation of the external force vector, with non-uniform sampling frequency
            {
                int n = 0;
                double dddt;

                if (t <= _T_S[0])
                {
                    return _D_S[0];
                }
                else if (_T_S[_N_S - 1] <= t)
                {
                    return _D_S[_N_S - 1];
                }

                while ((n < _N_S - 1) && (_T_S[n] < t))
                    n++;

                dddt = (_D_S[n] - _D_S[n - 1])/(_T_S[n] - _T_S[n - 1]);

                return _D_S[n - 1] + dddt*(t - _T_S[n - 1]);
            }
            double interp1d_linear_uniform(double t) const
            // First-order interpolation of the external force vector, with uniform sampling frequency
            {
                int n = 0;
                double dddt;

                if (t <= _T_S[0])
                {
                    return _D_S[0];
                }
                else if (_T_S[_N_S - 1] <= t)
                {
                    return _D_S[_N_S - 1];
                }

                n = (t - _T_S[0])/(_T_S[1] - _T_S[0]) + 1; // will be cast as int

                dddt = (_D_S[n] - _D_S[n - 1])/(_T_S[n] - _T_S[n - 1]);

                return _D_S[n - 1] + dddt*(t - _T_S[n - 1]);
            }
            void operator() (const state_type& x, state_type& xdot, double t)
            // Calculate the state rate from the state, external force and system parameters
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
    // The system observer model (required by odeint)
    {
        public:
            vector_type T;
            matrix_type X;
            vector_type D;

            matrix_type Xdot;
            vector_type F;

            int idx;

            list_type X_L;
            list_type Xdot_L;
            list_type F_L;

            Observer(int N) : idx(0) {
                // _N_S = N;
                _N = N;

                T.resize(N);
                X.resize(N);
                D.resize(N);
                Xdot.resize(N);
                F.resize(N);
                for (int n = 0; n < N; ++n)
                {
                    X[n].resize(2);
                    Xdot[n].resize(2);

                    boost::python::list x;
                    for (int i = 0; i < 2; ++i)
                    {
                        x.append(0.0);
                    }
                    X_L.append(x);
                    Xdot_L.append(x);
                    F_L.append(x);
                }
            }

            void operator() (const state_type& x, const double t)
            {
                // The index can exceed the length of X
                if ((idx >= 0) && (idx < _N)) {
                    T[idx] = t;
                    X[idx][0] = x[0];
                    X[idx][1] = x[1];
                    idx++;
                }
            }

            vector_type get_time_vector() {
                return T;
            }
            void set_time_vector(const vector_type& T_) {
                T = T_;
            }

        private:
            int _N;
    };

    int integrate(Plant& plant, Observer& observer, double t0, double dt, int N)
    // Perform the ODE integration over the time vector
    {
        dense_stepper_type stepper = boost::numeric::odeint::make_dense_output(1.0e-6, 1.0e-6, boost::numeric::odeint::runge_kutta_dopri5< state_type >());

        state_type x0 = plant.get_initial_state(); // Initial conditions

        observer.idx = 0;

        boost::numeric::odeint::integrate_n_steps(stepper, boost::ref(plant), x0, t0, dt, N - 1, boost::ref(observer));

        // Update the inertial force vector
        for (int n = 0; n < N; ++n)
        {
            observer.D[n] = plant.interp1d_zero(observer.T[n]);
            state_type x;
            state_type xdot;
            double f;
            for (int i = 0; i < 2; ++i)
            {
                x[i] = observer.X[n][i];
                observer.X_L[n][i] = x[i];
            }
            plant(x, xdot, observer.T[n]);
            for (int i = 0; i < 2; ++i)
            {
                observer.Xdot[n][i] = xdot[i];
                observer.Xdot_L[n][i] = xdot[i];
            }
            plant.forces(xdot, f);
            observer.F[n] = f;
            observer.F_L[n] = f;
        }

        return N;
    };
}

BOOST_PYTHON_MODULE(msdbx)
{
    using namespace boost::python;
    using namespace msd_boost;

    class_<vector_type>("vector_type")
        .def(vector_indexing_suite< vector_type >())
        ;
    class_<matrix_type>("matrix_type")
        .def(vector_indexing_suite< matrix_type >())
        ;
    class_<Plant>("Plant")
        .def("get_coeffs", &Plant::get_coeffs)
        .def("set_coeffs", &Plant::set_coeffs)
        .def("get_initial_state", &Plant::get_initial_state)
        .def("set_initial_state", &Plant::set_initial_state)
        .def("set_external_forces", &Plant::set_external_forces)
        ;
    class_<Observer>("Observer", init<int>())
        .def_readwrite("X", &Observer::X_L)
        .def_readwrite("Xdot", &Observer::Xdot_L)
        .def_readwrite("F", &Observer::F_L)
        .def("get_time_vector", &Observer::get_time_vector)
        .def("set_time_vector", &Observer::set_time_vector)
        ;

    def("integrate", integrate);
}
