#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <numeric>


template<typename T>
class SVM {
private:
    T C;
    T tol;
    T eps;

    std::vector<std::vector<T>> X;
    std::vector<T> y;
    std::vector<T> alphas;
    T b;

    /* Cache de erros
     E_i = f(x_i) - y_i
    */
    std::vector<T> error_cache;

    T kernel(int i1, int i2) const {
        T sum = 0.0;
        const std::vector<T>& x1 = X[i1];
        const std::vector<T>& x2 = X[i2];
        for (size_t k = 0; k < x1.size(); ++k) {
            sum += x1[k] * x2[k];
        }
        return sum;
    }

    T compute_output(int i) const {
        T sum = b;
        for (size_t j = 0; j < X.size(); ++j) {
            if (alphas[j] > 0) {
                sum += alphas[j] * y[j] * kernel(j, i);
            }
        }
        return sum;
    }

public:
    SVM(T C_param = 1.0, T tolerance = 0.01, T epsilon = 1e-5)
        : C(C_param), tol(tolerance), eps(epsilon), b(0.0) {
    }

    bool takeStep(int i, int j) {
        if (i == j) return false;

        T alph1 = alphas[i];
        T alph2 = alphas[j];
        T y1 = y[i];
        T y2 = y[j];

        T E1 = (alphas[i] > 0 && alphas[i] < C) ? error_cache[i] : compute_output(i) - y1;
        T E2 = (alphas[j] > 0 && alphas[j] < C) ? error_cache[j] : compute_output(j) - y2;

        T s = y1 * y2;

        T L, H;
        if (y1 != y2) {
            L = std::max(T(0.0), alph2 - alph1);
            H = std::min(C, C + alph2 - alph1);
        }
        else {
            L = std::max(T(0.0), alph2 + alph1 - C);
            H = std::min(C, alph2 + alph1);
        }

        if (L == H) return false;

        T k11 = kernel(i, i);
        T k12 = kernel(i, j);
        T k22 = kernel(j, j);
        T eta = 2 * k12 - k11 - k22;

        T a2;
        if (eta < 0) {
            a2 = alph2 - (y2 * (E1 - E2)) / eta;
            if (a2 < L) a2 = L;
            else if (a2 > H) a2 = H;
        }
        else {
            return false;
        }

        if (std::abs(a2 - alph2) < eps * (a2 + alph2 + eps)) return false;

        T a1 = alph1 + s * (alph2 - a2);

        T b1 = b - E1 - y1 * (a1 - alph1) * k11 - y2 * (a2 - alph2) * k12;
        T b2 = b - E2 - y1 * (a1 - alph1) * k12 - y2 * (a2 - alph2) * k22;

        if (a1 > 0 && a1 < C) b = b1;
        else if (a2 > 0 && a2 < C) b = b2;
        else b = (b1 + b2) / 2.0;

        alphas[i] = a1;
        alphas[j] = a2;

        for (size_t k = 0; k < alphas.size(); ++k) {
            if (alphas[k] > 0 && alphas[k] < C) {
                error_cache[k] = compute_output(k) - y[k];
            }
        }

        if (alphas[i] > 0 && alphas[i] < C) error_cache[i] = compute_output(i) - y[i];
        if (alphas[j] > 0 && alphas[j] < C) error_cache[j] = compute_output(j) - y[j];

        return true;
    }

    bool examineExample(int i) {
        T y2 = y[i];
        T alph2 = alphas[i];

        T E2 = (alph2 > 0 && alph2 < C) ? error_cache[i] : compute_output(i) - y2;

        T r2 = E2 * y2;
        if ((r2 < -tol && alph2 < C) || (r2 > tol && alph2 > 0)) {

            int best_j = -1;
            T max_delta = -1.0;

            for (size_t k = 0; k < alphas.size(); ++k) {
                if (alphas[k] > 0 && alphas[k] < C) {
                    T E1 = error_cache[k];
                    T delta = std::abs(E1 - E2);
                    if (delta > max_delta) {
                        max_delta = delta;
                        best_j = k;
                    }
                }
            }

            if (best_j != -1) {
                if (takeStep(best_j, i)) return true;
            }

            for (size_t k = 0; k < alphas.size(); ++k) {
                if (alphas[k] > 0 && alphas[k] < C) {
                    if (takeStep(k, i)) return true;
                }
            }

            for (size_t k = 0; k < alphas.size(); ++k) {
                if (takeStep(k, i)) return true;
            }
        }
        return false;
    }

    void fit(const std::vector<std::vector<T>>& X_train, const std::vector<T>& y_train) {
        X = X_train;
        y = y_train;
        size_t n = X.size();

        alphas.assign(n, T(0.0));
        error_cache.assign(n, T(0.0));
        b = T(0.0);

        int numChanged = 0;
        bool examineAll = true;

        while (numChanged > 0 || examineAll) {
            numChanged = 0;
            if (examineAll) {
                for (size_t i = 0; i < n; ++i) {
                    if (examineExample(i)) numChanged++;
                }
            }
            else {
                for (size_t i = 0; i < n; ++i) {
                    if (alphas[i] > 0 && alphas[i] < C) {
                        if (examineExample(i)) numChanged++;
                    }
                }
            }

            if (examineAll == true) {
                examineAll = false;
            }
            else if (numChanged == 0) {
                examineAll = true;
            }
        }
    }

    T predict(const std::vector<T>& input) const {
        T result = b;
        for (size_t i = 0; i < X.size(); ++i) {
            if (alphas[i] > 0) {
                T k_val = 0.0;
                const std::vector<T>& xi = X[i];
                for (size_t j = 0; j < input.size(); ++j) {
                    k_val += xi[j] * input[j];
                }
                result += alphas[i] * y[i] * k_val;
            }
        }
        return (result >= 0) ? T(1.0) : T(-1.0);
    }
    const std::vector<T>& get_alphas() const { return alphas; }
    const std::vector<std::vector<T>>& get_support_vectors() const { return X; }
    T get_bias() const { return b; }
};

void runSVM();