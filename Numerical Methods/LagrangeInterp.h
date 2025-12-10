#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <immintrin.h>
#include <windows.h>

/*
    Essa é a implementação básica de Lagrange Interp.
    porém a checagem do vetor x é de O(n^2), então não é a função ideal para plots.
*/
template <typename T>
double LagrangeInterp(const std::vector<T>& x, const std::vector<T>& y, const double& a) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("x e y precisam ser do mesmo tamanho.");
    }
    if (x.empty()) {
        return 0.0;
    }

    size_t n = x.size();
    /*
        checar se tem divisão por zero, checagem básica porém suficiente.
    */
    for (size_t i = 0; i < n; ++i) {
        for (size_t k = i + 1; k < n; ++k) {
            const T& val1 = x[i];
            const T& val2 = x[k];

            if (val1 == val2) {
                throw std::runtime_error("Valores duplicados em x.");
            }
        }
    }
    double result = 0.0;

    for (size_t j = 0; j < n; ++j) {
        const T& xj = x[j];
        const T& yj = y[j];

        double term = static_cast<double>(yj);

        for (size_t i = 0; i < n; ++i) {
            if (i != j) {
                const T& xi = x[i];

                // L_ij(x) = (x - x_i) / (x_j - x_i)
                double numerator = a - xi;
                double denominator = xj - xi;
                term *= (numerator / denominator);
            }
        }
        result += term;
    }
    return result;
}

/*
    Para plots, precisamos usar Baricentro que calcula os pesos uma vez e torna
    a checagem de x de O(n^2) para O(n).
*/

template <typename T>
class LagrageInterpBari {
private:
    std::vector<T> x_nodes;
    std::vector<T> y_nodes;
    std::vector<double> weights;

public:
    LagrageInterpBari(const std::vector<T>& x, const std::vector<T>& y)
        : x_nodes(x), y_nodes(y) {

        if (x.size() != y.size()) throw std::invalid_argument("Size mismatch");

        size_t n = x.size();
        weights.resize(n);

        // 1. Pre calcular os pesos O(N^2)
        for (size_t j = 0; j < n; ++j) {
            double w = 1.0;
            for (size_t i = 0; i < n; ++i) {
                if (i != j) {
                    if (x[j] == x[i]) throw std::runtime_error("Duplicate X found");
                    w *= (x[j] - x[i]);
                }
            }
            weights[j] = 1.0 / w;
        }
    }

    double evaluate(double a) const {
        double numerator = 0.0;
        double denominator = 0.0;

        for (size_t j = 0; j < x_nodes.size(); ++j) {
            /*
                Usando \varepsilon pequeno para evitar divisão por zero.
            */
            if (std::abs(a - x_nodes[j]) < 1e-9) {
                return y_nodes[j];
            }
            double temp = weights[j] / (a - x_nodes[j]);
            numerator += temp * y_nodes[j];
            denominator += temp;
        }

        return numerator / denominator;
    }
};


/*
Melhor dos dois mundos, Baricentro porém tentando otimizar o máximo possível a função,
na documentação .md explico como isso pode ter 100x a performance do Lagrange normal.
*/

inline double hmul_256(__m256d v) {
    // Permuta e multiplica: [0, 1, 2, 3] -> [0*2, 1*3, X, X]
    __m256d v_perm = _mm256_permute4x64_pd(v, 0b10110001);
    __m256d v_mul1 = _mm256_mul_pd(v, v_perm);

    // Extrai a parte baixa e alta para __m128d e multiplica
    __m128d v_low = _mm256_castpd256_pd128(v_mul1);
    __m128d v_high = _mm256_extractf128_pd(v_mul1, 1);
    __m128d v_res = _mm_mul_pd(v_low, v_high);

    // Extrai o escalar final
    return _mm_cvtsd_f64(v_res);
}

double LagrangeInterpOptm(const std::vector<double>& x, const std::vector<double>& y, double a) noexcept;

void runLagrange();