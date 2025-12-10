#include <iostream>
#include <vector>
#include "LagrangeInterp.h"

double LagrangeInterpOptm(const std::vector<double>& x, const std::vector<double>& y, double a) noexcept {
    const double* px = x.data();
    const double* py = y.data();
    const size_t n = x.size();

    const __m256d v_a = _mm256_set1_pd(static_cast<double>(a));

    double result = 0.0;

    for (size_t j = 0; j < n; ++j) {
        const double xj = px[j];
        const double yj = py[j];
        const __m256d v_xj = _mm256_set1_pd(xj);
        __m256d v_num_prod = _mm256_set1_pd(1.0);
        __m256d v_den_prod = _mm256_set1_pd(1.0);
        size_t i = 0;
        for (; i + 3 < j; i += 4) {
            __m256d v_xi = _mm256_loadu_pd(&px[i]);
            __m256d v_delta_x = _mm256_sub_pd(v_xj, v_xi); // xj - xi
            __m256d v_delta_a = _mm256_sub_pd(v_a, v_xi);  // a - xi
            v_den_prod = _mm256_mul_pd(v_den_prod, v_delta_x);
            v_num_prod = _mm256_mul_pd(v_num_prod, v_delta_a);
        }
        double scalar_num = 1.0;
        double scalar_den = 1.0;
        for (; i < j; ++i) {
            double xi = px[i];
            scalar_den *= (xj - xi);
            scalar_num *= (a - xi);
        }
        i = j + 1;
        for (; i + 3 < n; i += 4) {
            __m256d v_xi = _mm256_loadu_pd(&px[i]);

            __m256d v_delta_x = _mm256_sub_pd(v_xj, v_xi);
            __m256d v_delta_a = _mm256_sub_pd(v_a, v_xi);

            v_den_prod = _mm256_mul_pd(v_den_prod, v_delta_x);
            v_num_prod = _mm256_mul_pd(v_num_prod, v_delta_a);
        }
        for (; i < n; ++i) {
            double xi = px[i];
            scalar_den *= (xj - xi);
            scalar_num *= (a - xi);
        }
        scalar_num *= hmul_256(v_num_prod);
        scalar_den *= hmul_256(v_den_prod);
        result += yj * (scalar_num / scalar_den);
    }

    return result;
};

void runLagrange() {
    /*
    para printar caractéres unicode.
    */

    std::vector<double> xVal = { 1.0, 2.0, 3.0 };
    std::vector<double> yVal = { 2.0, 1.0, 2.0 };
    double a = 1.5;


    // 1. Chamada da sua função original (para comparação)
    double result = LagrangeInterp(xVal, yVal, a);

    // 2. Chamada da função AVX2 (A VERSÃO NUCLEAR)
    // O template 'T' é implicitamente 'double' aqui.
    double resultExtreme = LagrangeInterpOptm(xVal, yVal, a);

    // 3. Chamada da versão Baricêntrica
    LagrageInterpBari<double> interpol(xVal, yVal);
    double resultBari = interpol.evaluate(a);


    std::cout << "Ponto de avaliação 'a': " << a << "\n";
    std::cout << "------------------------------------------\n";
    std::cout << "1. Resultado Original (Lógica O(N²)): " << result << "\n";
    std::cout << "2. Resultado AVX2 (Otimizado p/ Hardware): " << resultExtreme << "\n";
    std::cout << "3. Resultado Baricêntrica (O(N) Lógica): " << resultBari << "\n";
    std::cout << "------------------------------------------\n";

    // Nota: Todos os resultados devem ser idênticos (neste caso, 1.75)
}