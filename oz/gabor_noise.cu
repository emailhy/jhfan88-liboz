/* 
 * Copyright (C) 2011 Adobe Systems Incorporated
 * Copyright (c) 2009 by Ares Lagae, Sylvain Lefebvre,
 * George Drettakis and Philip Dutre
 * 
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 * 
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */
#include <oz/gabor_noise.h>
#include <oz/generate.h>


namespace {

    class pseudo_random_number_generator {
    public:
        inline __device__ void seed(unsigned s) { x_ = s; }

        inline __device__ unsigned operator()() { x_ *= 3039177861u; return x_; }

        inline __device__ float uniform_0_1() { return float(operator()()) / float(UINT_MAX); }

        inline __device__ float uniform(float min, float max)
            { return min + (uniform_0_1() * (max - min)); }

        inline __device__ unsigned poisson(float mean) {
            float g_ = std::exp(-mean);
            unsigned em = 0;
            double t = uniform_0_1();
            while (t > g_) {
                ++em;
                t *= uniform_0_1();
            }
            return em;
        }

    private:
        unsigned x_;
    };


    static inline __device__ float gabor(float K, float a, float F_0, float omega_0, float x, float y) {
        float gaussian_envelop = K * std::exp(-CUDART_PI_F * (a * a) * ((x * x) + (y * y)));
        float sinusoidal_carrier = std::cos(2.0f * CUDART_PI_F * F_0 * ((x * std::cos(omega_0)) + (y * std::sin(omega_0))));
        return gaussian_envelop * sinusoidal_carrier;
    }


    static inline __device__ unsigned morton(unsigned x, unsigned y) {
        unsigned z = 0;
        for (unsigned i = 0; i < (sizeof(unsigned) * CHAR_BIT); ++i) {
            z |= ((x & (1 << i)) << i) | ((y & (1 << i)) << (i + 1));
        }
        return z;
    }


    struct noise_t {
        __device__  noise_t(float K, float a, float F_0, float number_of_impulses_per_kernel, unsigned random_offset)
            :  K_(K), a_(a), F_0_(F_0), random_offset_(random_offset)
        {
            kernel_radius_ = std::sqrt(-std::log(0.05f) / CUDART_PI_F) / a_;
            impulse_density_ = number_of_impulses_per_kernel / (CUDART_PI_F * kernel_radius_ * kernel_radius_);
        }

        inline __device__ float operator()(float x, float y) const {
            x /= kernel_radius_, y /= kernel_radius_;
            float int_x = std::floor(x), int_y = std::floor(y);
            float frac_x = x - int_x, frac_y = y - int_y;
            int i = int(int_x), j = int(int_y);
            float noise = 0;
            for (int di = -1; di <= +1; ++di) {
                for (int dj = -1; dj <= +1; ++dj) {
                    noise += cell(i + di, j + dj, frac_x - di, frac_y - dj);
                }
            }
            return noise;
        }

        inline __device__ float cell(int i, int j, float x, float y) const {
            //unsigned s = (((unsigned(j) % period_) * period_) + (unsigned(i) % period_)) + random_offset_; // periodic noise
            unsigned s = morton(i, j) + random_offset_; // nonperiodic noise
            if (s == 0) s = 1;
            pseudo_random_number_generator prng;
            prng.seed(s);
            double number_of_impulses_per_cell = impulse_density_ * kernel_radius_ * kernel_radius_;
            unsigned number_of_impulses = prng.poisson(number_of_impulses_per_cell);
            float noise = 0;
            for (unsigned i = 0; i < number_of_impulses; ++i) {
                float x_i = prng.uniform_0_1();
                float y_i = prng.uniform_0_1();
                float w_i = prng.uniform(-1.0f, +1.0f);
                float omega_0_i = prng.uniform(0.0f, (float)(2.0f * CUDART_PI_F));
                float x_i_x = x - x_i;
                float y_i_y = y - y_i;
                if (((x_i_x * x_i_x) + (y_i_y * y_i_y)) < 1.0f) {
                    //noise += w_i * gabor(K_, a_, F_0_, omega_0_, x_i_x * kernel_radius_, y_i_y * kernel_radius_); // anisotropic
                    noise += w_i * gabor(K_, a_, F_0_, omega_0_i, x_i_x * kernel_radius_, y_i_y * kernel_radius_); // isotropic
                }
            }
            return noise;
        }

        //__host__ float variance() const {
        //    float integral_gabor_filter_squared = ((K_ * K_) / (4.0 * a_ * a_)) * (1.0 + std::exp(-(2.0 * CUDART_PI_F * F_0_ * F_0_) / (a_ * a_)));
        //    return impulse_density_ * (1.0 / 3.0) * integral_gabor_filter_squared;
        //}

        float K_;
        float a_;
        float F_0_;
        float kernel_radius_;
        float impulse_density_;
        unsigned random_offset_;
    };


    struct GaborNoise : public oz::generator<float> { 
        unsigned w_;
        unsigned h_;
        float K_;
        float a_; 
        float F_0_; 
        float imp_per_kernel_; 
        float scale_;

        GaborNoise( unsigned w, unsigned h, float K, float a, float F_0, float imp_per_kernel, float scale )
            : w_(w), h_(h), K_(K), a_(a), F_0_(F_0), imp_per_kernel_(imp_per_kernel), scale_(scale) {}

        inline __device__ float operator()( int ix, int iy ) const {
            noise_t noise(K_, a_, F_0_, imp_per_kernel_, 0);
            float x = (float(ix) + 0.5f) - (float(w_) / 2.0f);
            float y = (float(h_ - iy - 1) + 0.5f) - (float(h_) / 2.0f);
            return 0.5f + (0.5f * (noise(x, y) / scale_));
        }
    };
}


oz::gpu_image oz::garbor_noise( unsigned w, unsigned h, float K, float a, float F_0, float imp_per_kernel ) {
    return generate(w, h, GaborNoise(w, h, K, a, F_0, imp_per_kernel, 1));
}
