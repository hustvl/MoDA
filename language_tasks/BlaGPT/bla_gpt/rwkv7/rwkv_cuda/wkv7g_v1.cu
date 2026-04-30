#include <stdio.h>
#include <assert.h>
#include "ATen/ATen.h"

typedef at::BFloat16 bf16;

template <typename F>
__global__ void kernel_forward(const int B, const int T, const int C, const int H,
                               const F *__restrict__ const _r, const F *__restrict__ const _w, const F *__restrict__ const _k, const F *__restrict__ const _v, const F *__restrict__ const _a, const F *__restrict__ const _b,
                               F *__restrict__ const _y, float *__restrict__ const _saa, float *__restrict__ const _sss)
{
    const int e = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;

    float state[_N_] = {0};
    __shared__ float r[_N_], k[_N_], w[_N_], a[_N_], b[_N_];

    float v[_T_];
    for (int _t = 0; _t < T; _t++)
    {
        const int t = e*T*C + h*_N_ + i + _t * C;
        v[_t] = float(_v[t]);
    }

    for (int _t = 0; _t < T; _t++)
    {
        const int t = e*T*C + h*_N_ + i + _t * C;
        __syncthreads();
        r[i] = float(_r[t]);
        w[i] = __expf(-__expf(float(_w[t])));
        k[i] = float(_k[t]);
        a[i] = float(_a[t]);
        b[i] = float(_b[t]);
        __syncthreads();

        float sa = 0;
        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            sa += a[j] * state[j];
        }
        _saa[t] = float(sa);

        float vv = v[_t];
        float y = 0;
        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float& s = state[j];
            s = s * w[j] + k[j] * vv + sa * b[j];
            y += s * r[j];
        }
        _y[t] = F(y);

        if ((_t+1) % _CHUNK_LEN_ == 0)
        {
            const int a = _t / _CHUNK_LEN_;
            const int c = _T_ / _CHUNK_LEN_;
            const int p = e*C*_N_*c + h*_N_*_N_*c + a*_N_*_N_ + i;
            #pragma unroll
            for (int j = 0; j < _N_; j++)
            {
                _sss[p + j*_N_] = float(state[j]);
            }
        }
    }
}

template <typename F>
__global__ void kernel_backward_zzz(const int B, const int T, const int C, const int H,
    const F *__restrict__ const _r, const F *__restrict__ const _w, const F *__restrict__ const _k, const F *__restrict__ const _a, const F *__restrict__ const _b, const F *__restrict__ const _gy,
    float *__restrict__ const _zzz)
{
    const int e = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;

    __shared__ float r[_N_], w[_N_], a[_N_], b[_N_];

    const int T_1 = e*T*C + (T-1)*C + h*_N_;
    float z[_N_];
    const float gy = _gy[T_1 + i];
    __syncthreads();
    r[i] = float(_r[T_1+i]);
    __syncthreads();
    #pragma unroll
    for (int j = 0; j < _N_; j++)
    {
        z[j] = gy * r[j];
    }

    for (int _t = T-2; _t > _CHUNK_LEN_-1; _t--)
    {
        const int t = e*T*C + h*_N_ + _t * C + i;
        const float gy = _gy[t];
        __syncthreads();
        r[i] = float(_r[t]);
        w[i] = __expf(-__expf(float(_w[t+C])));
        a[i] = float(_a[t+C]);
        b[i] = float(_b[t+C]);
        __syncthreads();

        float zz = 0;
        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            zz += b[j] * z[j];
        }
        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            z[j] = z[j] * w[j] + gy * r[j] + a[j] * zz;
            // printf("t %d i %d j %d z %f\n", _t, i, j, z[j]);
        }
        if (_t % _CHUNK_LEN_ == 0)
        {
            const int a = _t / _CHUNK_LEN_ - 1;
            const int c = _T_ / _CHUNK_LEN_ - 1;
            const int p = e*C*_N_*c + h*_N_*_N_*c + a*_N_*_N_ + i;
            #pragma unroll
            for (int j = 0; j < _N_; j++)
            {
                _zzz[p + j*_N_] = float(z[j]);
            }
        }
    }
}

template <typename F>
__global__ void kernel_backward_rwkv(const int B, const int T, const int C, const int H,
    const F *__restrict__ const _r, const F *__restrict__ const _w, const F *__restrict__ const _k, const F *__restrict__ const _v, const F *__restrict__ const _a, const F *__restrict__ const _b, const float *__restrict__ const _saa, const float *__restrict__ const _sss, const float *__restrict__ const _zzz,
    const F *__restrict__ const _gy, F *__restrict__ const _gr, F *__restrict__ const _gw, F *__restrict__ const _gk, F *__restrict__ const _gv, F *__restrict__ const _ga, F *__restrict__ const _gb)
{
    const int e = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int chunk = threadIdx.x;
    const int n_chunk = _T_ / _CHUNK_LEN_;

    float zzz[_N_*_N_] = {0}, sss[_N_*_N_] = {999}, saa[_N_] = {999};
    float r[_N_] = {999}, w[_N_] = {0}, w1[_N_] = {999}, winv[_N_] = {999}, ww[_N_] = {999};
    float k[_N_] = {0}, v[_N_] = {999}, a[_N_] = {0}, a1[_N_] = {999}, b[_N_] = {0}, b1[_N_] = {999}, gy[_N_] = {999};

    if (chunk != n_chunk - 1)
    {
        const int p = e*T*C + (chunk+1)*_CHUNK_LEN_*C + h*_N_;
        for (int i = 0; i < _N_; i++)
        {
            k[i] = float(_k[p+i]);
            a[i] = float(_a[p+i]);
            b[i] = float(_b[p+i]);
            w[i] = __expf(-__expf(float(_w[p+i])));
            const int p = e*C*_N_*(n_chunk-1) + h*_N_*_N_*(n_chunk-1) + chunk*_N_*_N_ + i*_N_;
            #pragma unroll
            for (int j = 0; j < _N_; j++)
            {
                zzz[i*_N_+j] = float(_zzz[p+j]);
            }
        }
    }
    for (int i = 0; i < _N_; i++)
    {
        const int p = e*C*_N_*n_chunk + h*_N_*_N_*n_chunk + chunk*_N_*_N_ + i*_N_;
        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            sss[i*_N_+j] = float(_sss[p+j]);
        }
    }

    for (int _t = _CHUNK_LEN_-1; _t > -1; _t--)
    {
        const int t = chunk * _CHUNK_LEN_ + _t;
        const int b_t_h = e*T*C + t*C + h*_N_;
        #pragma unroll
        for (int n = 0; n < _N_; n++)
        {
            w1[n] = w[n];
            a1[n] = a[n];
            b1[n] = b[n];

            r[n] = float(_r[b_t_h+n]);
            k[n] = float(_k[b_t_h+n]);
            v[n] = float(_v[b_t_h+n]);
            a[n] = float(_a[b_t_h+n]);
            b[n] = float(_b[b_t_h+n]);
            gy[n] = float(_gy[b_t_h+n]);
            saa[n] = float(_saa[b_t_h+n]);

            ww[n] = -__expf(float(_w[b_t_h+n]));
            w[n] = __expf(ww[n]);
            ww[n] = ww[n] * w[n];
            winv[n] = 1.0 / w[n];
        }

        for (int j = 0; j < _N_; j++)
        {
            float zz = 0;
            #pragma unroll
            for (int i = 0; i < _N_; i++)
            {
                zz += b1[i] * zzz[i*_N_+j];
            }
            const float gyj = gy[j];
            #pragma unroll
            for (int i = 0; i < _N_; i++)
            {
                zzz[i*_N_+j] = zzz[i*_N_+j] * w1[i] + gyj * r[i] + a1[i] * zz;
                // printf("t %d i %d j %d z %f\n",t,i,j,zzz[i*_N_+j]);
                // printf("t %d i %d j %d s %f\n",t,i,j,sss[i*_N_+j]);
            }
        }

        for (int i = 0; i < _N_; i++)
        {
            float gr = 0;
            #pragma unroll
            for (int j = 0; j < _N_; j++)
            {
                gr += gy[j] * sss[i*_N_+j];
            }
            _gr[b_t_h+i] = F(gr);
        }

        for (int i = 0; i < _N_; i++)
        {
            const float ki = k[i];
            const float bi = b[i];
            const float wi = winv[i];
            #pragma unroll
            for (int j = 0; j < _N_; j++)
            {
                sss[i*_N_+j] = (sss[i*_N_+j] - ki * v[j] - bi * saa[j]) * wi;
            }
        }

        float gv[_N_] = {0}; float as[_N_] = {0}; float bz[_N_] = {0};
        for (int i = 0; i < _N_; i++)
        {
            const float ki = k[i];
            const float ai = a[i];
            const float bi = b[i];
            float gw = 0;
            float gk = 0;
            #pragma unroll
            for (int j = 0; j < _N_; j++)
            {
                const float sij = sss[i*_N_+j];
                const float zij = zzz[i*_N_+j];
                gv[j] += ki * zij;
                as[j] += ai * sij;
                bz[j] += bi * zij;
                gw += sij * zij;
                gk += v[j] * zij;
            }
            _gw[b_t_h+i] = F(gw * ww[i]);
            _gk[b_t_h+i] = F(gk);
        }
        for (int i = 0; i < _N_; i++)
        {
            _gv[b_t_h+i] = F(gv[i]);
            float ga = 0;
            float gb = 0;
            #pragma unroll
            for (int j = 0; j < _N_; j++)
            {
                ga += bz[j] * sss[i*_N_+j];
                gb += as[j] * zzz[i*_N_+j];
            }
            _ga[b_t_h+i] = F(ga);
            _gb[b_t_h+i] = F(gb);
        }
    }
}

void cuda_forward(int B, int T, int C, int H, bf16 *r, bf16* w, bf16 *k, bf16 *v, bf16 *a, bf16 *b, bf16 *y, float *saa, float* sss)
{
    assert(H*_N_ == C);
    kernel_forward<<<dim3(B * H), dim3(_N_)>>>(B, T, C, H, r, w, k, v, a, b, y, saa, sss);
}

void cuda_backward(int B, int T, int C, int H, bf16 *r, bf16* w, bf16 *k, bf16 *v, bf16 *a, bf16 *b, float *saa, float* sss, float* zzz, bf16 *gy, bf16 *gr, bf16 *gw, bf16 *gk, bf16 *gv, bf16 *ga, bf16 *gb)
{
    assert(H*_N_ == C);
    assert(T%_CHUNK_LEN_ == 0);

    kernel_backward_zzz<<<dim3(B * H), dim3(_N_)>>>(B, T, C, H, r, w, k, a, b, gy, zzz);
    kernel_backward_rwkv<<<dim3(B * H), dim3(T/_CHUNK_LEN_)>>>(B, T, C, H, r, w, k, v, a, b, saa, sss, zzz, gy, gr, gw, gk, gv, ga, gb);
}