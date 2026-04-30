#include <cuda_bf16.h>
#include <stdio.h>

//TODO: static? inline? __align__(16)?

using bf = __nv_bfloat16;
using bf2 = __nv_bfloat162;
using uint = unsigned int;
__device__ inline float to_float(const bf & u) { return __bfloat162float(u); }
__device__ inline bf to_bf(const float & u) { return 	__float2bfloat16_rn(u); }
__device__ inline float2 to_float2(const bf2 & u) { return 	__bfloat1622float2(u); }
__device__ inline float2 to_float2(const float2 & u) { return u; }
__device__ inline bf2 to_bf2(const float2 & u) { return __float22bfloat162_rn(u); }
__device__ inline uint& as_uint(const bf2&x) { return *((uint*)(&x)); }
__device__ inline uint __smem(const void*x) { return __cvta_generic_to_shared(x); }

__device__ void __commit_group() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ void __wait_group() { asm volatile("cp.async.wait_all;\n" ::); }
template<int N> __device__ void __wait_groups() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

__device__ void __copy_wait() { __commit_group(); __wait_group(); }

__device__ void operator*=(float2&a, const float2&b) { a.x *= b.x; a.y *= b.y; }
__device__ void operator+=(float2&a, const float2&b) { a.x += b.x; a.y += b.y; }
__device__ float2 operator+(const float2&a, const float2&b) { return {a.x+b.x,a.y+b.y}; }
__device__ float2 operator*(const float2&a, const float2&b) { return {a.x*b.x,a.y*b.y}; }

struct STile;
struct RTile;
struct FTile;

struct GTile {
    bf*ga;
    int stride;
    __device__ GTile(bf*ga_, int stride_) : ga(ga_), stride(stride_) {}
    __device__ GTile& operator=(const RTile&);
};
struct GFTile {
    float*ga;
    int stride;
    __device__ GFTile(float*ga_, int stride_) : ga(ga_), stride(stride_) {}
    __device__ GFTile& operator=(const FTile&);
};
struct STileT { STile*st; };

struct __align__(16) STile {
    bf data[16*16];
    __device__ STile() {}
    __device__ STile(const RTile&o) { *this=o; }
    __device__ STile& operator=(const GTile&);
    __device__ STile& operator=(const RTile&);
    __device__ STileT t() { return STileT{this}; }
};
struct Product { const RTile*a, *b; };
struct ProductPlus { const RTile*a, *b; const FTile* c; };
struct RTile {
    bf2 data[4];
    __device__ RTile() {}
    __device__ void zero_() { data[0] = data[1] = data[2] = data[3] = to_bf2({0.f,0.f}); }
    __device__ RTile(const STile&o) { *this=o; }
    __device__ RTile(const STileT&o) { *this=o; }
    __device__ RTile(const FTile&o) { *this=o; }
    __device__ RTile& operator=(const STile&);
    __device__ RTile& operator=(const STileT&);
    __device__ RTile& operator=(const FTile&fa);
    __device__ RTile& operator=(const GTile&);
};
struct FTile {
    union {
        float2 data[4];
        float fdata[8];
    };
    __device__ void zero_() { data[0] = data[1] = data[2] = data[3] = {0.f,0.f}; }
    __device__ FTile() {}
    __device__ FTile(const FTile&o) { for (int i = 0; i < 4; i++) data[i] = o.data[i]; }
    __device__ FTile(const RTile&r) { *this=r; }
    __device__ FTile(const Product&p) { *this=p; }
    __device__ FTile(const ProductPlus&p) { *this=p; }
    __device__ FTile& operator=(const Product&);
    __device__ FTile& operator=(const RTile&);
    __device__ FTile& operator=(const ProductPlus&);
    __device__ FTile& operator+=(const Product&);
    __device__ FTile& operator+=(const FTile&o) { for (int i = 0; i < 4; i++) data[i] += o.data[i]; return *this; }
};

__device__ void print(STile t) {
    if (threadIdx.x == 0) {
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 16; j++) {
                printf("%f ", to_float(t.data[i*16+j]));
            }
            printf("\n");
        }
        printf("\n");
    }
}

template<class T>
__device__ void print(T t, int warpi = 0) {
    int tid = threadIdx.x - warpi*32;
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j += 2) {
            if (tid == i%8*4+j%8/2) {
                float2 xy = to_float2(t.data[i/8+j/8*2]);
                printf("%f %f ", xy.x, xy.y);
                //printf("T%d:{a%d,a%d} ", threadIdx.x, (i/8+j/8*2)*2, (i/8+j/8*2)*2+1);
            }
            __syncthreads();
        }
        if (tid == 0) printf("\n");
            __syncthreads();
    }
    if (tid == 0) printf("\n");
    __syncthreads();
}

template<class T>
__device__ void print8(T mat) {
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j += 2) {
            if (threadIdx.x == i%8*4+j%8/2) {
                float2 xy = to_float2(mat);
                printf("%f %f ", xy.x, xy.y);
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) printf("\n");
            __syncthreads();
    }
    if (threadIdx.x == 0) printf("\n");
    __syncthreads();
}



__device__ void load(STile&sa, bf*ga, int stride) {
    int i = threadIdx.x%32/2, j = threadIdx.x%2;
    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" :: "r"(__smem(&sa.data[i*16+j*8])), "l"(ga+stride*i+j*8), "n"(16));
}

__device__ void load(RTile&ra, const STile&sa) {
    int i = threadIdx.x%8, j = threadIdx.x%32/16, k = threadIdx.x/8%2;
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(as_uint(ra.data[0])), "=r"(as_uint(ra.data[1])), "=r"(as_uint(ra.data[2])), "=r"(as_uint(ra.data[3]))
            : "r"(__smem(&sa.data[i*16+j*8+k*8*16])));
}
__device__ void loadT(RTile&ra, const STile&sa) {
    int i = threadIdx.x%8, j = threadIdx.x%32/16, k = threadIdx.x/8%2;
    asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(as_uint(ra.data[0])), "=r"(as_uint(ra.data[1])), "=r"(as_uint(ra.data[2])), "=r"(as_uint(ra.data[3]))
            : "r"(__smem(&sa.data[i*16+j*8*16+k*8])));
}

__device__ static inline void __m16n8k16(float2&d0, float2&d1, const bf2 &a0, const bf2 &a1, const bf2 &a2, const bf2 &a3, const bf2 &b0, const bf2 &b1, const float2 &c0, const float2 &c1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
            : "=f"(d0.x), "=f"(d0.y), "=f"(d1.x), "=f"(d1.y)
            : "r"(as_uint(a0)), "r"(as_uint(a1)), "r"(as_uint(a2)), "r"(as_uint(a3)),
              "r"(as_uint(b0)), "r"(as_uint(b1)),
              "f"(c0.x), "f"(c0.y), "f"(c1.x), "f"(c1.y));
}
__device__ void mma(FTile&rd, const RTile&ra, const RTile&rb, const FTile&rc) { // d = a*b^T + c
    __m16n8k16(rd.data[0],rd.data[1], ra.data[0],ra.data[1],ra.data[2],ra.data[3], rb.data[0],rb.data[2], rc.data[0],rc.data[1]);
    __m16n8k16(rd.data[2],rd.data[3], ra.data[0],ra.data[1],ra.data[2],ra.data[3], rb.data[1],rb.data[3], rc.data[2],rc.data[3]);
}
__device__ static inline void __m16n8k16(float2&d0, float2&d1, const bf2 &a0, const bf2 &a1, const bf2 &a2, const bf2 &a3, const bf2 &b0, const bf2 &b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
            : "+f"(d0.x), "+f"(d0.y), "+f"(d1.x), "+f"(d1.y)
            : "r"(as_uint(a0)), "r"(as_uint(a1)), "r"(as_uint(a2)), "r"(as_uint(a3)),
              "r"(as_uint(b0)), "r"(as_uint(b1)),
              "f"(d0.x), "f"(d0.y), "f"(d1.x), "f"(d1.y));
}
__device__ void mma(FTile&rd, const RTile&ra, const RTile&rb) { // d += a*b^T
    __m16n8k16(rd.data[0],rd.data[1], ra.data[0],ra.data[1],ra.data[2],ra.data[3], rb.data[0],rb.data[2]);
    __m16n8k16(rd.data[2],rd.data[3], ra.data[0],ra.data[1],ra.data[2],ra.data[3], rb.data[1],rb.data[3]);
}
__device__ void mm(FTile&rd, const RTile&ra, const RTile&rb) { // d = a*b^T
    __m16n8k16(rd.data[0],rd.data[1], ra.data[0],ra.data[1],ra.data[2],ra.data[3], rb.data[0],rb.data[2], {0.f,0.f}, {0.f,0.f});
    __m16n8k16(rd.data[2],rd.data[3], ra.data[0],ra.data[1],ra.data[2],ra.data[3], rb.data[1],rb.data[3], {0.f,0.f}, {0.f,0.f});
}

__device__ void store(const FTile&ra, float*ga, int stride) {
    int i = threadIdx.x%32/4, j = threadIdx.x%4*2;
    *((float2*)&ga[ i   *stride+j  ]) = ra.data[0];
    *((float2*)&ga[(i+8)*stride+j  ]) = ra.data[1];
    *((float2*)&ga[ i   *stride+j+8]) = ra.data[2];
    *((float2*)&ga[(i+8)*stride+j+8]) = ra.data[3];
}

__device__ void store(const RTile&ra, bf*ga, int stride) {
    int i = threadIdx.x%32/4, j = threadIdx.x%4*2;
    *((bf2*)&ga[ i   *stride+j  ]) = ra.data[0];
    *((bf2*)&ga[(i+8)*stride+j  ]) = ra.data[1];
    *((bf2*)&ga[ i   *stride+j+8]) = ra.data[2];
    *((bf2*)&ga[(i+8)*stride+j+8]) = ra.data[3];
}
__device__ void load(RTile&ra, bf*ga, int stride) {
    int i = threadIdx.x%32/4, j = threadIdx.x%4*2;
    ra.data[0] = *((bf2*)&ga[ i   *stride+j  ]);
    ra.data[1] = *((bf2*)&ga[(i+8)*stride+j  ]);
    ra.data[2] = *((bf2*)&ga[ i   *stride+j+8]);
    ra.data[3] = *((bf2*)&ga[(i+8)*stride+j+8]);
}
__device__ void store(const RTile&ra, STile&sa) { //TODO: reduce bank conflicts?
    int i = threadIdx.x%32/4, j = threadIdx.x%4*2;
    *((bf2*)&sa.data[ i   *16+j  ]) = ra.data[0];
    *((bf2*)&sa.data[(i+8)*16+j  ]) = ra.data[1];
    *((bf2*)&sa.data[ i   *16+j+8]) = ra.data[2];
    *((bf2*)&sa.data[(i+8)*16+j+8]) = ra.data[3];
}

__device__ void convert(RTile&ra, const FTile&fa) {
    ra.data[0] = to_bf2(fa.data[0]);
    ra.data[1] = to_bf2(fa.data[1]);
    ra.data[2] = to_bf2(fa.data[2]);
    ra.data[3] = to_bf2(fa.data[3]);
}
__device__ void convert(FTile&fa, const RTile&ra) {
    fa.data[0] = to_float2(ra.data[0]);
    fa.data[1] = to_float2(ra.data[1]);
    fa.data[2] = to_float2(ra.data[2]);
    fa.data[3] = to_float2(ra.data[3]);
}

__device__ STile& STile::operator=(const GTile& ga) { load(*this, ga.ga, ga.stride); return *this; }
__device__ RTile& RTile::operator=(const GTile& ga) { load(*this, ga.ga, ga.stride); return *this; }
__device__ RTile& RTile::operator=(const STile& sa) { load(*this, sa); return *this; }
__device__ STile& STile::operator=(const RTile& ra) { store(ra, *this); return *this; }
__device__ RTile& RTile::operator=(const STileT& sa) { loadT(*this, *sa.st); return *this; }
__device__ Product operator%(const RTile&ra, const RTile&rb) { return Product{&ra,&rb}; }
__device__ ProductPlus operator+(const Product&prod, const FTile&rc) { return ProductPlus{prod.a,prod.b,&rc}; }
__device__ FTile& FTile::operator=(const Product& prod) { mm(*this, *prod.a, *prod.b); return *this; }
__device__ FTile& FTile::operator=(const ProductPlus& prod) { mma(*this, *prod.a, *prod.b, *prod.c); return *this; }
__device__ FTile& FTile::operator+=(const Product& prod) { mma(*this, *prod.a, *prod.b); return *this; }
__device__ RTile& RTile::operator=(const FTile&fa) { convert(*this,fa); return *this; }
__device__ FTile& FTile::operator=(const RTile&ra) { convert(*this,ra); return *this; }
__device__ GTile& GTile::operator=(const RTile&ra) { store(ra, this->ga, this->stride); return *this; }
__device__ GFTile& GFTile::operator=(const FTile&fa) { store(fa, this->ga, this->stride); return *this; }

// Is this kind of cumsum better than multiplying with a triangular matrix of ones?
template<int inclusive, int rev>
__device__ FTile cumsumv(FTile&w) {
    int tid = threadIdx.x%32, t = tid/4;

    FTile ret;
    if (inclusive) for (int i = 0; i < 4; i++) ret.data[i] = w.data[i];
    else for (int i = 0; i < 4; i++) ret.data[i] = float2{0.f,0.f};

    for (int b = 0; b < 3; b++) {
        for (int i = 0; i < 8; i++) {
            float other_w = __shfl_xor_sync(0xffffffff, w.fdata[i], 4<<b);
            if ((t>>b)%2 == !rev) ret.fdata[i] += other_w;
            w.fdata[i] += other_w;
        }
    }
    for (int i : {0,1,4,5}) {
        float &w0 = w.fdata[i^(2*!rev)], &w1 = w.fdata[i^(2*rev)];
        ret.fdata[i^(2*!rev)] += w1;
        w0 += w1;
        w1 = w0;
    }
    return ret;
}

template<int inclusive, int rev>
__device__ FTile cumprodv(FTile&w) {
    int tid = threadIdx.x%32, t = tid/4;

    FTile ret;
    if (inclusive) for (int i = 0; i < 4; i++) ret.data[i] = w.data[i];
    else for (int i = 0; i < 4; i++) ret.data[i] = float2{1.f,1.f};

    for (int b = 0; b < 3; b++) {
        for (int i = 0; i < 8; i++) {
            float other_w = __shfl_xor_sync(0xffffffff, w.fdata[i], 4<<b);
            if ((t>>b)%2 == !rev) ret.fdata[i] *= other_w;
            w.fdata[i] *= other_w;
        }
    }
    for (int i : {0,1,4,5}) {
        float &w0 = w.fdata[i^(2*!rev)], &w1 = w.fdata[i^(2*rev)];
        ret.fdata[i^(2*!rev)] *= w1;
        w0 *= w1;
        w1 = w0;
    }
    return ret;
}

__device__ FTile operator*(const FTile&a, const FTile&b) {
    FTile ret;
    for (int i = 0; i < 8; i++) ret.fdata[i] = a.fdata[i]*b.fdata[i];
    return ret;
}

template<int triangular = 0, int WARPS> // Lower triangular
__device__ FTile sum_warp(float2*share, const FTile&f) {
    int tid = threadIdx.x%32, warpi = threadIdx.x/32;
    FTile sum;
    sum.zero_();
    for (int i : {0,1,2,3}) {
        if (i == 2 && triangular) continue;
        for (int j = 0; j < WARPS; j++) {
            if (warpi == j) share[tid] = f.data[i];
            __syncthreads();
           sum.data[i].x += share[tid].x;
           sum.data[i].y += share[tid].y;
            __syncthreads();
        }
    }
    return sum;
}

__device__ RTile from_warp(const RTile&ra, int src, float4*share) {
    int tid = threadIdx.x%32, warpi = threadIdx.x/32;
    RTile ret;
    if (warpi == src) share[tid] = *((float4*)ra.data);
    __syncthreads();
    *((float4*)ret.data) = share[tid];
    __syncthreads();
    return ret;
}

// transpose(inv(I-f)) where f is strictly lower triangular
// Numerically unstable?
__device__ FTile tri_minvT(const FTile&f) {
    int i0 = threadIdx.x%32/4, j0 = threadIdx.x%4*2;

    auto getCol = [&](int col, float eye) {
        FTile ret;
        for (int k = 0; k < 8; k++) {
            int i = i0+k/2%2*8, j = j0+k%2+k/4*8;
            if (j == col && j < i)
                ret.fdata[k] = f.fdata[k]; // -
            else
                ret.fdata[k] = float(i==j)*eye;
        }
        return ret;
    };
    FTile prod = getCol(-1, 1.f);
    for (int i = 0; i < 15; i++)
        prod += (RTile)prod % getCol(i, 0.f);
    return prod;
}

template<int strict>
__device__ FTile tril(const FTile&f) {
    int i0 = threadIdx.x%32/4, j0 = threadIdx.x%4*2;
    FTile ret;
    for (int k = 0; k < 8; k++) {
        int i = i0+k/2%2*8, j = j0+k%2+k/4*8;
        if (strict) ret.fdata[k] = (i>j ? f.fdata[k] : 0.f);
        else ret.fdata[k] = (i>=j ? f.fdata[k] : 0.f);
    }
    return ret;
}

template<class F>
__device__ void apply_(FTile&tile, F f) {
    for (int i = 0; i < 8; i++) tile.fdata[i] = f(tile.fdata[i]);
}

__device__ bf2 transpose(bf2 a) {
    bf2 ret;
    asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(as_uint(ret)) : "r"(as_uint(a)));
    return ret;
}

__device__ RTile transpose(const RTile&ra) {
    RTile rb;
    rb.data[0] = transpose(ra.data[0]);
    rb.data[1] = transpose(ra.data[2]);
    rb.data[2] = transpose(ra.data[1]);
    rb.data[3] = transpose(ra.data[3]);
    return rb;
}

template<int strict>
__device__ FTile slow_dw(const RTile&A, const RTile&q, const RTile&k, STile*share) {
    share[0] = A;
    share[1] = q;
    share[2] = k;
    __syncthreads();
    if (threadIdx.x%32 == 0) {
        for (int k = 0; k < 16; k++) {
            for (int j = 0; j < 16; j++) {
                float sum = 0;
                for (int l = 0; l < k; l++) {
                    for (int r = k+strict; r < 16; r++) {
                        sum += to_float(share[0].data[r*16+l]) * to_float(share[1].data[r*16+j]) * to_float(share[2].data[l*16+j]);
                    }
                }
                share[3].data[k*16+j] = to_bf(sum);
            }
        }
    }
    __syncthreads();
    RTile ret = (RTile)share[3];
    __syncthreads();
    return ret;
}


__device__ static inline void __m16n8k8(float2&d0, float2&d1, const bf2 &a0, const bf2 &a1, const bf2 &b0) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
            : "=f"(d0.x), "=f"(d0.y), "=f"(d1.x), "=f"(d1.y) : "r"(as_uint(a0)), "r"(as_uint(a1)), "r"(as_uint(b0)), "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f));
}

template<int strict>
__device__ RTile fast_dw(const RTile&A, const RTile&q, const RTile&k) {
    float2 qkA8[4];
    RTile kt = transpose(k), qt = transpose(q);
    __m16n8k8(qkA8[0],qkA8[1], qt.data[2], qt.data[3], transpose(A.data[1]));
    __m16n8k8(qkA8[2],qkA8[3], kt.data[0], kt.data[1], A.data[1]);
    for (int x : {0,1}) {
        qkA8[x] *= to_float2(kt.data[x]);
        qkA8[2+x] *= to_float2(qt.data[2+x]);
    }

    int tid = threadIdx.x%32, j = threadIdx.x%4;
    // Non-inclusive cumsum
    for (int i = 0; i < 4; i++) {
        float sum = qkA8[i].x+qkA8[i].y;
        float psum = __shfl_xor_sync(0xffffffff, sum, 1);
        float ppsum = __shfl_xor_sync(0xffffffff, sum+psum, 2);
        if (i < 2) {
            psum = ppsum*(j>=2)+psum*(j%2);
            qkA8[i].y = psum + qkA8[i].x;
            qkA8[i].x = psum;
        } else {
            psum = ppsum*(j<2)+psum*(j%2==0);
            qkA8[i].x = psum + qkA8[i].y;
            qkA8[i].y = psum;
        }
    }

    float2 qkA4[4];
    {
        RTile k_q;
        for (int i = 0; i < 8; i++) ((bf*)k_q.data)[i] = (j<2?((bf*)kt.data)[i]:((bf*)qt.data)[i]);
        float lower_left = (tid >= 16 && j < 2);
        bf2 A0 = to_bf2(to_float2(A.data[0])*float2{lower_left,lower_left});
        bf2 A3 = to_bf2(to_float2(A.data[3])*float2{lower_left,lower_left});
        __m16n8k8(qkA4[0],qkA4[1], k_q.data[0], k_q.data[1], A0 + transpose(A0));
        __m16n8k8(qkA4[2],qkA4[3], k_q.data[2], k_q.data[3], A3 + transpose(A3));
        for (int i = 0; i < 4; i++)
            qkA4[i] *= to_float2(k_q.data[i]);
    }

    // Non-inclusive cumsum
    for (int i = 0; i < 4; i++) {
        float sum = qkA4[i].x+qkA4[i].y;
        float psum = __shfl_xor_sync(0xffffffff, sum, 1);
        psum *= (j%2 == j<2);
        qkA4[i] = {psum + qkA4[i].y*(j>=2), psum + qkA4[i].x*(j<2)};
    }

    FTile ret;
    ret.data[0] = qkA8[0]+qkA4[0];
    ret.data[1] = qkA8[1]+qkA4[1];
    ret.data[2] = qkA8[2]+qkA4[2];
    ret.data[3] = qkA8[3]+qkA4[3];

    for (int ci : {0,1}) {
        for (int ti : {0,1}) {
            int Ai = ti*3, di = ti*2+ci;
            unsigned mask = 0xffff<<(j>=2)*16;
            bf A8x  = __shfl_sync(mask, A.data[Ai].x,  8+(j>=2)*18);
            bf A12x = __shfl_sync(mask, A.data[Ai].x, 12+(j>=2)*18);
            bf A12y = __shfl_sync(mask, A.data[Ai].y, 12+(j>=2)*18);
            bf2 nq = __shfl_xor_sync(0xffffffff, qt.data[di], 1);
            bf2 pk = __shfl_xor_sync(0xffffffff, kt.data[di], 1);

            bool even = (j%2==0);
            float ax = to_float(even?A8x:A12x), ay = to_float(even?A12x:A12y), c = to_float(even?kt.data[di].x:qt.data[di].y);
            float2 b = to_float2(j%2?pk:nq);
            float d = (ax*b.x+ay*b.y)*c;
            ret.data[di].y += even*d;
            ret.data[di].x +=!even*d;
        }
    }

    if (!strict) {
        // Do we really need tril<1>()?
        ret += (kt % tril<1>(A)) * qt;
    }
    return transpose(ret);
}

__device__ void debug_set(RTile&ra, int i, int j, float v) {
    if (threadIdx.x%32 == i%8*4+j%8/2) ((bf*)ra.data)[i/8*2+j/8*4+j%2] = to_bf(v);
}