#include "tile.cuh"
#include <assert.h>
typedef bf * __restrict__ F_;

constexpr int WARPS = _C_/16;
constexpr int fw_stages = 1, bw_stages = 1;

__global__ void forward_kernel(int T, int H, F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_, F_ s0_, bf* y_, bf* s_, bf* sT_) {
    constexpr int C = _C_, K = 16;
    int bi = blockIdx.y, hi = blockIdx.x;
    extern __shared__ char smem_[];
    char*smem = smem_;

    STile *sw_ = (STile*)smem; smem += sizeof(STile)*fw_stages*WARPS;
    STile *sq_ = (STile*)smem; smem += sizeof(STile)*fw_stages*WARPS;
    STile *sk_ = (STile*)smem; smem += sizeof(STile)*fw_stages*WARPS;
    STile *sv_ = (STile*)smem; smem += sizeof(STile)*fw_stages*WARPS;
    STile *sa_ = (STile*)smem; smem += sizeof(STile)*fw_stages*WARPS;
    STile *sb_ = (STile*)smem; smem += sizeof(STile)*fw_stages*WARPS;
    char*share = (char*)smem; smem += sizeof(float4)*32;

    int stride = H*C;
    int warpi = threadIdx.x/32;

    auto push = [&](int t) {
        int off = bi*T*H*C + t*K*H*C + hi*C + warpi*16;
        int si = t%fw_stages;
        sw_[si*WARPS+warpi] = GTile(w_+off, stride);
        sq_[si*WARPS+warpi] = GTile(q_+off, stride);
        sk_[si*WARPS+warpi] = GTile(k_+off, stride);
        sv_[si*WARPS+warpi] = GTile(v_+off, stride);
        sa_[si*WARPS+warpi] = GTile(a_+off, stride);
        sb_[si*WARPS+warpi] = GTile(b_+off, stride);
    };
    for (int t = 0; t < fw_stages-1 && t < T/K; t++) push(t), __commit_group();

    RTile state[WARPS];
    for (int i = 0; i < WARPS; i++) {
        int off = bi*H*C*C + hi*C*C + warpi*16*C + i*16;
        state[i] = GTile(s0_+off, C);
    }

    for (int t = 0; t < T/K; t++) {
        __syncthreads();
        if (t+fw_stages-1 < T/K)
            push(t+fw_stages-1);
        __commit_group();
        __wait_groups<fw_stages-1>();
        __syncthreads();
        int si = t%fw_stages;
        STile &sw = sw_[si*WARPS+warpi], &sq = sq_[si*WARPS+warpi], &sk = sk_[si*WARPS+warpi], &sv = sv_[si*WARPS+warpi], &sa = sa_[si*WARPS+warpi], &sb = sb_[si*WARPS+warpi];

        FTile w = (RTile)sw;
        apply_(w, [](float x) { return __expf(-__expf(x)); });
        FTile fw = w;
        FTile non_incl_pref = cumprodv<0,0>(fw);
        FTile incl_pref = non_incl_pref * w;
        FTile inv_incl_pref = incl_pref;
        apply_(inv_incl_pref, [](float x) { return 1.f/x; });

        RTile wq = (RTile)sq *     incl_pref, kwi = (RTile)sk * inv_incl_pref;
        RTile wa = (RTile)sa * non_incl_pref, bwi = (RTile)sb * inv_incl_pref;
        RTile ab = sum_warp<1,WARPS>((float2*)share, tril<1>(wa % bwi));
        RTile ak = sum_warp<1,WARPS>((float2*)share, tril<1>(wa % kwi));

        RTile ab_invT;
        if (threadIdx.x < 32) ab_invT = tri_minvT(ab);
        ab_invT = from_warp(ab_invT, 0, (float4*)share);

        RTile vt = sv.t();
        FTile ab_ut = vt % ak;
        for (int i = 0; i < WARPS; i++)
            ab_ut += state[i] % from_warp(wa, i, (float4*)share);
        RTile ut = FTile(ab_ut % transpose(ab_invT));

        FTile y = sum_warp<1,WARPS>((float2*)share, tril<0>(wq % kwi)) % vt;
        y +=      sum_warp<1,WARPS>((float2*)share, tril<0>(wq % bwi)) % ut;
        for (int i = 0; i < WARPS; i++)
            y += from_warp(wq, i, (float4*)share) % state[i];

        int off = bi*T*H*C + t*K*H*C + hi*C + warpi*16;
        GTile(y_+off, stride) = RTile(y);

        RTile kwt = transpose(kwi*fw), bwt = transpose(bwi*fw);
        for (int i = 0; i < WARPS; i++) {
            int off = bi*H*(T/K)*C*C + hi*(T/K)*C*C + t*C*C + warpi*16*C + i*16;
            GTile(s_+off, C) = state[i];

            FTile fstate = state[i] * from_warp(fw, i, (float4*)share);
            fstate += vt % from_warp(kwt, i, (float4*)share);
            fstate += ut % from_warp(bwt, i, (float4*)share);
            state[i] = fstate;
        }
    }
    for (int i = 0; i < WARPS; i++) {
        int off = bi*H*C*C + hi*C*C + warpi*16*C + i*16;
        GTile(sT_+off, C) = state[i];
    }
}

void cuda_forward(int B, int T, int H, bf*w, bf*q, bf*k, bf*v, bf*z, bf*a, bf*s0, bf*y, bf*s, bf*sT) {
    assert(T%16 == 0);
    constexpr int threads = 32*WARPS, shared_mem = sizeof(STile)*fw_stages*WARPS*6+sizeof(float4)*32;
    static int reported = 0;
    if (!reported++) {
#if defined VERBOSE
        printf("forward_kernel() uses %d bytes of (dynamic) shared memory\n", shared_mem);
#endif
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, forward_kernel);
        int cur_mem = attr.maxDynamicSharedSizeBytes;
        if (shared_mem > cur_mem) {
#if defined VERBOSE
            printf("Increasing forward_kernel's MaxDynamicSharedMemorySize from %d to %d\n", cur_mem, shared_mem);
#endif
            assert(!cudaFuncSetAttribute(forward_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem));
        }
    }
    forward_kernel<<<dim3(H,B), dim3(threads), shared_mem>>>(T,H,w,q,k,v,z,a,s0,y,s,sT);
}


__global__ void backward_kernel(int T, int H, F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_, F_ dy_, F_ s_, F_ dsT_, bf* dw_, bf* dq_, bf* dk_, bf* dv_, bf* da_, bf* db_, bf* ds0_) {
    constexpr int C = _C_, K = 16;
    int bi = blockIdx.y, hi = blockIdx.x;
    extern __shared__ char smem_[];
    char*smem = smem_;

    STile *sw_ = (STile*)smem; smem += sizeof(STile)*bw_stages*WARPS;
    STile *sq_ = (STile*)smem; smem += sizeof(STile)*bw_stages*WARPS;
    STile *sk_ = (STile*)smem; smem += sizeof(STile)*bw_stages*WARPS;
    STile *sv_ = (STile*)smem; smem += sizeof(STile)*bw_stages*WARPS;
    STile *sa_ = (STile*)smem; smem += sizeof(STile)*bw_stages*WARPS;
    STile *sb_ = (STile*)smem; smem += sizeof(STile)*bw_stages*WARPS;
    STile *sdy_ = (STile*)smem; smem += sizeof(STile)*bw_stages*WARPS;
    STile *state_ = (STile*)smem; smem += sizeof(STile)*bw_stages*WARPS*WARPS;
    STile *dstate = (STile*)smem; smem += sizeof(STile)*bw_stages*WARPS*WARPS;
    char*share = (char*)smem; smem += sizeof(float4)*32;

    int stride = H*C;
    int warpi = threadIdx.x/32;

    auto push = [&](int t) {
        int off = bi*T*H*C + t*K*H*C + hi*C + warpi*16;
        int si = t%fw_stages;
        sw_[si*WARPS+warpi] = GTile(w_+off, stride);
        sq_[si*WARPS+warpi] = GTile(q_+off, stride);
        sk_[si*WARPS+warpi] = GTile(k_+off, stride);
        sv_[si*WARPS+warpi] = GTile(v_+off, stride);
        sa_[si*WARPS+warpi] = GTile(a_+off, stride);
        sb_[si*WARPS+warpi] = GTile(b_+off, stride);
        sdy_[si*WARPS+warpi] = GTile(dy_+off, stride);
        for (int i = 0; i < WARPS; i++) {
            int off2 = bi*H*(T/K)*C*C + hi*(T/K)*C*C + t*C*C + warpi*16*C + i*16;
            state_[si*WARPS*WARPS+warpi*WARPS+i] = GTile(s_+off2, C);
        }
    };

    for (int i = 0; i < WARPS; i++) {
        int off = bi*H*C*C + hi*C*C + warpi*16*C + i*16;
        dstate[warpi*WARPS+i] = GTile(dsT_+off, C);
        __commit_group();
    }

    for (int t = 0; t < bw_stages-1 && t < T/K; t++) push(T/K-1-t), __commit_group();

    for (int t = T/K-1; t >= 0; t--) {
        __syncthreads();
        if (t-bw_stages+1 >= 0)
            push(t-bw_stages+1);
        __commit_group();
        __wait_groups<bw_stages-1>();
        __syncthreads();
        int si = t%bw_stages;
        STile &sw = sw_[si*WARPS+warpi], &sq = sq_[si*WARPS+warpi], &sk = sk_[si*WARPS+warpi], &sv = sv_[si*WARPS+warpi], &sa = sa_[si*WARPS+warpi], &sb = sb_[si*WARPS+warpi], &sdy = sdy_[si*WARPS+warpi];
        STile*state = state_+si*WARPS*WARPS;

        FTile w = (RTile)sw;
        apply_(w, [](float x) { return __expf(-__expf(x)); });
        FTile fw = w;
        FTile non_incl_pref = cumprodv<0,0>(fw);
        FTile incl_pref = non_incl_pref * w;
        FTile inv_incl_pref = incl_pref;
        apply_(inv_incl_pref, [](float x) { return 1.f/x; });

        RTile wq = (RTile)sq *     incl_pref, kwi = (RTile)sk * inv_incl_pref;
        RTile wa = (RTile)sa * non_incl_pref, bwi = (RTile)sb * inv_incl_pref;
        RTile ab = sum_warp<1,WARPS>((float2*)share, tril<1>(wa % bwi));
        RTile ak = sum_warp<1,WARPS>((float2*)share, tril<1>(wa % kwi));

        RTile ab_invT;
        if (threadIdx.x < 32) ab_invT = tri_minvT(ab);
        ab_invT = from_warp(ab_invT, 0, (float4*)share);

        RTile vt = sv.t();
        FTile ab_ut = vt % ak;
        for (int i = 0; i < WARPS; i++)
            ab_ut += state[warpi*WARPS+i] % from_warp(wa, i, (float4*)share);
        RTile ut = FTile(ab_ut % transpose(ab_invT));

        RTile qb = sum_warp<1,WARPS>((float2*)share, tril<0>(wq % bwi));
        RTile qk = sum_warp<1,WARPS>((float2*)share, tril<0>(wq % kwi));

        RTile dyt = sdy.t();
        FTile dut = FTile(dyt % transpose(qb));
        for (int i = 0; i < WARPS; i++)
            dut += dstate[warpi*WARPS+i] % from_warp(bwi*fw, i, (float4*)share);
        RTile dab_ut = FTile(dut % ab_invT);
        FTile dv = transpose(qk) % dyt;
        dv += transpose(ak) % dab_ut;
        for (int i = 0; i < WARPS; i++)
            dv += from_warp(kwi*fw, i, (float4*)share) % dstate[warpi*WARPS+i];

        int off = bi*T*H*C + t*K*H*C + hi*C + warpi*16;
        GTile(dv_+off, stride) = RTile(dv);

        FTile dab = sum_warp<1,WARPS>((float2*)share, tril<1>(transpose(dab_ut) % transpose(ut)));
        FTile dak = sum_warp<1,WARPS>((float2*)share, tril<1>(transpose(dab_ut) % transpose(vt)));
        FTile dab_u_state0;
        dab_u_state0.zero_();
        for (int i = 0; i < WARPS; i++)
            dab_u_state0 += from_warp(transpose(dab_ut), i, (float4*)share) % state[i*WARPS+warpi].t();

        FTile da = dab_u_state0;
        da += dab % transpose(bwi);
        da += dak % transpose(kwi);
        da = non_incl_pref * da;
        GTile(da_+off, stride) = RTile(da);

        FTile dqb = sum_warp<1,WARPS>((float2*)share, tril<0>(transpose(dyt) % transpose(ut)));
        FTile dqk = sum_warp<1,WARPS>((float2*)share, tril<0>(transpose(dyt) % transpose(vt)));
        FTile dy_state0;
        dy_state0.zero_();
        for (int i = 0; i < WARPS; i++)
            dy_state0 += from_warp(transpose(dyt), i, (float4*)share) % state[i*WARPS+warpi].t();

        FTile dq = dy_state0;
        dq += dqb % transpose(bwi);
        dq += dqk % transpose(kwi);
        dq = incl_pref * dq;
        GTile(dq_+off, stride) = RTile(dq);

        RTile wqt = transpose(wq), wat = transpose(wa);

        FTile u_dstate;
        u_dstate.zero_();
        for (int i = 0; i < WARPS; i++)
            u_dstate += from_warp(transpose(ut), i, (float4*)share) % dstate[i*WARPS+warpi].t();

        FTile db = fw * u_dstate;
        db += transpose(dab) % wat;
        db += transpose(dqb) % wqt;
        db = inv_incl_pref * db;
        GTile(db_+off, stride) = RTile(db);

        FTile v_dstate;
        v_dstate.zero_();
        for (int i = 0; i < WARPS; i++)
            v_dstate += from_warp(transpose(vt), i, (float4*)share) % dstate[i*WARPS+warpi].t();

        FTile dk = fw * v_dstate;
        dk += transpose(dak) % wat;
        dk += transpose(dqk) % wqt;
        dk = inv_incl_pref * dk;
        GTile(dk_+off, stride) = RTile(dk);

        FTile dw;
        dw.zero_();
        RTile ones;
        for (int i = 0; i < 4; i++) ones.data[i] = to_bf2({1.f,1.f});
        for (int i = 0; i < WARPS; i++)
            dw += ones % transpose((RTile)state[i*WARPS+warpi]*(RTile)dstate[i*WARPS+warpi]);
        dw = fw * dw;
        dw += fast_dw<1>(dab,wa,bwi);
        dw += fast_dw<1>(dak,wa,kwi);
        dw += fast_dw<0>(dqb,wq,bwi);
        dw += fast_dw<0>(dqk,wq,kwi);
        FTile tmp;
        dw += cumsumv<0,0>(tmp = v_dstate*(fw*kwi));
        dw += cumsumv<0,0>(tmp = u_dstate*(fw*bwi));
        dw += cumsumv<0,1>(tmp = dab_u_state0*wa);
        dw += cumsumv<1,1>(tmp = dy_state0*wq);

        FTile dw_fac = (RTile)sw;
        apply_(dw_fac, [](float x) { return -__expf(x); });
        dw = dw * dw_fac;
        GTile(dw_+off, stride) = RTile(dw);

        __syncthreads();
        for (int i = 0; i < WARPS; i++) {
            FTile ndstate = (RTile)dstate[warpi*WARPS+i] * from_warp(fw, i, (float4*)share);
            ndstate += dyt % from_warp(wqt, i, (float4*)share);
            ndstate += dab_ut % from_warp(wat, i, (float4*)share);
            dstate[warpi*WARPS+i] = (RTile)ndstate;
        }
        __syncthreads();
    }
    for (int i = 0; i < WARPS; i++) {
        int off = bi*H*C*C + hi*C*C + warpi*16*C + i*16;
        GTile(ds0_+off, C) = dstate[warpi*WARPS+i];
    }
}

void cuda_backward(int B, int T, int H, bf*w, bf*q, bf*k, bf*v, bf*z, bf*a, bf*dy, bf*s, bf*dsT, bf*dw, bf*dq, bf*dk, bf*dv, bf*dz, bf*da, bf*ds0) {
    assert(T%16 == 0);
    constexpr int threads = 32*WARPS, shared_mem = sizeof(STile)*bw_stages*WARPS*(7+WARPS*2)+sizeof(float4)*32;
    static int reported = 0;
    if (!reported++) {
#if defined VERBOSE
        printf("backward_kernel() uses %d bytes of (dynamic) shared memory\n", shared_mem);
#endif
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, backward_kernel);
        int cur_mem = attr.maxDynamicSharedSizeBytes;
        if (shared_mem > cur_mem) {
#if defined VERBOSE
            printf("Increasing backward_kernel's MaxDynamicSharedMemorySize from %d to %d\n", cur_mem, shared_mem);
#endif
            assert(!cudaFuncSetAttribute(backward_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem));
        }
    }
    backward_kernel<<<dim3(H,B), dim3(threads), shared_mem>>>(T,H,w,q,k,v,z,a,dy,s,dsT,dw,dq,dk,dv,dz,da,ds0);
}