#include "common.h"
#include <cuda.h>
#include <thrust/scan.h>
#include <iostream>

#define NUM_THREADS 256

// Put any static global variables here that you will use throughout the simulation.
int blks;

__device__ void apply_force_gpu(particle_t& particle, particle_t& neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff)
        return;
    // r2 = fmax( r2, min_r*min_r );
    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);

    //
    //  very simple short-range repulsive force
    //
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

__global__ void compute_forces_gpu(particle_t* d_parts, int num_parts) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    d_parts[tid].ax = d_parts[tid].ay = 0;
    for (int j = 0; j < num_parts; j++)
        apply_force_gpu(d_parts[tid], d_parts[j]);
}

__global__ void move_gpu(particle_t* d_parts, int num_parts, double size) {

    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particle_t* p = &d_parts[tid];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;

    //
    //  bounce from walls
    //
    while (p->x < 0 || p->x > size) {
        p->x = p->x < 0 ? -(p->x) : 2 * size - p->x;
        p->vx = -(p->vx);
    }
    while (p->y < 0 || p->y > size) {
        p->y = p->y < 0 ? -(p->y) : 2 * size - p->y;
        p->vy = -(p->vy);
    }
}




static int nbinsx;
static particle_t* parts;
static int *part_ids, *d_part_ids;
static int *bin_starts, *d_bin_starts;
static int *bin_idx, *d_bin_idx; // used as counter when populating bins
static double dxbin; // length&width of each bin


void init_simulation(particle_t* d_parts, int num_parts, double size) {
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // d_parts live in GPU memory
    // Do not do any particle simulation here
    
    nbinsx = ((int)((double) size / (cutoff)) + 1);  // bin size greater or equal to cutoff length
    
    
    
    part_ids = new int[num_parts];
    cudaMalloc((void**)&d_part_ids, num_parts * sizeof(int));
    bin_starts = new int[nbinsx*nbinsx](); 
    cudaMalloc((void**)&d_bin_starts, (nbinsx*nbinsx) * sizeof(int));
    //bin_idx = new int[nbinsx*nbinsx]; 
    cudaMalloc((void**)&d_bin_idx, (nbinsx*nbinsx) * sizeof(int));
    
    parts = new particle_t[num_parts];
    cudaMemcpy(parts, d_parts, num_parts * sizeof(particle_t), cudaMemcpyDeviceToHost);
        
    dxbin = size / (double) nbinsx;    

    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;
}




__global__ void count_parts_gpu(particle_t* d_parts, int num_parts, int* d_bin_starts, int nbinsx, double dxbin) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;
    int ib = (int)(d_parts[tid].x / (dxbin));
    int jb = (int)(d_parts[tid].y / (dxbin));
    atomicAdd(d_bin_starts+(nbinsx)*ib+jb,1);    
}





__global__ void emplace_parts_gpu(particle_t* d_parts, int num_parts, int* d_bin_idx, int* d_part_ids, int nbinsx, double dxbin) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;
    int ib = (int)(d_parts[tid].x / dxbin);
    int jb = (int)(d_parts[tid].y / dxbin);
    int idx = atomicAdd(d_bin_idx+nbinsx*ib+jb,1);
    d_part_ids[idx] = tid;
}


void calc_bins(particle_t* d_parts, int num_parts){
    /// Reset counter to 0 ///
    cudaMemset(d_bin_starts, 0, nbinsx*nbinsx*sizeof(int));
    
    /// Count number of d_parts per bin ///
    count_parts_gpu<<<blks, NUM_THREADS>>>(d_parts, num_parts, d_bin_starts, nbinsx, dxbin);
    
    //cudaMemcpy(bin_starts, d_bin_starts, (nbinsx*nbinsx) * sizeof(int), cudaMemcpyDeviceToHost);
    //for (int b = 0; b < nbinsx*nbinsx; ++b) {
    //    std::cout << bin_starts[b];
    //}
    //std::cout << "\n";
    
    
    /// Prefix sum the counts ///
    thrust::exclusive_scan(thrust::device,d_bin_starts,d_bin_starts+nbinsx*nbinsx,d_bin_starts);
    
    /// Copy bin_starts into bin_idx ///
    cudaMemcpy(d_bin_idx, d_bin_starts, (nbinsx*nbinsx) * sizeof(int), cudaMemcpyDeviceToDevice);

    /// emplace parts ///  
    emplace_parts_gpu<<<blks, NUM_THREADS>>>(d_parts, num_parts, d_bin_idx,d_part_ids, nbinsx, dxbin);
    
    //cudaMemcpy(part_ids, d_part_ids, (num_parts) * sizeof(int), cudaMemcpyDeviceToHost);
    //for (int i = 0; i < num_parts; ++i) {
    //    std::cout << part_ids[i] << "-";
    //}
    //std::cout << "\n";
    
  
}

void simulate_one_step(particle_t* d_parts, int num_parts, double size) {
    // d_parts live in GPU memory
    // Rewrite this function
    
    calc_bins(d_parts,num_parts);
    // Finish full binning on gpu
    // then change compute_forces_gpu to take binning into account

    // Compute forces
    compute_forces_gpu<<<blks, NUM_THREADS>>>(d_parts, num_parts);

    // Move d_parts
    move_gpu<<<blks, NUM_THREADS>>>(d_parts, num_parts, size);
}
