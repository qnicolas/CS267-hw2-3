#include "common.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <list>

// Apply the force from neighbor to particle
void apply_force(particle_t& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;
    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

// Integrate the ODE
void move(particle_t& p, double size) {
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}

static int nbinsx;  // number of bins in one dimension (total number of bins = nbinsx^2)
static int* part_ids;
static int* bin_starts;
static int* bin_idx; // used as counter when populating bins

static double dxbin; // length&width of each bin


void init_simulation(particle_t* parts, int num_parts, double size) {
    nbinsx = ((int)((double) size / (cutoff)) + 1);  // bin size greater or equal to cutoff length
    
    part_ids = new int[num_parts];
    bin_starts = new int[nbinsx*nbinsx+1]();
    bin_idx = new int[nbinsx*nbinsx];
        
    dxbin = size / (double) nbinsx;    
}

void calc_bins(particle_t* parts, int num_parts){
    /// Reset counter to 0 ///
    for (int b = 1; b < nbinsx*nbinsx+1; ++b) {
        bin_starts[b] =0;
    }
    
    /// Count number of particles per bin ///
    for (int i = 0; i < num_parts; ++i) {
        int ib = (int)(parts[i].x / dxbin);
        int jb = (int)(parts[i].y / dxbin);
        bin_starts[1+nbinsx*ib+jb]++;
    }
    
    /// Prefix sum the counts ///
    for (int b = 1; b < nbinsx*nbinsx; ++b) {
        bin_starts[b+1] += bin_starts[b];
    }
    
    /////////////////////////////////////////////////////////////////////////////
    //if (bin_starts[nbinsx*nbinsx] != num_parts){
    //    std::cout << "shit -- binstarts[-1] = " << bin_starts[nbinsx*nbinsx] << "num_parts = " << num_parts << std::endl;
    //}
    /////////////////////////////////////////////////////////////////////////////
    
    /// Copy bin_starts into bin_idx ///
    for (int b = 0; b < nbinsx*nbinsx; ++b) {
        bin_idx[b] = bin_starts[b];
    }    
    
    /// emplace particles ///    
    for (int i = 0; i < num_parts; ++i) {
        int ib = (int)(parts[i].x / dxbin);
        int jb = (int)(parts[i].y / dxbin);
        part_ids[bin_idx[nbinsx*ib+jb]] = i;
        bin_idx[nbinsx*ib+jb] ++;
    }    
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    calc_bins(parts,num_parts);
    
    // Compute Forces
    for (int ib = 0; ib < nbinsx; ++ib) {
        for (int jb = 0; jb < nbinsx; ++jb) {
            for (int m = bin_starts[nbinsx*ib+jb]; m < bin_starts[nbinsx*ib+jb+1]; ++m){
                int i = part_ids[m];
                parts[i].ax = parts[i].ay = 0;
                for (int local_i = fmax(0,ib-1); local_i <= fmin(nbinsx-1,ib+1); ++local_i){
                    for (int local_j = fmax(0,jb-1); local_j <= fmin(nbinsx-1,jb+1); ++local_j){
                        for (int n = bin_starts[nbinsx*local_i+local_j]; n < bin_starts[nbinsx*local_i+local_j+1]; ++n){
                            int j = part_ids[n];
                            apply_force(parts[i], parts[j]);
                        }
                    }
                }
            }
        }
    }
    

    // Move Particles
    for (int ib = 0; ib < nbinsx; ++ib) {
        for (int jb = 0; jb < nbinsx; ++jb) {
            for (int m = bin_starts[nbinsx*ib+jb]; m < bin_starts[nbinsx*ib+jb+1]; ++m){
                int i = part_ids[m];
                move(parts[i], size);
            }
        }
    }    

}

