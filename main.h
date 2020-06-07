#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <mm_malloc.h>
#include <omp.h>

struct ParticleSoA {
  float *x;
  float *y;
  float *z;
  float *vx;
  float *vy;
  float *vz;
  int size;

  ParticleSoA(int size)
      : x(static_cast<float *>(_mm_malloc(sizeof(float) * size, 64))),
        y(static_cast<float *>(_mm_malloc(sizeof(float) * size, 64))),
        z(static_cast<float *>(_mm_malloc(sizeof(float) * size, 64))),
        vx(static_cast<float *>(_mm_malloc(sizeof(float) * size, 64))),
        vy(static_cast<float *>(_mm_malloc(sizeof(float) * size, 64))),
        vz(static_cast<float *>(_mm_malloc(sizeof(float) * size, 64))),
        size(size) {}

  ParticleSoA(const ParticleSoA &other) : ParticleSoA(other.size) {
    std::copy(other.x, other.x + size, x);
    std::copy(other.y, other.y + size, y);
    std::copy(other.z, other.z + size, z);
    std::copy(other.vx, other.vx + size, vx);
    std::copy(other.vy, other.vy + size, vy);
    std::copy(other.vz, other.vz + size, vz);
  }

  ~ParticleSoA() {
    _mm_free(x);
    _mm_free(y);
    _mm_free(z);
    _mm_free(vx);
    _mm_free(vy);
    _mm_free(vz);
  }
};

// Struct zur Beschreibung eines Teilchens
struct Particle {
  float x, y, z;    // Koordinaten des Teilchens
  float vx, vy, vz; // Geschwindigkeiten des Teilchens
};

// Prototypen
void MoveParticles(const int nr_Particles, Particle *const partikel,
                   const float dt);
void MoveParticlesSoA(ParticleSoA &particles, float dt);