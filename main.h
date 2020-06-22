#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <omp.h>

// Struct zur Beschreibung eines Teilchens
struct Particle {
  float x, y, z;    // Koordinaten des Teilchens
  float vx, vy, vz; // Geschwindigkeiten des Teilchens
};

// Prototypen
void MoveParticles(const int nr_Particles, Particle *const partikel,
                   const float dt);