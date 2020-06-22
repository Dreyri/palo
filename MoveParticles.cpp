#include "main.h"

#include <cmath>

void MoveParticlesSoA(ParticleSoA &particles, float dt) {

  // these are to indicate that none of the pointer alias each other
  // this is required because they are pointers to the same type so the c++
  // standard assumes that they can alias.
  float *__restrict__ x = particles.x;
  float *__restrict__ y = particles.y;
  float *__restrict__ z = particles.z;
  float *__restrict__ vx = particles.vx;
  float *__restrict__ vy = particles.vy;
  float *__restrict__ vz = particles.vz;
  int size = particles.size;

  // this indicates that our loop counter is a multiple of 16 and therefore the
  // compiler won't be required to generate peel loops
  __assume(size % 16 == 0);

  __assume_aligned(x, 64);
  __assume_aligned(y, 64);
  __assume_aligned(z, 64);
  __assume_aligned(vx, 64);
  __assume_aligned(vy, 64);
  __assume_aligned(vz, 64);

// the aligned attribute for simd does nothing
#pragma omp simd simdlen(16) // aligned(x, y, z, vx, vy, vz : 64)
  for (int i = 0; i != size; ++i) {
    float Fx = 0.f;
    float Fy = 0.f;
    float Fz = 0.f;

    for (int j = 0; j != size; ++j) {
      constexpr float softening = 1e-20;

      float dx = x[j] - x[i];
      float dy = y[j] - y[i];
      float dz = z[j] - z[i];
      float drSquared = dx * dx + dy * dy + dz * dz + softening;

      float drPower32 = std::sqrt(drSquared) * drSquared;

      Fx += dx / drPower32;
      Fy += dy / drPower32;
      Fz += dy / drPower32;
    }

    vx[i] += dt * Fx;
    vy[i] += dt * Fy;
    vz[i] += dt * Fy;
  }

  for (int i = 0; i < particles.size; ++i) {
    x[i] += vx[i] * dt;
    y[i] += vy[i] * dt;
    z[i] += vz[i] * dt;
  }
}

void MoveParticles(const int nr_Particles, Particle *const partikel,
                   const float dt) {
  // nicht noetig aber auch eine moeglichkeit alignment anzugeben
  //__assume_aligned(partikel, 64);

  // Schleife �ber alle Partikel
  for (int i = 0; i < nr_Particles; i++) {

    // Kraftkomponenten (x,y,z) der Kraft auf aktuellen Partikel (i)
    float Fx = 0, Fy = 0, Fz = 0;

    // Schleife �ber die anderen Partikel die Kraft auf Partikel i aus�ben
    // wir benutzen eine simdlen von 16 weil partikel 64 byte aligned sind und
    // 64 / 4 = 16
#pragma omp simd simdlen(16) aligned(partikel : 64)
    for (int j = 0; j < nr_Particles; j++) {

      // Abschw�chung als zus�tzlicher Abstand, um Singularit�t und
      // Selbst-Interaktion zu vermeiden
      const float softening = 1e-20;

      // Gravitationsgesetz
      // Berechne Abstand der Partikel i und j
      const float dx = partikel[j].x - partikel[i].x;
      const float dy = partikel[j].y - partikel[i].y;
      const float dz = partikel[j].z - partikel[i].z;
      const float drSquared = dx * dx + dy * dy + dz * dz + softening;
      // benutzung overloaded pow und exponent aendern zu float, die exponenten
      // berechnung wird beim kompilieren rausoptimiert.
      // hierdurch wird jetzt float pow statt double pow berechnet.
      // const float drPower32 = std::pow(drSquared, 3.f / 2.f);

      // sqrt ist eine schnellere funktion als pow, daher haben wir den pow
      // aufruf strength reduced zu einem sqrt aufruf.
      float drPower32 = std::sqrt(drSquared) * drSquared;

      // Addiere Kraftkomponenten zur Netto-Kraft
      Fx += dx / drPower32;
      Fy += dy / drPower32;
      Fz += dz / drPower32;
    }

    // Berechne �nderung der Geschwindigkeit des Partikel i durch einwirkende
    // Kraft
    partikel[i].vx += dt * Fx;
    partikel[i].vy += dt * Fy;
    partikel[i].vz += dt * Fz;
  }

  // Bewege Partikel entsprechend der aktuellen Geschwindigkeit
  for (int i = 0; i < nr_Particles; i++) {
    partikel[i].x += partikel[i].vx * dt;
    partikel[i].y += partikel[i].vy * dt;
    partikel[i].z += partikel[i].vz * dt;
  }
}