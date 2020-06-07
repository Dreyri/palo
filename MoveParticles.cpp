#include "main.h"

#include <cmath>

void MoveParticlesSoA(ParticleSoA &particles, float dt) {
  __assume_aligned(particles.x, 64);
  __assume_aligned(particles.y, 64);
  __assume_aligned(particles.z, 64);
  __assume_aligned(particles.vx, 64);
  __assume_aligned(particles.vy, 64);
  __assume_aligned(particles.vz, 64);

#pragma omp simd simdlen(16)
  for (int i = 0; i != particles.size; ++i) {
    float Fx = 0.f;
    float Fy = 0.f;
    float Fz = 0.f;

    __assume_aligned(particles.x, 64);
    __assume_aligned(particles.y, 64);
    __assume_aligned(particles.z, 64);

#pragma omp simd simdlen(16)
    for (int j = 0; j != particles.size; ++j) {
      constexpr float softening = 1e-20;

      float dx = particles.x[j] - particles.x[i];
      float dy = particles.y[j] - particles.y[i];
      float dz = particles.z[j] - particles.z[i];
      float drSquared = dx * dx + dy * dy + dz * dz + softening;

      float drPower32 = std::sqrt(drSquared) * drSquared;

      Fx += dx / drPower32;
      Fy += dy / drPower32;
      Fz += dy / drPower32;
    }

    particles.vx[i] += dt * Fx;
    particles.vy[i] += dt * Fy;
    particles.vz[i] += dt * Fy;
  }

  for (int i = 0; i < particles.size; ++i) {
    particles.x[i] += particles.vx[i] * dt;
    particles.y[i] += particles.vy[i] * dt;
    particles.z[i] += particles.vz[i] * dt;
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