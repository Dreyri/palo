#include "main.h"

#include <algorithm>
#include <array>
#include <cmath>

struct vec3 {
  float x;
  float y;
  float z;
};

void MoveParticles(const int nr_Particles, Particle *const partikel,
                   const float dt) {

  // this tilesize gives optimal speed, it also is a multiple of the vector
  // register length allowing for optimal utilisation.
  constexpr auto tile_size = 16384;

  // Schleife �ber alle Partikel
  // for (int i = 0; i < nr_Particles; i++) {
  int loop_count = nr_Particles / tile_size;
  for (int i_ = 0; i_ != loop_count; ++i_) {
    int i = i_ * tile_size;
    // populate current tile
    const auto iparticles = [&] {
      std::array<Particle, tile_size> arr;
      std::copy(partikel + i, partikel + i + tile_size, arr.begin());
      return arr;
    }();

    // Kraftkomponenten (x,y,z) der Kraft auf aktuellen Partikel (i)
    float Fx = 0, Fy = 0, Fz = 0;

    // Schleife �ber die anderen Partikel die Kraft auf Partikel i aus�ben
    // for (int j = 0; j < nr_Particles; j++) {
    for (int j_ = 0; j_ != loop_count; ++j_) {
      int j = j_ * tile_size;

      // populate jparticles tile
      const auto jparticles = [&] {
        std::array<Particle, tile_size> arr; // uninitialized
        std::copy(partikel + j, partikel + j + tile_size, arr.begin());
        return arr;
      }();

      // Abschw�chung als zus�tzlicher Abstand, um Singularit�t und
      // Selbst-Interaktion zu vermeiden
      constexpr float softening = 1e-20;

      // Gravitationsgesetz
      // Berechne Abstand der Partikel i und j

      vec3 Fs[tile_size];

      // vectorizable
      for (int k = 0; k != tile_size; ++k) {
        float dx = jparticles[k].x - iparticles[k].x;
        float dy = jparticles[k].y - iparticles[k].y;
        float dz = jparticles[k].z - iparticles[k].z;

        float drSquared = dx * dx + dy * dy + dz * dz + softening;

        /*
const float dx = partikel[j].x - partikel[i].x;
const float dy = partikel[j].y - partikel[i].y;
const float dz = partikel[j].z - partikel[i].z;
const float drSquared = dx * dx + dy * dy + dz * dz + softening;
        */
        // benutzung overloaded pow und exponent aendern zu float, die
        // exponenten berechnung wird beim kompilieren rausoptimiert. hierdurch
        // wird jetzt float pow statt double pow berechnet. const float
        // drPower32 = std::pow(drSquared, 3.f / 2.f);

        // sqrt ist eine schnellere funktion als pow, daher haben wir den pow
        // aufruf strength reduced zu einem sqrt aufruf.
        float drPower32 = std::sqrt(drSquared) * drSquared;

        Fs[k].x = dx / drPower32;
        Fs[k].y = dy / drPower32;
        Fs[k].z = dz / drPower32;

        // Addiere Kraftkomponenten zur Netto-Kraft
        // non vectorizable part
        // Fx += dx / drPower32;
        // Fy += dy / drPower32;
        // Fz += dz / drPower32;
      }

      // non vectorizable code part because of data dependency on F{x,y,z}
      for (int k = 0; k != tile_size; ++k) {
        Fx += Fs[k].x;
        Fy += Fs[k].y;
        Fz += Fs[k].z;
      }
    }

    // Berechne �nderung der Geschwindigkeit des Partikel i durch einwirkende
    // Kraft
    partikel[i].vx += dt * Fx;
    partikel[i].vy += dt * Fy;
    partikel[i].vz += dt * Fz;
  }

  // Bewege Partikel entsprechend der aktuellen Geschwindigkeit
  // this part is not vectorizable
  for (int i = 0; i < nr_Particles; i++) {
    partikel[i].x += partikel[i].vx * dt;
    partikel[i].y += partikel[i].vy * dt;
    partikel[i].z += partikel[i].vz * dt;
  }
}