#include "main.h"

#include <mm_malloc.h>

void initParticles(Particle *const partikel, const int nr_Particles) {
  srand(0);
  for (int i = 0; i < nr_Particles; i++) {
    partikel[i].x = float(rand()) / RAND_MAX;
    partikel[i].y = float(rand()) / RAND_MAX;
    partikel[i].z = float(rand()) / RAND_MAX;
    partikel[i].vx = float(rand()) / RAND_MAX;
    partikel[i].vy = float(rand()) / RAND_MAX;
    partikel[i].vz = float(rand()) / RAND_MAX;
  }
}

void initParticlesSoA(ParticleSoA &particles) {
  srand(0);

  for (int i = 0; i < particles.size; ++i) {
    particles.x[i] = float(rand()) / RAND_MAX;
    particles.y[i] = float(rand()) / RAND_MAX;
    particles.z[i] = float(rand()) / RAND_MAX;
    particles.vx[i] = float(rand()) / RAND_MAX;
    particles.vy[i] = float(rand()) / RAND_MAX;
    particles.vz[i] = float(rand()) / RAND_MAX;
  }
}

void copyParticles(Particle *const partikel_src, Particle *const partikel_dst,
                   const int nr_Particles) {
  for (int i = 0; i < nr_Particles; i++) {
    partikel_dst[i].x = partikel_src[i].x;
    partikel_dst[i].y = partikel_src[i].y;
    partikel_dst[i].z = partikel_src[i].z;
    partikel_dst[i].vx = partikel_src[i].vx;
    partikel_dst[i].vy = partikel_src[i].vy;
    partikel_dst[i].vz = partikel_src[i].vz;
  }
}

// use original and modified particle arrays to calculate the checksum.
// This gives us less tolerance for error/deviation.
float calcChecksum(Particle *original, Particle *modified,
                   int num_particles) noexcept {
  float res{};

  for (int i = 0; i != num_particles; ++i) {
    res += (original[i].x - modified[i].x);
    res += (original[i].y - modified[i].y);
    res += (original[i].z - modified[i].z);
    res += (original[i].vx - modified[i].vx);
    res += (original[i].vy - modified[i].vy);
    res += (original[i].vz - modified[i].vz);
  }

  return res / -245.958237;
}

float calcChecksumSoA(ParticleSoA &original, ParticleSoA &modified) noexcept {
  float res{};

  for (int i = 0; i != original.size; ++i) {
    res += (original.x[i] - modified.x[i]);
    res += (original.y[i] - modified.y[i]);
    res += (original.z[i] - modified.z[i]);
    res += (original.vx[i] - modified.vx[i]);
    res += (original.vy[i] - modified.vy[i]);
    res += (original.vz[i] - modified.vz[i]);
  }

  return res / -245.958237;
}

int main() {
  // Problemgr��e und Anzahl und Gr��e der Zeitschritte definieren
  constexpr int nrOfParticles = 16384; // 437000;
  constexpr int nrRuns =
      10; // Anzahl der L�ufe und der Zeitschritte der Simulation
  constexpr int skipRuns =
      3; // Anzahl der Messungen, die nicht in Mittelwert ber�cksichtigt werden
  constexpr float dt = 0.01f; // L�nge eines Zeitschrittes

  // do an aligned alloc for both particle arrays
  // the following form is sadly not supported by the intel compiler
  /*
  Particle *partikel_start =
      new (static_cast<std::align_val_t>(32)) Particle[nrOfParticles];
  Particle *partikel =
      new (static_cast<std::align_val_t>(32)) Particle[nrOfParticles];
  copyParticles(partikel_start, partikel, nrOfParticles);
  */
  // therefore we must do a c style aligned allocation, the one which is an
  // intrinsic for intel compiler
  /*
  Particle *partikel_start =
      static_cast<Particle *>(_mm_malloc(sizeof(Particle) * nrOfParticles, 64));
  Particle *partikel =
      static_cast<Particle *>(_mm_malloc(sizeof(Particle) * nrOfParticles, 64));
  */

  ParticleSoA partikel_soa_start(nrOfParticles);

  // Initiaslisierung der Partikel mit Zufallswerten
  // initParticles(partikel_start, nrOfParticles);

  initParticlesSoA(partikel_soa_start);

  // Messen der Performance
  double runtimeStep[nrRuns] = {0.}; // Sammlung der Laufzeiten der Steps
  double GFlopsStep[nrRuns] = {0.};  // Sammlung der Leistungen der Steps
  double meanRuntime = 0.;
  double stdRuntime = 0.;
  double meanGFlops = 0.;
  double stdGFlops = 0.;

  // Berechnung der Anzahl an GFLOPs der Berechnung
  const float NrOfGFLOPs =
      20.0 * 1e-9 * float(nrOfParticles) * float(nrOfParticles - 1);
  printf("#### Runtime Measurements Particle Simulation  ###\n");

  for (int run = 0; run < nrRuns; run++) {
    ParticleSoA partikel_soa(partikel_soa_start);
    // copyParticles(partikel_start, partikel, nrOfParticles);
    const double tStart = omp_get_wtime(); // Start der Zeitmessung
    MoveParticlesSoA(partikel_soa, dt);
    // MoveParticles(nrOfParticles, partikel,
    //              dt);                   // Funktion, die optimiert werden
    //              soll
    const double tEnd = omp_get_wtime(); // Ende der Zeitmessung

    // float checksum = calcChecksum(partikel_start, partikel, nrOfParticles);
    float checksum = calcChecksumSoA(partikel_soa_start, partikel_soa);
    runtimeStep[run] = tEnd - tStart;
    GFlopsStep[run] = NrOfGFLOPs / runtimeStep[run];
    if (run >= skipRuns) { // Berechnung Mittelwerte
      meanRuntime += runtimeStep[run];
      meanGFlops += GFlopsStep[run];
    }

    printf("Run %d: Runtime: %f03,\t GFLOPS/s: %f01, \t Checksum: %f \t %s\n",
           run, runtimeStep[run], GFlopsStep[run], checksum,
           (run < skipRuns ? "Not in Average" : ""));
    fflush(stdout); // Ausgabebuffer leeren
  }
  // Berechnung der Mittelwerte
  double nrRunsInStatistics = (double)(nrRuns - skipRuns);
  meanRuntime /= nrRunsInStatistics;
  meanGFlops /= nrRunsInStatistics;

  // Berechnung der Mittelwertfehler
  for (int i = skipRuns; i < nrRuns; i++) {
    stdRuntime +=
        (runtimeStep[i] - meanRuntime) * (runtimeStep[i] - meanRuntime);
    stdGFlops += (GFlopsStep[i] - meanGFlops) * (GFlopsStep[i] - meanGFlops);
  }
  stdRuntime =
      sqrt(stdRuntime / (nrRunsInStatistics * (nrRunsInStatistics - 1)));
  stdGFlops = sqrt(stdGFlops / (nrRunsInStatistics * (nrRunsInStatistics - 1)));

  // Ausgabe der Ergebnisse
  printf("\n\n####### Average Performance #########\n");
  printf("Average Runtime: %f03 +- %f Seconds \n", meanRuntime, stdRuntime);
  printf("Average Performance: %f03 +- %f03 GFLOPS/s \n", meanGFlops,
         stdGFlops);
  printf("#####################################\n");

  // delete[] partikel;
}