#include "main.h"

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

float calcChecksum(Particle *particles, int num_particles) noexcept {
  float res{};

  for (int i = 0; i != num_particles; ++i) {
    res += particles[i].x;
    res += particles[i].y;
    res += particles[i].z;
    res += particles[i].vx;
    res += particles[i].vy;
    res += particles[i].vz;
  }

  return res / 49382.42578103;
}

int main() {
  // Problemgr��e und Anzahl und Gr��e der Zeitschritte definieren
  constexpr int nrOfParticles = 16384;
  constexpr int nrRuns =
      10; // Anzahl der L�ufe und der Zeitschritte der Simulation
  constexpr int skipRuns =
      3; // Anzahl der Messungen, die nicht in Mittelwert ber�cksichtigt werden
  constexpr float dt = 0.01f; // L�nge eines Zeitschrittes

  Particle *partikel_start = new Particle[nrOfParticles];
  Particle *partikel = new Particle[nrOfParticles];
  copyParticles(partikel_start, partikel, nrOfParticles);

  // Initiaslisierung der Partikel mit Zufallswerten
  initParticles(partikel_start, nrOfParticles);

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
    copyParticles(partikel_start, partikel, nrOfParticles);
    const double tStart = omp_get_wtime(); // Start der Zeitmessung
    MoveParticles(nrOfParticles, partikel,
                  dt);                   // Funktion, die optimiert werden soll
    const double tEnd = omp_get_wtime(); // Ende der Zeitmessung

    float checksum = calcChecksum(partikel, nrOfParticles);

    runtimeStep[run] = tEnd - tStart;
    GFlopsStep[run] = NrOfGFLOPs / runtimeStep[run];
    if (run >= skipRuns) { // Berechnung Mittelwerte
      meanRuntime += runtimeStep[run];
      meanGFlops += GFlopsStep[run];
    }

    printf("Run %d: Runtime: %f03,\t GFLOPS/s: %f01, \t %.4f \t %s\n", run,
           runtimeStep[run], GFlopsStep[run], checksum,
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

  delete[] partikel;
}