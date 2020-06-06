CLANG_FLAGS="-march=native -O3 -DNDEBUG -Wall -Wextra -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize -fsave-optimization-record"
ICPC_FLAGS="-march=native -O3 -DNDEBUG -diag-file=stderr -qopt-report=5 -qopt-report-phase=vec"
GCC_FLAGS="-march=native -O3 -DNDEBUG -Wall -Wextra"

case $1 in
    intel)
        COMPILER=icpc
        CMAKE_FLAGS=${ICPC_FLAGS}
        ;;
    gcc)
        COMPILER=g++
        CMAKE_FLAGS=${GCC_FLAGS}
        ;;
    clang)
        COMPILER=clang++
        CMAKE_FLAGS=${CLANG_FLAGS}
        ;;
    *)
        echo "Missing positional argument, expected \`intel/gcc/clang\`"
        exit 1
esac

cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_VERBOSE_MAKEFILE=ON -DCMAKE_CXX_COMPILER=${COMPILER} -DCMAKE_CXX_FLAGS_RELEASE="${CMAKE_FLAGS}" -Bbuild/
cmake --build build/
