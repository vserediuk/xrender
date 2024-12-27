#include <iostream>
#include "engine.h"

int main(int argc, char* argv[]) {
    VKEngine engine;

    try {
        engine.run();
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
