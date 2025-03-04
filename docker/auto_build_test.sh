#!/bin/bash

# Print commands as they're executed
set -x

# Function to test if the container works
test_container() {
    local image_name=$1
    echo "Testing $image_name..."
    # Run a simple test that ONLY checks if TA-Lib is properly installed
    docker run --rm --entrypoint python $image_name -c "import talib; print('TA-Lib version:', talib.__version__)"
    return $?
}

# Try the first approach with pre-built wheel
echo "=== Method 1: Building with pre-built wheel ==="
if docker build -t btsm:method1 -f docker/Dockerfile .; then
    echo "Docker build with pre-built wheel successful!"
    if test_container "btsm:method1"; then
        echo "Container works! Using method 1."
        # Tag as latest
        docker tag btsm:method1 btsm:latest
        exit 0
    else
        echo "Container built, but TA-Lib doesn't work. Trying method 2."
    fi
else
    echo "Docker build with pre-built wheel failed. Trying method 2."
fi

# Try the second approach with conda
echo "=== Method 2: Building with conda ==="
if docker build -t btsm:method2 -f docker/Dockerfile.conda .; then
    echo "Docker build with conda successful!"
    if test_container "btsm:method2"; then
        echo "Container works! Using method 2."
        # Tag as latest
        docker tag btsm:method2 btsm:latest
        exit 0
    else
        echo "Container built, but TA-Lib doesn't work."
        exit 1
    fi
else
    echo "Both Docker build methods failed!"
    exit 1
fi