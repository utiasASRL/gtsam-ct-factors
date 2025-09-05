xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | docker exec -i gtsam-ct-factors-dev-container-1 xauth nmerge -
