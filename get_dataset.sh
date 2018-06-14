if [ ! -d "datasets/monet2photo" ]; then
    mkdir "datasets/monet2photo"
    wget "https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/monet2photo.zip"
    unzip "monet2photo.zip" -d ./datasets/
    rm "monet2photo.zip"
fi
