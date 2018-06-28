# Distributed Training

## Preparation of Datasets for n Workers
Links part of the training data to seperate folder from which the data reader should take the files.
```
WORKER_DIR='/work/projects/Project00755/datasets/imagenet/tfrecords_splitted/2_worker/worker_2'
mkdir -p ${WORKER_DIR}
cd ${WORKER_DIR}
for num in {00512..01023}
do
  ln /work/projects/Project00755/datasets/imagenet/tfrecords/train-${num}-of-01024 .
done
```