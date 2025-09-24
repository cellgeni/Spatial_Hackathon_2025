nextflow run main.nf \
  -profile lsf_singularity \
  --samplesheet "/lustre/scratch127/cellgen/cellgeni/sm57/hackathon_0925/module_image_alignment/nf-module/test_paths_2.txt" \
  -process.clusterOptions '-G team283'