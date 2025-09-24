## Description
This is Nextflow module built around image registration code which is based on palom. It allows to register H&E to fluorescent images with single cell resolution. It works for ndpi and tif images with any axis order, and outputs registered images, as well as (optionally) cropped parts of registered images to check quality of alignment. It does not ouptut any suingle transformation matrices (due to the nature of the palom - it aligns tiles of the image and then stitch them together).

## Usage
One can use it either locally usingthe docker image from quay.io/cellgeni/palom-env:latest or on Sanger LSF using singularity image stored in nfs. Examples of using module can be found in *run_module_local.sh* *and run_module_lsf.sh*. Before running you will need to prepare a file with paths of the images and the sample names, see example in *paths.txt* (first columns - sample name, second - path to H&E image, third - path to DAPI/fluorescent image)

## Comments and structure
At the moment the module structure is next: the process is stored in *modules/palom/register_multimodal_palom.nf*, subworkflow which uses only this module is in *subworkflows/run_palom.nf* and *main.nf* is used to run this subworkflow as workflow

## Future work
The current structure is quite chaotic (but it works!), so one of the optimisation steps can be to convert it to nf-core standards
