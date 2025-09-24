nextflow.enable.dsl=2

include { REGISTER_MULTIMODAL_PALOM } from '../modules/palom/register_multimodal_palom.nf'

workflow RUN_PALOM {
  take:
    // tuples: (sample_id, he_path, mip_path, outdir)
    samples_ch

  main:
    // Stage the Python script from the repo so each task gets its own copy
    def palom_script = file("${projectDir}/bin/RegisterOneImage_palom.py")

    samples_with_script = samples_ch.map { sid, he, mip, outdir ->
      tuple(sid, he, mip, palom_script, outdir)
    }

    REGISTER_MULTIMODAL_PALOM(samples_with_script)
}
