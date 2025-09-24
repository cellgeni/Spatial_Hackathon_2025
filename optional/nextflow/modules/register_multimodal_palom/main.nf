nextflow.enable.dsl=2

include { RUN_PALOM } from './subworkflows/run_palom.nf'

// --- Params & defaults ---
params.samplesheet         = params.samplesheet ?: "${projectDir}/samples.tsv"
params.default_outdir_base = params.default_outdir_base ?: "${workflow.launchDir}/results"

// --- Resolve & check ---
def samples_file = file(params.samplesheet)
if( !samples_file.exists() )
    error "Samplesheet not found: ${samples_file}"

// Build tuples: (sample_id, he_path, mip_path, outdir)
Channel
  .from( samples_file.readLines() )
  .filter { it?.trim() && !it.trim().startsWith('#') }
  .map { line ->
      def cols = line.split(/\t/)
      if( cols.size() < 3 ) error "Invalid line (need ≥3 tab-separated columns): ${line}"
      def sid  = cols[0].trim()
      def he   = cols[1].trim()
      def mip  = cols[2].trim()
      def base = (cols.size() >= 4 && cols[3].trim()) ? cols[3].trim() : params.default_outdir_base
      def out  = "${base}/${sid}"
      tuple(sid, he, mip, out)
  }
  .set { SAMPLES }

// --- Entry workflow ---
workflow {
  RUN_PALOM(SAMPLES)
}
