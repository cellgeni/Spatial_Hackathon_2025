process REGISTER_MULTIMODAL_PALOM {
  tag "${sample_id}"

  publishDir "${outdir}",
             mode: 'copy',
             overwrite: true

  input:
  tuple val(sample_id), val(he_path), val(mip_path), path(palom_script), val(outdir)

  output:
  path "${sample_id}", emit: outdir_path

  shell:
  '''
  set -euo pipefail

  # Make caches writable and block host Python paths
  export HOME=/tmp
  export XDG_CACHE_HOME=/tmp/.cache
  export MPLCONFIGDIR=/tmp/mpl
  export NUMBA_CACHE_DIR=/tmp/numba_cache
  export PYTHONNOUSERSITE=1
  export PYTHONDONTWRITEBYTECODE=1
  unset PYTHONPATH || true
  mkdir -p "$HOME" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"

  # Work dir subfolder for this sample
  mkdir -p "!{sample_id}"

  # Run the script from the module 'bin' staged by Nextflow
  /opt/conda/envs/palom/bin/python -s "!{palom_script}" \
    "!{sample_id}" \
    "!{he_path}" \
    "!{mip_path}" \
    "!{sample_id}"

  echo "=== Produced files (up to 2 levels) ==="
  find "!{sample_id}" -maxdepth 2 -type f -printf '%p\t%k KB\n' || true

  # Hard-fail if no files were created
  if [ -z "$(find "!{sample_id}" -type f -print -quit)" ]; then
    echo "ERROR: No output files were created in !{sample_id}" >&2
    exit 2
  fi
  '''
}
