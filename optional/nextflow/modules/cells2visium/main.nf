process CELLS2VISIUM {
    tag "$meta.id"
    label 'process_single'

    container "docker.io/cameronlloyd/cells2visium:0.0.1"

    input:
    tuple val(meta), path(image), path(spaceranger), val(sample)

    output:
    tuple val(meta), path("${prefix}/*.json"), emit: json
    tuple val(meta), path("${prefix}/*.tif") , emit: image
    path "versions.yml"                     , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args    = task.ext.args ?: ''
    def prefix   = task.ext.prefix ?: "${meta.id}"
    def VERSION = '0.0.1'
    """
    cells2visium \\
        ${args} \\
        ${image} \\
        ${spaceranger} \\
        ${sample} \\
        ${prefix}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        cells2visium: ${VERSION}
    END_VERSIONS
    """

    stub:
    def args    = task.ext.args ?: ''
    def prefix   = task.ext.prefix ?: "${meta.id}"
    def VERSION = '0.0.1'
    """
    echo $args
    
    mkdir "${prefix}"
    touch "${prefix}/${prefix}.json"
    touch "${prefix}/${prefix}_norm_image.tif"

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        cells2visium: ${VERSION}
    END_VERSIONS
    """
}
