# Spatial & Single-Cell Hackathon — CLI Scripts

This repo collects small CLI tools produced during the hackathon. The core expectation is simple:

- Put your **CLI script** in `bin/`.
- Add a **matching atomic Dockerfile** in `docker/` that installs only what’s needed to run the CLI.
- If you need extra tooling, create a separate **extended** Dockerfile that builds on top of the atomic image.

## Atomic image rule (practical)

- Atomic images should contain **only runtime dependencies** for the CLI.
- It’s **OK to use `pip` in atomic images** — just install the minimal packages needed to run the tool.
- Don’t add large, unrelated stacks (e.g., Java, ImageJ, Nextflow) to atomic images.
- If you need extras, make an **extended multistage Dockerfile** that `FROM` the atomic image and layers additional tools.

## Optional extras

- Nextflow / nf-core modules are optional. If you build them, put them under `optional/nextflow/` with a short README.
- Extended images (e.g., with OME-Zarr, Java, Nextflow) should live alongside atomic ones as separate `*.extend.Dockerfile` files.

## Repository layout

```bash
├── bin/ # CLI entrypoints (bash/python), executable
│ ├── curate-spatial
│ ├── submit-bia
│ └── integrate-files
├── docker/ # One Dockerfile per CLI (atomic)
│ ├── curate-spatial.Dockerfile
│ ├── submit-bia.Dockerfile
│ ├── integrate-files.Dockerfile
│ └── <name>.extend.Dockerfile # optional: heavier variants built FROM the atomic image
├── templates/ # Schemas/examples for curation work
│ ├── bia-metadata.schema.json
│ └── sample-metadata.yaml
├── optional/
│ └── nextflow/ # optional modules/pipelines (not required)
└── README.md
```


## Contribution checklist

1. Add your CLI to `bin/NAME` and make it executable.
2. Create `docker/NAME.Dockerfile`:
   - Install only the minimal runtime deps (OK to `pip install` the needed libs).
   - No unrelated heavyweight tools.
3. If you need extras, create `docker/NAME.extend.Dockerfile` that `FROM ghcr.io/<org>/NAME:<tag>` (or your local atomic tag) and adds those tools.
4. Update this README (or your tool’s mini-README) with a one-paragraph description and CLI `--help`.

## Licensing

MIT by default (teams may adjust if required).
