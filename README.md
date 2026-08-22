# Iron City — Graphic Novel Pipeline

A modular pipeline for generating graphic novel panels from chapter plans using AI.

## Architecture

The pipeline has six stages, each a separate module:

1. **Producer** (`src/producer.py`) — Generates Chapter Plans from narrative synopses using GPT-4o structured output.
2. **Parser** (`src/pipeline/parser.py`) — Validates Chapter Plans (4-stage) and emits self-contained PanelSpecs.
3. **Compiler** (`src/pipeline/compiler.py`) — Assembles 8-layer prompts from PanelSpecs (Style → Shot/Mood → Environment → Character → Scene Prompt → Exclusions → Negative Space → Reference Descriptions).
4. **Backend** (`src/pipeline/backend.py`) — Stateless adapter wrapping gpt-image-2 for image generation.
5. **Orchestrator** (`src/pipeline/orchestrator.py`) — Coordinates the full pipeline: compile → generate → post-process → write → provenance.
6. **Provenance** (`src/pipeline/provenance.py`) — Append-only JSONL store tracking every generation attempt.

## Quick Start

### Prerequisites

- Python 3.12+
- OpenAI API key (gpt-4o, gpt-4o-mini, and gpt-image-2 access)

### Setup

```bash
# Clone the project
cd iron-city

# Install dependencies
pip install -r requirements.txt

# Set your API key
export OPENAI_API_KEY="sk-your-key-here"

# (Optional) Or use a .env file
cp .env.example .env
# Edit .env with your key
```

### Usage

All commands run via `PYTHONPATH=src python -m pipeline.cli` from the project root.

```bash
# 1. Generate a Chapter Plan from a synopsis
PYTHONPATH=src python -m pipeline.cli produce --chapter 1 --synopsis "Alyssa wakes up in her apartment..."

# Or from a file
PYTHONPATH=src python -m pipeline.cli produce --chapter 1 --synopsis-file synopsis.txt

# 2. Parse the Chapter Plan into PanelSpecs
PYTHONPATH=src python -m pipeline.cli parse --chapter 1

# 3. Generate panel images
# Full chapter:
PYTHONPATH=src python -m pipeline.cli generate --chapter 1

# Single page:
PYTHONPATH=src python -m pipeline.cli generate --chapter 1 --page 1

# Single panel:
PYTHONPATH=src python -m pipeline.cli generate --chapter 1 --panel c01_pg1_l02_pn01

# 4. Regenerate a panel with feedback
# With director's note (full pipeline re-run):
PYTHONPATH=src python -m pipeline.cli regenerate --chapter 1 --panel c01_pg1_l02_pn03 \
    --feedback "Make the lighting warmer"

# Backend-only (reuse last scene prompt, faster):
PYTHONPATH=src python -m pipeline.cli regenerate --chapter 1 --panel c01_pg1_l02_pn01

# Force full pipeline without feedback:
PYTHONPATH=src python -m pipeline.cli regenerate --chapter 1 --panel c01_pg1_l02_pn01 --full-pipeline

# 5. Check status
# List all panels with generation status:
PYTHONPATH=src python -m pipeline.cli list panels --chapter 1

# Show provenance for a specific panel:
PYTHONPATH=src python -m pipeline.cli list provenance --panel c01_pg1_l02_pn03

# Chapter status summary:
PYTHONPATH=src python -m pipeline.cli status --chapter 1
```

### Verbose mode

Add `--verbose` to any command for debug-level logging:

```bash
PYTHONPATH=src python -m pipeline.cli generate --chapter 1 --page 1 --verbose
```

## Project Structure

```
iron-city/
├── project.yaml              # Root config (model, quality, seed, etc.)
├── style.yaml                # Visual style guide
├── requirements.txt          # Python dependencies
├── .env.example               # API key template
│
├── characters/                # Character YAML + reference images
│   ├── alyssa/
│   │   ├── alyssa.yaml
│   │   ├── ref_front.png
│   │   └── ref_three_quarter.png
│   └── hood/
│       └── hood.yaml
│
├── environments/             # Environment YAML + reference images
│   ├── city_exterior/
│   │   ├── city_exterior.yaml
│   │   └── ref_establishing.png
│   ├── city_bar_interior/
│   └── alyssa_apartment/
│
├── layouts/                  # Layout templates
│   ├── layout_01.yaml         # 2-panel
│   ├── layout_02.yaml         # 3-panel
│   └── layout_03.yaml         # 7-panel (2-3-2 grid)
│
├── chapters/                 # Generated Chapter Plans (YAML)
│   └── chapter_1.yaml
│
├── schemas/                  # JSON Schema (Draft-07) validators
│   ├── project.schema.json
│   ├── chapter_plan.schema.json
│   ├── character.schema.json
│   └── ...
│
├── output/                   # Generated output (gitignored)
│   ├── *.panelspec.json       # PanelSpec files
│   ├── *_attempt_*.png       # Generated panel images
│   ├── *.provenance.jsonl    # Provenance records
│   └── archive/               # Archived previous attempts
│
└── src/
    ├── producer.py            # Chapter Plan Producer
    └── pipeline/
        ├── cli.py             # Command-line interface
        ├── config.py          # Config loader
        ├── parser.py          # Chapter Plan Parser
        ├── compiler.py        # Prompt Compiler
        ├── backend.py         # Image Generation Backend
        ├── orchestrator.py    # Generation Orchestrator
        └── provenance.py      # Provenance Store
```

## Running Tests

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

## Design Document

See `DESIGN.md` for the full architecture specification.
