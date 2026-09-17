---
description: "Generate a technical specification document for a planned change following the project meta-spec template"
agent: "agent"
argument-hint: "What change to spec (e.g., 'extract reward system into RewardCalculator class')"
---

Generate a technical specification following [docs/meta-spec.md](../../docs/meta-spec.md). Save it to `docs/specs/<slug>.md` where `<slug>` is a kebab-case name derived from the task.

## Process

1. **Explore** the relevant code in read-only mode to understand current state, callers, and dependencies.
2. **Draft** the spec with all required sections from the meta-spec:
   - Goals & Non-Goals
   - Technical Decisions & Stack
   - Project Structure & Git Workflow
   - Testing & Conformance
   - Sequence & Workstreams (numbered execution phases)
   - Target Files (concrete list of files/directories that will change)
   - Commands & Code Style
   - Three-Tier Actions (Always Do / Ask First / Never Do)
   - Rejection Criteria
3. **Present** the spec for review. Do NOT implement any code changes.

## Constraints

- This prompt is planning only — produce the spec document, not the implementation.
- Be specific in Target Files — list exact file paths, not broad directories.
- Each Sequence step should be small enough to implement and validate independently.
- Reference existing docs rather than duplicating their content.
