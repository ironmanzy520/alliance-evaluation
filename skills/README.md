# Optional workflow skills

The main workflow does not depend on skills. Three instruction-only skills guide agents through the same program and contracts:

- [alliance-prepare-data](alliance-prepare-data/SKILL.md): establish input mappings, preview and convert dialogue, and create a plan only when needed and explicitly scoped.
- [alliance-run](alliance-run/SKILL.md): demonstrate the pipeline without a model or run an authorised evaluation.
- [alliance-review](alliance-review/SKILL.md): explain saved results and statistics, or perform a scoped read-only review.

## Discovery and installation

In the source repository, relative symlinks under `.agents/skills/` point to these directories. Codex supports this discovery location and symlinks; see the [official skill documentation](https://learn.chatgpt.com/docs/build-skills). Other agents can register the same directories or read a supplied `SKILL.md` path. They need local command and file access; automatic discovery is host-specific.

Keep each skill with the repository rather than copying its `SKILL.md` alone. Resolve references from the skill's real directory. Links open files; section names appear in the surrounding text instead of relying on previewer anchor support.

The wheel installs the program and runtime resources, not agent skills. The source distribution includes the skills for optional registration.

Skills organise preparation, execution and review; they do not add scoring logic, converters or model-call permissions. Inputs and runs need not be moved into the code directory. Model calls, data transmission and costs must still match the user's authorisation. Follow the shared [agent and maintenance guide](../AGENTS.md), including “Skill and code standards”, rather than duplicating those standards in each skill.
