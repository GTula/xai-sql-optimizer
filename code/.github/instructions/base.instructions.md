---
description: Backend development guidelines
applyTo: "**"
---

# Project Rules

## Scope
- These rules apply to all tasks in this repository.
- Prefer small, incremental changes over broad refactors.

## Architecture
- Do not move modules between packages unless explicitly requested.


## Code Style
- Keep it simple.
- Avoid duplication.
- Use clear, explicit names.
- Preserve existing public behavior unless the task requests a change.

## Error Handling
- Validate inputs early.
- Handle edge cases explicitly.
- Raise or return clear, actionable errors.

## Testing Expectations
- For code changes, run the relevant test or command to verify behavior.
- If tests fail for pre-existing reasons, report it clearly.

## Documentation Rules
- Keep `README.md` brief (project summary + setup/run steps only).
- Keep technical details in `docs/`.
- On every new implementation or behavior change, update `docs/DOCUMENTACION_ARCHIVOS.md` to reflect the current state.

## Prompt Traceability (Mandatory)
- Maintain a folder `prompts/` at repository root for AI prompt traceability.
- For each requirement worked with AI, create/update a prompt file using this naming convention:
	- `REQ_001_Catalog.md`
	- `REQ_002_Optimizer.md`
	- `REQ_003_IndexBTree.md`
- Each `prompts/REQ_###_*.md` should include:
	- pre-implementation decision analysis,
	- requirement summary,
	- constraints/assumptions,
	- final prompt used,
	- decisions taken,
	- follow-up tasks.

### Pre-Implementation Decision Analysis (Mandatory)
Before implementing any requirement, the corresponding `prompts/REQ_###_*.md` must document:
1. Proposed solution and why it is considered the recommended option.
2. At least one alternative considered.
3. Trade-offs (complexity, maintainability, performance, risk).
4. Final decision and rationale.

### Senior Decision Review in Chat (Mandatory)
Before coding each implementation, provide a short decision review in chat that includes:
1. Whether the current proposal is recommended or not.
2. Better alternatives (if any) and why.
3. Main risks and edge cases.
4. Final recommendation (go / adjust) before implementation.

This chat decision review must be reflected in `prompts/REQ_###_*.md` under the pre-implementation decision analysis.

## Delivery Checklist
Before finishing a task:
1. Code updated and validated.
2. `docs/DOCUMENTACION_ARCHIVOS.md` updated if behavior changed.
3. `prompts/REQ_###_*.md` created or updated for the requirement, including pre-implementation decision analysis.
4. Senior decision review communicated in chat before coding and captured in the corresponding `prompts/REQ_###_*.md`.
5. Note any open risks or pending improvements.