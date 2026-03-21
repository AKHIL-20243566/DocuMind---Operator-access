Run the mandatory pre-completion checklist on all work done in this conversation before we call the task complete. Work through each step carefully:

---

## Step 1 — Debug
Re-read every file that was changed in this task. For each one:
- Check for syntax errors, missing imports, undefined variables
- Trace the execution path end-to-end (input → processing → output)
- Identify any unhandled exceptions, missing null checks, or edge cases (empty strings, None, missing dict keys)
- Verify async/await is used correctly where applicable

List any bugs found.

---

## Step 2 — Recheck
- Confirm the changes actually solve the original requirement as stated
- Check for unintended side-effects on other parts of the codebase that weren't changed
- Verify no environment variable was replaced with a hardcoded value
- For backend changes: confirm all new endpoints apply `sanitize_input`, `check_prompt_injection`, `check_rate_limit`, and `verify_api_key` as appropriate
- For frontend changes: confirm API calls handle loading, error, and success states

List any mismatches or regressions found.

---

## Step 3 — Self-Heal
For every issue found in steps 1 and 2:
- Fix it now, do not defer
- After fixing, re-check that the fix is correct and didn't introduce a new issue
- Update `requirements.txt` or `package.json` if new dependencies were added

List every fix applied.

---

## Step 4 — Security Audit
Check each of the following and mark PASS or FLAG:

**Backend**
- [ ] User input goes through `sanitize_input()` (backend/security.py)
- [ ] Chat queries go through `check_prompt_injection()` (backend/security.py)
- [ ] Mutating endpoints call `check_rate_limit()` (backend/security.py)
- [ ] Protected endpoints call `verify_api_key()` (backend/security.py)
- [ ] No secrets or API keys are hardcoded — env vars only
- [ ] File uploads validate extension and size (SUPPORTED_TYPES, 10 MB limit)
- [ ] No path traversal risk from user-supplied filenames

**Frontend**
- [ ] No `dangerouslySetInnerHTML` with unsanitized content
- [ ] No sensitive data in localStorage or console.log
- [ ] API errors are caught and shown as friendly messages, not raw traces
- [ ] No hardcoded backend URLs (use env vars / import.meta.env)

**General**
- [ ] No debug print()/console.log() left with sensitive data
- [ ] .env is in .gitignore and not committed

---

## Final Output
After completing all steps, output a summary in this format:

```
TASK COMPLETE — Pre-Completion Report
======================================
Bugs fixed:       <list or "none">
Regressions:      <list or "none">
Self-heals:       <list or "none">
Security flags:   <list or "all clear">
Status:           READY / NEEDS ATTENTION
```
