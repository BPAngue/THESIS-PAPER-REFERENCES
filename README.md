# Contributing Guidelines

Thank you for contributing to this project!  
To maintain a clean and organized workflow, please follow the guidelines below when pushing new code or submitting pull requests.

---

## Branch Naming Convention

When contributing, **always create a new branch** using the following format: PSO-V<version_num>/<changes_done>

### Example
- PSO-V1.2/added-voting-mechanism
- PSO-V1.3/fixed-threading-issue
- PSO-V1.4/updated-ui-components


### Notes
- Replace `<version_num>` with the current project version (e.g., `1.2`, `2.0`).
- Replace `<changes_done>` with a short, descriptive summary of the modification.
- Use lowercase letters and hyphens (`-`) instead of spaces.
- Always pull the latest changes from the `main` branch before creating a new one.

---

## Commit Message Convention

Use **clear and descriptive commit messages** that explain what your change does.  
Follow this structure: <type>: <short summary>


### Common Types

| Type | Description |
|------|--------------|
| feat | Added a new feature |
| fix | Fixed a bug or issue |
| docs | Updated or added documentation |
| refactor | Refactored code without changing functionality |
| style | Code style changes (formatting, naming, etc.) |
| test | Added or updated tests |
| chore | Minor maintenance tasks |

### Example
- feat: added fitness-based voting system for PSO
- fix: resolved thread synchronization issue
- docs: updated README with installation steps

