# 🚀 Git Commit Template

## 📝 `<type>(<scope>): <short summary>`
- **`<type>`**: Type of commit (e.g., `feat`, `fix`, etc.)
- **`<scope>`**: Scope of the commit (e.g., `common`, `ui`, etc.)
- **`<short summary>`**: A concise summary in present tense. 
    - Do not capitalize the first letter.
    - No period at the end.
    - Keep it under 50 characters.

---

## 🛠️ Commit Types (`<type>`)
- **`feat`**: ✨ New feature
- **`fix`**: 🐛 Bug fix
- **`docs`**: 📚 Documentation changes
- **`style`**: 🎨 Code style changes (e.g., formatting)
- **`refactor`**: 🔄 Code refactoring without fixing bugs or adding features
- **`perf`**: ⚡ Performance improvements
- **`test`**: 🧪 Adding or updating tests
- **`build`**: 🏗️ Changes to build system or dependencies
- **`ci`**: 🤖 Changes to CI configuration or scripts
- **`chore`**: 🔧 Miscellaneous tasks (e.g., version bump)
- **`revert`**: ⏪ Reverting a previous commit

---

## 📂 Commit Scopes (`<scope>`)
- **`common`**: Shared components
- **`core`**: Core functionality
- **`ui`**: User interface
- **`api`**: API endpoints
- **`auth`**: Authentication and authorization
- **`database`**: Database and migrations
- **`i18n`**: Internationalization and localization
- **`config`**: System configuration
- **`tests`**: Testing

---

## ✍️ Summary (`<short summary>`)
- Use present tense (e.g., "add", not "added").
- Do not capitalize the first letter.
- No period at the end.
- Keep it under 50 characters.
- Clearly describe what has changed.

---

## ⚠️ Breaking Changes
- Start with `BREAKING CHANGE:`.
- Describe the change.
- Explain the reason and provide migration instructions if needed.

---

## ✅ Example Commit
```markdown
feat(auth): implement JWT authentication
```

---

## 🔑 Important Notes
- Commit a reasonable amount of changes, focusing on a specific purpose.
- Review your code before committing.
- Avoid committing commented-out code unless necessary (and explain why).
- Do not commit temporary or compiled files.
- Do not commit sensitive information (e.g., passwords, API keys, etc.).  (Quên là bay màu)
- Write your commits in English please
