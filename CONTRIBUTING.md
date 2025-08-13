# Contributing Guide

We accept improvements and suggestions to this repository.
If you'd like to contribute, please follow these steps:

1. Fork the repository on GitHub
2. Create a new branch for your feature or bugfix

   ```bash
   git checkout -b feature-name
   ```

3. Make your changes and commit them with clear, descriptive commit messages
4. Push your changes to your fork

   ```bash
   git push origin feature-name
   ```

5. Open a Pull Request on GitHub against the `main` branch
6. Wait for review and address any feedback

Please ensure your changes:

- Follow the existing code style
- Include appropriate documentation updates
- Add tests if introducing new features
- Pass all existing tests

## Internal Developers

### Initial Setup

1. Clone the repository from the original GitLab repo:

2. Add GitHub remote:

   ```bash
   git remote add github git@github.com:wenglor/AI-module-onnx.git
   ```

### Making Changes

1. Create feature branches from `develop`:

   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/your-feature-name
   ```

2. Make your changes, commit them, and push to GitLab:

   ```bash
   git push origin feature/your-feature-name
   ```

3. Open a Merge Request on GitLab:
   - Source branch: your feature branch
   - Target branch: `develop`
   - Add description and any relevant labels
   - Request review from team members

4. After approval, merge your changes into `develop`

### Publishing to GitHub

1. Create a release branch from `develop`:

   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b release/vX.Y.Z
   ```

2. Push the release branch to GitHub:

   ```bash
   git push github release/vX.Y.Z
   ```

3. Open a Pull Request on GitHub and merge:
   - Source branch: `release/vX.Y.Z`
   - Target branch: `main`

4. Synchronize the changes back to the Gitlab repository

### Synchronizing Repositories

1. Update local main from GitHub:

   ```bash
   git checkout main
   git pull github main
   ```

2. Push updates to GitLab:

   ```bash
   git push origin main
   ```

3. Synchronize develop branch:

   ```bash
   git checkout develop
   git merge main
   git push origin develop
   ```
